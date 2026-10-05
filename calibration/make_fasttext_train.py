"""Build fastText training data for Tier 2 from the shared exemplars, then train and evaluate.

Usage (from the project root):
    python calibration/make_fasttext_train.py

Reads calibration/exemplars*.jsonl, writes calibration/ft_train.txt and
calibration/ft_valid.txt, trains models/tier2.bin, scores it on the validation
file, and prints predictions for the probe prompts. Exemplar files may name the
text field either "prompt" or "text"; the first one present in a record is used.

Requires: pip install fasttext-wheel "numpy<2"
"""

import argparse
import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

TEXT_FIELDS = ("prompt", "text")
LABEL_FIELD = "label"
LABEL_MAP = {"jev_capable": "jev", "needs_llm": "llm"}  # Rust matches __label__jev / __label__llm
VALID_FRACTION = 0.2
SEED = 0

TRAIN_PATH = Path("calibration/ft_train.txt")
VALID_PATH = Path("calibration/ft_valid.txt")
MODEL_PATH = Path("models/tier2.bin")
EPOCH = 25
LR = 0.5
WORD_NGRAMS = 2

PROBES = (
    "What is 2+2?",
    "Write a 500-word essay on the history of Rome",
)


def normalize(s: str) -> str:
    """Must match Tier2Runner::normalize_input, or train and serve will skew."""
    return " ".join(s.lower().split())


def extract_text(rec: dict) -> str:
    for field in TEXT_FIELDS:
        value = rec.get(field)
        if value:
            return normalize(value)
    return ""


def build_splits(
    train_path: Path,
    valid_path: Path,
    holdout_path: Path | None = None,
    holdout_source: Path | None = None,
) -> None:
    by_label: dict[str, list[str]] = defaultdict(list)
    holdout: list[str] = []
    skipped = Counter()
    per_file = Counter()

    source_paths = [Path(path) for path in sorted(glob.glob("calibration/exemplars*.jsonl"))]
    if holdout_source is not None:
        holdout_source = holdout_source.resolve()
        if holdout_source not in {path.resolve() for path in source_paths}:
            raise ValueError(
                f"holdout source is not among calibration exemplar files: {holdout_source}"
            )
        if holdout_path is None:
            raise ValueError("holdout output path is required when a holdout source is selected")

    for path in source_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                label = LABEL_MAP.get(rec.get(LABEL_FIELD))
                text = extract_text(rec)
                if label is None or not text:
                    skipped[str(rec.get(LABEL_FIELD))] += 1
                    continue
                if holdout_source is not None and path.resolve() == holdout_source:
                    holdout.append(f"__label__{label} {text}")
                else:
                    by_label[label].append(text)
                per_file[path] += 1

    if holdout_source is not None and not holdout:
        raise ValueError(f"holdout source contains no usable labeled prompts: {holdout_source}")

    rng = random.Random(SEED)
    train, valid = [], []
    for label in sorted(by_label):
        texts = by_label[label]
        rng.shuffle(texts)
        n_valid = max(1, round(len(texts) * VALID_FRACTION))
        valid += [f"__label__{label} {t}" for t in texts[:n_valid]]
        train += [f"__label__{label} {t}" for t in texts[n_valid:]]

    rng.shuffle(train)
    rng.shuffle(valid)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with open(valid_path, "w", encoding="utf-8") as f:
        f.write("\n".join(valid) + "\n")

    print(f"rows read per file: {dict(per_file)}")
    print(f"wrote {len(train)} train / {len(valid)} validation to {train_path} and {valid_path}")
    print(f"train classes: {dict(Counter(r.split(' ', 1)[0] for r in train))}")
    print(f"valid classes: {dict(Counter(r.split(' ', 1)[0] for r in valid))}")
    if holdout_source is not None and holdout_path is not None:
        holdout_path.parent.mkdir(parents=True, exist_ok=True)
        with open(holdout_path, "w", encoding="utf-8") as f:
            f.write("\n".join(holdout) + "\n")
        print(f"wrote {len(holdout)} source-held-out rows to {holdout_path}")
    if skipped:
        print(f"skipped (unmapped label or empty text): {dict(skipped)}")


def train_and_evaluate(train_path: Path, valid_path: Path, model_path: Path) -> None:
    try:
        import fasttext
    except ImportError:
        raise SystemExit('fasttext not installed: pip install fasttext-wheel "numpy<2"')

    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=EPOCH,
        lr=LR,
        wordNgrams=WORD_NGRAMS,
        verbose=0,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    print(f"saved {model_path}, labels: {model.get_labels()}")

    n, precision, recall = model.test(str(valid_path))
    print(f"validation: n={n} precision@1={precision:.3f} recall@1={recall:.3f}")

    for prompt in PROBES:
        labels, probs = model.predict(normalize(prompt), k=2)
        scored = [(l, round(float(p), 4)) for l, p in zip(labels, probs)]
        print(f"probe {prompt!r}: {scored}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-source",
        type=Path,
        help="exclude this exemplar file from training and write it as a separate evaluation set",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write generated datasets and model here; required for source-held-out runs",
    )
    args = parser.parse_args()

    if args.holdout_source is not None and args.output_dir is None:
        parser.error(
            "--output-dir is required with --holdout-source to avoid overwriting normal artifacts"
        )

    if args.output_dir is None:
        train_path, valid_path, model_path = TRAIN_PATH, VALID_PATH, MODEL_PATH
        holdout_path = None
    else:
        train_path = args.output_dir / "ft_train.txt"
        valid_path = args.output_dir / "ft_valid.txt"
        model_path = args.output_dir / "tier2.bin"
        holdout_path = args.output_dir / "ft_holdout.txt" if args.holdout_source else None

    build_splits(train_path, valid_path, holdout_path, args.holdout_source)
    train_and_evaluate(train_path, valid_path, model_path)


if __name__ == "__main__":
    main()