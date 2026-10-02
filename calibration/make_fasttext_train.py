"""Build fastText training data for Tier 2 from the shared exemplars, then train and evaluate.

Usage (from the project root):
    python calibration/make_fasttext_train.py

Reads calibration/exemplars*.jsonl, writes calibration/ft_train.txt and
calibration/ft_valid.txt, trains models/tier2.bin, scores it on the validation
file, and prints predictions for the probe prompts. Exemplar files may name the
text field either "prompt" or "text"; the first one present in a record is used.

Requires: pip install fasttext-wheel "numpy<2"
"""

import glob
import json
import os
import random
from collections import Counter, defaultdict

TEXT_FIELDS = ("prompt", "text")
LABEL_FIELD = "label"
LABEL_MAP = {"jev_capable": "jev", "needs_llm": "llm"}  # Rust matches __label__jev / __label__llm
VALID_FRACTION = 0.2
SEED = 0

TRAIN_PATH = "calibration/ft_train.txt"
VALID_PATH = "calibration/ft_valid.txt"
MODEL_PATH = "models/tier2.bin"
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


def build_splits() -> None:
    by_label: dict[str, list[str]] = defaultdict(list)
    skipped = Counter()
    per_file = Counter()

    for path in sorted(glob.glob("calibration/exemplars*.jsonl")):
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
                by_label[label].append(text)
                per_file[path] += 1

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

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with open(VALID_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(valid) + "\n")

    print(f"rows read per file: {dict(per_file)}")
    print(f"wrote {len(train)} train / {len(valid)} valid")
    print(f"train classes: {dict(Counter(r.split(' ', 1)[0] for r in train))}")
    print(f"valid classes: {dict(Counter(r.split(' ', 1)[0] for r in valid))}")
    if skipped:
        print(f"skipped (unmapped label or empty text): {dict(skipped)}")


def train_and_evaluate() -> None:
    try:
        import fasttext
    except ImportError:
        raise SystemExit('fasttext not installed: pip install fasttext-wheel "numpy<2"')

    model = fasttext.train_supervised(
        input=TRAIN_PATH,
        epoch=EPOCH,
        lr=LR,
        wordNgrams=WORD_NGRAMS,
        verbose=0,
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"saved {MODEL_PATH}, labels: {model.get_labels()}")

    n, precision, recall = model.test(VALID_PATH)
    print(f"validation: n={n} precision@1={precision:.3f} recall@1={recall:.3f}")

    for prompt in PROBES:
        labels, probs = model.predict(normalize(prompt), k=2)
        scored = [(l, round(float(p), 4)) for l, p in zip(labels, probs)]
        print(f"probe {prompt!r}: {scored}")


def main() -> None:
    build_splits()
    train_and_evaluate()


if __name__ == "__main__":
    main()