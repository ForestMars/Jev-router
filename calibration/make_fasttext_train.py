"""Build fastText training data for Tier 2 from the shared exemplars.

Usage (from the project root):
    python calibration/make_fasttext_train.py

Reads calibration/exemplars*.jsonl, writes calibration/ft_train.txt and
calibration/ft_valid.txt. Adjust the three constants below to match the
exemplars schema; the field names here are assumptions.
"""

import glob
import json
import random
from collections import Counter, defaultdict

TEXT_FIELD = "prompt"
LABEL_FIELD = "label"
LABEL_MAP = {"jev_capable": "jev", "needs_llm": "llm"}  # Rust matches __label__jev / __label__llm
VALID_FRACTION = 0.2
SEED = 0


def normalize(s: str) -> str:
    """Must match Tier2Runner::normalize_input, or train and serve will skew."""
    return " ".join(s.lower().split())


def main() -> None:
    by_label: dict[str, list[str]] = defaultdict(list)
    skipped = Counter()

    for path in sorted(glob.glob("calibration/exemplars*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                label = LABEL_MAP.get(rec.get(LABEL_FIELD))
                text = normalize(rec.get(TEXT_FIELD, ""))
                if label is None or not text:
                    skipped[str(rec.get(LABEL_FIELD))] += 1
                    continue
                by_label[label].append(text)

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

    with open("calibration/ft_train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with open("calibration/ft_valid.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(valid) + "\n")

    train_counts = Counter(r.split(" ", 1)[0] for r in train)
    valid_counts = Counter(r.split(" ", 1)[0] for r in valid)
    print(f"wrote {len(train)} train / {len(valid)} valid")
    print(f"train classes: {dict(train_counts)}")
    print(f"valid classes: {dict(valid_counts)}")
    if skipped:
        print(f"skipped (unmapped label or empty text): {dict(skipped)}")


if __name__ == "__main__":
    main()