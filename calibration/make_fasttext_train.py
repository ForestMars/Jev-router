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
from collections import Counter

TEXT_FIELD = "prompt"
LABEL_FIELD = "label"
LABEL_MAP = {"jev_capable": "jev", "needs_llm": "llm"}  # Rust matches __label__jev / __label__llm
VALID_FRACTION = 0.2
SEED = 0


def normalize(s: str) -> str:
    """Must match Tier2Runner::normalize_input, or train and serve will skew."""
    return " ".join(s.lower().split())


def main() -> None:
    rows, skipped = [], Counter()
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
                rows.append(f"__label__{label} {text}")

    random.Random(SEED).shuffle(rows)
    cut = int(len(rows) * (1 - VALID_FRACTION))
    train, valid = rows[:cut], rows[cut:]

    with open("calibration/ft_train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with open("calibration/ft_valid.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(valid) + "\n")

    counts = Counter(r.split(" ", 1)[0] for r in rows)
    print(f"wrote {len(train)} train / {len(valid)} valid; class counts: {dict(counts)}")
    if skipped:
        print(f"skipped (unmapped label or empty text): {dict(skipped)}")


if __name__ == "__main__":
    main()
