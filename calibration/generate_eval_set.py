#!/usr/bin/env python3
"""Build the calibration/evaluation set from labeled prompts.

Reads seed exemplars and any hand-labeled prompts, deduplicates on
prompt text, and writes a stratified train/calibration split to
``eval_set.jsonl``.

Typical usage:
    $ python generate_eval_set.py
    $ python generate_eval_set.py exemplars_2.jsonl
"""
import json
import random
import sys
from pathlib import Path

# --- swap exemplar files here, or pass a filename as argv[1] ---
EXEMPLARS_FILENAME = "exemplars.jsonl"
EXEMPLARS_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(EXEMPLARS_FILENAME)
MANUAL_LABELS_PATH = Path("./manual_labels.jsonl") 
OUTPUT_PATH = Path("./eval_set.jsonl")

CALIBRATION_FRACTION = 0.30
SEED = 42


def load_exemplars(path: Path) -> list[dict]:
    """Load seed exemplars from a line-delimited JSON file.

    Args:
        path: Path to a ``.jsonl`` file where each line is a JSON
            object with at least ``prompt`` and ``label`` keys.

    Returns:
        A list of row dicts, each tagged with ``source: "exemplar"``
        if not already present.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If a non-empty line is not valid JSON.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r.setdefault("source", "exemplar")
                rows.append(r)
    return rows


def load_manual(path: Path) -> list[dict]:
    """Load hand-labeled prompts from a line-delimited JSON file.

    Args:
        path: Path to ``manual_labels.jsonl``. Missing files are
            treated as empty rather than an error, since manual
            labels are optional.

    Returns:
        A list of row dicts parsed from each non-empty line, or an
        empty list if ``path`` does not exist.

    Raises:
        json.JSONDecodeError: If a non-empty line is not valid JSON.
    """
    rows = []
    if not path.exists():
        print(f"warning: {path} not found, using exemplars only")
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_split(rows: list[dict], cal_fraction: float, seed: int) -> list[dict]:
    """Split rows into train/calibration sets, stratified by label.

    Each label's rows are shuffled independently and split so that
    every class contributes proportionally to the calibration set,
    avoiding a calibration set skewed toward whichever label happens
    to have more examples.

    Args:
        rows: Row dicts, each with a ``label`` key.
        cal_fraction: Fraction of each label's rows to assign to the
            calibration split (e.g. ``0.30`` for 30%).
        seed: Random seed, for reproducible splits across runs.

    Returns:
        The same rows, each with a ``split`` key added (``"train"``
        or ``"calibration"``), train rows first.
    """
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    train, cal = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_cal = max(1, int(len(items) * cal_fraction))
        cal.extend(items[:n_cal])
        train.extend(items[n_cal:])

    for r in train:
        r["split"] = "train"
    for r in cal:
        r["split"] = "calibration"
    return train + cal


def main() -> None:
    """Build and write the eval set.

    Loads exemplars and manual labels, deduplicates on prompt text,
    applies the stratified split, and writes the result to
    ``eval_set.jsonl``. Prints a per-label/per-split row count
    summary on completion.

    Raises:
        SystemExit: If ``EXEMPLARS_PATH`` does not exist.
    """
    if not EXEMPLARS_PATH.exists():
        raise SystemExit(f"exemplars file not found: {EXEMPLARS_PATH}")

    rows = load_exemplars(EXEMPLARS_PATH)

    seen = set()
    deduped = []
    for r in rows:
        if r["prompt"] not in seen:
            seen.add(r["prompt"])
            deduped.append(r)

    split = stratified_split(deduped, CALIBRATION_FRACTION, SEED)

    with open(OUTPUT_PATH, "w") as f:
        for r in split:
            f.write(json.dumps(r) + "\n")

    counts = {}
    for r in split:
        key = (r["label"], r["split"])
        counts[key] = counts.get(key, 0) + 1

    print(f"read exemplars from {EXEMPLARS_PATH}")
    print(f"wrote {len(split)} rows to {OUTPUT_PATH}")
    for (label, s), n in sorted(counts.items()):
        print(f"  {label:15s} {s:12s} {n}")


if __name__ == "__main__":
    main()