#!/usr/bin/env python3
"""Fit a Platt scaling map from raw model scores to calibrated probabilities.

Pulls calibration-split rows from ``eval_set.jsonl``, queries a
running Tier 2 sidecar for each prompt's raw score, fits a
single-feature logistic regression (Platt scaling), and writes the
resulting ``(a, b)`` map to ``calibration.json``.

Requires a Tier 2 sidecar already running on ``SIDECAR_ADDR``.

Typical usage:
    $ python fit_map.py
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

import grpc
import sys
sys.path.insert(0, "../sidecars/common")
from tier3_server import tier3_pb2, tier3_pb2_grpc

EVAL_PATH = Path("./eval_set.jsonl")
OUTPUT_PATH = Path("../models/harrier-270m/calibration.json")
SIDECAR_ADDR = "localhost:50051"


def fetch_raw_scores(rows: list[dict]) -> np.ndarray:
    """Fetch raw model scores for a set of prompts from the live sidecar.

    Opens a gRPC channel to ``SIDECAR_ADDR`` and calls ``Score`` once
    per row, in order.

    Args:
        rows: Row dicts, each with a ``prompt`` key.

    Returns:
        An ``(n, 1)`` array of raw scores, one per row, in the same
        order as ``rows``.

    Raises:
        grpc.RpcError: If the sidecar is unreachable or a call fails.
    """
    channel = grpc.insecure_channel(SIDECAR_ADDR)
    stub = tier3_pb2_grpc.Tier3ScorerStub(channel)
    scores = []
    for r in rows:
        resp = stub.Score(tier3_pb2.ScoreRequest(prompt=r["prompt"]))
        scores.append(resp.raw_score)
    return np.array(scores).reshape(-1, 1)


def main() -> None:
    """Fit and write the calibration map.

    Loads ``eval_set.jsonl``, restricts to calibration-split rows,
    fetches raw scores from the live sidecar, fits a Platt scaling
    logistic regression against the ``jev_capable`` / ``needs_llm``
    labels, and writes the fitted ``(a, b)`` coefficients plus split
    counts to ``calibration.json``.

    Raises:
        SystemExit: If fewer than 10 calibration rows are available.
    """
    rows = [json.loads(l) for l in open(EVAL_PATH) if l.strip()]
    cal_rows = [r for r in rows if r["split"] == "calibration"]
    if len(cal_rows) < 10:
        raise SystemExit(f"need >=10 calibration rows, have {len(cal_rows)}")

    X = fetch_raw_scores(cal_rows)
    y = np.array([1 if r["label"] == "jev_capable" else 0 for r in cal_rows])

    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(X, y)

    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])

    if a <= 0:
        print(f"warning: fitted slope a={a} is non-positive; map is not monotonic")

    out = {
        "method": "platt",
        "a": a,
        "b": b,
        "n_calibration": len(cal_rows),
        "n_jev_capable": int(y.sum()),
        "n_needs_llm": int((1 - y).sum()),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"wrote {OUTPUT_PATH}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()