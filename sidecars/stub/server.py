# Router System Smoke Test 

import sys

sys.path.insert(0, "../common")

from tier3_server import server, calibration, tier3_pb2


class StubModel:
    def __init__(self):
        self.model = self
        self.calibration = IdentityCalibration()
        self.model_id = "stub"

    def score(self, prompt):
        p = prompt.strip().lower()
        words = p.split()

        score = 0.5

        # Simplicity signals
        if p.endswith("?") and len(words) <= 4:
            score += 0.15
        if any(op in p for op in ("+", "-", "*", "/", "=")):
            score += 0.15
        if p.startswith(("what is", "who is", "when is", "where is")):
            score += 0.15

        # Complexity signals
        if len(words) > 25:
            score -= 0.35
        if any(k in p for k in ("essay", "write a", "explain why", "compare", "analyze")):
            score -= 0.35

        return max(0.0, min(1.0, score))


class IdentityCalibration:
    def map(self, raw):
        return raw


class StubServicer:
    def __init__(self, model, calibration, model_id):
        self.model = model
        self.calibration = calibration
        self.model_id = model_id

    def Score(self, request, context):
        raw = self.model.score(request.prompt)
        conf = self.calibration.map(raw)
        label = "jev_capable" if conf >= 0.5 else "needs_llm"

        print(
            f"[stub] prompt={request.prompt!r} "
            f"raw={raw} conf={conf} label={label}",
            flush=True,
        )

        return tier3_pb2.ScoreResponse(
            raw_score=raw,
            confidence=conf,
            label=label,
            model_id=self.model_id,
        )


if __name__ == "__main__":
    print("hello from the stub sidecar", flush=True)

    model = StubModel()
    cal = IdentityCalibration()

    server.serve(
        model,
        cal,
        "stub",
        50051,
    )