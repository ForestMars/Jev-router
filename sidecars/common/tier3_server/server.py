# sidecars/common/tier3_server/server.py 

import grpc
from concurrent import futures
from . import tier3_pb2, tier3_pb2_grpc

class Tier3Servicer(tier3_pb2_grpc.Tier3ScorerServicer):
    def __init__(self, model, calibration, model_id):
        self.model = model
        self.calibration = calibration
        self.model_id = model_id

    def Score(self, request, context):
        raw = self.model.score(request.prompt)
        conf = self.calibration.map(raw)
        label = "jev_capable" if conf >= 0.5 else "needs_llm"
        return tier3_pb2.ScoreResponse(
            raw_score=raw, confidence=conf, label=label, model_id=self.model_id
        )

def serve(model, calibration, model_id, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tier3_pb2_grpc.add_Tier3ScorerServicer_to_server(
        Tier3Servicer(model, calibration, model_id), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    print(f"harrier server listening on port {port}", flush=True)
    server.wait_for_termination()