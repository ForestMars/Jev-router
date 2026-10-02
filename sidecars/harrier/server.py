import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

from tier3_server import server, tier3_pb2
from harrier_model import HarrierModel
from calibration import IdentityCalibration


if __name__ == "__main__":
    print("hello from the harrier sidecar", flush=True)

    model = HarrierModel(
        model_path="../../models/harrier-270m/model.gguf",
        # model_path="../../models/harrier-270m/harrier-270m-q4_k.gguf",
        exemplars_path="../../calibration/exemplars.jsonl",
    )
    cal = IdentityCalibration()

    print(f"cal class: {cal.__class__.__name__}", flush=True)
    print(f"cal attrs: {vars(cal)}", flush=True)

    server.serve(model, cal, "harrier-270m", 50051)