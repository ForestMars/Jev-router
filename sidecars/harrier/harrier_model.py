"""Harrier-270M scoring model.

Encodes prompts as embeddings via a GGUF model in embedding mode
(llama.cpp), and scores by nearest-neighbor similarity against a set
of labeled exemplars loaded once at startup. Exemplars are runtime
reference data, not training data — swap in a different exemplar set
by pointing exemplars_path at another file, no retraining involved.
"""
import json
import math
from llama_cpp import Llama

import faulthandler, sys
# faulthandler.dump_traceback_later(20, exit=False, file=sys.stderr)


class HarrierModel:
    def __init__(self, model_path: str, exemplars_path: str, n_ctx: int = 512):
        print("loading model", flush=True)

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            embedding=True,
            verbose=False,
            pooling_type=3,  # 3 = LLAMA_POOLING_TYPE_LAST
            n_gpu_layers=1,
        )
        
        print("model loaded", flush=True)
        exemplars = self._load_exemplars(exemplars_path)
        print(f"loaded {len(exemplars)} exemplars, embedding now", flush=True)

        # self.exemplar_vectors = [
        #    (self._embed(e["prompt"]), e["label"]) for e in exemplars
        # ]

        self.exemplar_vectors = []
        for i, e in enumerate(exemplars):
            print(f"embedding {i}", flush=True)
            self.exemplar_vectors.append((self._embed(e["prompt"]), e["label"]))


    def _load_exemplars(self, path):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"no exemplars loaded from {path}")
        return rows

    INSTRUCTION = "Instruct: Retrieve semantically similar text\nQuery: "

    def _embed(self, text, is_query=False):
        if is_query:
            text = self.INSTRUCTION + text
        resp = self.llm.create_embedding(text)
        vec = resp["data"][0]["embedding"]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm else vec

    @staticmethod
    def _cosine(a, b):
        return sum(x * y for x, y in zip(a, b))  # both pre-normalized

    def score(self, prompt: str) -> float:
        query = self._embed(prompt, is_query=True)
        print(f"\n[harrier debug] prompt={prompt!r}", flush=True)
        for vec, label in self.exemplar_vectors:
            sim = self._cosine(query, vec)
            print(f"  {label:12s} sim={sim:.4f}", flush=True)

        best_jev = max(
            (self._cosine(query, v) for v, label in self.exemplar_vectors
            if label == "jev_capable"),
            default=-1.0,
        )
        best_needs_llm = max(
            (self._cosine(query, v) for v, label in self.exemplar_vectors
            if label == "needs_llm"),
            default=-1.0,
        )

        # return best_jev - best_needs_llm
        m = max(best_jev, best_needs_llm)
        e_jev = math.exp(best_jev - m)
        e_llm = math.exp(best_needs_llm - m)
            
        return e_jev / (e_jev + e_llm)