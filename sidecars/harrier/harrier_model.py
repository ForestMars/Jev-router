# /sidecars/harrier/harrier_model.py
"""Harrier-270M scoring model.

Encodes prompts as embeddings via a GGUF model in embedding mode
(llama.cpp), and scores by nearest-neighbor similarity against a set
of labeled exemplars loaded once at startup. Exemplars are runtime
reference data, not training data — swap in a different exemplar set
by pointing exemplars_path at another file, no retraining involved.
"""
import hashlib
import json
import math
import os
import time
from llama_cpp import Llama

import faulthandler, sys
# faulthandler.dump_traceback_later(20, exit=False, file=sys.stderr)

CACHE_VERSION = 1


class HarrierModel:
    def __init__(self, model_path: str, exemplars_path: str, n_ctx: int = 512):
        print("loading model", flush=True)

        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            embedding=True,
            verbose=False,
            pooling_type=3,  # 3 = LLAMA_POOLING_TYPE_LAST
            n_gpu_layers=1,
        )
        print(f"model loaded in {time.perf_counter() - t0:.2f}s", flush=True)

        t1 = time.perf_counter()
        exemplars = self._load_exemplars(exemplars_path)
        print(
            f"loaded {len(exemplars)} exemplars in {time.perf_counter() - t1:.3f}s",
            flush=True,
        )

        t2 = time.perf_counter()
        key = self._cache_key(model_path, n_ctx, exemplars)
        cache_path = f"{exemplars_path}.{key[:16]}.embcache.json"

        cached = self._read_cache(cache_path, key)
        if cached is not None:
            self.exemplar_vectors = cached
            print(
                f"exemplar embeddings loaded from cache in {time.perf_counter() - t2:.3f}s",
                flush=True,
            )
        else:
            self.exemplar_vectors = []
            for i, e in enumerate(exemplars):
                ti = time.perf_counter()
                self.exemplar_vectors.append((self._embed(e["prompt"]), e["label"]))
                print(f"embedded {i} in {time.perf_counter() - ti:.2f}s", flush=True)
            self._write_cache(cache_path, key, self.exemplar_vectors)
            print(
                f"all exemplars embedded and cached in {time.perf_counter() - t2:.2f}s",
                flush=True,
            )

    @staticmethod
    def _cache_key(model_path, n_ctx, exemplars):
        st = os.stat(model_path)
        h = hashlib.sha256()
        h.update(f"v{CACHE_VERSION}|{os.path.abspath(model_path)}|{st.st_size}|{st.st_mtime_ns}|{n_ctx}|".encode())
        for e in exemplars:
            h.update(json.dumps([e["prompt"], e["label"]], ensure_ascii=False).encode())
        return h.hexdigest()

    @staticmethod
    def _read_cache(path, key):
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("key") != key:
                return None
            return [(row["vec"], row["label"]) for row in data["rows"]]
        except (OSError, ValueError, KeyError):
            return None

    @staticmethod
    def _write_cache(path, key, vectors):
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(
                    {"key": key, "rows": [{"vec": v, "label": l} for v, l in vectors]},
                    f,
                )
            os.replace(tmp, path)
        except OSError as exc:
            print(f"cache write failed: {exc}", flush=True)

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