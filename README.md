# Ultra-Low Latency Cascade Router

An in-process, high-throughput Rust routing engine designed for sub-5ms decision pathways ($p_{99}$) across discriminative (Jev) and generative (LLM) processing tiers[cite: 1].

---

## Architecture Overview

The system uses a **5-tier deterministic fallback cascade**[cite: 1]. Each stage short-circuits the pipeline as soon as confidence constraints are met, ensuring simple queries exit in microsecond space while reserving expensive transformer passes for ambiguous prompts[cite: 1].

```text
                  [ Incoming Prompt Payload ]
                               │
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Tier 1: Static Rules & Regex Guardrails (< 0.1ms)         │
 │ - Exact signature hashes, length bounds, regex sets      │
 └─────────────────────────────┬────────────────────────────┘
                               │ (Unresolved)
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Tier 2: FastText Subword Classifier (0.2ms - 0.4ms)      │
 │ - Character n-gram embeddings & calibrated margin score   │
 └─────────────────────────────┬────────────────────────────┘
                               │ (Pass-Through / Low Margin)
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Tier 3: In-Process / Remote Evaluators (1.5ms - 8.0ms)   │
 │ - ModernBERT, Harrier, or Oryn inference engines         │
 └─────────────────────────────┬────────────────────────────┘
                               │ (Unresolved)
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Tier 4: Heavy Model Fallback                             │
 └─────────────────────────────┬────────────────────────────┘
                               │ (Unresolved)
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Tier 5: Frontier Model Escalation                        │
 └──────────────────────────────────────────────────────────┘
