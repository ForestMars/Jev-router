# Ultra-Low Latency Cascade Router

An in-process, high-throughput Rust routing engine designed for sub-5ms decision pathways (P99) across discriminative (Jev) and generative (LLM) processing tiers[cite: 1].

---

## 1. Architecture Overview

The system employs a **5-tier deterministic fallback cascade**[cite: 1]. Each stage short-circuits the pipeline as soon as confidence constraints are satisfied, routing simple traffic in microsecond space while reserving heavier transformers for ambiguous or semantically complex prompts[cite: 1].

                  [ Incoming Prompt Payload ]
                               │
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Tier 1: Static Rules & Regex Guardrails (< 0.1ms)        │
 │ - Exact signature hashes, length bounds, regex sets      │
 └─────────────────────────────┬────────────────────────────┘
                               │ (Unresolved)
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Tier 2: FastText Subword Classifier (0.2ms - 0.4ms)      │
 │ - Character n-gram embeddings & calibrated margin score  │
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
 │ Tier 4: Heavy Model Fallback (20ms - 50ms)               │
 └─────────────────────────────┬────────────────────────────┘
                               │ (Unresolved)
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Tier 5: Frontier Model Escalation (Variable)             │
 └──────────────────────────────────────────────────────────┘

---

## 2. Performance SLA & Latency Budget

| Tier | Engine / Mechanism | Latency Budget (P99) | Primary Purpose | Exit Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | regex::RegexSet & Hash Maps | < 0.1ms | Hard rules, static pings, payload bounds | ~25–35% |
| **Tier 2** | FastText Subword (libfasttext) | 0.2ms - 0.4ms | Character n-gram probabilistic matching & typo handling | ~40–50% |
| **Tier 3** | ModernBERT / Harrier / Oryn | 1.5ms - 8.0ms | Dense embedding evaluation & in-process model scoring[cite: 1] | ~15–25% |
| **Tier 4** | Secondary Heavy Model | 20ms - 50ms | High-dimensional structural classification[cite: 1] | ~3–5% |
| **Tier 5** | Frontier LLM | Variable | Final escalation for non-deterministic prompts[cite: 1] | <1% |

---

## 3. Directory Topology

src/
├── config/
│   ├── mod.rs          # Configuration loader
│   └── schema.rs       # Deserialization schemas for router thresholds
├── logging/
│   └── mod.rs          # Zero-allocation telemetry & tracing wrappers
├── router/
│   ├── cascade.rs      # Main 5-tier pipeline dispatcher[cite: 1]
│   ├── tier1.rs        # Tier 1: Static regex & rule guardrails[cite: 1]
│   ├── tier2.rs        # Tier 2: FastText subword classifier
│   ├── tier3.rs        # Tier 3: ModernBERT / Harrier / Oryn evaluator[cite: 1]
│   ├── tier4.rs        # Tier 4: Heavy fallback runner[cite: 1]
│   └── tier5.rs        # Tier 5: Frontier escalation runner
└── tier2/              # Sub-engines for Tier 3 evaluation[cite: 1]
    ├── calibration.rs  # Probability calibration utilities
    ├── trait.rs        # Core model traits & contract definitions
    ├── inprocess/      # Native ONNX / C++ in-process engines
    │   ├── fasttext.rs
    │   ├── modernbert.rs
    │   ├── harrier.rs
    │   └── oryn.rs
    └── remote/         # gRPC/RPC remote evaluators
        ├── client.rs
        └── proto.rs

---

## 4. Hardware Allocations & Dependencies

To maintain zero garbage collection pauses and sub-millisecond execution, the data plane relies on native C++ and Rust SIMD optimizations:

- Allocator Override: System glibc malloc is replaced globally with `mimalloc` to prevent multi-threaded thread contention under high throughput.
- Tier 1: `regex` crate utilizing RE2 linear-time semantics (no catastrophic backtracking).
- Tier 2: `fasttext` crate wrapping native C++ `libfasttext.a`.
- Tier 3: `ort` (ONNX Runtime bindings targeting AVX-512 / ARM Neon SIMD) and HuggingFace native `tokenizers` Rust library.

---

## 5. Quickstart & Testing

### Prerequisites

System dependencies required for C++ static linking:

# Debian / Ubuntu
sudo apt-get install build-essential clang

# macOS
xcode-select --install

### Build & Test

1. Train or generate the quantized FastText model binary (`router_fastpath.ftz`):
   python3 train_test.py

2. Run the full 5-tier cascade unit test suite:
   cargo test --all-targets

3. Run Criterion performance micro-benchmarks:
   cargo bench