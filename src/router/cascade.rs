// src/router/cascade.rs
//
// Every tier boundary logs exactly one of: TRY, RESOLVED, ESCALATE.
// Every request ends with exactly one DONE line.
// Faults (timeout, rpc_error, deadline_exhausted) escalate at WARN,
// ordinary low-confidence escalations at INFO.

use std::fmt;
use std::time::{Duration, Instant};

use tracing::{info, warn};

use super::tier2_runner::{Tier2Outcome, Tier2Route, Tier2Runner};
use super::tier3_runner::{Tier3Outcome, Tier3Route, Tier3Runner};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Jev,
    Llm,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Backend::Jev => write!(f, "jev"),
            Backend::Llm => write!(f, "llm"),
        }
    }
}

#[allow(dead_code)] // read by the axum handler once it dispatches on the decision
#[derive(Debug, Clone, Copy)]
pub struct Decision {
    pub backend: Backend,
    pub decided_by_tier: u8,
}

const FAULT_REASONS: [&str; 3] = ["timeout", "rpc_error", "deadline_exhausted"];

fn escalate(req: u64, from: u8, to: u8, reason: &str, detail: String, took: Duration) {
    let line = format!(
        "[req {req}] ESCALATE  tier={from} -> tier={to}  reason={reason}  {detail}  ({took:?})"
    );
    if FAULT_REASONS.contains(&reason) {
        warn!("{line}");
    } else {
        info!("{line}");
    }
}

fn resolved(
    req: u64,
    tier: u8,
    model: &str,
    backend: Backend,
    conf: Option<f32>,
    tier_took: Duration,
    total: Duration,
) -> Decision {
    let conf = conf.map_or_else(|| "n/a".to_string(), |c| format!("{c:.3}"));
    info!("[req {req}] RESOLVED  tier={tier} ({model}) -> {backend}  conf={conf}  ({tier_took:?})");
    info!("[req {req}] DONE      decided_by=tier{tier}  backend={backend}  total={total:?}");
    Decision { backend, decided_by_tier: tier }
}

pub async fn route(
    req: u64,
    prompt: &str,
    t2: &Tier2Runner,
    t3: &Tier3Runner,
    budget: Duration,
) -> Decision {
    let start = Instant::now();

    // Tier 1 slots in here once built, with the same TRY / RESOLVED / ESCALATE pattern.

    // ---- Tier 2: FastText, in-process ----
    info!("[req {req}] TRY tier=2 (fasttext)");
    let t = Instant::now();
    match t2.evaluate(prompt) {
        
        Tier2Outcome::Resolved { route, confidence, calibrated_score } => {
            info!("[req {req}] SCORE  tier=2 raw={confidence:.3} calibrated={calibrated_score:.3}");
            let backend = match route {
                Tier2Route::Jev => Backend::Jev,
                Tier2Route::LLM => Backend::Llm,
            };
            return resolved(req, 2, "fasttext", backend, Some(calibrated_score), t.elapsed(), start.elapsed());
        }

        Tier2Outcome::PassThrough { top_guess, score, reason } => {
            escalate(
                req, 2, 3, reason,
                format!("top_guess={top_guess:?} score={score:.3}"),
                t.elapsed(),
            );
        }
    }

    // ---- Tier 3: Harrier sidecar over gRPC ----
    let left = budget.saturating_sub(start.elapsed());
    info!("[req {req}] TRY       tier=3 (harrier)  budget_left={left:?}");
    let t = Instant::now();
    match t3.evaluate(prompt, left).await {
        Tier3Outcome::Resolved { route, confidence, model_id } => {
        let backend = match route {
            Tier3Route::Jev => Backend::Jev,
            Tier3Route::Llm => Backend::Llm,
        };
        return resolved(req, 3, &model_id, backend, Some(confidence), t.elapsed(), start.elapsed());
    }
        Tier3Outcome::PassThrough { top_guess, score, reason } => {
            escalate(
                req, 3, 4, reason,
                format!("top_guess={top_guess:?} score={score:.3}"),
                t.elapsed(),
            );
        }
    }

    // ---- Tier 4: the floor. Always terminates. ----
    info!("[req {req}] TRY       tier=4 (llm floor)");
    resolved(req, 4, "llm-floor", Backend::Llm, None, Duration::ZERO, start.elapsed())
}