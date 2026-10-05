// src/router/cascade.rs
//
// Every tier boundary logs exactly one of: TRY, RESOLVED, ESCALATE.
// Every request ends with exactly one DONE line.
// Faults (timeout, rpc_error, deadline_exhausted) escalate at WARN,
// ordinary low-confidence escalations at INFO.

use std::fmt;
use std::time::{Duration, Instant};

use chrono::Utc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::telemetry::{
    FinalRouting, Hashes, RouteRecord, Sampling, Tier1Outcome as Tier1Record,
    Tier2Outcome as Tier2Record, Tier3Outcome as Tier3Record,
};

use super::tier1_runner::{Tier1Engine, AMBIGUITY_UPPER_BOUND};
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

fn finish_route(
    decision: Decision,
    prompt: &str,
    start: Instant,
    timestamp: &chrono::DateTime<Utc>,
    hashes: &Hashes,
    tier1_outcome: Tier1Record,
    tier2_outcome: Option<Tier2Record>,
    tier3_outcome: Option<Tier3Record>,
) -> (Decision, RouteRecord) {
    let latency_us = start.elapsed().as_micros();
    let record = RouteRecord {
        request_id: Uuid::new_v4().to_string(),
        timestamp: timestamp.clone(),
        prompt_raw: prompt.to_string(),
        hashes: hashes.clone(),
        tier1_outcome,
        tier2_outcome,
        tier3_outcome,
        final_routing: FinalRouting {
            backend: decision.backend.to_string(),
            tier: decision.decided_by_tier,
            latency_us,
        },
        sampling: Sampling::default(),
    };
    (decision, record)
}

pub async fn route(
    req: u64,
    prompt: &str,
    t1: &Tier1Engine,
    t2: &Tier2Runner,
    t3: &Tier3Runner,
    budget: Duration,
    hashes: &Hashes,
) -> (Decision, RouteRecord) {
    let start = Instant::now();
    let timestamp = Utc::now();

    // ---- Tier 1: heuristic pattern engine, in-process ----
    info!("[req {req}] TRY tier=1 (heuristic) prompt={prompt:?}");
    let t = Instant::now();
    let r1 = t1.classify(prompt);
    let tier1_record = Tier1Record {
        confidence: r1.confidence,
        decided: r1.decided,
        reason: r1.reason.clone(),
    };
    info!(
        "[req {req}] SCORE  tier=1 conf={:.3} decided={} reason={}",
        r1.confidence, r1.decided, r1.reason
    );

    if r1.decided && r1.confidence > AMBIGUITY_UPPER_BOUND {
        let decision = resolved(req, 1, "heuristic", Backend::Jev, Some(r1.confidence), t.elapsed(), start.elapsed());
        return finish_route(decision, prompt, start, &timestamp, hashes, tier1_record, None, None);
    }

    escalate(
        req, 1, 2,
        if r1.decided { "not_jev" } else { "ambiguous" },
        format!("conf={:.3} reason={}", r1.confidence, r1.reason),
        t.elapsed(),
    );

    // ---- Tier 2: FastText, in-process ----
    info!("[req {req}] TRY tier=2 (fasttext) prompt={prompt:?}");
    let t = Instant::now();
    let mut tier2_record = None;
    let mut tier3_record = None;
    match t2.evaluate(prompt) {
        
        Tier2Outcome::Resolved { route, confidence, calibrated_score, raw_probabilities } => {
            tier2_record = Some(Tier2Record { raw_probabilities });
            info!("[req {req}] SCORE  tier=2 raw={confidence:.3} calibrated={calibrated_score:.3}");
            let backend = match route {
                Tier2Route::Jev => Backend::Jev,
                Tier2Route::LLM => Backend::Llm,
            };
            let decision = resolved(req, 2, "fasttext", backend, Some(calibrated_score), t.elapsed(), start.elapsed());
            return finish_route(decision, prompt, start, &timestamp, hashes, tier1_record, tier2_record, None);
        }

        Tier2Outcome::PassThrough { top_guess, score, reason, raw_probabilities } => {
            tier2_record = Some(Tier2Record { raw_probabilities });
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
    let tier3_result = t3.evaluate(prompt, left).await;
    let tier3_score = match &tier3_result {
        Tier3Outcome::Resolved { confidence, .. } => *confidence,
        Tier3Outcome::PassThrough { score, .. } => *score,
    };
    tier3_record = Some(Tier3Record { score: tier3_score });
    match tier3_result {
        Tier3Outcome::Resolved { route, confidence, model_id } => {
        let backend = match route {
            Tier3Route::Jev => Backend::Jev,
            Tier3Route::Llm => Backend::Llm,
        };
        let decision = resolved(req, 3, &model_id, backend, Some(confidence), t.elapsed(), start.elapsed());
        return finish_route(decision, prompt, start, &timestamp, hashes, tier1_record, tier2_record, tier3_record);
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
    info!("[req {req}] TRY tier=4 (llm floor)");
    let decision = resolved(req, 4, "llm-floor", Backend::Llm, None, Duration::ZERO, start.elapsed());
    finish_route(decision, prompt, start, &timestamp, hashes, tier1_record, tier2_record, tier3_record)
}