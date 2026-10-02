// src/router/tier3.rs
//
// Tier 3: Harrier, reached over gRPC (sidecars/harrier/server.py).
// Tier 2 (FastText, in-process) lives in router/tier2.rs.

use std::time::Duration;

use tokio::time::timeout;
use tonic::transport::Channel;

use super::tier3_proto::{tier3_scorer_client::Tier3ScorerClient, ScoreRequest};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier3Route {
    Jev,
    Llm,
}

#[derive(Debug, Clone)]
pub enum Tier3Outcome {
    Resolved {
        route: Tier3Route,
        confidence: f32,
        model_id: String,
    },
    PassThrough {
        top_guess: Option<Tier3Route>,
        score: f32,
        reason: &'static str,
    },
}

pub struct Tier3Runner {
    client: Tier3ScorerClient<Channel>,
    /// Bar for resolving to Jev. The expensive error, so this sits higher.
    jev_threshold: f32,
    /// Bar for resolving to the LLM. The free error, so this can sit lower.
    llm_threshold: f32,
    /// Hard cap on a single sidecar call, independent of the request budget.
    max_call: Duration,
}

impl Tier3Runner {
    /// `connect_lazy` means a down sidecar does not kill startup; the failure
    /// surfaces per call as a PassThrough("rpc_error" / "timeout") instead.
    pub fn new(
        endpoint: String,
        jev_threshold: f32,
        llm_threshold: f32,
        max_call: Duration,
    ) -> Result<Self, tonic::codegen::http::uri::InvalidUri> {
        let channel = Channel::from_shared(endpoint)?.connect_lazy();
        Ok(Self {
            client: Tier3ScorerClient::new(channel),
            jev_threshold,
            llm_threshold,
            max_call,
        })
    }

    /// `budget` is the time remaining on the request's global deadline.
    pub async fn evaluate(&self, prompt: &str, budget: Duration) -> Tier3Outcome {
        if budget.is_zero() {
            return pass(None, 0.0, "deadline_exhausted");
        }
        let deadline = budget.min(self.max_call);

        // Cloning a tonic client is cheap (shared channel), and score() needs &mut.
        let mut client = self.client.clone();
        let req = ScoreRequest { prompt: prompt.to_string() };

        let resp = match timeout(deadline, client.score(req)).await {
            Err(_) => return pass(None, 0.0, "timeout"),
            Ok(Err(_)) => return pass(None, 0.0, "rpc_error"),
            Ok(Ok(r)) => r.into_inner(),
        };

        decide(
            &resp.label,
            resp.confidence as f32,
            &resp.model_id,
            self.jev_threshold,
            self.llm_threshold,
        )
    }
}

/// Pure decision logic, split out so it is testable without a sidecar.
pub fn decide(
    label: &str,
    confidence: f32,
    model_id: &str,
    jev_threshold: f32,
    llm_threshold: f32,
) -> Tier3Outcome {
    let confidence = if confidence.is_nan() { 0.0 } else { confidence.clamp(0.0, 1.0) };

    let route = match label {
        "jev" => Tier3Route::Jev,
        "llm" => Tier3Route::Llm,
        _ => return pass(None, confidence, "unknown_label"),
    };

    let bar = match route {
        Tier3Route::Jev => jev_threshold,
        Tier3Route::Llm => llm_threshold,
    };

    if confidence >= bar {
        Tier3Outcome::Resolved {
            route,
            confidence,
            model_id: model_id.to_string(),
        }
    } else {
        pass(Some(route), confidence, "below_threshold")
    }
}

fn pass(top_guess: Option<Tier3Route>, score: f32, reason: &'static str) -> Tier3Outcome {
    Tier3Outcome::PassThrough { top_guess, score, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jev_needs_the_higher_bar() {
        // 0.90 clears the LLM bar but not the Jev bar.
        assert!(matches!(
            decide("jev", 0.90, "harrier", 0.95, 0.70),
            Tier3Outcome::PassThrough { reason: "below_threshold", .. }
        ));
        assert!(matches!(
            decide("llm", 0.90, "harrier", 0.95, 0.70),
            Tier3Outcome::Resolved { route: Tier3Route::Llm, .. }
        ));
    }

    #[test]
    fn jev_resolves_above_its_bar() {
        assert!(matches!(
            decide("jev", 0.97, "harrier", 0.95, 0.70),
            Tier3Outcome::Resolved { route: Tier3Route::Jev, .. }
        ));
    }

    #[test]
    fn unknown_label_and_nan_never_resolve() {
        assert!(matches!(
            decide("???", 0.99, "harrier", 0.95, 0.70),
            Tier3Outcome::PassThrough { reason: "unknown_label", .. }
        ));
        assert!(matches!(
            decide("jev", f32::NAN, "harrier", 0.95, 0.70),
            Tier3Outcome::PassThrough { .. }
        ));
    }
}