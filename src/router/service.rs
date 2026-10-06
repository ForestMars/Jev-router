use std::{sync::Arc, time::Duration};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::warn;

use crate::telemetry::{self, Hashes, RouteRecord};

use super::{
    cascade::{self, Backend},
    dispatch::DownstreamDispatcher,
    tier1_runner::Tier1Engine,
    tier2_runner::Tier2Runner,
    tier3_runner::Tier3Runner,
};

#[derive(Clone)]
pub struct AppState {
    tier1: Arc<Tier1Engine>,
    tier2: Arc<Tier2Runner>,
    tier3: Arc<Tier3Runner>,
    dispatcher: Arc<DownstreamDispatcher>,
    hashes: Arc<Hashes>,
    capture_tx: mpsc::Sender<RouteRecord>,
    budget: Duration,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    prompt: String,
}

pub fn app(
    tier1: Tier1Engine,
    tier2: Tier2Runner,
    tier3: Tier3Runner,
    dispatcher: DownstreamDispatcher,
    hashes: Hashes,
    capture_tx: mpsc::Sender<RouteRecord>,
    budget: Duration,
) -> Router {
    let state = AppState {
        tier1: Arc::new(tier1),
        tier2: Arc::new(tier2),
        tier3: Arc::new(tier3),
        dispatcher: Arc::new(dispatcher),
        hashes: Arc::new(hashes),
        capture_tx,
        budget,
    };
    Router::new()
        .route("/v1/completions", post(completion))
        .with_state(state)
}

async fn completion(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    let (_, mut record) = cascade::route(
        telemetry::next_req_id(),
        &request.prompt,
        &state.tier1,
        &state.tier2,
        &state.tier3,
        state.budget,
        &state.hashes,
    )
    .await;

    match state
        .dispatcher
        .execute(
            &record.request_id,
            if record.final_routing.backend == Backend::Jev.to_string() {
                Backend::Jev
            } else {
                Backend::Llm
            },
            &request.prompt,
        )
        .await
    {
        Ok(result) => {
            record.final_routing.backend = if result.tier == 4 { "jev" } else { "llm" }.into();
            record.final_routing.tier = result.tier;
            record.final_routing.latency_us += result.request_latency.as_micros();
            record.downstream_invocation_latency_ms =
                result.request_latency.as_secs_f64() * 1000.0;
            record.downstream_outcome = Some(result.outcome.into());
            record.downstream_tier = Some(format!("tier{}", result.tier));
            record.is_failover = result.is_failover;
            record.failure_reason = result.failure_reason;
            if result.is_failover {
                record.request_latency_failover_ms =
                    Some(result.request_latency.as_secs_f64() * 1000.0);
            } else {
                record.request_latency_primary_ms =
                    Some(result.request_latency.as_secs_f64() * 1000.0);
            }
            capture(&state.capture_tx, record).await;
            (StatusCode::OK, Json(result.payload)).into_response()
        }
        Err(error) => {
            record.final_routing.backend = "llm".into();
            record.final_routing.tier = error.tier;
            record.final_routing.latency_us += error.request_latency.as_micros();
            record.downstream_invocation_latency_ms =
                error.request_latency.as_secs_f64() * 1000.0;
            record.downstream_outcome = Some(error.outcome.into());
            record.downstream_tier = Some(format!("tier{}", error.tier));
            record.is_failover = error.is_failover;
            record.failure_reason = Some(error.failure_reason);
            if error.is_failover {
                record.request_latency_failover_ms =
                    Some(error.request_latency.as_secs_f64() * 1000.0);
            }
            let detail = error.error.to_string();
            capture(&state.capture_tx, record).await;
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "error": "generative_provider_unavailable", "detail": detail })),
            )
                .into_response()
        }
    }
}

async fn capture(sender: &mpsc::Sender<RouteRecord>, record: RouteRecord) {
    if let Err(error) = sender.send(record).await {
        warn!(error = %error, "routing capture writer is unavailable");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use serde_json::Value;
    use tower::ServiceExt;

    #[tokio::test]
    async fn completion_reaches_selected_backend() {
        let tier1 = Tier1Engine::new(
            crate::router::tier1_runner::Tier1Automaton::from_toml_file("config/tier1.toml")
                .unwrap(),
        );
        let tier2 = Tier2Runner::new("models/tier2.bin", 0.95).unwrap();
        let tier3 = Tier3Runner::new(
            "http://127.0.0.1:1".into(),
            0.95,
            0.7,
            Duration::from_millis(20),
        )
        .unwrap();
        let jev = super::super::dispatch::tests_endpoint("jev");
        let frontier = super::super::dispatch::tests_endpoint("frontier");
        let dispatcher = DownstreamDispatcher::new(
            jev,
            frontier,
            Duration::from_millis(100),
            2,
            Duration::from_secs(5),
            Duration::from_secs(30),
        )
        .unwrap();
        let (tx, _rx) = mpsc::channel(2);
        let application = app(
            tier1,
            tier2,
            tier3,
            dispatcher,
            Hashes {
                tier1_toml: String::new(),
                tier2_bin: String::new(),
            },
            tx,
            Duration::from_millis(30),
        );

        let response = application
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt":"What is 2+2?"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(payload.get("completion").is_some());
    }
}
