// src/main.rs

mod telemetry;
mod router;

use std::time::Duration;

use router::cascade;
use router::tier1_runner::{Tier1Automaton, Tier1Engine};
use router::tier2_runner::Tier2Runner;
use router::tier3_runner::Tier3Runner;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    telemetry::init();

    // 1. Layer for TRACES (Spans) -> Goes to Grafana Tempo
    // Captures start times, end times, parent-child request spans, and durations
    let otel_trace_layer = tracing_opentelemetry::layer()
        .with_tracer(init_opentelemetry_tracer()); 

    // 2. Layer for LOGS (Events) -> Goes to Grafana Loki
    // Captures individual log events (tracing::info!), formats as JSON, or ships via OTLP
    let loki_log_layer = tracing_loki::builder()
        .build_url("http://localhost:3100")
        .unwrap();

    // 3. Register BOTH layers together in the Subscriber Pipeline
    tracing_subscriber::registry()
        .with(otel_trace_layer) // Handles spans -> Tempo
        .with(loki_log_layer)  // Handles events -> Loki
        .init();

    // Placeholder thresholds and paths: use your tuned values.
    let tier1_path = std::env::var("TIER1_CONFIG").unwrap_or_else(|_| "config/tier1.toml".into());
    let t1 = Tier1Engine::new(Tier1Automaton::from_toml_file(&tier1_path)?);

    let model_path = std::env::var("FASTTEXT_MODEL").unwrap_or_else(|_| "models/tier2.bin".into());
    let t2 = Tier2Runner::new(&model_path, 0.95)?;
    let t3 = Tier3Runner::new(
        "http://[::1]:50051".into(),
        0.95, // jev bar
        0.70, // llm bar
        Duration::from_millis(500),
    )?;

    for prompt in [
        "What is 2+2?",
        "Write a 500-word essay on the history of Rome",
        "Convert 5 miles to kilometers",
    ] {
        let req = telemetry::next_req_id();
        let _decision = cascade::route(req, prompt, &t1, &t2, &t3, Duration::from_millis(600)).await;
    }
    Ok(())
}