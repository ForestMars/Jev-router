// src/main.rs

mod router;
mod telemetry;

use std::time::Duration;

use router::cascade;
use router::tier1_runner::{Tier1Automaton, Tier1Engine};
use router::tier2_runner::Tier2Runner;
use router::tier3_runner::Tier3Runner;

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace::{SdkTracerProvider, Tracer};
use sha2::{Digest, Sha256};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;
use url::Url;

fn sha256_file(path: &str) -> Result<String, std::io::Error> {
    let contents = std::fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(contents)))
}

// Helper function to set up OpenTelemetry Tracer for Tempo
fn init_opentelemetry_tracer() -> Tracer {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint("http://localhost:4317") // Tempo / OTel Collector OTLP endpoint
        .build()
        .expect("Failed to build OTLP exporter");

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .build();

    provider.tracer("cascade-router")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let otel_tracer = init_opentelemetry_tracer();

    // 1. Layer for TRACES (Spans) -> Goes to Grafana Tempo
    // Captures start times, end times, parent-child request spans, and durations
    let otel_trace_layer = tracing_opentelemetry::layer().with_tracer(otel_tracer);

    // 2. Layer for LOGS (Events) -> Goes to Grafana Loki
    // Captures individual log events (tracing::info!), formats as JSON, or ships via OTLP
    let (loki_layer, loki_task) = tracing_loki::builder()
        .label("service", "cascade-router")?
        .build_url(Url::parse("http://localhost:3100")?)?;

    let filter = match std::env::var("RUST_LOG") {
        Ok(filter) => EnvFilter::try_new(filter)?,
        Err(std::env::VarError::NotPresent) => EnvFilter::new("info"),
        Err(error) => return Err(error.into()),
    };

    // 3. Register BOTH layers together in the Subscriber Pipeline
    tracing_subscriber::registry()
        .with(filter)
        .with(otel_trace_layer) // Handles spans -> Tempo
        .with(loki_layer) // Handles events -> Loki
        .init();

    tokio::spawn(loki_task);
    tracing::info!("Cascade Router initialized successfully");

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<telemetry::RouteRecord>();
    let capture_task = tokio::spawn(async move {
        tokio::fs::create_dir_all("logs").await?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/routing_capture.jsonl")
            .await?;

        while let Some(record) = rx.recv().await {
            let json = serde_json::to_string(&record)?;
            file.write_all(format!("{json}\n").as_bytes()).await?;
        }

        file.flush().await?;
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    });

    // Placeholder thresholds and paths: use your tuned values.
    let tier1_path = std::env::var("TIER1_CONFIG").unwrap_or_else(|_| "config/tier1.toml".into());
    // let t1 = Tier1Engine::new(Tier1Automaton::from_toml_file(&tier1_path)?);
    let t1 = Tier1Engine::new(Tier1Automaton::from_toml_file(&tier1_path)?)?;

    let model_path = std::env::var("FASTTEXT_MODEL").unwrap_or_else(|_| "models/tier2.bin".into());
    let t2 = Tier2Runner::new(&model_path, 0.95)?;
    let hashes = telemetry::Hashes {
        tier1_toml: sha256_file(&tier1_path)?,
        tier2_bin: sha256_file(&model_path)?,
    };
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
        let (_decision, record) = cascade::route(
            req,
            prompt,
            &t1,
            &t2,
            &t3,
            Duration::from_millis(600),
            &hashes,
        )
        .await;
        tx.send(record).map_err(|error| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, error.to_string())
        })?;
    }
    drop(tx);
    capture_task
        .await?
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    Ok(())
}
