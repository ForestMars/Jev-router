// src/main.rs

mod logging;
mod router;

use std::time::Duration;

use router::cascade;
use router::tier2_runner::Tier2Runner;
use router::tier3_runner::Tier3Runner;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    logging::init();

    // Placeholder thresholds and path: use your tuned values.
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
    ] {
        let req = logging::next_req_id();
        let _decision = cascade::route(req, prompt, &t2, &t3, Duration::from_millis(600)).await;
    }
    Ok(())
}