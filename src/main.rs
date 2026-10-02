pub mod tier3 {
    tonic::include_proto!("tier3");
}

use tier3::tier3_scorer_client::Tier3ScorerClient;
use tier3::ScoreRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = Tier3ScorerClient::connect("http://[::1]:50051").await?;

    for prompt in [
        "What is 2+2?",
        "Write a 500-word essay on the history of Rome",
    ] {
        let resp = client
            .score(ScoreRequest { prompt: prompt.to_string() })
            .await?
            .into_inner();
        println!(
            "prompt={:?} -> label={} confidence={:.3} model={}",
            prompt, resp.label, resp.confidence, resp.model_id
        );
    }
    Ok(())
}