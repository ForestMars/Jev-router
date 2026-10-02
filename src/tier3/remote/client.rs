use tonic::transport::Channel;
use crate::tier3::remote::proto::{tier3_scorer_client::Tier3ScorerClient, ScoreRequest};

pub struct RemoteTier3 {
    client: Tier3ScorerClient<Channel>,
    model_id: String,
}

impl RemoteTier3 {
    pub async fn connect(endpoint: String, model_id: String) -> Result<Self, tonic::transport::Error> {
        Ok(Self {
            client: Tier3ScorerClient::connect(endpoint).await?,
            model_id,
        })
    }

    pub async fn score(&mut self, prompt: &str) -> Result<f32, tonic::Status> {
        let resp = self.client
            .score(ScoreRequest { prompt: prompt.to_string() })
            .await?
            .into_inner();
        Ok(resp.confidence)
    }
}