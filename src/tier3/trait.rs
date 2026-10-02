#[derive(Debug, Clone)]
pub struct Tier3Result {
    pub raw_score: f32,
    pub confidence: f32,
    pub label: String,
    pub model_id: String,
}

#[async_trait::async_trait]
pub trait Tier3Model: Send + Sync {
    async fn score(&mut self, prompt: &str) -> Result<Tier2Result, Box<dyn std::error::Error>>;
    fn model_id(&self) -> &str;
}