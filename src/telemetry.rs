#[path = "telemetry/mod.rs"]
mod logging;

pub use logging::{init, next_req_id};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRecord {
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
    pub prompt_raw: String,
    pub hashes: Hashes,
    pub tier1_outcome: Tier1Outcome,
    pub tier2_outcome: Option<Tier2Outcome>,
    pub tier3_outcome: Option<Tier3Outcome>,
    pub final_routing: FinalRouting,
    #[serde(default)]
    pub sampling: Sampling,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hashes {
    pub tier1_toml: String,
    pub tier2_bin: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Outcome {
    pub confidence: f32,
    pub decided: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier2Outcome {
    pub raw_probabilities: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier3Outcome {
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalRouting {
    pub backend: String,
    pub tier: u8,
    pub latency_us: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sampling {
    #[serde(default = "default_sample_prob")]
    pub sample_prob: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            sample_prob: default_sample_prob(),
        }
    }
}

fn default_sample_prob() -> f32 {
    1.0
}
