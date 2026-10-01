// src/router/tier2.rs

use fasttext::FastText;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier2Route {
    Jev,
    LLM,
}

#[derive(Debug, Clone)]
pub enum Tier2Outcome {
    Resolved {
        route: Tier2Route,
        confidence: f32,
        margin: f32,
    },
    PassThrough {
        top_guess: Tier2Route,
        score: f32,
        reason: &'static str,
    },
}

pub struct Tier2Runner {
    model: FastText,
    confidence_threshold: f32,
}

impl Tier2Runner {
    pub fn new<P: AsRef<Path>>(model_path: P, confidence_threshold: f32) -> Result<Self, String> {
        let mut model = FastText::new();
        model
            .load_model(model_path.as_ref().to_str().ok_or("Invalid path string")?)
            .map_err(|e| format!("Failed loading Tier 2 FastText model: {:?}", e))?;

        Ok(Self {
            model,
            confidence_threshold,
        })
    }

    #[inline]
    pub fn evaluate(&self, prompt: &str) -> Tier2Outcome {
        if prompt.is_empty() {
            return Tier2Outcome::PassThrough {
                top_guess: Tier2Route::LLM,
                score: 0.0,
                reason: "empty_input",
            };
        }

        let predictions = match self.model.predict(prompt, 2, 0.0) {
            Ok(preds) if !preds.is_empty() => preds,
            _ => {
                return Tier2Outcome::PassThrough {
                    top_guess: Tier2Route::LLM,
                    score: 0.0,
                    reason: "prediction_failed",
                }
            }
        };

        let p1 = predictions[0].prob;
        let p2 = if predictions.len() > 1 {
            predictions[1].prob
        } else {
            0.0
        };
        let margin = p1 - p2;
        let calibrated_score = p1 * (1.0 + margin);

        let route = match predictions[0].label.trim_start_matches("__label__") {
            "jev" => Tier2Route::Jev,
            _ => Tier2Route::LLM,
        };

        if calibrated_score >= self.confidence_threshold {
            Tier2Outcome::Resolved {
                route,
                confidence: p1,
                margin,
            }
        } else {
            Tier2Outcome::PassThrough {
                top_guess: route,
                score: calibrated_score,
                reason: "below_confidence_threshold",
            }
        }
    }
}
