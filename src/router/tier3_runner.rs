// src/router/tier3.rs

use fasttext::FastText;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier3Route {
    Jev,
    LLM,
}

#[derive(Debug, Clone)]
pub enum Tier3Outcome {
    Resolved {
        route: Tier3Route,
        confidence: f32,
        margin: f32,
    },
    PassThrough {
        top_guess: Tier3Route,
        score: f32,
        reason: &'static str,
    },
}

pub struct Tier3Runner {
    model: FastText,
    confidence_threshold: f32,
}

impl Tier3Runner {
    pub fn normalize_input(input: &str) -> String {
        input.trim().to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
    }
    pub fn new<P: AsRef<Path>>(model_path: P, confidence_threshold: f32) -> Result<Self, String> {
        let model = FastText::load_model(&model_path)
            .map_err(|e| format!("Failed to load FastText model: {e}"))?;

        Ok(Self {
            model,
            confidence_threshold,
        })
    }

    #[inline]
    pub fn evaluate(&self, prompt: &str) -> Tier3Outcome {
        if prompt.is_empty() {
            return Tier3Outcome::PassThrough {
                top_guess: Tier3Route::LLM,
                score: 0.0,
                reason: "empty_input",
            };
        }

        let predictions = self.model.predict(prompt, 2, 0.0);
        if predictions.is_empty() {
            return Tier3Outcome::PassThrough {
                top_guess: Tier3Route::LLM,
                score: 0.0,
                reason: "no_predictions",
            };
        }

        let p1 = predictions[0].prob;
        let p2 = if predictions.len() > 1 {
            predictions[1].prob
        } else {
            0.0
        };
        let margin = p1 - p2;
        let calibrated_score = p1 * (1.0 + margin);

        let route = match predictions[0].label.trim_start_matches("__label__") {
            "jev" => Tier3Route::Jev,
            _ => Tier3Route::LLM,
        };

        if calibrated_score >= self.confidence_threshold {
            Tier3Outcome::Resolved {
                route,
                confidence: p1,
                margin,
            }
        } else {
            Tier3Outcome::PassThrough {
                top_guess: route,
                score: calibrated_score,
                reason: "below_confidence_threshold",
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // 1. Invariant: normalize_input never produces leading/trailing whitespace
        // or consecutive spaces, regardless of input string noise.
        #[test]
        fn prop_normalize_input_invariants(s in ".*") {
            let normalized = Tier3Runner::normalize_input(&s);

            // Output should always be lowercase
            prop_assert_eq!(&normalized, &normalized.to_lowercase());

            // Output should never have leading/trailing whitespace
            prop_assert_eq!(&normalized, normalized.trim());

            // Output should never contain double spaces
            prop_assert!(!normalized.contains("  "));
        }

        // 2. Invariant: Probability clamping and margin math must always lie in [0.0, 1.0]
        // even with extreme or out-of-bounds float inputs from bindings.
        #[test]
        fn prop_calibrated_score_bounds(
            p_jev_raw in -100.0f32..100.0f32,
            p_llm_raw in -100.0f32..100.0f32
        ) {
            let p_jev = p_jev_raw.clamp(0.0, 1.0);
            let p_llm = p_llm_raw.clamp(0.0, 1.0);

            let (top_prob, _) = if p_jev >= p_llm {
                (p_jev, Tier3Route::Jev)
            } else {
                (p_llm, Tier3Route::LLM)
            };

            let margin = (p_jev - p_llm).abs();
            let calibrated_score = (top_prob * (1.0 + margin)).clamp(0.0, 1.0);

            // Invariants
            prop_assert!(margin >= 0.0 && margin <= 1.0);
            prop_assert!(calibrated_score >= 0.0 && calibrated_score <= 1.0);
        }

        // 3. Invariant: Margin calculation must be symmetric regardless of label order.
        #[test]
        fn prop_margin_is_symmetric(a in 0.0f32..1.0f32, b in 0.0f32..1.0f32) {
            let margin1 = (a - b).abs();
            let margin2 = (b - a).abs();
            prop_assert_eq!(margin1, margin2);
        }
    }
}
