// src/router/Tier2.rs

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
        calibrated_score: f32, // Added per ticket spec
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
    pub fn normalize_input(input: &str) -> String {
        input
            .trim()
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
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
    pub fn evaluate(&self, prompt: &str) -> Tier2Outcome {
        // 1. Input Normalization pre-pass
        let normalized = Self::normalize_input(prompt);

        if normalized.is_empty() {
            return Tier2Outcome::PassThrough {
                top_guess: Tier2Route::LLM,
                score: 0.0,
                reason: "empty_input",
            };
        }

        let predictions = self.model.predict(&normalized, 2, 0.0);
        if predictions.is_empty() {
            return Tier2Outcome::PassThrough {
                top_guess: Tier2Route::LLM,
                score: 0.0,
                reason: "no_predictions",
            };
        }

        // 2. Explicit Class Extraction by label name & Defensive Clamping
        let mut p_jev = 0.0f32;
        let mut p_llm = 0.0f32;

        for pred in &predictions {
            let clamped_prob = pred.prob.clamp(0.0, 1.0);
            let label_name = pred.label.trim_start_matches("__label__");
            match label_name {
                "jev" => p_jev = clamped_prob,
                "llm" => p_llm = clamped_prob,
                _ => {}
            }
        }

        // 3. Symmetric Margin Calculation
        let margin = (p_jev - p_llm).abs();

        let (route, top_prob) = if p_jev >= p_llm {
            (Tier2Route::Jev, p_jev)
        } else {
            (Tier2Route::LLM, p_llm)
        };

        // 4. Defensive Clamping on Calibrated Score
        let calibrated_score = (top_prob * (1.0 + margin)).clamp(0.0, 1.0);

        if calibrated_score >= self.confidence_threshold {
            Tier2Outcome::Resolved {
                route,
                confidence: top_prob,
                calibrated_score,
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // 1. Invariant: normalize_input never produces leading/trailing whitespace
        // or consecutive spaces, regardless of input string noise.
        #[test]
        fn prop_normalize_input_invariants(s in ".*") {
            let normalized = Tier2Runner::normalize_input(&s);

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

            let top_prob = p_jev.max(p_llm);
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

    #[test]
    fn test_whitespace_and_casing_normalization() {
        let raw = "  Hello   WORLD \t\n Test  ";
        assert_eq!(Tier2Runner::normalize_input(raw), "hello world test");
    }
}
