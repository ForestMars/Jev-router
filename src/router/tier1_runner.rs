// src/router/tier1_runner.rs

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use arc_swap::ArcSwap;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use strsim::damerau_levenshtein;

pub const BINARY_FASTPATH_MAX_CHARS: usize = 250;
pub const AMBIGUITY_LOWER_BOUND: f32 = 0.35;
pub const AMBIGUITY_UPPER_BOUND: f32 = 0.78;

#[derive(Debug, Clone, Deserialize)]
pub struct PatternGroup {
    #[serde(default)]
    pub exact: Vec<String>,
    #[serde(default)]
    pub fuzzy: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FuzzyConfig {
    #[serde(default = "default_max_distance")]
    pub max_distance: usize,
    #[serde(default = "default_imperative_zone_bytes")]
    pub imperative_zone_bytes: usize,
}

fn default_max_distance() -> usize { 1 }
fn default_imperative_zone_bytes() -> usize { 48 }

#[derive(Debug, Clone, Deserialize)]
pub struct Tier1Config {
    pub fuzzy: FuzzyConfig,
    pub jev_strong: PatternGroup,
    pub jev_moderate: PatternGroup,
    pub llm_strong: PatternGroup,
    pub llm_prefixes: PatternGroup,
    pub polar_starts: PatternGroup,
}

#[derive(Debug, Clone)]
pub struct Tier1Result {
    pub confidence: f32,
    pub reason: String,
    pub decided: bool,
}

/// Compiled runtime automaton + fuzzy target lists.
pub struct Tier1Automaton {
    config: Tier1Config,
    jev_strong: AhoCorasick,
    jev_moderate: AhoCorasick,
    llm_strong: AhoCorasick,
    llm_prefixes: AhoCorasick,
    polar_starts: AhoCorasick,
}

impl Tier1Automaton {
    /// Deserializes TOML config file and compiles Aho-Corasick automatons.
    pub fn from_toml_file<P: AsRef<Path>>(path: P) -> Result<Self, anyhow::Error> {
        let content = fs::read_to_string(path)?;
        let config: Tier1Config = toml::from_str(&content)?;
        Self::from_config(config)
    }

    pub fn from_config(config: Tier1Config) -> Result<Self, anyhow::Error> {
        let build_dfa = |patterns: &[String]| {
            AhoCorasickBuilder::new()
                .ascii_case_insensitive(true)
                .match_kind(MatchKind::Standard)
                .build(patterns)
        };

        Ok(Self {
            jev_strong: build_dfa(&config.jev_strong.exact)?,
            jev_moderate: build_dfa(&config.jev_moderate.exact)?,
            llm_strong: build_dfa(&config.llm_strong.exact)?,
            llm_prefixes: build_dfa(&config.llm_prefixes.exact)?,
            polar_starts: build_dfa(&config.polar_starts.exact)?,
            config,
        })
    }
}

/// Lock-free, hot-reloadable Tier 1 engine with hybrid exact/fuzzy matching.
pub struct Tier1Engine {
    automaton: ArcSwap<Tier1Automaton>,
}

impl Tier1Engine {
    pub fn new(initial_automaton: Tier1Automaton) -> Self {
        Self {
            automaton: ArcSwap::from_pointee(initial_automaton),
        }
    }

    /// Atomically swaps the active pattern state in < 100ns without reader locks.
    #[allow(dead_code)]
    pub fn reload(&self, new_automaton: Tier1Automaton) {
        self.automaton.store(Arc::new(new_automaton));
    }

    pub fn classify(&self, prompt: &str) -> Tier1Result {
        let text = prompt.trim();

        if text.is_empty() {
            return Tier1Result {
                confidence: 0.0,
                reason: "empty prompt — defaulting to LLM".into(),
                decided: true,
            };
        }

        let guard = self.automaton.load();
        let max_bytes = guard.config.fuzzy.imperative_zone_bytes;
        let max_dist = guard.config.fuzzy.max_distance;

        // 1. Exact Aho-Corasick Positional Scans
        let matches_llm_prefix = guard
            .llm_prefixes
            .find_iter(text)
            .any(|m| is_prefix_match(text, m.start()));

        let starts_polar = guard
            .polar_starts
            .find_iter(text)
            .any(|m| is_prefix_match(text, m.start()));

        let mut jev_strong_hits = guard
            .jev_strong
            .find_iter(text)
            .filter(|m| m.start() <= max_bytes)
            .count();

        let jev_mod_hits = guard
            .jev_moderate
            .find_iter(text)
            .filter(|m| m.start() <= max_bytes)
            .count();

        let mut llm_strong_hits = guard
            .llm_strong
            .find_iter(text)
            .filter(|m| m.start() <= max_bytes)
            .count();

        // 2. Fuzzy Token Matcher over Imperative Zone (Catches typos like "summrize", "clasyfy")
        if jev_strong_hits == 0 || llm_strong_hits == 0 {
            let imperative_slice = if text.len() > max_bytes {
                &text[..max_bytes]
            } else {
                text
            };

            for token in imperative_slice.split_whitespace() {
                // Strip punctuation attached to token
                let clean_token = token.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                if clean_token.len() <= 3 {
                    continue; // Skip noise tokens
                }

                if jev_strong_hits == 0 {
                    for target in &guard.config.jev_strong.fuzzy {
                        if damerau_levenshtein(&clean_token, target) <= max_dist {
                            jev_strong_hits += 1;
                            break;
                        }
                    }
                }

                if llm_strong_hits == 0 {
                    for target in &guard.config.llm_strong.fuzzy {
                        if damerau_levenshtein(&clean_token, target) <= max_dist {
                            llm_strong_hits += 1;
                            break;
                        }
                    }
                }
            }
        }

        // 3. Routing Policy Evaluation
        if matches_llm_prefix || llm_strong_hits >= 2 {
            return Tier1Result {
                confidence: 0.1,
                reason: format!("LLM imperative signal (prefix={matches_llm_prefix}, hits={llm_strong_hits})"),
                decided: true,
            };
        }

        if llm_strong_hits >= 1 && jev_strong_hits == 0 && jev_mod_hits == 0 {
            return Tier1Result {
                confidence: 0.2,
                reason: "LLM directive in imperative zone with zero Jev matches".into(),
                decided: true,
            };
        }

        let is_binary_question = starts_polar
            && text.ends_with('?')
            && text.len() <= BINARY_FASTPATH_MAX_CHARS
            && !matches_llm_prefix
            && llm_strong_hits == 0;

        if is_binary_question {
            return Tier1Result {
                confidence: 0.9,
                reason: "short polar yes/no question without generative markers".into(),
                decided: true,
            };
        }

        if jev_strong_hits >= 2 {
            return Tier1Result {
                confidence: 0.85,
                reason: format!("{jev_strong_hits} strong Jev matches in imperative zone"),
                decided: true,
            };
        }

        let combined_jev = (jev_strong_hits as f32 * 0.35) + (jev_mod_hits as f32 * 0.15);
        let combined_llm = llm_strong_hits as f32 * 0.3;

        let raw_confidence = 0.5 + combined_jev - combined_llm;
        let confidence = raw_confidence.clamp(0.0, 1.0);
        let decided = confidence < AMBIGUITY_LOWER_BOUND || confidence > AMBIGUITY_UPPER_BOUND;

        Tier1Result {
            confidence,
            reason: if decided {
                format!("heuristic confident: jev={jev_strong_hits}+{jev_mod_hits}, llm={llm_strong_hits}")
            } else {
                format!("heuristic ambiguous: jev={jev_strong_hits}+{jev_mod_hits}, llm={llm_strong_hits}")
            },
            decided,
        }
    }
}

#[inline]
fn is_prefix_match(prompt: &str, matched_start_offset: usize) -> bool {
    let prefix = &prompt[..matched_start_offset];
    prefix.chars().all(|c| c.is_whitespace() || c == '"' || c == '\'' || c == '`' || c == '-')
}
