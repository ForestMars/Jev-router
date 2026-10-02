pub async fn route(prompt: &str, tier3: &mut dyn Tier2Model) -> Decision {
    // Tier 1: deterministic (stub for now)
    if let Some(d) = tier1_check(prompt) { return d; }

    // Tier 2: 

    // Tier 3: call sidecar
    match tier3.score(prompt).await {
        Ok(result) if result.confidence >= 0.5 => {
            return Decision::new(&result.label, &result.model_id);
        }
        _ => {} // fall through to Tier 4
    }

    // Tier 4: Jev (stub)
    Decision::new("jev", "tier3")
}