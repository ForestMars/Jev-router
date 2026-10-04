// src/logging/mod.rs
//
// Cargo.toml:
//   tracing = "0.1"
//   tracing-subscriber = { version = "0.3", features = ["env-filter"] }

use std::sync::atomic::{AtomicU64, Ordering};

use tracing_subscriber::EnvFilter;

static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Call once at the top of main(). Level comes from RUST_LOG, default info.
pub fn init() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with_target(false)
        .compact()
        .init();
}

/// One id per request, so interleaved concurrent requests stay readable.
pub fn next_req_id() -> u64 {
    REQ_COUNTER.fetch_add(1, Ordering::Relaxed)
}