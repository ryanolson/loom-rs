//! Multi-Armed Bandit (MAB) adaptive scheduler for compute work.
//!
//! This module provides an adaptive scheduler that decides per-function whether
//! to execute synchronous work **inline on Tokio** or **offload to Rayon**.
//! It uses a 2-arm bandit with Thompson Sampling to learn optimal strategies.
//!
//! # Key Features
//!
//! - **Per-function learning**: Each unique function maintains its own statistics
//! - **Thompson Sampling**: Balances exploration vs exploitation
//! - **Guardrails**: Hard rules prevent Tokio starvation even when learning
//! - **Compute hints**: Optional trait for cold-start guidance
//! - **Calibration**: Optional startup measurement of offload overhead
//!
//! # Usage Patterns
//!
//! ## Stream Mode (immediate feedback)
//!
//! Use `adaptive_map()` on streams - each stream owns its own scheduler:
//!
//! ```ignore
//! use loom_rs::ComputeStreamExt;
//!
//! stream.adaptive_map(|x| expensive_work(x))
//! ```
//!
//! ## Handler Mode (delayed feedback)
//!
//! Use the shared scheduler for handler patterns:
//!
//! ```ignore
//! let sched = runtime.mab_scheduler();
//! let key = FunctionKey::from_type::<MyHandler>();
//! let ctx = runtime.collect_context();
//!
//! let (id, arm) = sched.choose(key, &ctx);
//! let result = match arm {
//!     Arm::InlineTokio => execute_inline(),
//!     Arm::OffloadRayon => execute_offload().await,
//! };
//! sched.finish(id, cost_us, Some(fn_us));
//! ```
//!
//! # Guardrails
//!
//! Four guardrails prevent Tokio starvation:
//!
//! - **GR0**: Single-worker protection (very conservative when `tokio_workers=1`)
//! - **GR1**: Hard blocking threshold (never inline if `ema > t_block_hard_us`)
//! - **GR2**: Pressure-sensitive threshold (tighter limit under high pressure)
//! - **GR3**: Strike suppression (suppress inline after repeated slow executions)
//!
//! # Compute Hints
//!
//! Implement [`ComputeHintProvider`] on input types to guide cold-start behavior:
//!
//! ```ignore
//! use loom_rs::mab::{ComputeHint, ComputeHintProvider};
//!
//! struct MyMessage { payload: Vec<u8> }
//!
//! impl ComputeHintProvider for MyMessage {
//!     fn compute_hint(&self) -> ComputeHint {
//!         match self.payload.len() {
//!             0..=1024 => ComputeHint::Low,
//!             1025..=65536 => ComputeHint::Medium,
//!             _ => ComputeHint::High,
//!         }
//!     }
//! }
//! ```

mod calibration;
mod knobs;
mod scheduler;
mod types;

pub use calibration::{calibrate, CalibrationConfig, CalibrationResult};
pub use knobs::MabKnobs;
pub use scheduler::MabScheduler;
pub use types::{
    Arm, ArmStats, ComputeHint, ComputeHintProvider, Context, DecisionId, FunctionKey, KeyStats,
};
