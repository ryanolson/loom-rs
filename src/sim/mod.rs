//! Discrete-event simulation (DES) runtime for loom-rs.
//!
//! Provides a [`SimulationRuntime`] that drives application code under virtual time.
//! Simulated components interact with the scheduler via [`SimHandle`].

mod handle;
mod queue;
mod runtime;
mod time;

pub use handle::{DelayFuture, SimHandle};
pub use runtime::{SimulationRuntime, StepOutcome};
pub use time::SimTime;
