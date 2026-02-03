//! Core types for the Multi-Armed Bandit (MAB) scheduler.
//!
//! This module defines the fundamental types used throughout the MAB system:
//! - `FunctionKey`: Identifies unique compute functions
//! - `Arm`: The two execution strategies (inline vs offload)
//! - `DecisionId`: Tracks pending decisions for delayed feedback
//! - `Context`: Runtime metrics used for scheduling decisions
//! - `ArmStats`/`KeyStats`: Statistical accumulators for learning
//! - `ComputeHint`/`ComputeHintProvider`: Cold-start guidance from user code

use std::any::TypeId;
use std::hash::{Hash, Hasher};

/// Identifies a unique compute function for per-function learning.
///
/// Created from a type ID (via `FunctionKey::from_type::<F>()`) or a raw u64.
/// Each unique function key maintains its own MAB statistics.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FunctionKey(pub u64);

impl FunctionKey {
    /// Create a function key from a type.
    ///
    /// Typically used with the closure type of a handler:
    /// ```ignore
    /// let key = FunctionKey::from_type::<MyHandler>();
    /// ```
    pub fn from_type<T: 'static>() -> Self {
        let type_id = TypeId::of::<T>();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        type_id.hash(&mut hasher);
        FunctionKey(hasher.finish())
    }

    /// Create a function key from a string identifier.
    ///
    /// Useful when you want a stable key across runs:
    /// ```ignore
    /// let key = FunctionKey::from_name("my_handler");
    /// ```
    pub fn from_name(s: &str) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut hasher);
        FunctionKey(hasher.finish())
    }

    /// Create a function key from a type's name, without requiring `'static`.
    ///
    /// This is useful for scoped closures where the type captures non-`'static`
    /// references. Uses `std::any::type_name` which works on any type.
    ///
    /// Note: Type names are not guaranteed to be stable across compiler versions,
    /// but this is fine for runtime-only MAB learning that doesn't persist.
    ///
    /// ```ignore
    /// let key = FunctionKey::from_type_name::<MyHandler>();
    /// ```
    pub fn from_type_name<T: ?Sized>() -> Self {
        let type_name = std::any::type_name::<T>();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        type_name.hash(&mut hasher);
        FunctionKey(hasher.finish())
    }
}

/// The two arms of the bandit: inline execution on Tokio or offload to Rayon.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Arm {
    /// Execute synchronously on the current Tokio worker thread.
    /// Low overhead (~0ns) but blocks the worker during execution.
    InlineTokio,
    /// Offload to Rayon thread pool and await completion.
    /// Higher overhead (~100-500ns) but doesn't block Tokio workers.
    OffloadRayon,
}

/// Unique identifier for a pending decision, used for delayed feedback.
///
/// When using the handler pattern (shared scheduler across invocations),
/// the scheduler returns a `DecisionId` that must be passed to `finish()`
/// after the work completes to record the outcome.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DecisionId(pub u64);

/// Runtime context used for scheduling decisions.
///
/// Contains current system metrics that inform the guardrails and
/// pressure-adjusted cost modeling.
#[derive(Clone, Copy, Debug, Default)]
pub struct Context {
    /// Number of Tokio worker threads in this runtime
    pub tokio_workers: u32,
    /// Number of tracked async tasks currently in flight
    pub inflight_tasks: u32,
    /// Recent spawn rate (tasks per second)
    pub spawn_rate_per_s: f32,
    /// Number of Rayon threads in this runtime
    pub rayon_threads: u32,
    /// Estimated Rayon queue depth (submitted - started)
    pub rayon_queue_depth: u32,
}

/// Statistics for a single arm (inline or offload).
///
/// Uses Welford's online algorithm for numerically stable variance calculation.
/// The `n_eff` is a decayed effective sample count for recent-weighted learning.
#[derive(Clone, Copy, Debug)]
pub struct ArmStats {
    /// Effective sample count (decayed over time)
    pub n_eff: f64,
    /// Mean of ln(cost_us)
    pub mu: f64,
    /// Sum of squared deviations from mean (for variance calculation)
    pub s2: f64,
}

impl Default for ArmStats {
    fn default() -> Self {
        Self {
            n_eff: 0.0,
            mu: 0.0,
            s2: 0.0,
        }
    }
}

impl ArmStats {
    /// Create new arm stats with an initial estimate.
    ///
    /// Used to seed the prior for a new function key.
    pub fn with_prior(mu: f64, variance: f64, n_eff: f64) -> Self {
        Self {
            n_eff,
            mu,
            s2: variance * n_eff.max(1.0), // Convert variance to sum of squared deviations
        }
    }

    /// Get the variance estimate.
    ///
    /// Returns a minimum variance to avoid numerical issues.
    pub fn variance(&self) -> f64 {
        if self.n_eff < 2.0 {
            // Not enough samples, return a diffuse prior variance
            1.0
        } else {
            (self.s2 / (self.n_eff - 1.0)).max(0.01)
        }
    }

    /// Get the standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Per-function statistics tracking both arms.
#[derive(Clone, Debug)]
pub struct KeyStats {
    /// Statistics for inline execution
    pub inline: ArmStats,
    /// Statistics for offload execution
    pub offload: ArmStats,
    /// Exponential moving average of function execution time (microseconds)
    pub ema_fn_us: f64,
    /// Strike counter for GR3 (inline failures due to blocking)
    pub inline_strikes: f64,
    /// Number of forced offload explorations remaining (for High hint)
    pub hint_explore_remaining: u32,
}

impl Default for KeyStats {
    fn default() -> Self {
        Self {
            inline: ArmStats::default(),
            offload: ArmStats::default(),
            ema_fn_us: 0.0,
            inline_strikes: 0.0,
            hint_explore_remaining: 0,
        }
    }
}

impl KeyStats {
    /// Create new key stats with initial EMA estimate.
    pub fn with_ema(ema_fn_us: f64) -> Self {
        Self {
            ema_fn_us,
            ..Default::default()
        }
    }
}

/// Compute cost hint for cold-start guidance.
///
/// The MAB uses this to bias initial exploration but doesn't trust it fully
/// until validated by actual observations. Implement [`ComputeHintProvider`]
/// on your input types to provide these hints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputeHint {
    /// No hint available - use default Thompson Sampling exploration
    #[default]
    Unknown,
    /// Expected < 50us (likely inline-safe)
    Low,
    /// Expected 50-500us (borderline, explore both arms)
    Medium,
    /// Expected > 500us (should test offload early)
    High,
}

/// Trait for items that can provide a compute cost hint.
///
/// Implement this on your input types to guide cold-start behavior.
/// The scheduler uses hints only during the cold-start phase (first few
/// observations per function) and then relies on learned data.
///
/// # Example
///
/// ```ignore
/// use loom_rs::mab::ComputeHintProvider;
///
/// struct MyMessage {
///     payload: Vec<u8>,
/// }
///
/// impl ComputeHintProvider for MyMessage {
///     fn compute_hint(&self) -> ComputeHint {
///         match self.payload.len() {
///             0..=1024 => ComputeHint::Low,
///             1025..=65536 => ComputeHint::Medium,
///             _ => ComputeHint::High,
///         }
///     }
/// }
/// ```
pub trait ComputeHintProvider {
    /// Return a hint about the expected compute cost for this item.
    ///
    /// Default implementation returns `ComputeHint::Unknown`.
    fn compute_hint(&self) -> ComputeHint {
        ComputeHint::Unknown
    }
}

// === Default implementations for common types ===

impl ComputeHintProvider for () {}

impl<T: ComputeHintProvider> ComputeHintProvider for &T {
    fn compute_hint(&self) -> ComputeHint {
        (*self).compute_hint()
    }
}

impl<T: ComputeHintProvider> ComputeHintProvider for &mut T {
    fn compute_hint(&self) -> ComputeHint {
        (**self).compute_hint()
    }
}

impl<T: ComputeHintProvider> ComputeHintProvider for Box<T> {
    fn compute_hint(&self) -> ComputeHint {
        (**self).compute_hint()
    }
}

impl<T: ComputeHintProvider> ComputeHintProvider for std::sync::Arc<T> {
    fn compute_hint(&self) -> ComputeHint {
        (**self).compute_hint()
    }
}

impl<T: ComputeHintProvider> ComputeHintProvider for std::rc::Rc<T> {
    fn compute_hint(&self) -> ComputeHint {
        (**self).compute_hint()
    }
}

impl<T: ComputeHintProvider> ComputeHintProvider for Option<T> {
    fn compute_hint(&self) -> ComputeHint {
        match self {
            Some(t) => t.compute_hint(),
            None => ComputeHint::Unknown,
        }
    }
}

impl<T: ComputeHintProvider, E> ComputeHintProvider for Result<T, E> {
    fn compute_hint(&self) -> ComputeHint {
        match self {
            Ok(t) => t.compute_hint(),
            Err(_) => ComputeHint::Unknown,
        }
    }
}

// Primitive types default to Unknown
impl ComputeHintProvider for i8 {}
impl ComputeHintProvider for i16 {}
impl ComputeHintProvider for i32 {}
impl ComputeHintProvider for i64 {}
impl ComputeHintProvider for i128 {}
impl ComputeHintProvider for isize {}
impl ComputeHintProvider for u8 {}
impl ComputeHintProvider for u16 {}
impl ComputeHintProvider for u32 {}
impl ComputeHintProvider for u64 {}
impl ComputeHintProvider for u128 {}
impl ComputeHintProvider for usize {}
impl ComputeHintProvider for f32 {}
impl ComputeHintProvider for f64 {}
impl ComputeHintProvider for bool {}
impl ComputeHintProvider for char {}
impl ComputeHintProvider for String {}
impl ComputeHintProvider for &str {}
impl<T> ComputeHintProvider for Vec<T> {}
impl<T, const N: usize> ComputeHintProvider for [T; N] {}
impl<T> ComputeHintProvider for [T] {}

// Tuples
impl<A: ComputeHintProvider> ComputeHintProvider for (A,) {
    fn compute_hint(&self) -> ComputeHint {
        self.0.compute_hint()
    }
}

impl<A: ComputeHintProvider, B: ComputeHintProvider> ComputeHintProvider for (A, B) {
    fn compute_hint(&self) -> ComputeHint {
        // Take the more pessimistic hint
        match (self.0.compute_hint(), self.1.compute_hint()) {
            (ComputeHint::High, _) | (_, ComputeHint::High) => ComputeHint::High,
            (ComputeHint::Medium, _) | (_, ComputeHint::Medium) => ComputeHint::Medium,
            (ComputeHint::Low, _) | (_, ComputeHint::Low) => ComputeHint::Low,
            _ => ComputeHint::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_key_from_type() {
        fn my_function() -> i32 {
            42
        }

        let key1 = FunctionKey::from_type::<fn() -> i32>();
        let key2 = FunctionKey::from_type::<fn() -> i32>();
        let key3 = FunctionKey::from_type::<fn() -> String>();

        // Same type should produce same key
        assert_eq!(key1, key2);
        // Different types should produce different keys
        assert_ne!(key1, key3);

        // Just to use the function
        let _ = my_function();
    }

    #[test]
    fn test_function_key_from_name() {
        let key1 = FunctionKey::from_name("my_handler");
        let key2 = FunctionKey::from_name("my_handler");
        let key3 = FunctionKey::from_name("other_handler");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_arm_stats_variance() {
        // Empty stats should have diffuse prior
        let empty = ArmStats::default();
        assert_eq!(empty.variance(), 1.0);

        // With prior
        let with_prior = ArmStats::with_prior(5.0, 0.5, 10.0);
        assert!((with_prior.variance() - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_compute_hint_provider_delegation() {
        struct TestItem;
        impl ComputeHintProvider for TestItem {
            fn compute_hint(&self) -> ComputeHint {
                ComputeHint::High
            }
        }

        let item = TestItem;
        assert_eq!(item.compute_hint(), ComputeHint::High);

        let boxed = Box::new(TestItem);
        assert_eq!(boxed.compute_hint(), ComputeHint::High);

        let arced = std::sync::Arc::new(TestItem);
        assert_eq!(arced.compute_hint(), ComputeHint::High);

        let optional: Option<TestItem> = Some(TestItem);
        assert_eq!(optional.compute_hint(), ComputeHint::High);

        let none: Option<TestItem> = None;
        assert_eq!(none.compute_hint(), ComputeHint::Unknown);
    }

    #[test]
    fn test_compute_hint_tuple_takes_pessimistic() {
        struct Low;
        impl ComputeHintProvider for Low {
            fn compute_hint(&self) -> ComputeHint {
                ComputeHint::Low
            }
        }

        struct High;
        impl ComputeHintProvider for High {
            fn compute_hint(&self) -> ComputeHint {
                ComputeHint::High
            }
        }

        let tuple = (Low, High);
        assert_eq!(tuple.compute_hint(), ComputeHint::High);
    }

    #[test]
    fn test_primitives_return_unknown() {
        assert_eq!(42i32.compute_hint(), ComputeHint::Unknown);
        assert_eq!(1.234f64.compute_hint(), ComputeHint::Unknown);
        assert_eq!("hello".compute_hint(), ComputeHint::Unknown);
        assert_eq!(String::from("world").compute_hint(), ComputeHint::Unknown);
    }
}
