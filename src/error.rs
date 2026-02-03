//! Error types for loom-rs.

use thiserror::Error;

/// Errors that can occur when building or using a Loom runtime.
#[derive(Debug, Error)]
pub enum LoomError {
    /// Error parsing CPU set string.
    #[error("invalid cpuset format: {0}")]
    InvalidCpuSet(String),

    /// CPU ID is not available on this system.
    #[error("CPU {0} is not available on this system")]
    CpuNotAvailable(usize),

    /// No CPUs available after applying constraints.
    #[error("no CPUs available after applying constraints")]
    NoCpusAvailable,

    /// Error extracting configuration from figment.
    #[error("configuration error: {0}")]
    Config(#[from] Box<figment::Error>),

    /// Error building tokio runtime.
    #[error("failed to build tokio runtime: {0}")]
    TokioRuntime(#[from] std::io::Error),

    /// Error building rayon thread pool.
    #[error("failed to build rayon thread pool: {0}")]
    RayonThreadPool(#[from] rayon::ThreadPoolBuildError),

    /// Error setting thread affinity.
    #[error("failed to set thread affinity for CPU {0}")]
    AffinityFailed(usize),

    /// CUDA-related errors (feature-gated).
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// NVML initialization or query error (feature-gated).
    #[cfg(feature = "cuda")]
    #[error("NVML error: {0}")]
    Nvml(#[from] nvml_wrapper::error::NvmlError),

    /// Thread count mismatch - not enough CPUs for requested threads.
    #[error("requested {requested} threads but only {available} CPUs available")]
    InsufficientCpus { requested: usize, available: usize },

    /// CUDA device cpuset has no overlap with process affinity mask.
    #[cfg(feature = "cuda")]
    #[error(
        "CUDA device cpuset {cuda_cpuset} has no overlap with process affinity mask {process_mask}"
    )]
    CudaCpusetNoOverlap {
        /// The CUDA device's local CPU set
        cuda_cpuset: String,
        /// The process affinity mask
        process_mask: String,
    },
}

/// Result type alias for Loom operations.
pub type Result<T> = std::result::Result<T, LoomError>;
