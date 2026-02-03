//! Configuration types for loom-rs runtime.

use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
use crate::cuda::CudaDeviceSelector;

use crate::mab::{CalibrationConfig, MabKnobs};
use crate::pool::DEFAULT_POOL_SIZE;
use prometheus::Registry;

/// Configuration for the Loom runtime.
///
/// This struct can be deserialized from TOML, YAML, JSON, or environment variables
/// using figment.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoomConfig {
    /// Thread name prefix (default: "loom")
    #[serde(default = "default_prefix")]
    pub prefix: String,

    /// Number of tokio worker threads (default: 1)
    #[serde(default)]
    pub tokio_threads: Option<usize>,

    /// Number of rayon threads (default: remaining CPUs after tokio threads)
    #[serde(default)]
    pub rayon_threads: Option<usize>,

    /// Size of compute task pool per result type (default: 64)
    #[serde(default = "default_compute_pool_size")]
    pub compute_pool_size: usize,

    /// Whether to pin threads to specific CPUs (default: true)
    #[serde(default = "default_pin_threads")]
    pub pin_threads: bool,

    /// CUDA device selection (feature-gated)
    #[cfg(feature = "cuda")]
    #[serde(default)]
    pub cuda_device: Option<CudaDeviceSelector>,

    /// MAB scheduler configuration knobs.
    /// If None, default knobs are used.
    #[serde(default)]
    pub mab_knobs: Option<MabKnobs>,

    /// Calibration configuration.
    /// If None or disabled, calibration is skipped.
    #[serde(default)]
    pub calibration: Option<CalibrationConfig>,

    /// Prometheus registry for metrics exposition.
    /// If provided, metrics will be registered for scraping.
    /// Not serializable - must be set programmatically.
    #[serde(skip)]
    pub prometheus_registry: Option<Registry>,
}

fn default_compute_pool_size() -> usize {
    DEFAULT_POOL_SIZE
}

fn default_prefix() -> String {
    "loom".to_string()
}

fn default_pin_threads() -> bool {
    true
}

impl Default for LoomConfig {
    fn default() -> Self {
        Self {
            prefix: default_prefix(),
            tokio_threads: None,
            rayon_threads: None,
            compute_pool_size: default_compute_pool_size(),
            pin_threads: default_pin_threads(),
            #[cfg(feature = "cuda")]
            cuda_device: None,
            mab_knobs: None,
            calibration: None,
            prometheus_registry: None,
        }
    }
}

impl LoomConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the effective number of tokio threads.
    ///
    /// Returns the configured value or 1 as the default.
    pub fn effective_tokio_threads(&self) -> usize {
        self.tokio_threads.unwrap_or(1)
    }

    /// Get the effective number of rayon threads.
    ///
    /// Returns the configured value or calculates based on available CPUs
    /// minus tokio threads.
    pub fn effective_rayon_threads(&self, available_cpus: usize) -> usize {
        self.rayon_threads
            .unwrap_or_else(|| available_cpus.saturating_sub(self.effective_tokio_threads()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LoomConfig::default();
        assert_eq!(config.prefix, "loom");
        assert!(config.tokio_threads.is_none());
        assert!(config.rayon_threads.is_none());
        assert_eq!(config.compute_pool_size, 64);
        assert!(config.pin_threads);
        assert!(config.mab_knobs.is_none());
        assert!(config.calibration.is_none());
    }

    #[test]
    fn test_effective_tokio_threads() {
        let mut config = LoomConfig::default();
        assert_eq!(config.effective_tokio_threads(), 1);

        config.tokio_threads = Some(4);
        assert_eq!(config.effective_tokio_threads(), 4);
    }

    #[test]
    fn test_effective_rayon_threads() {
        let mut config = LoomConfig::default();
        // With 8 CPUs and 1 tokio thread, should get 7 rayon threads
        assert_eq!(config.effective_rayon_threads(8), 7);

        config.tokio_threads = Some(2);
        // With 8 CPUs and 2 tokio threads, should get 6 rayon threads
        assert_eq!(config.effective_rayon_threads(8), 6);

        config.rayon_threads = Some(4);
        // Explicit override
        assert_eq!(config.effective_rayon_threads(8), 4);
    }

    #[test]
    fn test_deserialize_config() {
        let toml = r#"
            prefix = "myapp"
            tokio_threads = 2
            rayon_threads = 6
            compute_pool_size = 128
        "#;

        let config: LoomConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.prefix, "myapp");
        assert_eq!(config.tokio_threads, Some(2));
        assert_eq!(config.rayon_threads, Some(6));
        assert_eq!(config.compute_pool_size, 128);
        // pin_threads should default to true when not specified
        assert!(config.pin_threads);
    }

    #[test]
    fn test_pin_threads_default() {
        let config = LoomConfig::default();
        assert!(config.pin_threads, "pin_threads should default to true");
    }

    #[test]
    fn test_pin_threads_deserialize() {
        let toml = r#"
            prefix = "test"
            pin_threads = false
        "#;

        let config: LoomConfig = toml::from_str(toml).unwrap();
        assert!(!config.pin_threads, "pin_threads should be false when set");
    }

    #[test]
    fn test_pin_threads_serialize() {
        let config = LoomConfig {
            pin_threads: false,
            ..Default::default()
        };

        let serialized = toml::to_string(&config).unwrap();
        assert!(
            serialized.contains("pin_threads = false"),
            "serialized config should contain pin_threads = false"
        );
    }
}
