//! Builder pattern for constructing Loom runtimes.
//!
//! The builder supports multiple configuration sources using figment:
//! - Default values
//! - Config files (TOML, YAML, JSON)
//! - Environment variables
//! - Programmatic overrides
//! - CLI arguments via clap

use crate::config::LoomConfig;
use crate::error::Result;
use crate::mab::{CalibrationConfig, MabKnobs};
use crate::runtime::LoomRuntime;

use figment::providers::{Env, Format, Json, Serialized, Toml, Yaml};
use figment::Figment;
use prometheus::Registry;
use std::path::Path;

#[cfg(feature = "cuda")]
use crate::cuda::CudaDeviceSelector;

/// Builder for constructing a `LoomRuntime`.
///
/// Configuration sources are merged in the following order (later sources override earlier):
/// 1. Default values
/// 2. Config files (in order added)
/// 3. Environment variables
/// 4. Programmatic overrides
///
/// # Examples
///
/// ```ignore
/// use loom_rs::LoomBuilder;
///
/// let runtime = LoomBuilder::new()
///     .file("loom.toml")
///     .env_prefix("LOOM")
///     .prefix("myapp")
///     .tokio_threads(2)
///     .build()?;
/// ```
pub struct LoomBuilder {
    figment: Figment,
    prometheus_registry: Option<Registry>,
}

impl Default for LoomBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LoomBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoomBuilder")
            .field("figment", &self.figment)
            .field(
                "prometheus_registry",
                &self.prometheus_registry.as_ref().map(|_| "<Registry>"),
            )
            .finish()
    }
}

impl LoomBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            figment: Figment::from(Serialized::defaults(LoomConfig::default())),
            prometheus_registry: None,
        }
    }

    /// Add a configuration file.
    ///
    /// Supports TOML, YAML, and JSON formats (detected by extension).
    /// Files are merged in the order they are added.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the configuration file
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let builder = LoomBuilder::new()
    ///     .file("loom.toml")
    ///     .file("loom.local.toml"); // Overrides values from loom.toml
    /// ```
    pub fn file<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path = path.as_ref();
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        self.figment = match extension.to_lowercase().as_str() {
            "toml" => self.figment.merge(Toml::file(path)),
            "yaml" | "yml" => self.figment.merge(Yaml::file(path)),
            "json" => self.figment.merge(Json::file(path)),
            _ => {
                // Default to TOML
                self.figment.merge(Toml::file(path))
            }
        };
        self
    }

    /// Add environment variables with a prefix.
    ///
    /// Environment variables are expected in the format `{PREFIX}_{KEY}`,
    /// e.g., `LOOM_CPUSET`, `LOOM_TOKIO_THREADS`.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The environment variable prefix (without trailing underscore)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Will read MYAPP_CPUSET, MYAPP_TOKIO_THREADS, etc.
    /// let builder = LoomBuilder::new().env_prefix("MYAPP");
    /// ```
    pub fn env_prefix(mut self, prefix: &str) -> Self {
        self.figment = self.figment.merge(Env::prefixed(prefix).split("_"));
        self
    }

    /// Set the thread name prefix.
    ///
    /// Thread names will be formatted as `{prefix}-tokio-{NNNN}` and
    /// `{prefix}-rayon-{NNNN}`.
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.figment = self
            .figment
            .merge(Serialized::default("prefix", prefix.into()));
        self
    }

    /// Set the CPU set string.
    ///
    /// Format: `"0-7,16-23"` for ranges, `"0,2,4,6"` for individual CPUs.
    pub fn cpuset(mut self, cpuset: impl Into<String>) -> Self {
        self.figment = self
            .figment
            .merge(Serialized::default("cpuset", cpuset.into()));
        self
    }

    /// Set the number of tokio worker threads.
    ///
    /// Default is 1 thread.
    pub fn tokio_threads(mut self, n: usize) -> Self {
        self.figment = self.figment.merge(Serialized::default("tokio_threads", n));
        self
    }

    /// Set the number of rayon threads.
    ///
    /// Default is the remaining CPUs after tokio threads are allocated.
    pub fn rayon_threads(mut self, n: usize) -> Self {
        self.figment = self.figment.merge(Serialized::default("rayon_threads", n));
        self
    }

    /// Set the compute pool size per result type.
    ///
    /// Each unique result type `R` used with `spawn_compute::<F, R>()` gets its own
    /// pool of this size. Default is 64.
    ///
    /// # Guidelines
    ///
    /// - Set to your maximum expected concurrency per type
    /// - Higher values use more memory but reduce allocation
    /// - Undersized pools fall back to allocation (still correct)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let runtime = LoomBuilder::new()
    ///     .compute_pool_size(128)  // For high-concurrency workloads
    ///     .build()?;
    /// ```
    pub fn compute_pool_size(mut self, size: usize) -> Self {
        self.figment = self
            .figment
            .merge(Serialized::default("compute_pool_size", size));
        self
    }

    /// Set the MAB scheduler knobs.
    ///
    /// These control the adaptive scheduling decisions. Most users don't need
    /// to modify these. See [`MabKnobs`] for details.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use loom_rs::{LoomBuilder, MabKnobs};
    ///
    /// let runtime = LoomBuilder::new()
    ///     .mab_knobs(MabKnobs::default().with_k_starve(0.2))
    ///     .build()?;
    /// ```
    pub fn mab_knobs(mut self, knobs: MabKnobs) -> Self {
        self.figment = self.figment.merge(Serialized::default("mab_knobs", knobs));
        self
    }

    /// Enable calibration at runtime startup.
    ///
    /// Calibration measures the overhead of offloading work to rayon,
    /// which helps the MAB make better decisions for borderline workloads.
    ///
    /// Default: disabled (for fast unit test startup).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let runtime = LoomBuilder::new()
    ///     .calibrate(true)
    ///     .build()?;
    /// ```
    pub fn calibrate(mut self, enabled: bool) -> Self {
        let config = CalibrationConfig {
            enabled,
            ..Default::default()
        };
        self.figment = self
            .figment
            .merge(Serialized::default("calibration", config));
        self
    }

    /// Set calibration configuration.
    ///
    /// Allows full control over calibration parameters.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use loom_rs::mab::CalibrationConfig;
    ///
    /// let runtime = LoomBuilder::new()
    ///     .calibration_config(
    ///         CalibrationConfig::new()
    ///             .enabled()
    ///             .sample_count(500)
    ///     )
    ///     .build()?;
    /// ```
    pub fn calibration_config(mut self, config: CalibrationConfig) -> Self {
        self.figment = self
            .figment
            .merge(Serialized::default("calibration", config));
        self
    }

    /// Provide an external Prometheus registry for metrics exposition.
    ///
    /// When a registry is provided, loom runtime metrics will be registered
    /// and available for Prometheus scraping.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use prometheus::Registry;
    ///
    /// let registry = Registry::new();
    /// let runtime = LoomBuilder::new()
    ///     .prometheus_registry(registry.clone())
    ///     .build()?;
    ///
    /// // Later: expose via HTTP endpoint
    /// let encoder = prometheus::TextEncoder::new();
    /// let metric_families = registry.gather();
    /// // encoder.encode(&metric_families, &mut buffer)?;
    /// ```
    pub fn prometheus_registry(mut self, registry: Registry) -> Self {
        self.prometheus_registry = Some(registry);
        self
    }

    /// Set the CUDA device by ID.
    ///
    /// This will configure the runtime to use CPUs local to the specified
    /// CUDA device's NUMA node.
    #[cfg(feature = "cuda")]
    pub fn cuda_device_id(mut self, id: u32) -> Self {
        self.figment = self.figment.merge(Serialized::default(
            "cuda_device",
            CudaDeviceSelector::DeviceId(id),
        ));
        self
    }

    /// Set the CUDA device by UUID.
    ///
    /// This will configure the runtime to use CPUs local to the specified
    /// CUDA device's NUMA node.
    #[cfg(feature = "cuda")]
    pub fn cuda_device_uuid(mut self, uuid: impl Into<String>) -> Self {
        self.figment = self.figment.merge(Serialized::default(
            "cuda_device",
            CudaDeviceSelector::Uuid(uuid.into()),
        ));
        self
    }

    /// Apply CLI argument overrides.
    ///
    /// This method applies any non-None values from the `LoomArgs` struct.
    pub fn with_cli_args(mut self, args: &LoomArgs) -> Self {
        if let Some(ref prefix) = args.loom_prefix {
            self.figment = self
                .figment
                .merge(Serialized::default("prefix", prefix.clone()));
        }
        if let Some(ref cpuset) = args.loom_cpuset {
            self.figment = self
                .figment
                .merge(Serialized::default("cpuset", cpuset.clone()));
        }
        if let Some(threads) = args.loom_tokio_threads {
            self.figment = self
                .figment
                .merge(Serialized::default("tokio_threads", threads));
        }
        if let Some(threads) = args.loom_rayon_threads {
            self.figment = self
                .figment
                .merge(Serialized::default("rayon_threads", threads));
        }
        #[cfg(feature = "cuda")]
        if let Some(ref device) = args.loom_cuda_device {
            // Parse device string - could be a number or UUID
            if let Ok(id) = device.parse::<u32>() {
                self.figment = self.figment.merge(Serialized::default(
                    "cuda_device",
                    CudaDeviceSelector::DeviceId(id),
                ));
            } else {
                self.figment = self.figment.merge(Serialized::default(
                    "cuda_device",
                    CudaDeviceSelector::Uuid(device.clone()),
                ));
            }
        }
        self
    }

    /// Build the runtime.
    ///
    /// This extracts the configuration and constructs the tokio and rayon
    /// runtimes with CPU pinning.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Configuration extraction fails
    /// - CPU set is invalid or contains unavailable CPUs
    /// - Runtime construction fails
    pub fn build(self) -> Result<LoomRuntime> {
        let mut config: LoomConfig = self.figment.extract().map_err(Box::new)?;
        config.prometheus_registry = self.prometheus_registry;
        let pool_size = config.compute_pool_size;
        LoomRuntime::from_config(config, pool_size)
    }
}

/// CLI arguments for Loom configuration.
///
/// Use with clap's `Parser` derive macro. These arguments can be applied
/// to a `LoomBuilder` using `with_cli_args`.
///
/// # Examples
///
/// ```ignore
/// use clap::Parser;
/// use loom_rs::{LoomBuilder, LoomArgs};
///
/// #[derive(Parser)]
/// struct MyArgs {
///     #[command(flatten)]
///     loom: LoomArgs,
///     // ... other args
/// }
///
/// let args = MyArgs::parse();
/// let runtime = LoomBuilder::new()
///     .with_cli_args(&args.loom)
///     .build()?;
/// ```
#[derive(Debug, Default, Clone, clap::Args)]
pub struct LoomArgs {
    /// Thread name prefix
    #[arg(long)]
    pub loom_prefix: Option<String>,

    /// CPU set (e.g., "0-7,16-23")
    #[arg(long)]
    pub loom_cpuset: Option<String>,

    /// Number of tokio worker threads
    #[arg(long)]
    pub loom_tokio_threads: Option<usize>,

    /// Number of rayon threads
    #[arg(long)]
    pub loom_rayon_threads: Option<usize>,

    /// CUDA device ID or UUID
    #[cfg(feature = "cuda")]
    #[arg(long)]
    pub loom_cuda_device: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let config: LoomConfig = LoomBuilder::new().figment.extract().unwrap();
        assert_eq!(config.prefix, "loom");
        assert!(config.cpuset.is_none());
        assert!(config.tokio_threads.is_none());
        assert!(config.rayon_threads.is_none());
    }

    #[test]
    fn test_builder_programmatic_override() {
        let config: LoomConfig = LoomBuilder::new()
            .prefix("myapp")
            .cpuset("0-3")
            .tokio_threads(2)
            .rayon_threads(6)
            .figment
            .extract()
            .unwrap();

        assert_eq!(config.prefix, "myapp");
        assert_eq!(config.cpuset, Some("0-3".to_string()));
        assert_eq!(config.tokio_threads, Some(2));
        assert_eq!(config.rayon_threads, Some(6));
    }

    #[test]
    fn test_builder_cli_args() {
        let args = LoomArgs {
            loom_prefix: Some("cliapp".to_string()),
            loom_cpuset: Some("4-7".to_string()),
            loom_tokio_threads: Some(1),
            loom_rayon_threads: Some(3),
            #[cfg(feature = "cuda")]
            loom_cuda_device: None,
        };

        let config: LoomConfig = LoomBuilder::new()
            .prefix("original")
            .with_cli_args(&args)
            .figment
            .extract()
            .unwrap();

        // CLI args should override programmatic values
        assert_eq!(config.prefix, "cliapp");
        assert_eq!(config.cpuset, Some("4-7".to_string()));
        assert_eq!(config.tokio_threads, Some(1));
        assert_eq!(config.rayon_threads, Some(3));
    }

    #[test]
    fn test_builder_partial_cli_args() {
        let args = LoomArgs {
            loom_prefix: Some("cliapp".to_string()),
            loom_cpuset: None,
            loom_tokio_threads: None,
            loom_rayon_threads: None,
            #[cfg(feature = "cuda")]
            loom_cuda_device: None,
        };

        let config: LoomConfig = LoomBuilder::new()
            .prefix("original")
            .cpuset("0-3")
            .with_cli_args(&args)
            .figment
            .extract()
            .unwrap();

        // Only prefix should be overridden
        assert_eq!(config.prefix, "cliapp");
        assert_eq!(config.cpuset, Some("0-3".to_string()));
    }
}
