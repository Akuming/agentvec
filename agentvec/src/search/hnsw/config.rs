//! HNSW configuration parameters.

use serde::{Deserialize, Serialize};

/// Configuration for HNSW index construction and search.
///
/// The default parameters are tuned for a balance of recall and performance
/// at the 10K-100K vector scale typical of AI agent memory workloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Number of connections to establish per node during construction.
    /// Higher values improve recall but increase memory and build time.
    /// Typical range: 8-64. Default: 16.
    pub m: usize,

    /// Maximum connections at layer 0 (base layer).
    /// Usually set to 2*M for better recall at the densest layer.
    /// Default: 32.
    pub m_max0: usize,

    /// Maximum connections at layers above 0.
    /// Usually equal to M.
    /// Default: 16.
    pub m_max: usize,

    /// Size of dynamic candidate list during construction.
    /// Higher values improve graph quality but slow down inserts.
    /// Typical range: 100-500. Default: 200.
    pub ef_construction: usize,

    /// Default size of dynamic candidate list during search.
    /// Can be overridden per-query. Higher values improve recall.
    /// Typical range: 10-200. Default: 50.
    pub ef_search: usize,

    /// Level generation factor (1/ln(M)).
    /// Controls the probability of a node appearing at higher layers.
    /// Computed automatically from M if not specified.
    pub ml: f64,

    /// Minimum number of vectors before HNSW is auto-enabled.
    /// Below this threshold, exact search is used.
    /// Default: 1000.
    pub min_vectors: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max0: 2 * m,
            m_max: m,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
            min_vectors: 1000,
        }
    }
}

impl HnswConfig {
    /// Create a new configuration with the specified M value.
    /// Other parameters are derived from M.
    pub fn with_m(m: usize) -> Self {
        Self {
            m,
            m_max0: 2 * m,
            m_max: m,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
            min_vectors: 1000,
        }
    }

    /// Create a configuration optimized for high recall (>95%).
    /// Good balance of construction speed and search quality.
    pub fn high_recall() -> Self {
        Self {
            m: 32,
            m_max0: 64,
            m_max: 32,
            ef_construction: 200,
            ef_search: 150,
            ml: 1.0 / 32.0_f64.ln(),
            min_vectors: 1000,
        }
    }

    /// Create a balanced configuration for production at 100K+ scale.
    /// Targets >95% recall with faster construction than previous settings.
    pub fn production() -> Self {
        Self {
            m: 32,
            m_max0: 64,
            m_max: 32,
            ef_construction: 250,
            ef_search: 200,
            ml: 1.0 / 32.0_f64.ln(),
            min_vectors: 1000,
        }
    }

    /// Create a configuration for maximum recall (>98%).
    /// Slower construction but highest quality graph.
    pub fn max_recall() -> Self {
        Self {
            m: 48,
            m_max0: 96,
            m_max: 48,
            ef_construction: 400,
            ef_search: 300,
            ml: 1.0 / 48.0_f64.ln(),
            min_vectors: 1000,
        }
    }

    /// Create a configuration optimized for speed.
    /// Faster construction and search, but lower recall (~90%).
    pub fn fast() -> Self {
        Self {
            m: 12,
            m_max0: 24,
            m_max: 12,
            ef_construction: 100,
            ef_search: 50,
            ml: 1.0 / 12.0_f64.ln(),
            min_vectors: 1000,
        }
    }

    /// Create a configuration optimized for high throughput builds.
    /// Targets ~1000 vec/s build rate with ~90% recall.
    /// Use higher ef_search to compensate for lower construction quality.
    pub fn high_throughput() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            m_max: 16,
            ef_construction: 100,
            ef_search: 300, // Higher search ef compensates for lower construction quality
            ml: 1.0 / 16.0_f64.ln(),
            min_vectors: 1000,
        }
    }

    /// Create a configuration optimized for PQ-accelerated construction.
    /// Uses higher ef_construction to compensate for PQ approximation errors.
    /// Achieves >95% recall at >1000 vec/s with PQ acceleration.
    pub fn pq_optimized() -> Self {
        Self {
            m: 32,
            m_max0: 64,
            m_max: 32,
            ef_construction: 1200, // Higher ef for larger scale (250K+)
            ef_search: 400,        // Higher search ef for better recall
            ml: 1.0 / 32.0_f64.ln(),
            min_vectors: 1000,
        }
    }

    /// Set the minimum vectors threshold for auto-enabling HNSW.
    pub fn with_min_vectors(mut self, min: usize) -> Self {
        self.min_vectors = min;
        self
    }

    /// Set the ef_search parameter.
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Set the ef_construction parameter.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HnswConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.m_max0, 32);
        assert_eq!(config.m_max, 16);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 50);
        assert!((config.ml - 1.0 / 16.0_f64.ln()).abs() < 0.001);
        assert_eq!(config.min_vectors, 1000);
    }

    #[test]
    fn test_with_m() {
        let config = HnswConfig::with_m(32);
        assert_eq!(config.m, 32);
        assert_eq!(config.m_max0, 64);
        assert_eq!(config.m_max, 32);
    }

    #[test]
    fn test_builder_pattern() {
        let config = HnswConfig::default()
            .with_min_vectors(500)
            .with_ef_search(100);
        assert_eq!(config.min_vectors, 500);
        assert_eq!(config.ef_search, 100);
    }

    #[test]
    fn test_serialization() {
        let config = HnswConfig::default();
        let serialized = bincode::serialize(&config).unwrap();
        let deserialized: HnswConfig = bincode::deserialize(&serialized).unwrap();
        assert_eq!(config.m, deserialized.m);
        assert_eq!(config.ef_construction, deserialized.ef_construction);
    }
}
