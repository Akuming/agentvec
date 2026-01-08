//! Configuration types for AgentVec.

use serde::{Deserialize, Serialize};

use crate::search::HnswConfig;

/// Maximum allowed vector dimensions.
pub const MAX_DIMENSIONS: usize = 65536;

/// Threshold above which a warning is logged for high-dimensional vectors.
pub const WARN_DIMENSIONS: usize = 4096;

/// Distance metric for vector similarity calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Metric {
    /// Cosine similarity (normalized dot product).
    ///
    /// Vectors are L2-normalized on ingestion, so cosine similarity
    /// becomes a simple dot product during search. This is the default
    /// and works well with most embedding models (OpenAI, Cohere, etc.).
    Cosine,

    /// Raw dot product similarity.
    ///
    /// Use this when vectors are pre-normalized or when the magnitude
    /// carries semantic meaning.
    Dot,

    /// Euclidean distance (L2).
    ///
    /// Returns the squared L2 distance. Lower values indicate more similar vectors.
    /// Commonly used with image embeddings.
    L2,
}

impl Default for Metric {
    fn default() -> Self {
        Metric::Cosine
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Metric::Cosine => write!(f, "cosine"),
            Metric::Dot => write!(f, "dot"),
            Metric::L2 => write!(f, "l2"),
        }
    }
}

impl std::str::FromStr for Metric {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(Metric::Cosine),
            "dot" => Ok(Metric::Dot),
            "l2" | "euclidean" => Ok(Metric::L2),
            _ => Err(format!(
                "Invalid metric '{}'. Expected: cosine, dot, or l2",
                s
            )),
        }
    }
}

/// Configuration for a collection.
///
/// This is stored in the database and used to validate operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Collection name (unique within database).
    pub name: String,

    /// Vector dimensions (fixed at creation).
    pub dimensions: usize,

    /// Distance metric.
    pub metric: Metric,

    /// Unix timestamp when collection was created.
    pub created_at: u64,

    /// Optional HNSW index configuration.
    /// If None, exact search is used. If Some, HNSW is used for approximate search.
    #[serde(default)]
    pub hnsw: Option<HnswConfig>,
}

impl CollectionConfig {
    /// Create a new collection configuration.
    ///
    /// # Arguments
    ///
    /// * `name` - Collection name
    /// * `dimensions` - Vector dimensions
    /// * `metric` - Distance metric
    ///
    /// # Returns
    ///
    /// A new `CollectionConfig` with `created_at` set to current time.
    pub fn new(name: impl Into<String>, dimensions: usize, metric: Metric) -> Self {
        Self {
            name: name.into(),
            dimensions,
            metric,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            hnsw: None,
        }
    }

    /// Create a new collection configuration with HNSW indexing enabled.
    pub fn with_hnsw(name: impl Into<String>, dimensions: usize, metric: Metric, hnsw_config: HnswConfig) -> Self {
        Self {
            name: name.into(),
            dimensions,
            metric,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            hnsw: Some(hnsw_config),
        }
    }

    /// Calculate the byte size of a single vector slot.
    #[inline]
    pub const fn slot_size_bytes(&self) -> usize {
        self.dimensions * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_default() {
        assert_eq!(Metric::default(), Metric::Cosine);
    }

    #[test]
    fn test_metric_display() {
        assert_eq!(Metric::Cosine.to_string(), "cosine");
        assert_eq!(Metric::Dot.to_string(), "dot");
        assert_eq!(Metric::L2.to_string(), "l2");
    }

    #[test]
    fn test_metric_from_str() {
        assert_eq!("cosine".parse::<Metric>().unwrap(), Metric::Cosine);
        assert_eq!("COSINE".parse::<Metric>().unwrap(), Metric::Cosine);
        assert_eq!("dot".parse::<Metric>().unwrap(), Metric::Dot);
        assert_eq!("l2".parse::<Metric>().unwrap(), Metric::L2);
        assert_eq!("euclidean".parse::<Metric>().unwrap(), Metric::L2);
        assert!("invalid".parse::<Metric>().is_err());
    }

    #[test]
    fn test_collection_config_new() {
        let config = CollectionConfig::new("test", 384, Metric::Cosine);
        assert_eq!(config.name, "test");
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.metric, Metric::Cosine);
        assert!(config.created_at > 0);
    }

    #[test]
    fn test_slot_size_bytes() {
        let config = CollectionConfig::new("test", 384, Metric::Cosine);
        assert_eq!(config.slot_size_bytes(), 384 * 4); // 384 f32s = 1536 bytes
    }

    #[test]
    fn test_config_serialization() {
        let config = CollectionConfig::new("test", 384, Metric::Cosine);
        let bytes = bincode::serialize(&config).unwrap();
        let decoded: CollectionConfig = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.name, config.name);
        assert_eq!(decoded.dimensions, config.dimensions);
        assert_eq!(decoded.metric, config.metric);
    }
}
