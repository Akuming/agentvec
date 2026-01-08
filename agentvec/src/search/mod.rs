//! Search functionality for AgentVec.
//!
//! This module provides:
//!
//! - [`distance`]: Distance/similarity functions (dot, L2, cosine)
//! - [`exact`]: Brute-force exact search
//! - [`hnsw`]: HNSW approximate nearest neighbor search
//! - [`quantize`]: Scalar quantization for vector compression
//! - [`pq`]: Product quantization for fast approximate distance computation
//! - [`SearchResult`]: Result type for search operations

pub mod distance;
mod exact;
pub mod hnsw;
pub mod pq;
pub mod quantize;

pub use distance::{dot_product, dot_product_scalar, l2_squared, l2_squared_scalar, normalize_l2, DistanceMetric};
pub use quantize::{ScalarQuantizer, QuantizedVectors, QueryTables, SignedQuantizer, SignedQuantizedVectors};
pub use pq::{ProductQuantizer, PQConfig, PQDistanceTable, PQVectors};
pub use exact::ExactSearch;
pub use hnsw::{HnswConfig, HnswIndex};

use serde_json::Value as JsonValue;

/// Result of a search operation.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Record ID.
    pub id: String,

    /// Similarity/distance score.
    ///
    /// For cosine and dot: higher is more similar.
    /// For L2: lower is more similar.
    pub score: f32,

    /// Record metadata.
    pub metadata: JsonValue,
}

impl SearchResult {
    /// Create a new search result.
    pub fn new(id: impl Into<String>, score: f32, metadata: JsonValue) -> Self {
        Self {
            id: id.into(),
            score,
            metadata,
        }
    }
}

/// Helper for maintaining top-k results.
#[derive(Debug)]
pub struct TopK {
    /// Maximum number of results to keep.
    k: usize,

    /// Current results, sorted by score descending (for similarity metrics).
    results: Vec<(f32, usize)>, // (score, index)

    /// Whether higher scores are better (true for similarity, false for distance).
    higher_is_better: bool,
}

impl TopK {
    /// Create a new top-k tracker.
    ///
    /// # Arguments
    ///
    /// * `k` - Maximum number of results
    /// * `higher_is_better` - True for similarity metrics (cosine, dot), false for distance (L2)
    pub fn new(k: usize, higher_is_better: bool) -> Self {
        Self {
            k,
            results: Vec::with_capacity(k + 1),
            higher_is_better,
        }
    }

    /// Try to add a result.
    ///
    /// # Arguments
    ///
    /// * `score` - The score for this result
    /// * `index` - An index or identifier for the result
    ///
    /// # Returns
    ///
    /// True if the result was added to top-k.
    pub fn push(&mut self, score: f32, index: usize) -> bool {
        // Check if we should add this result
        if self.results.len() >= self.k {
            let worst = if self.higher_is_better {
                self.results.last().map(|(s, _)| *s).unwrap_or(f32::NEG_INFINITY)
            } else {
                self.results.last().map(|(s, _)| *s).unwrap_or(f32::INFINITY)
            };

            let dominated = if self.higher_is_better {
                score <= worst
            } else {
                score >= worst
            };

            if dominated {
                return false;
            }
        }

        // Add the result
        self.results.push((score, index));

        // Sort and truncate
        if self.higher_is_better {
            self.results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            self.results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        if self.results.len() > self.k {
            self.results.truncate(self.k);
        }

        true
    }

    /// Get the current worst score in the top-k.
    ///
    /// Returns None if no results yet.
    pub fn worst_score(&self) -> Option<f32> {
        self.results.last().map(|(s, _)| *s)
    }

    /// Get the results as (score, index) pairs.
    pub fn results(&self) -> &[(f32, usize)] {
        &self.results
    }

    /// Consume and return the results.
    pub fn into_results(self) -> Vec<(f32, usize)> {
        self.results
    }

    /// Get the number of results.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_search_result() {
        let result = SearchResult::new("test_id", 0.95, json!({"key": "value"}));
        assert_eq!(result.id, "test_id");
        assert!((result.score - 0.95).abs() < f32::EPSILON);
        assert_eq!(result.metadata["key"], "value");
    }

    #[test]
    fn test_topk_higher_is_better() {
        let mut topk = TopK::new(3, true);

        topk.push(0.5, 0);
        topk.push(0.8, 1);
        topk.push(0.3, 2);
        topk.push(0.9, 3);
        topk.push(0.7, 4);

        let results = topk.into_results();
        assert_eq!(results.len(), 3);

        // Should be sorted descending
        assert_eq!(results[0].1, 3); // 0.9
        assert_eq!(results[1].1, 1); // 0.8
        assert_eq!(results[2].1, 4); // 0.7
    }

    #[test]
    fn test_topk_lower_is_better() {
        let mut topk = TopK::new(3, false);

        topk.push(0.5, 0);
        topk.push(0.8, 1);
        topk.push(0.3, 2);
        topk.push(0.9, 3);
        topk.push(0.1, 4);

        let results = topk.into_results();
        assert_eq!(results.len(), 3);

        // Should be sorted ascending (lower is better)
        assert_eq!(results[0].1, 4); // 0.1
        assert_eq!(results[1].1, 2); // 0.3
        assert_eq!(results[2].1, 0); // 0.5
    }

    #[test]
    fn test_topk_worst_score() {
        let mut topk = TopK::new(2, true);

        assert!(topk.worst_score().is_none());

        topk.push(0.5, 0);
        assert!((topk.worst_score().unwrap() - 0.5).abs() < f32::EPSILON);

        topk.push(0.8, 1);
        assert!((topk.worst_score().unwrap() - 0.5).abs() < f32::EPSILON);

        topk.push(0.9, 2);
        assert!((topk.worst_score().unwrap() - 0.8).abs() < f32::EPSILON);
    }
}
