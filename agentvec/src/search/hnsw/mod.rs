//! HNSW (Hierarchical Navigable Small World) index implementation.
//!
//! HNSW is an approximate nearest neighbor (ANN) algorithm that provides
//! efficient search with sub-linear time complexity. It works by building
//! a multi-layer graph where each layer is a navigable small world graph.
//!
//! # Algorithm Overview
//!
//! - **Multi-layer structure**: Higher layers are sparser and allow fast
//!   navigation to the region of interest. Layer 0 is the densest.
//!
//! - **Insertion**: Each new vector is assigned a random layer using an
//!   exponential distribution. Connections are established at all layers
//!   from the assigned layer down to layer 0.
//!
//! - **Search**: Starting from the entry point at the top layer, greedy
//!   search finds the closest node. This process continues layer by layer
//!   until reaching layer 0, where beam search with `ef` candidates
//!   provides the final results.
//!
//! # Configuration
//!
//! Key parameters:
//! - `M`: Number of connections per node (typically 16-32)
//! - `ef_construction`: Beam width during index building (typically 100-400)
//! - `ef_search`: Beam width during search (typically 50-200)
//!
//! Higher values improve recall but increase memory and latency.
//!
//! # Example
//!
//! ```ignore
//! use agentvec::search::hnsw::{HnswConfig, HnswIndex};
//!
//! // Create index with default parameters
//! let config = HnswConfig::default();
//! let mut index = HnswIndex::new(config, Metric::Cosine);
//!
//! // Insert vectors
//! index.insert(slot, id, &vectors);
//!
//! // Search
//! let results = index.search(&query, k, ef_search, &vectors, &deleted);
//! ```

mod config;
mod graph;
mod insert;
mod parallel;
mod persist;
mod search;

pub use config::HnswConfig;
pub use graph::{HnswGraph, HnswNode};
pub use parallel::build_parallel;

use std::collections::HashSet;
use std::path::Path;

use parking_lot::RwLock;

use crate::config::Metric;
use crate::error::Result;
use crate::search::SearchResult;
use crate::storage::{MetadataStorage, VectorStorage};

/// HNSW index for approximate nearest neighbor search.
///
/// This is a thread-safe wrapper around `HnswGraph` that provides
/// the public API for HNSW operations.
pub struct HnswIndex {
    /// The underlying graph structure.
    graph: RwLock<HnswGraph>,

    /// Distance metric used for this index.
    metric: Metric,

    /// Set of deleted slots (excluded from search results).
    deleted: RwLock<HashSet<u64>>,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(config: HnswConfig, metric: Metric) -> Self {
        Self {
            graph: RwLock::new(HnswGraph::new(config)),
            metric,
            deleted: RwLock::new(HashSet::new()),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> HnswConfig {
        self.graph.read().config().clone()
    }

    /// Get the number of nodes in the index.
    pub fn len(&self) -> usize {
        self.graph.read().node_count()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.graph.read().is_empty()
    }

    /// Get the number of deleted nodes.
    pub fn deleted_count(&self) -> usize {
        self.deleted.read().len()
    }

    /// Insert a vector into the index.
    ///
    /// # Arguments
    ///
    /// * `slot` - Slot offset in VectorStorage
    /// * `id` - Record ID
    /// * `vectors` - Vector storage containing the vector data
    ///
    /// # Returns
    ///
    /// The internal node ID assigned to the vector.
    pub fn insert(&self, slot: u64, id: String, vectors: &VectorStorage) -> u32 {
        self.graph.write().insert(slot, id, vectors, self.metric)
    }

    /// Insert multiple vectors incrementally into an existing index.
    ///
    /// Uses reduced ef_construction for speed. For incremental inserts,
    /// lower ef is acceptable since the graph already has good structure.
    ///
    /// # Arguments
    ///
    /// * `records` - Vector of (slot, id) pairs to insert
    /// * `vectors` - Vector storage containing the vector data
    ///
    /// # Returns
    ///
    /// Number of vectors inserted.
    pub fn insert_batch(&self, records: &[(u64, String)], vectors: &VectorStorage) -> usize {
        // Filter out already-indexed records
        let new_records: Vec<_> = {
            let graph = self.graph.read();
            records
                .iter()
                .filter(|(slot, _)| graph.get_node_id_by_slot(*slot).is_none())
                .cloned()
                .collect()
        };

        if new_records.is_empty() {
            return 0;
        }

        let count = new_records.len();

        // Use lower ef for incremental inserts (75 balances speed and recall)
        // The existing graph structure helps guide new insertions
        let incremental_ef = 75.min(self.graph.read().config().ef_construction);

        let mut graph = self.graph.write();
        for (slot, id) in &new_records {
            graph.insert_with_ef(*slot, id.clone(), vectors, self.metric, incremental_ef);
        }

        count
    }

    /// Check if a slot is already indexed.
    pub fn contains_slot(&self, slot: u64) -> bool {
        self.graph.read().get_node_id_by_slot(slot).is_some()
    }

    /// Mark a slot as deleted.
    ///
    /// Deleted slots are excluded from search results but remain in the
    /// graph structure until a rebuild.
    pub fn mark_deleted(&self, slot: u64) {
        self.deleted.write().insert(slot);
        self.graph.write().mark_deleted(slot);
    }

    /// Clear deleted slots (after rebuild or compaction).
    pub fn clear_deleted(&self) {
        self.deleted.write().clear();
    }

    /// Search for nearest neighbors.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector (should be normalized for cosine metric)
    /// * `k` - Number of results to return
    /// * `ef` - Size of dynamic candidate list (use config.ef_search if None)
    /// * `vectors` - Vector storage for reading vector data
    ///
    /// # Returns
    ///
    /// Vector of (node_id, slot, id, distance) for the k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        vectors: &VectorStorage,
    ) -> Vec<(u32, u64, String, f32)> {
        let graph = self.graph.read();
        let deleted = self.deleted.read();
        let ef = ef.unwrap_or(graph.config().ef_search).max(k);

        let results = graph.search(query, k, ef, vectors, self.metric, &deleted);

        results
            .into_iter()
            .filter_map(|(node_id, distance)| {
                graph.get_node(node_id).map(|node| {
                    (node_id, node.slot, node.id.clone(), distance)
                })
            })
            .collect()
    }

    /// Search and return SearchResult objects with metadata.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `k` - Number of results
    /// * `ef` - Beam width (or None for default)
    /// * `vectors` - Vector storage
    /// * `metadata` - Metadata storage for record lookup
    ///
    /// # Returns
    ///
    /// Vector of SearchResult with scores and metadata.
    pub fn search_with_metadata(
        &self,
        query: &[f32],
        k: usize,
        ef: Option<usize>,
        vectors: &VectorStorage,
        metadata: &MetadataStorage,
    ) -> Result<Vec<SearchResult>> {
        let candidates = self.search(query, k * 2, ef, vectors); // Over-fetch for filtering

        let txn = metadata.begin_read()?;
        let mut results = Vec::with_capacity(k);

        for (_node_id, _slot, id, score) in candidates {
            if let Some(record) = metadata.get_record(&txn, &id)? {
                if record.is_active() {
                    results.push(SearchResult::new(&record.id, score, record.metadata()));
                    if results.len() >= k {
                        break;
                    }
                }
            }
        }

        Ok(results)
    }

    /// Save the index to a file.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.graph.read().save(path)
    }

    /// Load an index from a file.
    ///
    /// The config is used only for reference (e.g., ef_search default).
    /// The actual graph parameters are loaded from the file.
    pub fn load(path: &Path, _config: HnswConfig, metric: Metric) -> Result<Self> {
        let graph = HnswGraph::load(path)?;
        Ok(Self {
            graph: RwLock::new(graph),
            metric,
            deleted: RwLock::new(HashSet::new()),
        })
    }

    /// Check if an index file exists.
    pub fn exists(path: &Path) -> bool {
        HnswGraph::exists(path)
    }

    /// Build an index from existing records using parallel construction.
    ///
    /// This method uses multi-threaded parallel processing to significantly
    /// speed up index construction compared to sequential insertion.
    ///
    /// # Arguments
    ///
    /// * `config` - HNSW configuration
    /// * `metric` - Distance metric
    /// * `vectors` - Vector storage
    /// * `metadata` - Metadata storage to iterate records
    ///
    /// # Returns
    ///
    /// A new HnswIndex containing all active records.
    pub fn build_from_records(
        config: HnswConfig,
        metric: Metric,
        vectors: &VectorStorage,
        metadata: &MetadataStorage,
    ) -> Result<Self> {
        // Collect all active records first
        let mut records = Vec::new();
        let txn = metadata.begin_read()?;
        metadata.iter_records(&txn, |record| {
            if record.is_active() {
                records.push((record.slot_offset, record.id.clone()));
            }
            Ok(true)
        })?;
        drop(txn);

        // Build using parallel construction with repair pass
        let graph = build_parallel(config.clone(), metric, records, vectors);

        Ok(Self {
            graph: RwLock::new(graph),
            metric,
            deleted: RwLock::new(HashSet::new()),
        })
    }

    /// Load an existing index or build from records if not found.
    pub fn load_or_build(
        path: &Path,
        config: HnswConfig,
        metric: Metric,
        vectors: &VectorStorage,
        metadata: &MetadataStorage,
    ) -> Result<Self> {
        if Self::exists(path) {
            match Self::load(path, config.clone(), metric) {
                Ok(index) => return Ok(index),
                Err(e) => {
                    // Log error and rebuild
                    eprintln!("Failed to load HNSW index, rebuilding: {}", e);
                }
            }
        }

        // Build from scratch
        let index = Self::build_from_records(config, metric, vectors, metadata)?;

        // Save for next time
        if let Err(e) = index.save(path) {
            eprintln!("Failed to save HNSW index: {}", e);
        }

        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_storage(dim: usize, count: usize) -> (TempDir, VectorStorage) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");
        let mut storage = VectorStorage::create(&path, dim).unwrap();

        for i in 0..count {
            let slot = storage.allocate_slot().unwrap();
            let mut vec = vec![0.0f32; dim];
            vec[i % dim] = 1.0;
            storage.write_slot(slot, &vec).unwrap();
        }

        (dir, storage)
    }

    #[test]
    fn test_hnsw_index_basic() {
        let config = HnswConfig::with_m(4);
        let index = HnswIndex::new(config, Metric::Cosine);

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let config = HnswConfig::with_m(4);
        let index = HnswIndex::new(config, Metric::Cosine);

        let (_dir, vectors) = create_test_storage(8, 20);

        // Insert vectors
        for i in 0..20 {
            index.insert(i as u64, format!("vec{}", i), &vectors);
        }

        assert_eq!(index.len(), 20);

        // Search
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 5, None, &vectors);

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            // For cosine (higher is better), scores should be descending
            assert!(results[i - 1].3 >= results[i].3);
        }
    }

    #[test]
    fn test_hnsw_delete() {
        let config = HnswConfig::with_m(4);
        let index = HnswIndex::new(config, Metric::Cosine);

        let (_dir, vectors) = create_test_storage(4, 10);

        for i in 0..10 {
            index.insert(i as u64, format!("vec{}", i), &vectors);
        }

        // Delete slot 0
        index.mark_deleted(0);

        assert_eq!(index.deleted_count(), 1);

        // Search should not return deleted node
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 10, None, &vectors);

        for (_, slot, _, _) in &results {
            assert_ne!(*slot, 0, "Deleted slot should not appear in results");
        }
    }

    #[test]
    fn test_hnsw_save_and_load() {
        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("hnsw.index");

        let config = HnswConfig::with_m(4);
        let index = HnswIndex::new(config.clone(), Metric::Cosine);

        let (_vec_dir, vectors) = create_test_storage(4, 10);

        for i in 0..10 {
            index.insert(i as u64, format!("vec{}", i), &vectors);
        }

        // Save
        index.save(&index_path).unwrap();

        // Load
        let loaded = HnswIndex::load(&index_path, config, Metric::Cosine).unwrap();

        assert_eq!(loaded.len(), index.len());

        // Search should work on loaded index
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = loaded.search(&query, 5, Some(100), &vectors);
        // With orthogonal test vectors, we get at least the 3 identical vectors
        assert!(results.len() >= 3);
    }
}
