//! Main database implementation.
//!
//! AgentVec is the top-level handle for the vector database.
//! It manages collections and provides the main API.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::collection::Collection;
use crate::config::{CollectionConfig, Metric, MAX_DIMENSIONS, WARN_DIMENSIONS};
use crate::error::{AgentVecError, Result};
use crate::recovery::RecoveryStats;
use crate::storage::MetadataStorage;

/// The main AgentVec database handle.
///
/// Manages collections and provides the top-level API.
/// Thread-safe and can be shared across threads.
pub struct AgentVec {
    /// Path to the database directory.
    path: PathBuf,

    /// Global metadata storage.
    metadata: MetadataStorage,

    /// Cached collection handles.
    collections: RwLock<HashMap<String, Arc<Collection>>>,

    /// Recovery stats from database open.
    recovery_stats: RecoveryStats,
}

impl AgentVec {
    /// Open or create a database at the given path.
    ///
    /// If the database exists, it will be opened and recovery will be performed.
    /// If it doesn't exist, a new database will be created.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database directory
    ///
    /// # Returns
    ///
    /// The database handle.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use agentvec::AgentVec;
    ///
    /// let db = AgentVec::open("./my_database.avdb").unwrap();
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create directory structure
        std::fs::create_dir_all(&path)?;
        std::fs::create_dir_all(path.join("collections"))?;

        let metadata_path = path.join("meta.redb");
        let metadata = MetadataStorage::open(&metadata_path)?;

        // Run recovery
        let recovery_stats = Self::recover(&metadata, &path)?;

        Ok(Self {
            path,
            metadata,
            collections: RwLock::new(HashMap::new()),
            recovery_stats,
        })
    }

    /// Get or create a collection.
    ///
    /// If the collection exists, returns it. If it doesn't exist,
    /// creates it with the specified dimensions and metric.
    ///
    /// # Arguments
    ///
    /// * `name` - Collection name
    /// * `dim` - Vector dimensions
    /// * `metric` - Distance metric
    ///
    /// # Returns
    ///
    /// An Arc to the collection handle.
    ///
    /// # Errors
    ///
    /// Returns error if collection exists with different dimensions.
    pub fn collection(
        &self,
        name: &str,
        dim: usize,
        metric: Metric,
    ) -> Result<Arc<Collection>> {
        // Validate dimensions
        Self::validate_dimensions(dim)?;

        // Check cache first
        {
            let collections = self.collections.read();
            if let Some(col) = collections.get(name) {
                // Validate dimensions match
                if col.dimensions() != dim {
                    return Err(AgentVecError::DimensionMismatch {
                        expected: col.dimensions(),
                        got: dim,
                    });
                }
                if col.metric() != metric {
                    return Err(AgentVecError::CollectionExists(format!(
                        "Collection '{}' exists with metric {:?}, not {:?}",
                        name,
                        col.metric(),
                        metric
                    )));
                }
                return Ok(Arc::clone(col));
            }
        }

        // Check if registered in metadata
        let txn = self.metadata.begin_read()?;
        let existing_config = self.metadata.get_collection(&txn, name)?;
        drop(txn);

        if let Some(config) = existing_config {
            // Validate dimensions match
            if config.dimensions != dim {
                return Err(AgentVecError::DimensionMismatch {
                    expected: config.dimensions,
                    got: dim,
                });
            }
            if config.metric != metric {
                return Err(AgentVecError::CollectionExists(format!(
                    "Collection '{}' exists with metric {:?}, not {:?}",
                    name, config.metric, metric
                )));
            }

            // Open existing collection
            let col_path = self.collection_path(name);
            let col = Arc::new(Collection::open(col_path, config)?);

            let mut collections = self.collections.write();
            collections.insert(name.to_string(), Arc::clone(&col));

            return Ok(col);
        }

        // Create new collection
        let config = CollectionConfig::new(name, dim, metric);

        // Register in metadata
        let txn = self.metadata.begin_write()?;
        self.metadata.register_collection(&txn, &config)?;
        txn.commit()?;

        // Create collection directory and open
        let col_path = self.collection_path(name);
        let col = Arc::new(Collection::open(col_path, config)?);

        let mut collections = self.collections.write();
        collections.insert(name.to_string(), Arc::clone(&col));

        Ok(col)
    }

    /// Get an existing collection.
    ///
    /// # Arguments
    ///
    /// * `name` - Collection name
    ///
    /// # Returns
    ///
    /// The collection if it exists, error otherwise.
    pub fn get_collection(&self, name: &str) -> Result<Arc<Collection>> {
        // Check cache
        {
            let collections = self.collections.read();
            if let Some(col) = collections.get(name) {
                return Ok(Arc::clone(col));
            }
        }

        // Check metadata
        let txn = self.metadata.begin_read()?;
        let config = self.metadata.get_collection(&txn, name)?;
        drop(txn);

        match config {
            Some(config) => {
                let col_path = self.collection_path(name);
                let col = Arc::new(Collection::open(col_path, config)?);

                let mut collections = self.collections.write();
                collections.insert(name.to_string(), Arc::clone(&col));

                Ok(col)
            }
            None => Err(AgentVecError::CollectionNotFound(name.to_string())),
        }
    }

    /// List all collection names.
    pub fn collections(&self) -> Result<Vec<String>> {
        let txn = self.metadata.begin_read()?;
        self.metadata.list_collections(&txn)
    }

    /// Delete a collection and all its data.
    ///
    /// # Arguments
    ///
    /// * `name` - Collection name
    ///
    /// # Returns
    ///
    /// True if collection existed and was deleted.
    pub fn drop_collection(&self, name: &str) -> Result<bool> {
        // Remove from cache
        {
            let mut collections = self.collections.write();
            collections.remove(name);
        }

        // Check if exists
        let txn = self.metadata.begin_read()?;
        let exists = self.metadata.get_collection(&txn, name)?.is_some();
        drop(txn);

        if !exists {
            return Ok(false);
        }

        // Remove from metadata
        let txn = self.metadata.begin_write()?;
        self.metadata.remove_collection(&txn, name)?;
        txn.commit()?;

        // Delete collection directory
        let col_path = self.collection_path(name);
        if col_path.exists() {
            std::fs::remove_dir_all(&col_path)?;
        }

        Ok(true)
    }

    /// Get recovery stats from the last open.
    pub fn recovery_stats(&self) -> &RecoveryStats {
        &self.recovery_stats
    }

    /// Sync all collections to disk.
    pub fn sync(&self) -> Result<()> {
        let collections = self.collections.read();
        for col in collections.values() {
            col.sync()?;
        }
        Ok(())
    }

    /// Get the database path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the path for a collection.
    fn collection_path(&self, name: &str) -> PathBuf {
        self.path.join("collections").join(name)
    }

    /// Validate dimensions.
    fn validate_dimensions(dim: usize) -> Result<()> {
        if dim == 0 {
            return Err(AgentVecError::InvalidInput(
                "Dimensions must be greater than 0".into(),
            ));
        }

        if dim > MAX_DIMENSIONS {
            return Err(AgentVecError::DimensionsTooLarge {
                max: MAX_DIMENSIONS,
                got: dim,
            });
        }

        if dim > WARN_DIMENSIONS {
            // High dimensions may impact search performance
            // TODO: Add tracing/logging when tracing feature is implemented
            let _ = dim; // Silence unused warning in this branch
        }

        Ok(())
    }

    /// Run recovery on database open.
    fn recover(_metadata: &MetadataStorage, _path: &Path) -> Result<RecoveryStats> {
        let start = std::time::Instant::now();
        let mut stats = RecoveryStats::new();

        // TODO: Implement full recovery protocol
        // For now, just count existing records

        stats.duration_ms = start.elapsed().as_millis() as u64;
        Ok(stats)
    }
}

// Ensure AgentVec is Send + Sync
static_assertions::assert_impl_all!(AgentVec: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn test_open_create() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.avdb");

        // Create
        let db = AgentVec::open(&db_path).unwrap();
        assert!(db.collections().unwrap().is_empty());
        drop(db);

        // Reopen
        let db = AgentVec::open(&db_path).unwrap();
        assert!(db.collections().unwrap().is_empty());
    }

    #[test]
    fn test_collection_create() {
        let dir = tempdir().unwrap();
        let db = AgentVec::open(dir.path().join("test.avdb")).unwrap();

        let col = db.collection("test", 384, Metric::Cosine).unwrap();
        assert_eq!(col.dimensions(), 384);
        assert_eq!(col.metric(), Metric::Cosine);

        let collections = db.collections().unwrap();
        assert!(collections.contains(&"test".to_string()));
    }

    #[test]
    fn test_collection_get_existing() {
        let dir = tempdir().unwrap();
        let db = AgentVec::open(dir.path().join("test.avdb")).unwrap();

        // Create
        let col1 = db.collection("test", 384, Metric::Cosine).unwrap();

        // Get same
        let col2 = db.collection("test", 384, Metric::Cosine).unwrap();

        // Should be same Arc
        assert!(Arc::ptr_eq(&col1, &col2));
    }

    #[test]
    fn test_collection_dimension_mismatch() {
        let dir = tempdir().unwrap();
        let db = AgentVec::open(dir.path().join("test.avdb")).unwrap();

        db.collection("test", 384, Metric::Cosine).unwrap();

        // Try to open with different dimensions
        let result = db.collection("test", 512, Metric::Cosine);
        assert!(matches!(result, Err(AgentVecError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_collection_metric_mismatch() {
        let dir = tempdir().unwrap();
        let db = AgentVec::open(dir.path().join("test.avdb")).unwrap();

        db.collection("test", 384, Metric::Cosine).unwrap();

        // Try to open with different metric
        let result = db.collection("test", 384, Metric::L2);
        assert!(matches!(result, Err(AgentVecError::CollectionExists(_))));
    }

    #[test]
    fn test_get_collection() {
        let dir = tempdir().unwrap();
        let db = AgentVec::open(dir.path().join("test.avdb")).unwrap();

        // Not found
        let result = db.get_collection("nonexistent");
        assert!(matches!(result, Err(AgentVecError::CollectionNotFound(_))));

        // Create
        db.collection("test", 384, Metric::Cosine).unwrap();

        // Found
        let col = db.get_collection("test").unwrap();
        assert_eq!(col.dimensions(), 384);
    }

    #[test]
    fn test_drop_collection() {
        let dir = tempdir().unwrap();
        let db = AgentVec::open(dir.path().join("test.avdb")).unwrap();

        db.collection("test", 384, Metric::Cosine).unwrap();
        assert!(db.collections().unwrap().contains(&"test".to_string()));

        let dropped = db.drop_collection("test").unwrap();
        assert!(dropped);

        assert!(!db.collections().unwrap().contains(&"test".to_string()));

        // Drop again should return false
        let dropped = db.drop_collection("test").unwrap();
        assert!(!dropped);
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.avdb");

        // Create and add data
        {
            let db = AgentVec::open(&db_path).unwrap();
            let col = db.collection("memories", 3, Metric::Cosine).unwrap();
            col.add(&[1.0, 2.0, 3.0], json!({"key": "value"}), Some("id1"), None)
                .unwrap();
            col.sync().unwrap();
        }

        // Reopen and verify
        {
            let db = AgentVec::open(&db_path).unwrap();
            let collections = db.collections().unwrap();
            assert!(collections.contains(&"memories".to_string()));

            let col = db.get_collection("memories").unwrap();
            assert_eq!(col.len().unwrap(), 1);

            let record = col.get("id1").unwrap();
            assert!(record.is_some());
            assert_eq!(record.unwrap().metadata["key"], "value");
        }
    }

    #[test]
    fn test_multiple_collections() {
        let dir = tempdir().unwrap();
        let db = AgentVec::open(dir.path().join("test.avdb")).unwrap();

        let episodic = db.collection("episodic", 384, Metric::Cosine).unwrap();
        let semantic = db.collection("semantic", 1536, Metric::Dot).unwrap();
        let spatial = db.collection("spatial", 128, Metric::L2).unwrap();

        assert_eq!(episodic.dimensions(), 384);
        assert_eq!(semantic.dimensions(), 1536);
        assert_eq!(spatial.dimensions(), 128);

        let collections = db.collections().unwrap();
        assert_eq!(collections.len(), 3);
    }

    #[test]
    fn test_invalid_dimensions() {
        let dir = tempdir().unwrap();
        let db = AgentVec::open(dir.path().join("test.avdb")).unwrap();

        // Zero dimensions
        let result = db.collection("test", 0, Metric::Cosine);
        assert!(matches!(result, Err(AgentVecError::InvalidInput(_))));

        // Too large
        let result = db.collection("test", 100_000, Metric::Cosine);
        assert!(matches!(result, Err(AgentVecError::DimensionsTooLarge { .. })));
    }

    #[test]
    fn test_recovery_stats() {
        let dir = tempdir().unwrap();
        let db = AgentVec::open(dir.path().join("test.avdb")).unwrap();

        let stats = db.recovery_stats();
        // Fresh database should have no recovery needed
        assert!(!stats.had_recovery());
    }
}
