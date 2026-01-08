//! Mobile bindings for AgentVec vector database.
//!
//! This crate provides UniFFI-based bindings for iOS (Swift) and Android (Kotlin).
//! It wraps the core AgentVec library with a mobile-friendly API.

use std::sync::Arc;
use serde_json::Value as JsonValue;

// Re-export core types
use agentvec::{
    AgentVec as CoreAgentVec,
    Collection as CoreCollection,
    Metric as CoreMetric,
    SearchResult as CoreSearchResult,
    CompactStats as CoreCompactStats,
    ImportStats as CoreImportStats,
    RecoveryStats as CoreRecoveryStats,
    Filter,
};

// Include the UniFFI scaffolding
uniffi::include_scaffolding!("agentvec");

// ========== Enums ==========

/// Distance metric for vector similarity.
#[derive(Debug, Clone, Copy)]
pub enum Metric {
    Cosine,
    Dot,
    L2,
}

impl From<Metric> for CoreMetric {
    fn from(m: Metric) -> Self {
        match m {
            Metric::Cosine => CoreMetric::Cosine,
            Metric::Dot => CoreMetric::Dot,
            Metric::L2 => CoreMetric::L2,
        }
    }
}

impl From<CoreMetric> for Metric {
    fn from(m: CoreMetric) -> Self {
        match m {
            CoreMetric::Cosine => Metric::Cosine,
            CoreMetric::Dot => Metric::Dot,
            CoreMetric::L2 => Metric::L2,
        }
    }
}

// ========== Error Types ==========

/// Error types for mobile bindings.
/// The error message is included in the Display output and will be available
/// as the exception message in Swift/Kotlin.
#[derive(Debug, thiserror::Error)]
pub enum AgentVecError {
    #[error("I/O error: {0}")]
    IoError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("{0}")]
    Other(String),
}

impl From<agentvec::AgentVecError> for AgentVecError {
    fn from(e: agentvec::AgentVecError) -> Self {
        use agentvec::AgentVecError as E;
        match e {
            E::Io(err) => AgentVecError::IoError(err.to_string()),
            E::Database(msg) => AgentVecError::DatabaseError(msg),
            E::DimensionMismatch { expected, got } => {
                AgentVecError::DimensionMismatch(format!("expected {}, got {}", expected, got))
            }
            E::InvalidInput(msg) => AgentVecError::InvalidInput(msg),
            E::NotFound(msg) => AgentVecError::NotFound(msg),
            E::CollectionNotFound(name) => AgentVecError::CollectionNotFound(name),
            E::Serialization(msg) => AgentVecError::Serialization(msg),
            _ => AgentVecError::Other(e.to_string()),
        }
    }
}

// ========== Data Types ==========

/// Search result from a vector query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub metadata_json: String,
}

impl From<CoreSearchResult> for SearchResult {
    fn from(r: CoreSearchResult) -> Self {
        Self {
            id: r.id,
            score: r.score,
            metadata_json: serde_json::to_string(&r.metadata).unwrap_or_else(|_| "{}".to_string()),
        }
    }
}

/// A record with its vector data included.
/// Use this when you need to retrieve the actual vector values.
#[derive(Debug, Clone)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata_json: String,
}

/// Statistics from a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactStats {
    pub expired_removed: u64,
    pub tombstones_removed: u64,
    pub bytes_freed: u64,
    pub duration_ms: u64,
}

impl From<CoreCompactStats> for CompactStats {
    fn from(s: CoreCompactStats) -> Self {
        Self {
            expired_removed: s.expired_removed as u64,
            tombstones_removed: s.tombstones_removed as u64,
            bytes_freed: s.bytes_freed,
            duration_ms: s.duration_ms,
        }
    }
}

/// Statistics from an import operation.
#[derive(Debug, Clone)]
pub struct ImportStats {
    pub imported: u64,
    pub skipped: u64,
    pub failed: u64,
    pub duration_ms: u64,
}

impl From<CoreImportStats> for ImportStats {
    fn from(s: CoreImportStats) -> Self {
        Self {
            imported: s.imported as u64,
            skipped: s.skipped as u64,
            failed: s.failed as u64,
            duration_ms: s.duration_ms,
        }
    }
}

/// Recovery statistics from database open.
#[derive(Debug, Clone)]
pub struct RecoveryStats {
    pub promoted: u64,
    pub rolled_back: u64,
    pub tombstones: u64,
}

impl From<&CoreRecoveryStats> for RecoveryStats {
    fn from(s: &CoreRecoveryStats) -> Self {
        Self {
            promoted: s.promoted as u64,
            rolled_back: s.rolled_back as u64,
            tombstones: s.tombstones as u64,
        }
    }
}

// ========== AgentVec Database ==========

/// AgentVec vector database for mobile.
pub struct AgentVec {
    inner: CoreAgentVec,
}

impl AgentVec {
    /// Open or create a database at the given path.
    pub fn open(path: String) -> Result<Self, AgentVecError> {
        let inner = CoreAgentVec::open(&path)?;
        Ok(Self { inner })
    }

    /// Get or create a collection.
    pub fn collection(
        &self,
        name: String,
        dimensions: u32,
        metric: Metric,
    ) -> Result<Arc<Collection>, AgentVecError> {
        let col = self.inner.collection(&name, dimensions as usize, metric.into())?;
        Ok(Arc::new(Collection { inner: col }))
    }

    /// Get an existing collection.
    pub fn get_collection(&self, name: String) -> Result<Arc<Collection>, AgentVecError> {
        let col = self.inner.get_collection(&name)?;
        Ok(Arc::new(Collection { inner: col }))
    }

    /// List all collection names.
    pub fn collections(&self) -> Result<Vec<String>, AgentVecError> {
        Ok(self.inner.collections()?)
    }

    /// Delete a collection.
    pub fn drop_collection(&self, name: String) -> Result<(), AgentVecError> {
        self.inner.drop_collection(&name)?;
        Ok(())
    }

    /// Flush all pending writes.
    pub fn sync(&self) -> Result<(), AgentVecError> {
        Ok(self.inner.sync()?)
    }

    /// Get recovery statistics.
    pub fn recovery_stats(&self) -> RecoveryStats {
        RecoveryStats::from(self.inner.recovery_stats())
    }
}

// ========== Collection ==========

/// Collection of vectors with metadata.
pub struct Collection {
    inner: Arc<CoreCollection>,
}

impl Collection {
    /// Add a vector to the collection.
    pub fn add(
        &self,
        vector: Vec<f32>,
        metadata_json: String,
        id: Option<String>,
        ttl: Option<u64>,
    ) -> Result<String, AgentVecError> {
        let metadata: JsonValue = serde_json::from_str(&metadata_json)
            .map_err(|e| AgentVecError::Serialization(format!("Invalid metadata JSON: {}", e)))?;

        let id = self.inner.add(
            &vector,
            metadata,
            id.as_deref(),
            ttl,
        )?;

        Ok(id)
    }

    /// Insert or update a vector.
    pub fn upsert(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata_json: String,
        ttl: Option<u64>,
    ) -> Result<(), AgentVecError> {
        let metadata: JsonValue = serde_json::from_str(&metadata_json)
            .map_err(|e| AgentVecError::Serialization(format!("Invalid metadata JSON: {}", e)))?;

        self.inner.upsert(&id, &vector, metadata, ttl)?;
        Ok(())
    }

    /// Search for nearest neighbors.
    pub fn search(
        &self,
        vector: Vec<f32>,
        k: u32,
        where_json: Option<String>,
    ) -> Result<Vec<SearchResult>, AgentVecError> {
        let filter = match where_json {
            Some(json_str) => {
                let json: JsonValue = serde_json::from_str(&json_str)
                    .map_err(|e| AgentVecError::Serialization(format!("Invalid filter JSON: {}", e)))?;
                Some(Filter::from_json(&json))
            }
            None => None,
        };

        let results = self.inner.search(&vector, k as usize, filter)?;
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    /// Get a record by ID.
    pub fn get(&self, id: String) -> Result<Option<SearchResult>, AgentVecError> {
        let result = self.inner.get(&id)?;
        Ok(result.map(SearchResult::from))
    }

    /// Get a record with its vector data by ID.
    /// This is useful when you need to retrieve the actual vector values.
    pub fn get_with_vector(&self, id: String) -> Result<Option<VectorRecord>, AgentVecError> {
        let result = self.inner.get_with_vector(&id)?;
        Ok(result.map(|r| VectorRecord {
            id: r.id,
            vector: r.vector,
            metadata_json: serde_json::to_string(&r.metadata)
                .unwrap_or_else(|_| "{}".to_string()),
        }))
    }

    /// Delete a record by ID.
    pub fn delete(&self, id: String) -> Result<bool, AgentVecError> {
        Ok(self.inner.delete(&id)?)
    }

    /// Compact the collection.
    pub fn compact(&self) -> Result<CompactStats, AgentVecError> {
        let stats = self.inner.compact()?;
        Ok(CompactStats::from(stats))
    }

    /// Get the number of active records.
    pub fn len(&self) -> Result<u64, AgentVecError> {
        Ok(self.inner.len()? as u64)
    }

    /// Check if the collection is empty.
    pub fn is_empty(&self) -> Result<bool, AgentVecError> {
        Ok(self.inner.is_empty()?)
    }

    /// Preload vectors into memory.
    pub fn preload(&self) -> Result<(), AgentVecError> {
        Ok(self.inner.preload()?)
    }

    /// Flush pending writes.
    pub fn sync(&self) -> Result<(), AgentVecError> {
        Ok(self.inner.sync()?)
    }

    /// Export to file.
    pub fn export_to_file(&self, path: String) -> Result<u64, AgentVecError> {
        let count = self.inner.export_to_file(&path)?;
        Ok(count as u64)
    }

    /// Import from file.
    pub fn import_from_file(&self, path: String) -> Result<ImportStats, AgentVecError> {
        let stats = self.inner.import_from_file(&path)?;
        Ok(ImportStats::from(stats))
    }

    /// Get vector dimensions.
    pub fn dimensions(&self) -> u32 {
        self.inner.dimensions() as u32
    }

    /// Get collection name.
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get distance metric.
    pub fn metric(&self) -> Metric {
        Metric::from(self.inner.metric())
    }

    /// Get vector storage size in bytes.
    pub fn vectors_size_bytes(&self) -> u64 {
        self.inner.vectors_size_bytes()
    }
}
