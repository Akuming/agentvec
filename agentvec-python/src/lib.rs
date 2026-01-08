//! Python bindings for AgentVec vector database.
//!
//! This crate provides PyO3-based Python bindings for the core AgentVec library.
//! It is built using maturin and distributed as a Python wheel.

use std::sync::Arc;

use pyo3::exceptions::{PyIOError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value as JsonValue;

use ::agentvec::{AgentVec, Collection, CompactStats, Filter, Metric, RecoveryStats, SearchResult};

// ============ Helper Functions ============

/// Convert a Python dict to serde_json::Value.
fn pydict_to_json(py: Python<'_>, dict: &Bound<'_, PyDict>) -> PyResult<JsonValue> {
    // Use Python's json module to serialize, then parse in Rust
    let json_mod = py.import("json")?;
    let json_str: String = json_mod.call_method1("dumps", (dict,))?.extract()?;
    serde_json::from_str(&json_str)
        .map_err(|e| PyValueError::new_err(format!("Invalid JSON: {e}")))
}

/// Convert serde_json::Value to a Python object.
fn json_to_pyobject(py: Python<'_>, value: &JsonValue) -> PyResult<PyObject> {
    let json_mod = py.import("json")?;
    let json_str = serde_json::to_string(value)
        .map_err(|e| PyValueError::new_err(format!("JSON serialization error: {e}")))?;
    let result = json_mod.call_method1("loads", (json_str,))?;
    Ok(result.into())
}

/// Parse metric string to Metric enum.
fn parse_metric(s: &str) -> PyResult<Metric> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(Metric::Cosine),
        "dot" => Ok(Metric::Dot),
        "l2" | "euclidean" => Ok(Metric::L2),
        _ => Err(PyValueError::new_err(format!(
            "Invalid metric '{s}'. Expected: cosine, dot, l2"
        ))),
    }
}

// ============ PySearchResult ============

/// Search result from a vector query.
#[pyclass(name = "SearchResult")]
#[derive(Clone)]
pub struct PySearchResult {
    /// Record ID.
    #[pyo3(get)]
    pub id: String,

    /// Similarity/distance score.
    #[pyo3(get)]
    pub score: f32,

    /// Record metadata (stored as JSON string for conversion).
    metadata_json: String,
}

#[pymethods]
impl PySearchResult {
    /// Get the metadata as a Python dict.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        let value: JsonValue = serde_json::from_str(&self.metadata_json)
            .map_err(|e| PyValueError::new_err(format!("Invalid metadata JSON: {e}")))?;
        json_to_pyobject(py, &value)
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id='{}', score={:.4}, metadata={})",
            self.id, self.score, self.metadata_json
        )
    }
}

impl From<SearchResult> for PySearchResult {
    fn from(r: SearchResult) -> Self {
        Self {
            id: r.id,
            score: r.score,
            metadata_json: serde_json::to_string(&r.metadata).unwrap_or_else(|_| "{}".to_string()),
        }
    }
}

// ============ PyCompactStats ============

/// Statistics from a compaction operation.
#[pyclass(name = "CompactStats")]
#[derive(Clone)]
pub struct PyCompactStats {
    /// Number of expired records removed.
    #[pyo3(get)]
    pub expired_removed: usize,

    /// Number of tombstone records removed.
    #[pyo3(get)]
    pub tombstones_removed: usize,

    /// Bytes freed (approximate).
    #[pyo3(get)]
    pub bytes_freed: u64,

    /// Duration in milliseconds.
    #[pyo3(get)]
    pub duration_ms: u64,
}

#[pymethods]
impl PyCompactStats {
    fn __repr__(&self) -> String {
        format!(
            "CompactStats(expired_removed={}, tombstones_removed={}, bytes_freed={}, duration_ms={})",
            self.expired_removed, self.tombstones_removed, self.bytes_freed, self.duration_ms
        )
    }
}

impl From<CompactStats> for PyCompactStats {
    fn from(s: CompactStats) -> Self {
        Self {
            expired_removed: s.expired_removed,
            tombstones_removed: s.tombstones_removed,
            bytes_freed: s.bytes_freed,
            duration_ms: s.duration_ms,
        }
    }
}

// ============ PyRecoveryStats ============

/// Statistics from crash recovery.
#[pyclass(name = "RecoveryStats")]
#[derive(Clone)]
pub struct PyRecoveryStats {
    /// Records promoted from Pending to Active.
    #[pyo3(get)]
    pub promoted: usize,

    /// Records rolled back (incomplete writes).
    #[pyo3(get)]
    pub rolled_back: usize,

    /// Tombstone records found.
    #[pyo3(get)]
    pub tombstones: usize,
}

#[pymethods]
impl PyRecoveryStats {
    fn __repr__(&self) -> String {
        format!(
            "RecoveryStats(promoted={}, rolled_back={}, tombstones={})",
            self.promoted, self.rolled_back, self.tombstones
        )
    }
}

impl From<&RecoveryStats> for PyRecoveryStats {
    fn from(s: &RecoveryStats) -> Self {
        Self {
            promoted: s.promoted,
            rolled_back: s.rolled_back,
            tombstones: s.tombstones,
        }
    }
}

// ============ PyImportStats ============

/// Statistics from an import operation.
#[pyclass(name = "ImportStats")]
#[derive(Clone)]
pub struct PyImportStats {
    /// Number of records successfully imported.
    #[pyo3(get)]
    pub imported: usize,

    /// Number of records skipped.
    #[pyo3(get)]
    pub skipped: usize,

    /// Number of records that failed to import.
    #[pyo3(get)]
    pub failed: usize,

    /// Duration in milliseconds.
    #[pyo3(get)]
    pub duration_ms: u64,
}

#[pymethods]
impl PyImportStats {
    fn __repr__(&self) -> String {
        format!(
            "ImportStats(imported={}, skipped={}, failed={}, duration_ms={})",
            self.imported, self.skipped, self.failed, self.duration_ms
        )
    }
}

impl From<::agentvec::ImportStats> for PyImportStats {
    fn from(s: ::agentvec::ImportStats) -> Self {
        Self {
            imported: s.imported,
            skipped: s.skipped,
            failed: s.failed,
            duration_ms: s.duration_ms,
        }
    }
}

// ============ PyCollection ============

/// A collection of vectors with associated metadata.
#[pyclass(name = "Collection")]
pub struct PyCollection {
    inner: Arc<Collection>,
}

#[pymethods]
impl PyCollection {
    /// Add a vector to the collection.
    ///
    /// Args:
    ///     vector: The vector as a list of floats.
    ///     metadata: Associated metadata as a dict.
    ///     id: Optional custom ID (auto-generated if None).
    ///     ttl: Optional time-to-live in seconds.
    ///
    /// Returns:
    ///     The record ID.
    #[pyo3(signature = (vector, metadata, id=None, ttl=None))]
    fn add(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        metadata: &Bound<'_, PyDict>,
        id: Option<&str>,
        ttl: Option<u64>,
    ) -> PyResult<String> {
        let meta = pydict_to_json(py, metadata)?;
        self.inner
            .add(&vector, meta, id, ttl)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Insert or update a vector by ID.
    ///
    /// This is idempotent - calling with the same ID will replace the previous vector.
    ///
    /// Args:
    ///     id: The record ID.
    ///     vector: The vector as a list of floats.
    ///     metadata: Associated metadata as a dict.
    ///     ttl: Optional time-to-live in seconds.
    #[pyo3(signature = (id, vector, metadata, ttl=None))]
    fn upsert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: &Bound<'_, PyDict>,
        ttl: Option<u64>,
    ) -> PyResult<()> {
        let meta = pydict_to_json(py, metadata)?;
        self.inner
            .upsert(id, &vector, meta, ttl)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Batch add vectors (10-100x faster than looped add).
    ///
    /// Args:
    ///     vectors: List of vectors (each a list of floats).
    ///     metadatas: List of metadata dicts.
    ///     ids: Optional list of custom IDs.
    ///     ttls: Optional list of TTLs in seconds.
    ///
    /// Returns:
    ///     List of record IDs.
    #[pyo3(signature = (vectors, metadatas, ids=None, ttls=None))]
    fn add_batch(
        &self,
        py: Python<'_>,
        vectors: Vec<Vec<f32>>,
        metadatas: Vec<Bound<'_, PyDict>>,
        ids: Option<Vec<String>>,
        ttls: Option<Vec<Option<u64>>>,
    ) -> PyResult<Vec<String>> {
        // Convert metadatas
        let metas: Vec<JsonValue> = metadatas
            .iter()
            .map(|d| pydict_to_json(py, d))
            .collect::<PyResult<Vec<_>>>()?;

        // Convert vectors to slices
        let vec_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        // Convert ids to &str slices
        let id_strs: Option<Vec<&str>> = ids.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());

        self.inner
            .add_batch(
                &vec_refs,
                &metas,
                id_strs.as_ref().map(|v| v.as_slice()),
                ttls.as_ref().map(|v| v.as_slice()),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Search for nearest neighbors.
    ///
    /// Args:
    ///     vector: The query vector.
    ///     k: Number of results to return.
    ///     where_: Optional metadata filter as a dict.
    ///
    /// Returns:
    ///     List of SearchResult objects.
    #[pyo3(signature = (vector, k, where_=None))]
    #[pyo3(name = "search")]
    fn search_py(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        k: usize,
        where_: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PySearchResult>> {
        let filter = match where_ {
            Some(dict) => {
                let json = pydict_to_json(py, dict)?;
                Some(Filter::from_json(&json))
            }
            None => None,
        };

        self.inner
            .search(&vector, k, filter)
            .map(|results| results.into_iter().map(PySearchResult::from).collect())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get a record by ID.
    ///
    /// Args:
    ///     id: The record ID.
    ///
    /// Returns:
    ///     SearchResult if found and not expired, None otherwise.
    fn get(&self, id: &str) -> PyResult<Option<PySearchResult>> {
        self.inner
            .get(id)
            .map(|opt| opt.map(PySearchResult::from))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Delete a record by ID.
    ///
    /// Args:
    ///     id: The record ID.
    ///
    /// Returns:
    ///     True if the record existed and was deleted.
    fn delete(&self, id: &str) -> PyResult<bool> {
        self.inner
            .delete(id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Compact the collection (remove expired records, defragment storage).
    ///
    /// Returns:
    ///     CompactStats with details about what was cleaned up.
    fn compact(&self) -> PyResult<PyCompactStats> {
        self.inner
            .compact()
            .map(PyCompactStats::from)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Get the number of active (non-expired) records.
    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .len()
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Preload vectors into memory for faster search.
    fn preload(&self) -> PyResult<()> {
        self.inner
            .preload()
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Flush pending writes to disk.
    fn sync(&self) -> PyResult<()> {
        self.inner
            .sync()
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Vector storage size in bytes.
    #[getter]
    fn vectors_size_bytes(&self) -> u64 {
        self.inner.vectors_size_bytes()
    }

    /// Collection dimensions.
    #[getter]
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    /// Collection name.
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Distance metric (cosine, dot, l2).
    #[getter]
    fn metric(&self) -> &str {
        match self.inner.metric() {
            Metric::Cosine => "cosine",
            Metric::Dot => "dot",
            Metric::L2 => "l2",
        }
    }

    /// Export the collection to a file.
    ///
    /// Creates an NDJSON file with all records including vectors and metadata.
    ///
    /// Args:
    ///     path: Output file path.
    ///
    /// Returns:
    ///     Number of records exported.
    fn export_to_file(&self, path: &str) -> PyResult<usize> {
        self.inner
            .export_to_file(path)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Import records from a file.
    ///
    /// Reads an NDJSON file and imports all records. The file must have
    /// matching dimensions to this collection.
    ///
    /// Args:
    ///     path: Input file path.
    ///
    /// Returns:
    ///     ImportStats with details about the import.
    fn import_from_file(&self, path: &str) -> PyResult<PyImportStats> {
        self.inner
            .import_from_file(path)
            .map(PyImportStats::from)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Collection(name='{}', dimensions={}, metric='{}')",
            self.inner.name(),
            self.inner.dimensions(),
            self.metric()
        )
    }
}

// ============ PyAgentVec ============

/// AgentVec vector database.
///
/// A lightweight, serverless, single-file vector database for AI agent memory.
#[pyclass(name = "AgentVec")]
pub struct PyAgentVec {
    inner: AgentVec,
}

#[pymethods]
impl PyAgentVec {
    /// Open or create a database at the given path.
    ///
    /// Args:
    ///     path: Path to the database directory (will be created if needed).
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        AgentVec::open(path)
            .map(|inner| Self { inner })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Get or create a collection.
    ///
    /// Args:
    ///     name: Collection name.
    ///     dim: Vector dimensions.
    ///     metric: Distance metric ("cosine", "dot", or "l2").
    ///
    /// Returns:
    ///     The Collection object.
    #[pyo3(signature = (name, dim, metric="cosine"))]
    fn collection(&self, name: &str, dim: usize, metric: &str) -> PyResult<PyCollection> {
        let m = parse_metric(metric)?;
        self.inner
            .collection(name, dim, m)
            .map(|c| PyCollection { inner: c })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get an existing collection.
    ///
    /// Args:
    ///     name: Collection name.
    ///
    /// Returns:
    ///     The Collection object.
    ///
    /// Raises:
    ///     KeyError: If collection does not exist.
    fn get_collection(&self, name: &str) -> PyResult<PyCollection> {
        self.inner
            .get_collection(name)
            .map(|c| PyCollection { inner: c })
            .map_err(|e| PyKeyError::new_err(e.to_string()))
    }

    /// List all collection names.
    ///
    /// Returns:
    ///     List of collection names.
    fn collections(&self) -> PyResult<Vec<String>> {
        self.inner
            .collections()
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Delete a collection and all its data.
    ///
    /// Args:
    ///     name: Collection name.
    fn drop_collection(&self, name: &str) -> PyResult<()> {
        self.inner
            .drop_collection(name)
            .map(|_| ())  // Discard the bool return value
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Flush all pending writes to disk.
    fn sync(&self) -> PyResult<()> {
        self.inner
            .sync()
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Get recovery statistics from database open.
    ///
    /// Returns:
    ///     RecoveryStats with details about crash recovery.
    fn recovery_stats(&self) -> PyRecoveryStats {
        PyRecoveryStats::from(self.inner.recovery_stats())
    }

    fn __repr__(&self) -> String {
        match self.inner.collections() {
            Ok(cols) => format!("AgentVec(collections={})", cols.len()),
            Err(_) => "AgentVec()".to_string(),
        }
    }
}

// ============ Module ============

/// AgentVec Python module.
///
/// A lightweight, serverless vector database for AI agent memory.
#[pymodule]
fn agentvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAgentVec>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyCompactStats>()?;
    m.add_class::<PyRecoveryStats>()?;
    m.add_class::<PyImportStats>()?;
    Ok(())
}
