//! JavaScript/WASM bindings for AgentVec vector database.
//!
//! This is a self-contained in-memory implementation for WASM environments.
//! Data is stored in memory and not persisted to disk.
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import { AgentVec, Metric } from 'agentvec';
//!
//! const db = AgentVec.open('./agent_memory');
//! const memories = db.collection('episodic', 384, Metric.Cosine);
//!
//! // Add a vector
//! const id = memories.add(
//!     new Float32Array([0.1, 0.2, ...]),
//!     { type: 'conversation', user: 'alice' },
//!     null,  // auto-generate ID
//!     3600   // TTL: 1 hour
//! );
//!
//! // Search
//! const results = memories.search(queryVector, 10, null);
//! ```

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// ========== Enums ==========

/// Distance metric for vector similarity.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    /// Cosine similarity (normalized dot product). Best for text embeddings.
    Cosine = 0,
    /// Raw dot product. Higher is more similar.
    Dot = 1,
    /// Euclidean (L2) distance. Lower is more similar.
    L2 = 2,
}

// ========== Internal Types ==========

/// Internal record storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Record {
    id: String,
    vector: Vec<f32>,
    metadata: JsonValue,
    created_at: u64,
    expires_at: Option<u64>,
    deleted: bool,
}

impl Record {
    fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(exp) => current_unix_time() >= exp,
            None => false,
        }
    }

    fn is_active(&self) -> bool {
        !self.deleted && !self.is_expired()
    }
}

/// Get current Unix timestamp in seconds.
fn current_unix_time() -> u64 {
    // Use JavaScript's Date.now() for WASM compatibility
    (js_sys::Date::now() / 1000.0) as u64
}

/// Internal collection data.
#[derive(Debug)]
struct CollectionData {
    name: String,
    dimensions: usize,
    metric: Metric,
    records: HashMap<String, Record>,
}

/// Internal database data.
#[derive(Debug)]
struct DatabaseData {
    path: String,
    collections: HashMap<String, Arc<RwLock<CollectionData>>>,
}

// ========== Distance Functions ==========

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn compute_score(query: &[f32], vector: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => cosine_similarity(query, vector),
        Metric::Dot => dot_product(query, vector),
        Metric::L2 => -l2_distance(query, vector), // Negate so higher is better
    }
}

// ========== Filter Implementation ==========

/// Check if a record matches a filter.
fn matches_filter(metadata: &JsonValue, filter: &JsonValue) -> bool {
    match filter {
        JsonValue::Object(filter_obj) => {
            for (key, condition) in filter_obj {
                let value = metadata.get(key);
                if !check_condition(value, condition) {
                    return false;
                }
            }
            true
        }
        _ => true,
    }
}

fn check_condition(value: Option<&JsonValue>, condition: &JsonValue) -> bool {
    // If condition is an operator object
    if let JsonValue::Object(ops) = condition {
        // Check for MongoDB-style operators
        for (op, op_val) in ops {
            match op.as_str() {
                "$eq" => {
                    if value != Some(op_val) {
                        return false;
                    }
                }
                "$ne" => {
                    if value == Some(op_val) {
                        return false;
                    }
                }
                "$gt" => {
                    if !compare_values(value, op_val, |a, b| a > b) {
                        return false;
                    }
                }
                "$gte" => {
                    if !compare_values(value, op_val, |a, b| a >= b) {
                        return false;
                    }
                }
                "$lt" => {
                    if !compare_values(value, op_val, |a, b| a < b) {
                        return false;
                    }
                }
                "$lte" => {
                    if !compare_values(value, op_val, |a, b| a <= b) {
                        return false;
                    }
                }
                "$in" => {
                    if let JsonValue::Array(arr) = op_val {
                        if !arr.iter().any(|v| Some(v) == value) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                "$nin" => {
                    if let JsonValue::Array(arr) = op_val {
                        if arr.iter().any(|v| Some(v) == value) {
                            return false;
                        }
                    }
                }
                _ => {
                    // Not an operator, treat as nested object equality
                    return value == Some(condition);
                }
            }
        }
        true
    } else {
        // Direct equality comparison
        value == Some(condition)
    }
}

fn compare_values<F>(value: Option<&JsonValue>, target: &JsonValue, cmp: F) -> bool
where
    F: Fn(f64, f64) -> bool,
{
    match (value, target) {
        (Some(JsonValue::Number(a)), JsonValue::Number(b)) => {
            if let (Some(a_f), Some(b_f)) = (a.as_f64(), b.as_f64()) {
                cmp(a_f, b_f)
            } else {
                false
            }
        }
        _ => false,
    }
}

// ========== Data Types ==========

/// Search result from a vector query.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SearchResult {
    id: String,
    score: f32,
    metadata: String, // JSON string
}

#[wasm_bindgen]
impl SearchResult {
    /// Get the record ID.
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    /// Get the similarity/distance score.
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Get the metadata as a JSON string.
    #[wasm_bindgen(getter, js_name = metadataJson)]
    pub fn metadata_json(&self) -> String {
        self.metadata.clone()
    }

    /// Get the metadata as a JavaScript object.
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> Result<JsValue, JsError> {
        let json: JsonValue = serde_json::from_str(&self.metadata)
            .map_err(|e| JsError::new(&format!("Invalid JSON: {}", e)))?;
        serde_wasm_bindgen::to_value(&json)
            .map_err(|e| JsError::new(&format!("Serialization error: {}", e)))
    }
}

/// Statistics from a compaction operation.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct CompactStats {
    expired_removed: u64,
    tombstones_removed: u64,
    bytes_freed: u64,
    duration_ms: u64,
}

#[wasm_bindgen]
impl CompactStats {
    #[wasm_bindgen(getter, js_name = expiredRemoved)]
    pub fn expired_removed(&self) -> u64 {
        self.expired_removed
    }

    #[wasm_bindgen(getter, js_name = tombstonesRemoved)]
    pub fn tombstones_removed(&self) -> u64 {
        self.tombstones_removed
    }

    #[wasm_bindgen(getter, js_name = bytesFreed)]
    pub fn bytes_freed(&self) -> u64 {
        self.bytes_freed
    }

    #[wasm_bindgen(getter, js_name = durationMs)]
    pub fn duration_ms(&self) -> u64 {
        self.duration_ms
    }
}

/// Recovery statistics from database open.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct RecoveryStats {
    promoted: u64,
    rolled_back: u64,
    tombstones: u64,
}

#[wasm_bindgen]
impl RecoveryStats {
    #[wasm_bindgen(getter)]
    pub fn promoted(&self) -> u64 {
        self.promoted
    }

    #[wasm_bindgen(getter, js_name = rolledBack)]
    pub fn rolled_back(&self) -> u64 {
        self.rolled_back
    }

    #[wasm_bindgen(getter)]
    pub fn tombstones(&self) -> u64 {
        self.tombstones
    }
}

// ========== AgentVec Database ==========

/// AgentVec vector database.
///
/// The main entry point for creating and managing vector collections.
/// In WASM mode, all data is stored in memory and not persisted.
#[wasm_bindgen]
pub struct AgentVec {
    inner: Arc<RwLock<DatabaseData>>,
}

#[wasm_bindgen]
impl AgentVec {
    /// Open or create a database at the given path.
    /// In WASM mode, the path is used as an identifier but data is not persisted.
    #[wasm_bindgen(constructor)]
    pub fn open(path: &str) -> Result<AgentVec, JsError> {
        let data = DatabaseData {
            path: path.to_string(),
            collections: HashMap::new(),
        };

        Ok(Self {
            inner: Arc::new(RwLock::new(data)),
        })
    }

    /// Get or create a collection.
    pub fn collection(
        &self,
        name: &str,
        dimensions: u32,
        metric: Metric,
    ) -> Result<Collection, JsError> {
        let mut db = self.inner.write()
            .map_err(|e| JsError::new(&format!("Lock error: {}", e)))?;

        if let Some(col) = db.collections.get(name) {
            return Ok(Collection {
                inner: Arc::clone(col),
            });
        }

        let col_data = CollectionData {
            name: name.to_string(),
            dimensions: dimensions as usize,
            metric,
            records: HashMap::new(),
        };

        let col = Arc::new(RwLock::new(col_data));
        db.collections.insert(name.to_string(), Arc::clone(&col));

        Ok(Collection { inner: col })
    }

    /// Get an existing collection.
    #[wasm_bindgen(js_name = getCollection)]
    pub fn get_collection(&self, name: &str) -> Result<Collection, JsError> {
        let db = self.inner.read()
            .map_err(|e| JsError::new(&format!("Lock error: {}", e)))?;

        match db.collections.get(name) {
            Some(col) => Ok(Collection {
                inner: Arc::clone(col),
            }),
            None => Err(JsError::new(&format!("Collection '{}' not found", name))),
        }
    }

    /// List all collection names.
    pub fn collections(&self) -> Result<Vec<String>, JsError> {
        let db = self.inner.read()
            .map_err(|e| JsError::new(&format!("Lock error: {}", e)))?;

        Ok(db.collections.keys().cloned().collect())
    }

    /// Delete a collection.
    #[wasm_bindgen(js_name = dropCollection)]
    pub fn drop_collection(&self, name: &str) -> Result<bool, JsError> {
        let mut db = self.inner.write()
            .map_err(|e| JsError::new(&format!("Lock error: {}", e)))?;

        Ok(db.collections.remove(name).is_some())
    }

    /// Flush all pending writes (no-op in WASM mode).
    pub fn sync(&self) -> Result<(), JsError> {
        Ok(())
    }

    /// Get recovery statistics (always zero in WASM mode).
    #[wasm_bindgen(js_name = recoveryStats)]
    pub fn recovery_stats(&self) -> RecoveryStats {
        RecoveryStats {
            promoted: 0,
            rolled_back: 0,
            tombstones: 0,
        }
    }
}

// ========== Collection ==========

/// A collection of vectors with metadata.
#[wasm_bindgen]
pub struct Collection {
    inner: Arc<RwLock<CollectionData>>,
}

#[wasm_bindgen]
impl Collection {
    /// Add a vector to the collection.
    pub fn add(
        &self,
        vector: &[f32],
        metadata: JsValue,
        id: Option<String>,
        ttl: Option<u64>,
    ) -> Result<String, JsError> {
        let metadata_json = parse_metadata(metadata)?;

        let mut col = self.inner.write()
            .map_err(|e| JsError::new(&format!("Lock error: {}", e)))?;

        // Validate dimensions
        if vector.len() != col.dimensions {
            return Err(JsError::new(&format!(
                "Dimension mismatch: expected {}, got {}",
                col.dimensions,
                vector.len()
            )));
        }

        // Generate or use provided ID
        let record_id = id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let now = current_unix_time();
        let record = Record {
            id: record_id.clone(),
            vector: vector.to_vec(),
            metadata: metadata_json,
            created_at: now,
            expires_at: ttl.map(|t| now + t),
            deleted: false,
        };

        col.records.insert(record_id.clone(), record);

        Ok(record_id)
    }

    /// Insert or update a vector.
    pub fn upsert(
        &self,
        id: &str,
        vector: &[f32],
        metadata: JsValue,
        ttl: Option<u64>,
    ) -> Result<(), JsError> {
        let metadata_json = parse_metadata(metadata)?;

        let mut col = self.inner.write()
            .map_err(|e| JsError::new(&format!("Lock error: {}", e)))?;

        // Validate dimensions
        if vector.len() != col.dimensions {
            return Err(JsError::new(&format!(
                "Dimension mismatch: expected {}, got {}",
                col.dimensions,
                vector.len()
            )));
        }

        let now = current_unix_time();
        let record = Record {
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata: metadata_json,
            created_at: now,
            expires_at: ttl.map(|t| now + t),
            deleted: false,
        };

        col.records.insert(id.to_string(), record);

        Ok(())
    }

    /// Search for nearest neighbors.
    pub fn search(
        &self,
        vector: &[f32],
        k: u32,
        filter: JsValue,
    ) -> Result<Vec<SearchResult>, JsError> {
        let filter_json = parse_filter(filter)?;

        let col = self.inner.read()
            .map_err(|e| JsError::new(&format!("Lock error: {}", e)))?;

        // Validate dimensions
        if vector.len() != col.dimensions {
            return Err(JsError::new(&format!(
                "Dimension mismatch: expected {}, got {}",
                col.dimensions,
                vector.len()
            )));
        }

        // Score all active records
        let mut scores: Vec<(String, f32, JsonValue)> = col
            .records
            .values()
            .filter(|r| r.is_active())
            .filter(|r| {
                filter_json
                    .as_ref()
                    .map(|f| matches_filter(&r.metadata, f))
                    .unwrap_or(true)
            })
            .map(|r| {
                let score = compute_score(vector, &r.vector, col.metric);
                (r.id.clone(), score, r.metadata.clone())
            })
            .collect();

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        let results: Vec<SearchResult> = scores
            .into_iter()
            .take(k as usize)
            .map(|(id, score, metadata)| SearchResult {
                id,
                score,
                metadata: serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string()),
            })
            .collect();

        Ok(results)
    }

    /// Get a record by ID.
    pub fn get(&self, id: &str) -> Result<Option<SearchResult>, JsError> {
        let col = self.inner.read()
            .map_err(|e| JsError::new(&format!("Lock error: {}", e)))?;

        match col.records.get(id) {
            Some(record) if record.is_active() => Ok(Some(SearchResult {
                id: record.id.clone(),
                score: 1.0,
                metadata: serde_json::to_string(&record.metadata)
                    .unwrap_or_else(|_| "{}".to_string()),
            })),
            _ => Ok(None),
        }
    }

    /// Delete a record by ID.
    pub fn delete(&self, id: &str) -> Result<bool, JsError> {
        let mut col = self.inner.write()
            .map_err(|e| JsError::new(&format!("Lock error: {}", e)))?;

        if let Some(record) = col.records.get_mut(id) {
            if record.is_active() {
                record.deleted = true;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Compact the collection (remove expired and deleted records).
    pub fn compact(&self) -> Result<CompactStats, JsError> {
        let start = js_sys::Date::now();

        let mut col = self.inner.write()
            .map_err(|e| JsError::new(&format!("Lock error: {}", e)))?;

        let mut expired_removed = 0u64;
        let mut tombstones_removed = 0u64;

        // Remove expired and deleted records
        col.records.retain(|_, record| {
            if record.deleted {
                tombstones_removed += 1;
                return false;
            }
            if record.is_expired() {
                expired_removed += 1;
                return false;
            }
            true
        });

        let duration_ms = (js_sys::Date::now() - start) as u64;

        Ok(CompactStats {
            expired_removed,
            tombstones_removed,
            bytes_freed: 0, // Not tracked in WASM mode
            duration_ms,
        })
    }

    /// Get the number of active records.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> Result<u64, JsError> {
        let col = self.inner.read()
            .map_err(|e| JsError::new(&format!("Lock error: {}", e)))?;

        let count = col.records.values().filter(|r| r.is_active()).count();
        Ok(count as u64)
    }

    /// Check if the collection is empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> Result<bool, JsError> {
        let len = self.length()?;
        Ok(len == 0)
    }

    /// Preload vectors into memory (no-op in WASM mode).
    pub fn preload(&self) -> Result<(), JsError> {
        Ok(())
    }

    /// Flush pending writes (no-op in WASM mode).
    pub fn sync(&self) -> Result<(), JsError> {
        Ok(())
    }

    /// Get the vector dimensions.
    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> u32 {
        self.inner.read().map(|c| c.dimensions as u32).unwrap_or(0)
    }

    /// Get the collection name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.read().map(|c| c.name.clone()).unwrap_or_default()
    }

    /// Get the distance metric.
    #[wasm_bindgen(getter)]
    pub fn metric(&self) -> Metric {
        self.inner.read().map(|c| c.metric).unwrap_or(Metric::Cosine)
    }

    /// Get the vector storage size in bytes (estimated).
    #[wasm_bindgen(getter, js_name = vectorsSizeBytes)]
    pub fn vectors_size_bytes(&self) -> u64 {
        self.inner.read().map(|c| {
            let active_count = c.records.values().filter(|r| r.is_active()).count();
            (active_count * c.dimensions * 4) as u64 // 4 bytes per f32
        }).unwrap_or(0)
    }
}

// ========== Helper Functions ==========

/// Parse metadata from JS value (object or JSON string).
fn parse_metadata(value: JsValue) -> Result<JsonValue, JsError> {
    if value.is_undefined() || value.is_null() {
        return Ok(JsonValue::Object(serde_json::Map::new()));
    }

    if let Some(s) = value.as_string() {
        serde_json::from_str(&s)
            .map_err(|e| JsError::new(&format!("Invalid metadata JSON: {}", e)))
    } else {
        serde_wasm_bindgen::from_value(value)
            .map_err(|e| JsError::new(&format!("Invalid metadata: {}", e)))
    }
}

/// Parse filter from JS value.
fn parse_filter(value: JsValue) -> Result<Option<JsonValue>, JsError> {
    if value.is_undefined() || value.is_null() {
        return Ok(None);
    }

    let json: JsonValue = if let Some(s) = value.as_string() {
        serde_json::from_str(&s)
            .map_err(|e| JsError::new(&format!("Invalid filter JSON: {}", e)))?
    } else {
        serde_wasm_bindgen::from_value(value)
            .map_err(|e| JsError::new(&format!("Invalid filter: {}", e)))?
    };

    Ok(Some(json))
}

// ========== Tests ==========

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((l2_distance(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_filter_equality() {
        let metadata = serde_json::json!({"user": "alice", "score": 85});
        let filter = serde_json::json!({"user": "alice"});
        assert!(matches_filter(&metadata, &filter));

        let filter2 = serde_json::json!({"user": "bob"});
        assert!(!matches_filter(&metadata, &filter2));
    }

    #[test]
    fn test_filter_comparison() {
        let metadata = serde_json::json!({"score": 85});

        let filter_gt = serde_json::json!({"score": {"$gt": 80}});
        assert!(matches_filter(&metadata, &filter_gt));

        let filter_lt = serde_json::json!({"score": {"$lt": 80}});
        assert!(!matches_filter(&metadata, &filter_lt));
    }

    #[test]
    fn test_filter_in() {
        let metadata = serde_json::json!({"status": "active"});

        let filter = serde_json::json!({"status": {"$in": ["active", "pending"]}});
        assert!(matches_filter(&metadata, &filter));

        let filter2 = serde_json::json!({"status": {"$in": ["deleted", "expired"]}});
        assert!(!matches_filter(&metadata, &filter2));
    }
}
