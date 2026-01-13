use serde::Serialize;
use serde_json::Value as JsonValue;

/// Standard API response wrapper
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }
}

impl ApiResponse<()> {
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(msg.into()),
        }
    }
}

/// Collection info response
#[derive(Debug, Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimensions: usize,
    pub metric: String,
    pub count: usize,
}

/// Search result item
#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub id: String,
    pub score: f32,
    pub metadata: JsonValue,
}

/// Full record with vector
#[derive(Debug, Serialize)]
pub struct FullRecordResponse {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: JsonValue,
}

/// Add vector response
#[derive(Debug, Serialize)]
pub struct AddVectorResponse {
    pub id: String,
}

/// Batch add response
#[derive(Debug, Serialize)]
pub struct AddBatchResponse {
    pub ids: Vec<String>,
    pub count: usize,
}

/// Compact operation stats
#[derive(Debug, Serialize)]
pub struct CompactStatsResponse {
    pub expired_removed: usize,
    pub tombstones_removed: usize,
    pub bytes_freed: u64,
    pub duration_ms: u64,
}

/// Collection statistics
#[derive(Debug, Serialize)]
pub struct CollectionStatsResponse {
    pub name: String,
    pub count: usize,
    pub dimensions: usize,
    pub metric: String,
    pub vector_storage_bytes: u64,
    pub pending_writes: usize,
    pub has_hnsw_index: bool,
    pub hnsw_node_count: Option<usize>,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}
