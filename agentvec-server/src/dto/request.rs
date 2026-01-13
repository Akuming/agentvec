use serde::Deserialize;
use serde_json::Value as JsonValue;

fn default_metric() -> String {
    "cosine".to_string()
}

fn default_k() -> usize {
    10
}

/// Request to create a new collection
#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimensions: usize,
    #[serde(default = "default_metric")]
    pub metric: String,
}

/// Request to add a vector
#[derive(Debug, Deserialize)]
pub struct AddVectorRequest {
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: JsonValue,
    pub id: Option<String>,
    pub ttl: Option<u64>,
}

/// Request to upsert a vector
#[derive(Debug, Deserialize)]
pub struct UpsertVectorRequest {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: JsonValue,
    pub ttl: Option<u64>,
}

/// Request to add vectors in batch
#[derive(Debug, Deserialize)]
pub struct AddBatchRequest {
    pub vectors: Vec<Vec<f32>>,
    pub metadatas: Vec<JsonValue>,
    pub ids: Option<Vec<String>>,
    pub ttls: Option<Vec<Option<u64>>>,
}

/// Request to search vectors
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    #[serde(default = "default_k")]
    pub k: usize,
    pub filter: Option<JsonValue>,
}
