use axum::{
    extract::{Path, State},
    Json,
};
use agentvec::Filter;

use crate::{
    dto::{
        request::{AddBatchRequest, AddVectorRequest, SearchRequest, UpsertVectorRequest},
        response::{
            AddBatchResponse, AddVectorResponse, ApiResponse, FullRecordResponse, SearchResultItem,
        },
    },
    error::{ApiError, ApiResult},
    state::AppState,
};

/// POST /collections/{name}/add - Add a single vector
pub async fn add_vector(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<AddVectorRequest>,
) -> ApiResult<AddVectorResponse> {
    let col = state.db.get_collection(&name).map_err(ApiError::from)?;

    let id = col
        .add(&req.vector, req.metadata, req.id.as_deref(), req.ttl)
        .map_err(ApiError::from)?;

    Ok(Json(ApiResponse::success(AddVectorResponse { id })))
}

/// POST /collections/{name}/add_batch - Add vectors in batch
pub async fn add_batch(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<AddBatchRequest>,
) -> ApiResult<AddBatchResponse> {
    let col = state.db.get_collection(&name).map_err(ApiError::from)?;

    let ids_refs: Option<Vec<&str>> = req
        .ids
        .as_ref()
        .map(|ids| ids.iter().map(|s| s.as_str()).collect());

    let ids = col
        .add_batch(
            &req.vectors,
            &req.metadatas,
            ids_refs.as_deref(),
            req.ttls.as_deref(),
        )
        .map_err(ApiError::from)?;

    let count = ids.len();
    Ok(Json(ApiResponse::success(AddBatchResponse { ids, count })))
}

/// POST /collections/{name}/upsert - Upsert a vector
pub async fn upsert_vector(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertVectorRequest>,
) -> ApiResult<AddVectorResponse> {
    let col = state.db.get_collection(&name).map_err(ApiError::from)?;

    col.upsert(&req.id, &req.vector, req.metadata, req.ttl)
        .map_err(ApiError::from)?;

    Ok(Json(ApiResponse::success(AddVectorResponse { id: req.id })))
}

/// POST /collections/{name}/search - Search vectors
pub async fn search_vectors(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> ApiResult<Vec<SearchResultItem>> {
    let col = state.db.get_collection(&name).map_err(ApiError::from)?;

    let filter = req.filter.map(|f| Filter::from_json(&f));

    let results = col
        .search(&req.vector, req.k, filter)
        .map_err(ApiError::from)?;

    let items: Vec<SearchResultItem> = results
        .into_iter()
        .map(|r| SearchResultItem {
            id: r.id,
            score: r.score,
            metadata: r.metadata,
        })
        .collect();

    Ok(Json(ApiResponse::success(items)))
}

/// GET /collections/{name}/records/{id} - Get record by ID
pub async fn get_record(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, String)>,
) -> ApiResult<FullRecordResponse> {
    let col = state.db.get_collection(&name).map_err(ApiError::from)?;

    match col.get_with_vector(&id).map_err(ApiError::from)? {
        Some(record) => Ok(Json(ApiResponse::success(FullRecordResponse {
            id: record.id,
            vector: record.vector,
            metadata: record.metadata,
        }))),
        None => Err(ApiError::not_found(format!("Record '{}' not found", id))),
    }
}

/// DELETE /collections/{name}/records/{id} - Delete record by ID
pub async fn delete_record(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, String)>,
) -> ApiResult<bool> {
    let col = state.db.get_collection(&name).map_err(ApiError::from)?;

    let deleted = col.delete(&id).map_err(ApiError::from)?;

    if deleted {
        Ok(Json(ApiResponse::success(true)))
    } else {
        Err(ApiError::not_found(format!("Record '{}' not found", id)))
    }
}
