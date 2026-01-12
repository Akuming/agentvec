use axum::{
    extract::{Path, State},
    Json,
};
use agentvec::Metric;

use crate::{
    dto::{
        request::CreateCollectionRequest,
        response::{ApiResponse, CollectionInfo, CollectionStatsResponse, CompactStatsResponse},
    },
    error::{ApiError, ApiResult},
    state::AppState,
};

/// GET /collections - List all collections
pub async fn list_collections(State(state): State<AppState>) -> ApiResult<Vec<CollectionInfo>> {
    let names = state.db.collections().map_err(ApiError::from)?;

    let mut collections = Vec::with_capacity(names.len());
    for name in names {
        let col = state.db.get_collection(&name).map_err(ApiError::from)?;
        collections.push(CollectionInfo {
            name: col.name().to_string(),
            dimensions: col.dimensions(),
            metric: col.metric().to_string(),
            count: col.len().map_err(ApiError::from)?,
        });
    }

    Ok(Json(ApiResponse::success(collections)))
}

/// POST /collections - Create a new collection
pub async fn create_collection(
    State(state): State<AppState>,
    Json(req): Json<CreateCollectionRequest>,
) -> ApiResult<CollectionInfo> {
    let metric: Metric = req
        .metric
        .parse()
        .map_err(|e: String| ApiError::bad_request(e))?;

    let col = state
        .db
        .collection(&req.name, req.dimensions, metric)
        .map_err(ApiError::from)?;

    Ok(Json(ApiResponse::success(CollectionInfo {
        name: col.name().to_string(),
        dimensions: col.dimensions(),
        metric: col.metric().to_string(),
        count: col.len().map_err(ApiError::from)?,
    })))
}

/// DELETE /collections/{name} - Drop a collection
pub async fn drop_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<bool> {
    let dropped = state.db.drop_collection(&name).map_err(ApiError::from)?;

    if dropped {
        Ok(Json(ApiResponse::success(true)))
    } else {
        Err(ApiError::not_found(format!(
            "Collection '{}' not found",
            name
        )))
    }
}

/// GET /collections/{name}/stats - Get collection statistics
pub async fn collection_stats(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<CollectionStatsResponse> {
    let col = state.db.get_collection(&name).map_err(ApiError::from)?;

    Ok(Json(ApiResponse::success(CollectionStatsResponse {
        name: col.name().to_string(),
        count: col.len().map_err(ApiError::from)?,
        dimensions: col.dimensions(),
        metric: col.metric().to_string(),
        vector_storage_bytes: col.vectors_size_bytes(),
        pending_writes: col.pending_count(),
        has_hnsw_index: col.has_hnsw_index(),
        hnsw_node_count: col.hnsw_node_count(),
    })))
}

/// POST /collections/{name}/compact - Compact collection
pub async fn compact_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<CompactStatsResponse> {
    let col = state.db.get_collection(&name).map_err(ApiError::from)?;
    let stats = col.compact().map_err(ApiError::from)?;

    Ok(Json(ApiResponse::success(CompactStatsResponse {
        expired_removed: stats.expired_removed,
        tombstones_removed: stats.tombstones_removed,
        bytes_freed: stats.bytes_freed,
        duration_ms: stats.duration_ms,
    })))
}
