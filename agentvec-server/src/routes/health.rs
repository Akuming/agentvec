use axum::Json;

use crate::dto::response::{ApiResponse, HealthResponse};

pub async fn health() -> Json<ApiResponse<HealthResponse>> {
    Json(ApiResponse::success(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    }))
}
