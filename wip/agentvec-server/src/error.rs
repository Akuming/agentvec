use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use agentvec::AgentVecError;
use serde_json::json;

use crate::dto::response::ApiResponse;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, msg)
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, msg)
    }

    pub fn conflict(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, msg)
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, msg)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "success": false,
            "error": self.message,
        }));
        (self.status, body).into_response()
    }
}

impl From<AgentVecError> for ApiError {
    fn from(err: AgentVecError) -> Self {
        match &err {
            AgentVecError::NotFound(id) => {
                ApiError::not_found(format!("Record not found: {}", id))
            }
            AgentVecError::CollectionNotFound(name) => {
                ApiError::not_found(format!("Collection not found: {}", name))
            }
            AgentVecError::CollectionExists(name) => {
                ApiError::conflict(format!("Collection already exists: {}", name))
            }
            AgentVecError::DimensionMismatch { expected, got } => ApiError::bad_request(format!(
                "Dimension mismatch: expected {}, got {}",
                expected, got
            )),
            AgentVecError::DimensionsTooLarge { max, got } => ApiError::bad_request(format!(
                "Dimensions too large: max {}, got {}",
                max, got
            )),
            AgentVecError::InvalidInput(msg) => ApiError::bad_request(msg.clone()),
            AgentVecError::InvalidFormat(msg) => ApiError::bad_request(msg.clone()),
            AgentVecError::Io(_)
            | AgentVecError::Database(_)
            | AgentVecError::Corruption(_)
            | AgentVecError::Serialization(_)
            | AgentVecError::Transaction(_)
            | AgentVecError::Lock(_) => ApiError::internal(err.to_string()),
            _ => ApiError::internal(err.to_string()),
        }
    }
}

pub type ApiResult<T> = Result<Json<ApiResponse<T>>, ApiError>;
