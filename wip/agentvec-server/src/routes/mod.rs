use axum::{
    routing::{delete, get, post},
    Router,
};

use crate::state::AppState;

mod collections;
mod health;
mod records;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Health
        .route("/health", get(health::health))
        // Collections
        .route("/collections", get(collections::list_collections))
        .route("/collections", post(collections::create_collection))
        .route("/collections/:name", delete(collections::drop_collection))
        .route(
            "/collections/:name/stats",
            get(collections::collection_stats),
        )
        .route(
            "/collections/:name/compact",
            post(collections::compact_collection),
        )
        // Records
        .route("/collections/:name/add", post(records::add_vector))
        .route("/collections/:name/add_batch", post(records::add_batch))
        .route("/collections/:name/upsert", post(records::upsert_vector))
        .route("/collections/:name/search", post(records::search_vectors))
        .route(
            "/collections/:name/records/:id",
            get(records::get_record),
        )
        .route(
            "/collections/:name/records/:id",
            delete(records::delete_record),
        )
        .with_state(state)
}
