use std::time::Duration;

use agentvec::AgentVec;
use clap::Parser;
use tokio::signal;
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use agentvec_server::{config::ServerConfig, routes, state::AppState};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = ServerConfig::parse();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Opening database at {:?}", config.db_path);
    let db = AgentVec::open(&config.db_path)?;
    let state = AppState::new(db);

    // Build router with middleware
    let app = routes::create_router(state.clone())
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::new(Duration::from_secs(config.timeout)));

    let app = if config.cors {
        tracing::info!("CORS enabled for all origins");
        app.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
    } else {
        app
    };

    let addr = config.bind_addr();
    tracing::info!("Starting AgentVec server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await?;

    tracing::info!("Server shutdown complete");
    Ok(())
}

async fn shutdown_signal(state: AppState) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, syncing database...");

    if let Err(e) = state.db.sync() {
        tracing::error!("Error syncing database on shutdown: {}", e);
    }
}
