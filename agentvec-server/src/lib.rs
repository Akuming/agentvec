//! AgentVec HTTP Server library.
//!
//! This crate provides the HTTP server implementation for AgentVec.
//! It exposes the AgentVec vector database functionality via a RESTful JSON API.

pub mod config;
pub mod dto;
pub mod error;
pub mod routes;
pub mod state;

pub use config::ServerConfig;
pub use state::AppState;
