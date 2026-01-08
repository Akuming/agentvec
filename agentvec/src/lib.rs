//! AgentVec: A lightweight, serverless vector database for AI agent memory.
//!
//! AgentVec is designed for transactional, row-oriented workloads typical of local AI agents.
//! It prioritizes low RAM usage, instant startup, efficient in-place updates, and memory decay (TTL).
//!
//! # Quick Start
//!
//! ```no_run
//! use agentvec::{AgentVec, Metric};
//! use serde_json::json;
//!
//! // Open or create database
//! let db = AgentVec::open("./agent_memory.avdb").unwrap();
//!
//! // Create a collection
//! let collection = db.collection("memories", 384, Metric::Cosine).unwrap();
//!
//! // Add a vector
//! let id = collection.add(
//!     &[0.1; 384],
//!     json!({"type": "conversation"}),
//!     None,
//!     None,
//! ).unwrap();
//!
//! // Search
//! let results = collection.search(&[0.1; 384], 10, None).unwrap();
//! ```
//!
//! # Architecture
//!
//! AgentVec uses a hybrid storage model:
//! - **Vectors**: Memory-mapped file for zero-copy access
//! - **Metadata**: redb (pure Rust ACID key-value store) for transactional safety
//!
//! Each collection is stored in its own directory with independent vector dimensions.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

mod config;
mod db;
mod error;
mod collection;
mod filter;
mod recovery;
mod export;

pub mod storage;
pub mod search;

// Re-export public API
pub use config::{CollectionConfig, Metric};
pub use db::AgentVec;
pub use error::{AgentVecError, Result};
pub use collection::{Collection, CompactStats, WriteConfig};
pub use filter::Filter;
pub use search::SearchResult;
pub use search::HnswConfig;
pub use recovery::RecoveryStats;
pub use export::{ExportHeader, ExportRecord, ImportStats};
