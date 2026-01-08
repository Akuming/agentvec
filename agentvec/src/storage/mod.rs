//! Storage layer for AgentVec.
//!
//! This module provides the low-level storage primitives:
//!
//! - [`VectorStorage`]: Memory-mapped vector file with fixed-slot allocation
//! - [`MetadataStorage`]: redb-based transactional metadata storage
//! - [`Freelist`]: Slot recycling for deleted vectors

mod vectors;
mod metadata;
mod freelist;

pub use vectors::VectorStorage;
pub use metadata::{MetadataStorage, Record, current_unix_time};
pub use freelist::{Freelist, FreelistManager};
