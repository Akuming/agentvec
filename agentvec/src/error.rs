//! Error types for AgentVec operations.

use std::io;
use thiserror::Error;

/// Result type alias for AgentVec operations.
pub type Result<T> = std::result::Result<T, AgentVecError>;

/// Errors that can occur during AgentVec operations.
#[derive(Debug, Error)]
pub enum AgentVecError {
    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Database error from redb.
    #[error("Database error: {0}")]
    Database(String),

    /// Vector dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension count.
        expected: usize,
        /// Actual dimension count provided.
        got: usize,
    },

    /// Vector dimensions exceed maximum allowed.
    #[error("Dimensions too large: maximum is {max}, got {got}")]
    DimensionsTooLarge {
        /// Maximum allowed dimensions.
        max: usize,
        /// Actual dimension count provided.
        got: usize,
    },

    /// Invalid dimension specification (count mismatch).
    #[error("Invalid dimensions: expected {expected}, got {got}")]
    InvalidDimensions {
        /// Expected dimensions.
        expected: usize,
        /// Actual dimensions.
        got: usize,
    },

    /// Invalid input parameter.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid file format.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Deserialization error.
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// Record not found by ID.
    #[error("Record not found: {0}")]
    NotFound(String),

    /// Collection not found by name.
    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    /// Collection already exists with different configuration.
    #[error("Collection already exists: {0}")]
    CollectionExists(String),

    /// Data corruption detected.
    #[error("Data corruption: {0}")]
    Corruption(String),

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Transaction error.
    #[error("Transaction error: {0}")]
    Transaction(String),

    /// Lock acquisition error.
    #[error("Lock error: {0}")]
    Lock(String),
}

impl From<redb::Error> for AgentVecError {
    fn from(err: redb::Error) -> Self {
        AgentVecError::Database(err.to_string())
    }
}

impl From<redb::DatabaseError> for AgentVecError {
    fn from(err: redb::DatabaseError) -> Self {
        AgentVecError::Database(err.to_string())
    }
}

impl From<redb::TableError> for AgentVecError {
    fn from(err: redb::TableError) -> Self {
        AgentVecError::Database(err.to_string())
    }
}

impl From<redb::TransactionError> for AgentVecError {
    fn from(err: redb::TransactionError) -> Self {
        AgentVecError::Transaction(err.to_string())
    }
}

impl From<redb::CommitError> for AgentVecError {
    fn from(err: redb::CommitError) -> Self {
        AgentVecError::Transaction(err.to_string())
    }
}

impl From<redb::StorageError> for AgentVecError {
    fn from(err: redb::StorageError) -> Self {
        AgentVecError::Database(err.to_string())
    }
}

impl From<redb::CompactionError> for AgentVecError {
    fn from(err: redb::CompactionError) -> Self {
        AgentVecError::Database(err.to_string())
    }
}

impl From<bincode::Error> for AgentVecError {
    fn from(err: bincode::Error) -> Self {
        AgentVecError::Serialization(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = AgentVecError::DimensionMismatch {
            expected: 384,
            got: 512,
        };
        let msg = e.to_string();
        assert!(msg.contains("384"), "Error message should contain expected dimension");
        assert!(msg.contains("512"), "Error message should contain actual dimension");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: AgentVecError = io_err.into();
        assert!(matches!(err, AgentVecError::Io(_)));
    }

    #[test]
    fn test_dimension_too_large() {
        let e = AgentVecError::DimensionsTooLarge {
            max: 65536,
            got: 100000,
        };
        assert!(e.to_string().contains("65536"));
        assert!(e.to_string().contains("100000"));
    }

    #[test]
    fn test_all_error_variants_have_display() {
        // Ensure all variants can be displayed without panic
        let errors: Vec<AgentVecError> = vec![
            AgentVecError::Io(io::Error::new(io::ErrorKind::Other, "test")),
            AgentVecError::Database("test".into()),
            AgentVecError::DimensionMismatch { expected: 1, got: 2 },
            AgentVecError::DimensionsTooLarge { max: 1, got: 2 },
            AgentVecError::InvalidDimensions { expected: 384, got: 512 },
            AgentVecError::NotFound("test".into()),
            AgentVecError::CollectionNotFound("test".into()),
            AgentVecError::CollectionExists("test".into()),
            AgentVecError::Corruption("test".into()),
            AgentVecError::Serialization("test".into()),
            AgentVecError::Transaction("test".into()),
            AgentVecError::Lock("test".into()),
            AgentVecError::InvalidFormat("test".into()),
            AgentVecError::Deserialization("test".into()),
            AgentVecError::InvalidInput("test".into()),
        ];

        for err in errors {
            let _ = err.to_string();
        }
    }
}
