//! Metadata storage using redb.
//!
//! Stores record metadata (ID, slot offset, JSON metadata, timestamps, status)
//! in a transactional key-value store for crash safety.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use redb::{Database, ReadTransaction, ReadableTable, TableDefinition, WriteTransaction};

use crate::config::CollectionConfig;
use crate::error::Result;
use crate::recovery::RecordStatus;

/// Table definition for records: ID -> Record (bincode serialized)
const RECORDS_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("records");

/// Table definition for collection config
const COLLECTIONS_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("collections");

/// A record stored in the database.
///
/// Note: Metadata is stored as a JSON string internally because bincode
/// cannot serialize serde_json::Value directly (it uses deserialize_any).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// Unique identifier for the record.
    pub id: String,

    /// Slot index in the vector storage.
    pub slot_offset: u64,

    /// User-provided metadata stored as JSON string.
    /// Use `metadata()` and `set_metadata()` to access as JsonValue.
    metadata_json: String,

    /// Unix timestamp when record was created.
    pub created_at: u64,

    /// Optional Unix timestamp when record expires (TTL).
    pub expires_at: Option<u64>,

    /// Record status for crash recovery.
    pub status: RecordStatus,
}

impl Record {
    /// Create a new record.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier
    /// * `slot_offset` - Slot index in vector storage
    /// * `metadata` - User metadata
    /// * `ttl` - Optional time-to-live in seconds
    /// * `status` - Initial status (usually Pending during write)
    pub fn new(
        id: impl Into<String>,
        slot_offset: u64,
        metadata: JsonValue,
        ttl: Option<u64>,
        status: RecordStatus,
    ) -> Self {
        let now = current_unix_time();
        Self {
            id: id.into(),
            slot_offset,
            metadata_json: serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string()),
            created_at: now,
            expires_at: ttl.map(|t| now + t),
            status,
        }
    }

    /// Get the metadata as a JsonValue.
    pub fn metadata(&self) -> JsonValue {
        serde_json::from_str(&self.metadata_json).unwrap_or(JsonValue::Null)
    }

    /// Set the metadata from a JsonValue.
    pub fn set_metadata(&mut self, metadata: JsonValue) {
        self.metadata_json = serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string());
    }

    /// Check if the record is expired.
    ///
    /// A record is expired if the current time is at or past the expiry time.
    /// TTL=0 means the record is expired immediately upon creation.
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(exp) => current_unix_time() >= exp,
            None => false,
        }
    }

    /// Check if the record is active (not expired and not tombstone).
    pub fn is_active(&self) -> bool {
        self.status == RecordStatus::Active && !self.is_expired()
    }

    /// Serialize the record for storage.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(Into::into)
    }

    /// Deserialize a record from storage.
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(Into::into)
    }
}

/// Get current Unix timestamp in seconds.
pub fn current_unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Metadata storage manager.
///
/// Handles all redb operations for records and collection configuration.
pub struct MetadataStorage {
    /// The redb database.
    db: Database,
}

impl MetadataStorage {
    /// Create or open metadata storage at the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the meta.redb file
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let db = Database::create(path.as_ref())?;
        Ok(Self { db })
    }

    /// Begin a read transaction.
    pub fn begin_read(&self) -> Result<ReadTransaction> {
        self.db.begin_read().map_err(Into::into)
    }

    /// Begin a write transaction.
    pub fn begin_write(&self) -> Result<WriteTransaction> {
        self.db.begin_write().map_err(Into::into)
    }

    // ==================== Record Operations ====================

    /// Insert a record.
    pub fn insert_record(&self, txn: &WriteTransaction, record: &Record) -> Result<()> {
        let mut table = txn.open_table(RECORDS_TABLE)?;
        let data = record.serialize()?;
        table.insert(record.id.as_str(), data.as_slice())?;
        Ok(())
    }

    /// Get a record by ID.
    pub fn get_record(&self, txn: &ReadTransaction, id: &str) -> Result<Option<Record>> {
        let table = match txn.open_table(RECORDS_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        match table.get(id)? {
            Some(data) => Ok(Some(Record::deserialize(data.value())?)),
            None => Ok(None),
        }
    }

    /// Update a record's status.
    pub fn update_record_status(
        &self,
        txn: &WriteTransaction,
        id: &str,
        status: RecordStatus,
    ) -> Result<bool> {
        let read_txn = self.db.begin_read()?;
        let record = match self.get_record(&read_txn, id)? {
            Some(mut r) => {
                r.status = status;
                r
            }
            None => return Ok(false),
        };
        drop(read_txn);

        self.insert_record(txn, &record)?;
        Ok(true)
    }

    /// Delete a record.
    pub fn delete_record(&self, txn: &WriteTransaction, id: &str) -> Result<bool> {
        let mut table = txn.open_table(RECORDS_TABLE)?;
        let existed = table.remove(id)?.is_some();
        Ok(existed)
    }

    /// Iterate over all records.
    pub fn iter_records<F>(&self, txn: &ReadTransaction, mut f: F) -> Result<()>
    where
        F: FnMut(Record) -> Result<bool>, // Return false to stop iteration
    {
        let table = match txn.open_table(RECORDS_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(()),
            Err(e) => return Err(e.into()),
        };

        for result in table.iter()? {
            let (_, value) = result?;
            let record = Record::deserialize(value.value())?;
            if !f(record)? {
                break;
            }
        }

        Ok(())
    }

    /// Count active (non-expired, non-tombstone) records.
    pub fn count_active(&self, txn: &ReadTransaction) -> Result<usize> {
        let mut count = 0;
        self.iter_records(txn, |record| {
            if record.is_active() {
                count += 1;
            }
            Ok(true)
        })?;
        Ok(count)
    }

    // ==================== Collection Operations ====================

    /// Register a collection.
    pub fn register_collection(
        &self,
        txn: &WriteTransaction,
        config: &CollectionConfig,
    ) -> Result<()> {
        let mut table = txn.open_table(COLLECTIONS_TABLE)?;
        let data = bincode::serialize(config)?;
        table.insert(config.name.as_str(), data.as_slice())?;
        Ok(())
    }

    /// Get collection configuration.
    pub fn get_collection(
        &self,
        txn: &ReadTransaction,
        name: &str,
    ) -> Result<Option<CollectionConfig>> {
        let table = match txn.open_table(COLLECTIONS_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        match table.get(name)? {
            Some(data) => {
                let config: CollectionConfig = bincode::deserialize(data.value())?;
                Ok(Some(config))
            }
            None => Ok(None),
        }
    }

    /// List all collection names.
    pub fn list_collections(&self, txn: &ReadTransaction) -> Result<Vec<String>> {
        let table = match txn.open_table(COLLECTIONS_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };

        let mut names = Vec::new();
        for result in table.iter()? {
            let (key, _) = result?;
            names.push(key.value().to_string());
        }

        Ok(names)
    }

    /// Remove a collection registration.
    pub fn remove_collection(&self, txn: &WriteTransaction, name: &str) -> Result<bool> {
        let mut table = txn.open_table(COLLECTIONS_TABLE)?;
        let existed = table.remove(name)?.is_some();
        Ok(existed)
    }

    /// Compact the database.
    pub fn compact(&mut self) -> Result<()> {
        self.db.compact()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Metric;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn test_record_creation() {
        let record = Record::new("test_id", 42, json!({"key": "value"}), None, RecordStatus::Active);

        assert_eq!(record.id, "test_id");
        assert_eq!(record.slot_offset, 42);
        assert_eq!(record.metadata()["key"], "value");
        assert!(record.created_at > 0);
        assert!(record.expires_at.is_none());
        assert!(!record.is_expired());
        assert!(record.is_active());
    }

    #[test]
    fn test_record_with_ttl() {
        let record = Record::new("test", 0, json!({}), Some(3600), RecordStatus::Active);

        assert!(record.expires_at.is_some());
        assert!(!record.is_expired()); // Shouldn't be expired immediately
    }

    #[test]
    fn test_record_expired() {
        let mut record = Record::new("test", 0, json!({}), Some(0), RecordStatus::Active);
        // Set expires_at to past
        record.expires_at = Some(0);

        assert!(record.is_expired());
        assert!(!record.is_active());
    }

    #[test]
    fn test_record_serialization() {
        let record = Record::new("test", 42, json!({"nested": {"value": 123}}), Some(3600), RecordStatus::Active);

        let data = record.serialize().unwrap();
        let decoded = Record::deserialize(&data).unwrap();

        assert_eq!(decoded.id, record.id);
        assert_eq!(decoded.slot_offset, record.slot_offset);
        assert_eq!(decoded.metadata(), record.metadata());
        assert_eq!(decoded.created_at, record.created_at);
        assert_eq!(decoded.expires_at, record.expires_at);
        assert_eq!(decoded.status, record.status);
    }

    #[test]
    fn test_metadata_storage_records() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.redb");

        let storage = MetadataStorage::open(&path).unwrap();

        // Insert
        let record = Record::new("id1", 0, json!({"x": 1}), None, RecordStatus::Active);
        {
            let txn = storage.begin_write().unwrap();
            storage.insert_record(&txn, &record).unwrap();
            txn.commit().unwrap();
        }

        // Get
        {
            let txn = storage.begin_read().unwrap();
            let retrieved = storage.get_record(&txn, "id1").unwrap();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap().id, "id1");
        }

        // Count
        {
            let txn = storage.begin_read().unwrap();
            let count = storage.count_active(&txn).unwrap();
            assert_eq!(count, 1);
        }

        // Delete
        {
            let txn = storage.begin_write().unwrap();
            let deleted = storage.delete_record(&txn, "id1").unwrap();
            assert!(deleted);
            txn.commit().unwrap();
        }

        // Verify deleted
        {
            let txn = storage.begin_read().unwrap();
            let retrieved = storage.get_record(&txn, "id1").unwrap();
            assert!(retrieved.is_none());
        }
    }

    #[test]
    fn test_metadata_storage_collections() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.redb");

        let storage = MetadataStorage::open(&path).unwrap();

        let config = CollectionConfig::new("test_collection", 384, Metric::Cosine);

        // Register
        {
            let txn = storage.begin_write().unwrap();
            storage.register_collection(&txn, &config).unwrap();
            txn.commit().unwrap();
        }

        // Get
        {
            let txn = storage.begin_read().unwrap();
            let retrieved = storage.get_collection(&txn, "test_collection").unwrap();
            assert!(retrieved.is_some());
            let c = retrieved.unwrap();
            assert_eq!(c.name, "test_collection");
            assert_eq!(c.dimensions, 384);
            assert_eq!(c.metric, Metric::Cosine);
        }

        // List
        {
            let txn = storage.begin_read().unwrap();
            let names = storage.list_collections(&txn).unwrap();
            assert_eq!(names, vec!["test_collection"]);
        }

        // Remove
        {
            let txn = storage.begin_write().unwrap();
            let removed = storage.remove_collection(&txn, "test_collection").unwrap();
            assert!(removed);
            txn.commit().unwrap();
        }

        // Verify removed
        {
            let txn = storage.begin_read().unwrap();
            let names = storage.list_collections(&txn).unwrap();
            assert!(names.is_empty());
        }
    }

    #[test]
    fn test_update_record_status() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.redb");

        let storage = MetadataStorage::open(&path).unwrap();

        // Insert with Pending status
        let record = Record::new("id1", 0, json!({}), None, RecordStatus::Pending);
        {
            let txn = storage.begin_write().unwrap();
            storage.insert_record(&txn, &record).unwrap();
            txn.commit().unwrap();
        }

        // Update to Active
        {
            let txn = storage.begin_write().unwrap();
            let updated = storage.update_record_status(&txn, "id1", RecordStatus::Active).unwrap();
            assert!(updated);
            txn.commit().unwrap();
        }

        // Verify status
        {
            let txn = storage.begin_read().unwrap();
            let record = storage.get_record(&txn, "id1").unwrap().unwrap();
            assert_eq!(record.status, RecordStatus::Active);
        }
    }

    #[test]
    fn test_iter_records() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meta.redb");

        let storage = MetadataStorage::open(&path).unwrap();

        // Insert multiple records
        {
            let txn = storage.begin_write().unwrap();
            for i in 0..5 {
                let record = Record::new(
                    format!("id{}", i),
                    i as u64,
                    json!({"idx": i}),
                    None,
                    RecordStatus::Active,
                );
                storage.insert_record(&txn, &record).unwrap();
            }
            txn.commit().unwrap();
        }

        // Iterate
        {
            let txn = storage.begin_read().unwrap();
            let mut count = 0;
            storage.iter_records(&txn, |_record| {
                count += 1;
                Ok(true)
            }).unwrap();
            assert_eq!(count, 5);
        }
    }
}
