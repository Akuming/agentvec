//! Freelist management for recycling deleted vector slots.
//!
//! The freelist tracks which slots in the vector storage are available for reuse.
//! It is persisted in redb for crash safety, with an in-memory cache for performance.
//!
//! # Design
//!
//! - Authoritative state is stored in redb
//! - In-memory Vec<u64> caches the freelist for fast pop/push
//! - On startup, freelist is loaded from redb
//! - All mutations go through redb for crash safety

use std::sync::Mutex;

use crate::error::Result;

/// Freelist for tracking available vector slots.
///
/// Uses LIFO (stack) ordering - most recently freed slots are reused first.
/// This improves cache locality for recently touched data.
pub struct Freelist {
    /// In-memory cache of free slot indices.
    slots: Mutex<Vec<u64>>,
}

impl Freelist {
    /// Create a new empty freelist.
    pub fn new() -> Self {
        Self {
            slots: Mutex::new(Vec::new()),
        }
    }

    /// Create a freelist from a list of free slots.
    pub fn from_slots(slots: Vec<u64>) -> Self {
        Self {
            slots: Mutex::new(slots),
        }
    }

    /// Push a freed slot onto the freelist.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot index to add to the freelist
    pub fn push(&self, slot: u64) {
        let mut slots = self.slots.lock().unwrap();
        slots.push(slot);
    }

    /// Pop a slot from the freelist for reuse.
    ///
    /// # Returns
    ///
    /// The slot index if available, None if freelist is empty.
    pub fn pop(&self) -> Option<u64> {
        let mut slots = self.slots.lock().unwrap();
        slots.pop()
    }

    /// Check if a slot is in the freelist.
    ///
    /// Note: This is O(n) - use sparingly.
    pub fn contains(&self, slot: u64) -> bool {
        let slots = self.slots.lock().unwrap();
        slots.contains(&slot)
    }

    /// Get the number of free slots.
    pub fn len(&self) -> usize {
        let slots = self.slots.lock().unwrap();
        slots.len()
    }

    /// Check if the freelist is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all slots in the freelist (for persistence).
    pub fn get_all(&self) -> Vec<u64> {
        let slots = self.slots.lock().unwrap();
        slots.clone()
    }

    /// Clear the freelist.
    pub fn clear(&self) {
        let mut slots = self.slots.lock().unwrap();
        slots.clear();
    }

    /// Serialize the freelist for storage.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let slots = self.slots.lock().unwrap();
        bincode::serialize(&*slots).map_err(Into::into)
    }

    /// Deserialize a freelist from storage.
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Ok(Self::new());
        }
        let slots: Vec<u64> = bincode::deserialize(data)?;
        Ok(Self::from_slots(slots))
    }
}

impl Default for Freelist {
    fn default() -> Self {
        Self::new()
    }
}

/// Manages freelist persistence in redb.
pub struct FreelistManager {
    /// The in-memory freelist.
    freelist: Freelist,
}

impl FreelistManager {
    /// Table name for freelist storage in redb.
    pub const TABLE_NAME: &'static str = "freelist";

    /// Key for the serialized freelist data.
    pub const DATA_KEY: &'static str = "slots";

    /// Create a new freelist manager with empty freelist.
    pub fn new() -> Self {
        Self {
            freelist: Freelist::new(),
        }
    }

    /// Load freelist from redb transaction.
    pub fn load_from_redb(
        txn: &redb::ReadTransaction,
    ) -> Result<Self> {
        let table_def: redb::TableDefinition<&str, &[u8]> =
            redb::TableDefinition::new(Self::TABLE_NAME);

        let table = match txn.open_table(table_def) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => {
                return Ok(Self::new());
            }
            Err(e) => return Err(e.into()),
        };

        let freelist = match table.get(Self::DATA_KEY)? {
            Some(data) => Freelist::deserialize(data.value())?,
            None => Freelist::new(),
        };

        Ok(Self { freelist })
    }

    /// Save freelist to redb transaction.
    pub fn save_to_redb(
        &self,
        txn: &redb::WriteTransaction,
    ) -> Result<()> {
        let table_def: redb::TableDefinition<&str, &[u8]> =
            redb::TableDefinition::new(Self::TABLE_NAME);

        let mut table = txn.open_table(table_def)?;
        let data = self.freelist.serialize()?;
        table.insert(Self::DATA_KEY, data.as_slice())?;

        Ok(())
    }

    /// Get access to the freelist.
    pub fn freelist(&self) -> &Freelist {
        &self.freelist
    }

    /// Push a slot and save to redb.
    pub fn push_and_save(
        &self,
        slot: u64,
        txn: &redb::WriteTransaction,
    ) -> Result<()> {
        self.freelist.push(slot);
        self.save_to_redb(txn)
    }

    /// Pop a slot (doesn't save - caller should save after write succeeds).
    pub fn pop(&self) -> Option<u64> {
        self.freelist.pop()
    }

    /// Get the number of free slots.
    pub fn len(&self) -> usize {
        self.freelist.len()
    }

    /// Check if freelist is empty.
    pub fn is_empty(&self) -> bool {
        self.freelist.is_empty()
    }
}

impl Default for FreelistManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freelist_push_pop() {
        let fl = Freelist::new();

        assert!(fl.is_empty());
        assert_eq!(fl.pop(), None);

        fl.push(5);
        fl.push(10);
        fl.push(15);

        assert_eq!(fl.len(), 3);
        assert!(!fl.is_empty());

        // LIFO order
        assert_eq!(fl.pop(), Some(15));
        assert_eq!(fl.pop(), Some(10));
        assert_eq!(fl.pop(), Some(5));
        assert_eq!(fl.pop(), None);
    }

    #[test]
    fn test_freelist_contains() {
        let fl = Freelist::new();
        fl.push(42);
        fl.push(100);

        assert!(fl.contains(42));
        assert!(fl.contains(100));
        assert!(!fl.contains(0));
    }

    #[test]
    fn test_freelist_serialize() {
        let fl = Freelist::new();
        fl.push(1);
        fl.push(2);
        fl.push(3);

        let data = fl.serialize().unwrap();
        let fl2 = Freelist::deserialize(&data).unwrap();

        assert_eq!(fl2.len(), 3);
        assert_eq!(fl2.pop(), Some(3));
        assert_eq!(fl2.pop(), Some(2));
        assert_eq!(fl2.pop(), Some(1));
    }

    #[test]
    fn test_freelist_deserialize_empty() {
        let fl = Freelist::deserialize(&[]).unwrap();
        assert!(fl.is_empty());
    }

    #[test]
    fn test_freelist_clear() {
        let fl = Freelist::new();
        fl.push(1);
        fl.push(2);
        fl.clear();
        assert!(fl.is_empty());
    }

    #[test]
    fn test_freelist_get_all() {
        let fl = Freelist::new();
        fl.push(1);
        fl.push(2);
        fl.push(3);

        let all = fl.get_all();
        assert_eq!(all, vec![1, 2, 3]);
    }

    #[test]
    fn test_freelist_manager() {
        let mgr = FreelistManager::new();

        assert!(mgr.is_empty());
        assert_eq!(mgr.pop(), None);

        mgr.freelist().push(42);
        assert_eq!(mgr.len(), 1);
        assert_eq!(mgr.pop(), Some(42));
        assert!(mgr.is_empty());
    }
}
