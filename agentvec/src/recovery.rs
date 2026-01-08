//! Crash recovery protocol for AgentVec.
//!
//! AgentVec uses a write-ahead reservation protocol to ensure crash safety:
//!
//! 1. Reserve slot in freelist (in redb transaction)
//! 2. Write metadata with status = Pending
//! 3. Commit transaction (crash-safe checkpoint)
//! 4. Write vector to mmap'd slot
//! 5. Sync vector file
//! 6. Update status = Active (in redb transaction)
//!
//! On recovery, we scan for Pending records:
//! - If vector is valid (not NaN tombstone), promote to Active
//! - Otherwise, rollback: delete metadata, return slot to freelist

/// Statistics from crash recovery process.
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Number of pending records promoted to active.
    ///
    /// These were records where the vector was successfully written
    /// but the status update was interrupted.
    pub promoted: usize,

    /// Number of pending records rolled back.
    ///
    /// These were records where the vector write was never completed.
    pub rolled_back: usize,

    /// Number of tombstone records found.
    ///
    /// These are soft-deleted records awaiting compaction.
    pub tombstones: usize,

    /// Number of expired records found.
    ///
    /// These are records past their TTL, awaiting compaction.
    pub expired: usize,

    /// Total active records after recovery.
    pub active_records: usize,

    /// Recovery duration in milliseconds.
    pub duration_ms: u64,
}

impl RecoveryStats {
    /// Create new recovery stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if any recovery actions were needed.
    #[must_use]
    pub fn had_recovery(&self) -> bool {
        self.promoted > 0 || self.rolled_back > 0
    }
}

impl std::fmt::Display for RecoveryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.had_recovery() {
            write!(
                f,
                "Recovery: {} promoted, {} rolled back, {} tombstones, {} expired ({} ms)",
                self.promoted,
                self.rolled_back,
                self.tombstones,
                self.expired,
                self.duration_ms
            )
        } else {
            write!(f, "No recovery needed ({} active records)", self.active_records)
        }
    }
}

/// Status of a record in the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RecordStatus {
    /// Record is active and valid.
    Active,

    /// Record write is pending (vector may not be written yet).
    ///
    /// Recovery will check if vector exists and promote or rollback.
    Pending,

    /// Record is soft-deleted (tombstone).
    ///
    /// The slot can be reused after compaction.
    Tombstone,
}

impl Default for RecordStatus {
    fn default() -> Self {
        RecordStatus::Active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recovery_stats_default() {
        let stats = RecoveryStats::new();
        assert_eq!(stats.promoted, 0);
        assert_eq!(stats.rolled_back, 0);
        assert!(!stats.had_recovery());
    }

    #[test]
    fn test_had_recovery() {
        let mut stats = RecoveryStats::new();
        assert!(!stats.had_recovery());

        stats.promoted = 1;
        assert!(stats.had_recovery());

        stats.promoted = 0;
        stats.rolled_back = 1;
        assert!(stats.had_recovery());
    }

    #[test]
    fn test_display_no_recovery() {
        let mut stats = RecoveryStats::new();
        stats.active_records = 100;
        let s = stats.to_string();
        assert!(s.contains("No recovery needed"));
        assert!(s.contains("100"));
    }

    #[test]
    fn test_display_with_recovery() {
        let mut stats = RecoveryStats::new();
        stats.promoted = 5;
        stats.rolled_back = 2;
        stats.tombstones = 10;
        stats.duration_ms = 50;
        let s = stats.to_string();
        assert!(s.contains("5 promoted"));
        assert!(s.contains("2 rolled back"));
        assert!(s.contains("10 tombstones"));
    }

    #[test]
    fn test_record_status_default() {
        assert_eq!(RecordStatus::default(), RecordStatus::Active);
    }

    #[test]
    fn test_record_status_serialization() {
        let status = RecordStatus::Pending;
        let bytes = bincode::serialize(&status).unwrap();
        let decoded: RecordStatus = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded, RecordStatus::Pending);
    }
}
