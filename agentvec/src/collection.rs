//! Collection implementation.
//!
//! A collection is a namespace within an AgentVec database that holds vectors
//! of a fixed dimension with a specific distance metric.
//!
//! ## Write Coalescing
//!
//! To achieve fast single-record inserts, the collection uses write coalescing:
//! - Writes are buffered in memory and the vector is written to mmap immediately
//! - The buffer is flushed to redb when it reaches a threshold or on explicit sync
//! - Search includes both committed and pending records
//! - On crash, pending (unflushed) records are lost but the database remains consistent

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};
use serde_json::Value as JsonValue;

use crate::config::{CollectionConfig, Metric};
use crate::error::{AgentVecError, Result};
use crate::filter::Filter;
use crate::recovery::RecordStatus;
use crate::search::distance::{normalize_l2, DistanceMetric};
use crate::search::hnsw::HnswIndex;
use crate::search::{ExactSearch, SearchResult};
use crate::storage::{FreelistManager, MetadataStorage, Record, VectorStorage};

/// Default maximum pending writes before auto-flush.
const DEFAULT_MAX_PENDING: usize = 100;

/// Default maximum time (ms) before auto-flush.
const DEFAULT_MAX_PENDING_MS: u64 = 100;

/// A pending write waiting to be flushed to redb.
#[derive(Debug, Clone)]
struct PendingWrite {
    /// The record (with Active status, ready to commit).
    record: Record,
    /// The normalized vector data.
    vector: Vec<f32>,
}

/// Configuration for write coalescing behavior.
#[derive(Debug, Clone)]
pub struct WriteConfig {
    /// Maximum pending writes before auto-flush (0 = flush immediately).
    pub max_pending: usize,
    /// Maximum milliseconds before auto-flush (0 = no time-based flush).
    pub max_pending_ms: u64,
}

impl Default for WriteConfig {
    fn default() -> Self {
        Self {
            max_pending: DEFAULT_MAX_PENDING,
            max_pending_ms: DEFAULT_MAX_PENDING_MS,
        }
    }
}

impl WriteConfig {
    /// Create a config that flushes immediately (ACID per write, slower).
    pub fn immediate() -> Self {
        Self {
            max_pending: 0,
            max_pending_ms: 0,
        }
    }

    /// Create a config optimized for throughput (batch writes, faster).
    pub fn throughput() -> Self {
        Self {
            max_pending: 1000,
            max_pending_ms: 200,
        }
    }
}

/// A collection of vectors with associated metadata.
///
/// Collections are thread-safe and support concurrent reads.
/// Writes are serialized via internal locking.
///
/// ## Write Coalescing
///
/// By default, writes are buffered for better throughput. Call `sync()` to
/// ensure all writes are durable, or use `WriteConfig::immediate()` for
/// per-write durability.
pub struct Collection {
    /// Collection configuration.
    config: CollectionConfig,

    /// Write coalescing configuration.
    write_config: WriteConfig,

    /// Path to the collection directory.
    path: PathBuf,

    /// Vector storage (mmap'd file).
    vectors: RwLock<VectorStorage>,

    /// Metadata storage (redb).
    metadata: MetadataStorage,

    /// Freelist for slot recycling.
    freelist: RwLock<FreelistManager>,

    /// Pending writes buffer (not yet committed to redb).
    pending: Mutex<Vec<PendingWrite>>,

    /// Timestamp of first pending write (for time-based flush).
    pending_since: AtomicU64,

    /// Optional HNSW index for approximate nearest neighbor search.
    /// Wrapped in RwLock for lazy building.
    hnsw_index: RwLock<Option<HnswIndex>>,

    /// Flag indicating HNSW index needs to be (re)built.
    /// Set to true when vectors are added, false after build.
    hnsw_dirty: AtomicBool,
}

impl Collection {
    /// Open or create a collection with default write configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the collection directory
    /// * `config` - Collection configuration
    ///
    /// # Returns
    ///
    /// The collection handle.
    pub fn open(path: impl AsRef<Path>, config: CollectionConfig) -> Result<Self> {
        Self::open_with_write_config(path, config, WriteConfig::default())
    }

    /// Open or create a collection with custom write configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the collection directory
    /// * `config` - Collection configuration
    /// * `write_config` - Write coalescing configuration
    ///
    /// # Returns
    ///
    /// The collection handle.
    pub fn open_with_write_config(
        path: impl AsRef<Path>,
        config: CollectionConfig,
        write_config: WriteConfig,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Create directory if needed
        std::fs::create_dir_all(&path)?;

        let vectors_path = path.join("vectors.bin");
        let metadata_path = path.join("meta.redb");

        // Open or create vector storage
        let vectors = if vectors_path.exists() {
            VectorStorage::open(&vectors_path)?
        } else {
            VectorStorage::create(&vectors_path, config.dimensions)?
        };

        // Verify dimensions match
        if vectors.dimensions() != config.dimensions {
            return Err(AgentVecError::DimensionMismatch {
                expected: config.dimensions,
                got: vectors.dimensions(),
            });
        }

        // Open metadata storage
        let metadata = MetadataStorage::open(&metadata_path)?;

        // Load freelist
        let freelist = {
            let txn = metadata.begin_read()?;
            FreelistManager::load_from_redb(&txn)?
        };

        // Load existing HNSW index if available (don't build yet - deferred)
        let (hnsw_index, hnsw_dirty) = if let Some(hnsw_config) = &config.hnsw {
            let hnsw_path = path.join("hnsw.index");
            if hnsw_path.exists() {
                // Try to load existing index
                match HnswIndex::load(&hnsw_path, hnsw_config.clone(), config.metric) {
                    Ok(index) => {
                        // Check if index is up to date with records
                        let txn = metadata.begin_read()?;
                        let record_count = metadata.count_active(&txn)?;
                        let index_count = index.len();
                        if index_count == record_count {
                            (Some(index), false) // Index is current
                        } else {
                            // Index is stale, needs rebuild
                            (None, true)
                        }
                    }
                    Err(_) => (None, true), // Failed to load, will rebuild
                }
            } else {
                // No index file, needs build if we have records
                let txn = metadata.begin_read()?;
                let record_count = metadata.count_active(&txn)?;
                (None, record_count > 0)
            }
        } else {
            (None, false) // HNSW not configured
        };

        Ok(Self {
            config,
            write_config,
            path,
            vectors: RwLock::new(vectors),
            metadata,
            freelist: RwLock::new(freelist),
            pending: Mutex::new(Vec::new()),
            pending_since: AtomicU64::new(0),
            hnsw_index: RwLock::new(hnsw_index),
            hnsw_dirty: AtomicBool::new(hnsw_dirty),
        })
    }

    /// Get the collection configuration.
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    /// Get the collection name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get the vector dimensions.
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    /// Get the distance metric.
    pub fn metric(&self) -> Metric {
        self.config.metric
    }

    /// Get the path to the collection directory.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Add a new vector to the collection.
    ///
    /// By default, writes are buffered for better performance. Call `sync()`
    /// to ensure durability, or use `add_sync()` for immediate durability.
    ///
    /// # Arguments
    ///
    /// * `vector` - The vector to add
    /// * `metadata` - Associated metadata
    /// * `id` - Optional custom ID (auto-generated if None)
    /// * `ttl` - Optional time-to-live in seconds
    ///
    /// # Returns
    ///
    /// The record ID.
    pub fn add(
        &self,
        vector: &[f32],
        metadata: JsonValue,
        id: Option<&str>,
        ttl: Option<u64>,
    ) -> Result<String> {
        self.validate_vector(vector)?;

        let id = id
            .map(String::from)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // Normalize vector for cosine metric
        let normalized = self.maybe_normalize(vector);

        // Allocate slot (reuse from freelist or allocate new)
        let slot = self.allocate_slot()?;

        // Write vector to mmap immediately (no sync yet)
        {
            let mut vectors = self.vectors.write();
            vectors.write_slot(slot, &normalized)?;
            // Don't sync here - we'll sync when flushing
        }

        // Create record with Active status (will be committed on flush)
        let record = Record::new(&id, slot, metadata, ttl, RecordStatus::Active);

        // Add to pending buffer
        {
            let mut pending = self.pending.lock();

            // Set pending_since if this is the first pending write
            if pending.is_empty() {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                self.pending_since.store(now, Ordering::SeqCst);
            }

            pending.push(PendingWrite {
                record,
                vector: normalized,
            });
        }

        // Check if we should auto-flush
        self.maybe_flush()?;

        Ok(id)
    }

    /// Add a new vector with immediate durability (slower).
    ///
    /// This bypasses the write buffer and commits immediately.
    /// Use this when you need guaranteed durability per write.
    pub fn add_sync(
        &self,
        vector: &[f32],
        metadata: JsonValue,
        id: Option<&str>,
        ttl: Option<u64>,
    ) -> Result<String> {
        let id = self.add(vector, metadata, id, ttl)?;
        self.flush_pending()?;
        Ok(id)
    }

    /// Check if pending buffer should be flushed and do so if needed.
    fn maybe_flush(&self) -> Result<()> {
        let should_flush = {
            let pending = self.pending.lock();
            if pending.is_empty() {
                return Ok(());
            }

            // Check size threshold
            if self.write_config.max_pending > 0 && pending.len() >= self.write_config.max_pending {
                true
            }
            // Check time threshold
            else if self.write_config.max_pending_ms > 0 {
                let pending_since = self.pending_since.load(Ordering::SeqCst);
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                now.saturating_sub(pending_since) >= self.write_config.max_pending_ms
            } else {
                // If max_pending is 0, flush immediately
                self.write_config.max_pending == 0
            }
        };

        if should_flush {
            self.flush_pending()?;
        }

        Ok(())
    }

    /// Flush all pending writes to durable storage.
    ///
    /// This commits all buffered writes to redb in a single transaction.
    pub fn flush_pending(&self) -> Result<()> {
        let pending_writes = {
            let mut pending = self.pending.lock();
            if pending.is_empty() {
                return Ok(());
            }
            std::mem::take(&mut *pending)
        };

        // Sync vector file first
        {
            let vectors = self.vectors.read();
            vectors.sync()?;
        }

        // Mark HNSW as dirty (will be rebuilt on next search or sync)
        if self.config.hnsw.is_some() {
            self.hnsw_dirty.store(true, Ordering::SeqCst);
        }

        // Commit all records in a single transaction
        let txn = self.metadata.begin_write()?;
        for pw in &pending_writes {
            self.metadata.insert_record(&txn, &pw.record)?;
        }

        // Save freelist state
        self.freelist.read().save_to_redb(&txn)?;

        txn.commit()?;

        // Reset pending_since
        self.pending_since.store(0, Ordering::SeqCst);

        Ok(())
    }

    /// Get the number of pending (unflushed) writes.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().len()
    }

    /// Insert or update a vector by ID.
    ///
    /// This is idempotent - calling with the same ID multiple times
    /// will replace the previous vector and metadata.
    ///
    /// Handles both pending (unflushed) and committed records.
    ///
    /// # Arguments
    ///
    /// * `id` - The record ID
    /// * `vector` - The vector
    /// * `metadata` - Associated metadata
    /// * `ttl` - Optional time-to-live in seconds
    pub fn upsert(
        &self,
        id: &str,
        vector: &[f32],
        metadata: JsonValue,
        ttl: Option<u64>,
    ) -> Result<()> {
        self.validate_vector(vector)?;

        // Check pending records first
        {
            let mut pending = self.pending.lock();
            if let Some(pos) = pending.iter().position(|pw| pw.record.id == id) {
                let pw = pending.remove(pos);
                // Reclaim the slot
                self.freelist.write().freelist().push(pw.record.slot_offset);
            }
        }

        // Check if record exists in committed storage
        let existing = {
            let txn = self.metadata.begin_read()?;
            self.metadata.get_record(&txn, id)?
        };

        if let Some(old_record) = existing {
            // Delete old record first (reclaim slot)
            self.delete_internal(id, old_record.slot_offset)?;
        }

        // Add new record
        self.add(vector, metadata, Some(id), ttl)?;

        Ok(())
    }

    /// Add multiple vectors in a batch.
    ///
    /// This is significantly faster than calling add() in a loop.
    /// All records are committed atomically.
    ///
    /// # Arguments
    ///
    /// * `vectors` - The vectors to add
    /// * `metadatas` - Associated metadata for each vector
    /// * `ids` - Optional custom IDs (auto-generated if None)
    /// * `ttls` - Optional TTLs for each vector
    ///
    /// # Returns
    ///
    /// The record IDs.
    pub fn add_batch(
        &self,
        vectors: &[impl AsRef<[f32]>],
        metadatas: &[JsonValue],
        ids: Option<&[&str]>,
        ttls: Option<&[Option<u64>]>,
    ) -> Result<Vec<String>> {
        if vectors.len() != metadatas.len() {
            return Err(AgentVecError::InvalidInput(
                "vectors and metadatas must have same length".into(),
            ));
        }

        if let Some(ids) = ids {
            if ids.len() != vectors.len() {
                return Err(AgentVecError::InvalidInput(
                    "ids must have same length as vectors".into(),
                ));
            }
        }

        if let Some(ttls) = ttls {
            if ttls.len() != vectors.len() {
                return Err(AgentVecError::InvalidInput(
                    "ttls must have same length as vectors".into(),
                ));
            }
        }

        // Validate all vectors first
        for vector in vectors {
            self.validate_vector(vector.as_ref())?;
        }

        // Prepare normalized vectors
        let normalized_vectors: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| self.maybe_normalize(v.as_ref()))
            .collect();

        // Allocate slots and prepare records
        let mut result_ids = Vec::with_capacity(vectors.len());
        let mut records = Vec::with_capacity(vectors.len());
        let mut slots = Vec::with_capacity(vectors.len());

        for i in 0..vectors.len() {
            let id = ids
                .and_then(|ids| ids.get(i).map(|s| s.to_string()))
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

            let ttl = ttls.and_then(|t| t.get(i).copied()).flatten();
            let slot = self.allocate_slot()?;

            let record = Record::new(&id, slot, metadatas[i].clone(), ttl, RecordStatus::Pending);

            result_ids.push(id);
            records.push(record);
            slots.push(slot);
        }

        // Begin transaction for all metadata
        let txn = self.metadata.begin_write()?;

        for record in &records {
            self.metadata.insert_record(&txn, record)?;
        }

        // Save freelist state
        self.freelist.read().save_to_redb(&txn)?;

        // Commit reservation
        txn.commit()?;

        // Write all vectors
        {
            let mut vectors_storage = self.vectors.write();
            for (i, slot) in slots.iter().enumerate() {
                vectors_storage.write_slot(*slot, &normalized_vectors[i])?;
            }
            vectors_storage.sync()?;
        }

        // Mark HNSW as dirty (will be rebuilt on next search or sync)
        if self.config.hnsw.is_some() {
            self.hnsw_dirty.store(true, Ordering::SeqCst);
        }

        // Update all to Active
        let txn = self.metadata.begin_write()?;
        for id in &result_ids {
            self.metadata.update_record_status(&txn, id, RecordStatus::Active)?;
        }
        txn.commit()?;

        Ok(result_ids)
    }

    /// Ensure the HNSW index is built if configured and dirty.
    ///
    /// This is called automatically before search. Uses incremental insertion
    /// when an index already exists, which is much faster than full rebuild.
    fn ensure_hnsw_built(&self) -> Result<()> {
        // Quick check without locking
        if self.config.hnsw.is_none() || !self.hnsw_dirty.load(Ordering::SeqCst) {
            return Ok(());
        }

        // Double-check with write lock
        let mut hnsw_guard = self.hnsw_index.write();
        if !self.hnsw_dirty.load(Ordering::SeqCst) {
            return Ok(()); // Another thread built it
        }

        let hnsw_config = self.config.hnsw.as_ref().unwrap();

        // Collect all active records
        let mut all_records = Vec::new();
        let txn = self.metadata.begin_read()?;
        self.metadata.iter_records(&txn, |record| {
            if record.is_active() {
                all_records.push((record.slot_offset, record.id.clone()));
            }
            Ok(true)
        })?;
        drop(txn);

        // Check if we have an existing index to update incrementally
        if let Some(ref existing_index) = *hnsw_guard {
            let existing_count = existing_index.len();
            let total_count = all_records.len();

            if existing_count > 0 && total_count > existing_count {
                // Incremental update: only insert new records
                let new_records: Vec<_> = all_records
                    .into_iter()
                    .filter(|(slot, _)| !existing_index.contains_slot(*slot))
                    .collect();

                if !new_records.is_empty() {
                    let vectors = self.vectors.read();
                    let inserted = existing_index.insert_batch(&new_records, &vectors);
                    if inserted > 0 {
                        // Index was updated
                    }
                }

                self.hnsw_dirty.store(false, Ordering::SeqCst);
                return Ok(());
            }
        }

        // Full rebuild: no existing index, or index is empty, or needs complete rebuild
        let new_index = HnswIndex::build_from_records(
            hnsw_config.clone(),
            self.config.metric,
            &self.vectors.read(),
            &self.metadata,
        )?;

        *hnsw_guard = Some(new_index);
        self.hnsw_dirty.store(false, Ordering::SeqCst);

        Ok(())
    }

    /// Search for the k nearest neighbors.
    ///
    /// This includes both committed and pending (unflushed) records.
    /// Uses HNSW index for approximate search if configured, otherwise exact search.
    ///
    /// # Arguments
    ///
    /// * `vector` - The query vector
    /// * `k` - Number of results to return
    /// * `filter` - Optional metadata filter
    ///
    /// # Returns
    ///
    /// The search results, sorted by score.
    pub fn search(
        &self,
        vector: &[f32],
        k: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>> {
        self.validate_vector(vector)?;

        // Ensure HNSW index is built if configured (deferred building)
        self.ensure_hnsw_built()?;

        // Normalize query for cosine metric
        let query = self.maybe_normalize(vector);

        // Collect pending records for search
        let pending_data: Vec<(Record, Vec<f32>)> = {
            let pending = self.pending.lock();
            pending
                .iter()
                .map(|pw| (pw.record.clone(), pw.vector.clone()))
                .collect()
        };

        // Search committed records
        let hnsw_guard = self.hnsw_index.read();
        let mut results = if let Some(ref hnsw) = *hnsw_guard {
            // Use HNSW for approximate search (if no filter, or filter post-search)
            if filter.is_none() {
                // Fast path: no filter, use HNSW directly
                hnsw.search_with_metadata(
                    &query,
                    k,
                    None,
                    &self.vectors.read(),
                    &self.metadata,
                )?
            } else {
                // With filter: over-fetch from HNSW, then filter
                // Fetch more candidates to account for filtering
                let over_fetch = (k * 4).min(hnsw.len());
                let candidates = hnsw.search_with_metadata(
                    &query,
                    over_fetch,
                    None,
                    &self.vectors.read(),
                    &self.metadata,
                )?;

                // Apply filter
                let filter_ref = filter.as_ref().unwrap();
                candidates
                    .into_iter()
                    .filter(|r| filter_ref.matches(&r.metadata))
                    .take(k)
                    .collect()
            }
        } else {
            // Fall back to exact search
            ExactSearch::search(
                &query,
                k,
                filter.as_ref(),
                self.config.metric,
                &self.vectors.read(),
                &self.metadata,
            )?
        };

        // If there are pending records, include them in search
        if !pending_data.is_empty() {
            let pending_refs: Vec<(Record, &[f32])> = pending_data
                .iter()
                .map(|(r, v)| (r.clone(), v.as_slice()))
                .collect();

            let pending_results = ExactSearch::search_with_data(
                &query,
                k,
                filter.as_ref(),
                self.config.metric,
                &pending_refs,
            );

            // Merge results: combine and re-sort
            results.extend(pending_results);
            let higher_is_better = self.config.metric.higher_is_better();
            if higher_is_better {
                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
            }
            results.truncate(k);
        }

        Ok(results)
    }

    /// Get a record by ID.
    ///
    /// This checks both pending (unflushed) and committed records.
    ///
    /// # Arguments
    ///
    /// * `id` - The record ID
    ///
    /// # Returns
    ///
    /// The record if found and not expired, None otherwise.
    pub fn get(&self, id: &str) -> Result<Option<SearchResult>> {
        // Check pending records first (unflushed writes)
        {
            let pending = self.pending.lock();
            for pw in pending.iter() {
                if pw.record.id == id && pw.record.is_active() {
                    return Ok(Some(SearchResult::new(
                        &pw.record.id,
                        0.0,
                        pw.record.metadata(),
                    )));
                }
            }
        }

        // Check committed records
        let txn = self.metadata.begin_read()?;
        let record = match self.metadata.get_record(&txn, id)? {
            Some(r) => r,
            None => return Ok(None),
        };

        // Check if expired or not active
        if !record.is_active() {
            return Ok(None);
        }

        // Get vector for score (in case caller wants it)
        let vectors = self.vectors.read();
        if vectors.is_tombstone(record.slot_offset)? {
            return Ok(None);
        }

        Ok(Some(SearchResult::new(
            &record.id,
            0.0, // Score not applicable for get()
            record.metadata(),
        )))
    }

    /// Delete a record by ID.
    ///
    /// This handles both pending (unflushed) and committed records.
    ///
    /// # Arguments
    ///
    /// * `id` - The record ID
    ///
    /// # Returns
    ///
    /// True if the record existed and was deleted.
    pub fn delete(&self, id: &str) -> Result<bool> {
        // Check pending records first (remove from buffer, reclaim slot)
        {
            let mut pending = self.pending.lock();
            if let Some(pos) = pending.iter().position(|pw| pw.record.id == id) {
                let pw = pending.remove(pos);
                // Reclaim the slot (vector was written but not committed)
                self.freelist.write().freelist().push(pw.record.slot_offset);
                return Ok(true);
            }
        }

        // Check committed records
        let txn = self.metadata.begin_read()?;
        let record = match self.metadata.get_record(&txn, id)? {
            Some(r) => r,
            None => return Ok(false),
        };
        drop(txn);

        self.delete_internal(id, record.slot_offset)?;
        Ok(true)
    }

    /// Internal delete implementation.
    fn delete_internal(&self, id: &str, slot: u64) -> Result<()> {
        // Mark vector as tombstone
        {
            let mut vectors = self.vectors.write();
            vectors.mark_tombstone(slot)?;
        }

        // Mark deleted in HNSW index if enabled, and set dirty flag
        if self.config.hnsw.is_some() {
            if let Some(ref hnsw) = *self.hnsw_index.read() {
                hnsw.mark_deleted(slot);
            }
            self.hnsw_dirty.store(true, Ordering::SeqCst);
        }

        // Add slot to freelist
        self.freelist.write().freelist().push(slot);

        // Begin transaction
        let txn = self.metadata.begin_write()?;

        // Update record status to Tombstone
        self.metadata.update_record_status(&txn, id, RecordStatus::Tombstone)?;

        // Save freelist
        self.freelist.read().save_to_redb(&txn)?;

        txn.commit()?;

        // Actually delete the record
        let txn = self.metadata.begin_write()?;
        self.metadata.delete_record(&txn, id)?;
        txn.commit()?;

        Ok(())
    }

    /// Get the number of active (non-expired) records.
    ///
    /// This includes both pending (unflushed) and committed records.
    pub fn len(&self) -> Result<usize> {
        let committed = {
            let txn = self.metadata.begin_read()?;
            self.metadata.count_active(&txn)?
        };
        let pending = self.pending.lock().len();
        Ok(committed + pending)
    }

    /// Check if the collection is empty.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Compact the collection.
    ///
    /// Removes expired records and defragments vector storage.
    ///
    /// # Returns
    ///
    /// Statistics about the compaction.
    pub fn compact(&self) -> Result<CompactStats> {
        let start = std::time::Instant::now();
        let mut stats = CompactStats::default();

        // Find expired and tombstone records
        let txn = self.metadata.begin_read()?;
        let mut to_remove = Vec::new();

        self.metadata.iter_records(&txn, |record| {
            if record.is_expired() {
                to_remove.push((record.id.clone(), record.slot_offset, true));
                stats.expired_removed += 1;
            } else if record.status == RecordStatus::Tombstone {
                to_remove.push((record.id.clone(), record.slot_offset, false));
                stats.tombstones_removed += 1;
            }
            Ok(true)
        })?;
        drop(txn);

        // Remove each record
        for (id, slot, _is_expired) in to_remove {
            // Mark tombstone if not already
            {
                let mut vectors = self.vectors.write();
                if vectors.is_valid(slot).unwrap_or(false) {
                    vectors.mark_tombstone(slot)?;
                    stats.bytes_freed += (self.config.dimensions * 4) as u64;
                }
            }

            // Add to freelist
            if !self.freelist.read().freelist().contains(slot) {
                self.freelist.write().freelist().push(slot);
            }

            // Delete metadata
            let txn = self.metadata.begin_write()?;
            self.metadata.delete_record(&txn, &id)?;
            self.freelist.read().save_to_redb(&txn)?;
            txn.commit()?;
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;

        Ok(stats)
    }

    /// Hint to OS to preload vectors into memory.
    pub fn preload(&self) -> Result<()> {
        self.vectors.read().preload()
    }

    /// Get the size of vector storage in bytes.
    pub fn vectors_size_bytes(&self) -> u64 {
        self.vectors.read().file_size()
    }

    /// Validate that a vector has the correct dimensions.
    fn validate_vector(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimensions {
            return Err(AgentVecError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        Ok(())
    }

    /// Normalize vector if using cosine metric.
    fn maybe_normalize(&self, vector: &[f32]) -> Vec<f32> {
        match self.config.metric {
            Metric::Cosine => normalize_l2(vector),
            Metric::Dot | Metric::L2 => vector.to_vec(),
        }
    }

    /// Allocate a slot (from freelist or new).
    fn allocate_slot(&self) -> Result<u64> {
        // Try freelist first
        if let Some(slot) = self.freelist.write().pop() {
            return Ok(slot);
        }

        // Allocate new slot
        let mut vectors = self.vectors.write();
        vectors.allocate_slot()
    }

    /// Sync all data to disk.
    ///
    /// This flushes any pending writes and ensures all data is durable.
    /// Also persists the HNSW index if enabled.
    pub fn sync(&self) -> Result<()> {
        self.flush_pending()?;
        self.vectors.read().sync()?;

        // Build HNSW index if dirty (deferred building)
        self.ensure_hnsw_built()?;

        // Persist HNSW index if enabled
        if let Some(ref hnsw) = *self.hnsw_index.read() {
            let hnsw_path = self.path.join("hnsw.index");
            hnsw.save(&hnsw_path)?;
        }

        Ok(())
    }

    /// Check if HNSW indexing is enabled for this collection.
    pub fn has_hnsw_index(&self) -> bool {
        self.config.hnsw.is_some()
    }

    /// Get the number of nodes in the HNSW index (if enabled and built).
    pub fn hnsw_node_count(&self) -> Option<usize> {
        self.hnsw_index.read().as_ref().map(|idx| idx.len())
    }

    /// Check if the HNSW index needs to be rebuilt.
    pub fn hnsw_is_dirty(&self) -> bool {
        self.hnsw_dirty.load(Ordering::SeqCst)
    }

    // ========== Export/Import ==========

    /// Export the collection to a file.
    ///
    /// Creates an NDJSON file with all records including vectors and metadata.
    /// The export includes both committed and pending (unflushed) records.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    ///
    /// # Returns
    ///
    /// The number of records exported.
    ///
    /// # Example
    ///
    /// ```ignore
    /// collection.export_to_file("backup.ndjson")?;
    /// ```
    pub fn export_to_file<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        use crate::export::{ExportRecord, ExportWriter};
        use std::fs::File;

        // Flush pending writes first to ensure consistent export
        self.flush_pending()?;

        // Get total record count
        let record_count = self.len()?;

        let file = File::create(path.as_ref())
            .map_err(|e| AgentVecError::Io(e))?;

        let metric_str = match self.config.metric {
            Metric::Cosine => "cosine",
            Metric::Dot => "dot",
            Metric::L2 => "l2",
        };

        let mut writer = ExportWriter::new(
            file,
            &self.config.name,
            self.config.dimensions,
            metric_str,
            record_count,
        )?;

        let vectors = self.vectors.read();
        let txn = self.metadata.begin_read()?;

        self.metadata.iter_records(&txn, |record| {
            // Skip non-active records
            if !record.is_active() {
                return Ok(true);
            }

            // Read the vector
            let vector_data = vectors.read_slot_ref(record.slot_offset)?;
            let vector: Vec<f32> = vector_data.to_vec();

            // Calculate TTL from expiry time
            let ttl = record.expires_at.map(|exp| {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                if exp > now { exp - now } else { 0 }
            });

            let export_record = ExportRecord {
                id: record.id.clone(),
                vector,
                metadata: record.metadata(),
                created_at: Some(record.created_at),
                ttl,
            };

            writer.write_record(&export_record)?;
            Ok(true)
        })?;

        writer.finish()
    }

    /// Import records from a file.
    ///
    /// Reads an NDJSON file and imports all records. Validates that the file
    /// dimensions match the collection dimensions.
    ///
    /// # Arguments
    ///
    /// * `path` - Input file path
    ///
    /// # Returns
    ///
    /// Import statistics.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let stats = collection.import_from_file("backup.ndjson")?;
    /// println!("Imported {} records", stats.imported);
    /// ```
    pub fn import_from_file<P: AsRef<Path>>(&self, path: P) -> Result<crate::export::ImportStats> {
        use crate::export::{ExportReader, ImportStats};
        use std::fs::File;
        use std::io::BufReader;
        use std::time::Instant;

        let start = Instant::now();

        let file = File::open(path.as_ref())
            .map_err(|e| AgentVecError::Io(e))?;

        let reader = BufReader::new(file);
        let mut export_reader = ExportReader::new(reader)?;

        let header = export_reader.header();

        // Validate dimensions
        if header.dimensions != self.config.dimensions {
            return Err(AgentVecError::InvalidDimensions {
                expected: self.config.dimensions,
                got: header.dimensions,
            });
        }

        let mut stats = ImportStats::default();

        while let Some(record) = export_reader.read_record()? {
            // Validate vector dimensions
            if record.vector.len() != self.config.dimensions {
                stats.failed += 1;
                continue;
            }

            // Import using upsert (handles duplicates gracefully)
            match self.upsert(&record.id, &record.vector, record.metadata, record.ttl) {
                Ok(_) => stats.imported += 1,
                Err(_) => stats.failed += 1,
            }
        }

        // Flush pending writes
        self.flush_pending()?;

        stats.duration_ms = start.elapsed().as_millis() as u64;

        Ok(stats)
    }

    /// Export collection records as an iterator.
    ///
    /// This is useful for streaming exports or custom export formats.
    /// Yields `ExportRecord` for each active record.
    pub fn export_records(&self) -> Result<Vec<crate::export::ExportRecord>> {
        use crate::export::ExportRecord;

        // Flush pending writes first
        self.flush_pending()?;

        let mut records = Vec::new();
        let vectors = self.vectors.read();
        let txn = self.metadata.begin_read()?;

        self.metadata.iter_records(&txn, |record| {
            if !record.is_active() {
                return Ok(true);
            }

            let vector_data = vectors.read_slot_ref(record.slot_offset)?;
            let vector: Vec<f32> = vector_data.to_vec();

            let ttl = record.expires_at.map(|exp| {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                if exp > now { exp - now } else { 0 }
            });

            records.push(ExportRecord {
                id: record.id.clone(),
                vector,
                metadata: record.metadata(),
                created_at: Some(record.created_at),
                ttl,
            });

            Ok(true)
        })?;

        Ok(records)
    }
}

/// Statistics from a compact operation.
#[derive(Debug, Clone, Default)]
pub struct CompactStats {
    /// Number of expired records removed.
    pub expired_removed: usize,

    /// Number of tombstone records removed.
    pub tombstones_removed: usize,

    /// Bytes freed (approximate).
    pub bytes_freed: u64,

    /// Duration in milliseconds.
    pub duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn create_test_collection(dim: usize) -> (tempfile::TempDir, Collection) {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("test", dim, Metric::Cosine);
        let col = Collection::open(dir.path().join("col"), config).unwrap();
        (dir, col)
    }

    #[test]
    fn test_add_and_get() {
        let (_dir, col) = create_test_collection(3);

        let id = col
            .add(&[1.0, 2.0, 3.0], json!({"key": "value"}), None, None)
            .unwrap();

        let result = col.get(&id).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata["key"], "value");
    }

    #[test]
    fn test_add_with_custom_id() {
        let (_dir, col) = create_test_collection(3);

        let id = col
            .add(&[1.0, 2.0, 3.0], json!({}), Some("my_id"), None)
            .unwrap();

        assert_eq!(id, "my_id");
        assert!(col.get("my_id").unwrap().is_some());
    }

    #[test]
    fn test_delete() {
        let (_dir, col) = create_test_collection(3);

        let id = col.add(&[1.0, 2.0, 3.0], json!({}), None, None).unwrap();
        assert!(col.get(&id).unwrap().is_some());

        let deleted = col.delete(&id).unwrap();
        assert!(deleted);

        assert!(col.get(&id).unwrap().is_none());
    }

    #[test]
    fn test_upsert() {
        let (_dir, col) = create_test_collection(3);

        // First upsert
        col.upsert("id1", &[1.0, 0.0, 0.0], json!({"v": 1}), None)
            .unwrap();
        assert_eq!(col.len().unwrap(), 1);

        // Second upsert (update)
        col.upsert("id1", &[0.0, 1.0, 0.0], json!({"v": 2}), None)
            .unwrap();
        assert_eq!(col.len().unwrap(), 1);

        let result = col.get("id1").unwrap().unwrap();
        assert_eq!(result.metadata["v"], 2);
    }

    #[test]
    fn test_search() {
        let (_dir, col) = create_test_collection(3);

        col.add(&[1.0, 0.0, 0.0], json!({"name": "x"}), Some("x"), None)
            .unwrap();
        col.add(&[0.0, 1.0, 0.0], json!({"name": "y"}), Some("y"), None)
            .unwrap();
        col.add(&[0.0, 0.0, 1.0], json!({"name": "z"}), Some("z"), None)
            .unwrap();

        let results = col.search(&[1.0, 0.0, 0.0], 2, None).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "x"); // Most similar
    }

    #[test]
    fn test_search_with_filter() {
        let (_dir, col) = create_test_collection(3);

        col.add(&[1.0, 0.0, 0.0], json!({"type": "a"}), Some("1"), None)
            .unwrap();
        col.add(&[0.9, 0.1, 0.0], json!({"type": "b"}), Some("2"), None)
            .unwrap();
        col.add(&[0.8, 0.2, 0.0], json!({"type": "a"}), Some("3"), None)
            .unwrap();

        let filter = Filter::new().eq("type", "a");
        let results = col.search(&[1.0, 0.0, 0.0], 10, Some(filter)).unwrap();

        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.metadata["type"], "a");
        }
    }

    #[test]
    fn test_add_batch() {
        let (_dir, col) = create_test_collection(3);

        let vectors = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let metadatas = vec![json!({"i": 0}), json!({"i": 1}), json!({"i": 2})];

        let ids = col.add_batch(&vectors, &metadatas, None, None).unwrap();

        assert_eq!(ids.len(), 3);
        assert_eq!(col.len().unwrap(), 3);
    }

    #[test]
    fn test_ttl_expiry() {
        let (_dir, col) = create_test_collection(3);

        col.add(&[1.0, 0.0, 0.0], json!({}), Some("exp"), Some(0))
            .unwrap();

        // Should be expired immediately (TTL=0 means expires now)
        std::thread::sleep(std::time::Duration::from_millis(10));

        // get() should return None for expired
        assert!(col.get("exp").unwrap().is_none());

        // Search should exclude expired
        let results = col.search(&[1.0, 0.0, 0.0], 10, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_dimension_validation() {
        let (_dir, col) = create_test_collection(3);

        let result = col.add(&[1.0, 2.0], json!({}), None, None); // Wrong dim
        assert!(matches!(result, Err(AgentVecError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let col_path = dir.path().join("col");

        // Create and add
        {
            let config = CollectionConfig::new("test", 3, Metric::Cosine);
            let col = Collection::open(&col_path, config).unwrap();
            col.add(&[1.0, 2.0, 3.0], json!({"key": "value"}), Some("id1"), None)
                .unwrap();
            col.sync().unwrap();
        }

        // Reopen and verify
        {
            let config = CollectionConfig::new("test", 3, Metric::Cosine);
            let col = Collection::open(&col_path, config).unwrap();
            let result = col.get("id1").unwrap();
            assert!(result.is_some());
            assert_eq!(result.unwrap().metadata["key"], "value");
        }
    }

    #[test]
    fn test_compact() {
        let (_dir, col) = create_test_collection(3);

        // Add some records
        col.add(&[1.0, 0.0, 0.0], json!({}), Some("keep"), None)
            .unwrap();
        col.add(&[0.0, 1.0, 0.0], json!({}), Some("delete"), None)
            .unwrap();
        col.add(&[0.0, 0.0, 1.0], json!({}), Some("expire"), Some(0))
            .unwrap();

        // Flush pending writes before delete/compact
        col.sync().unwrap();

        // Delete one
        col.delete("delete").unwrap();

        // Wait for expiry
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Compact
        let stats = col.compact().unwrap();

        assert!(stats.expired_removed >= 1 || stats.tombstones_removed >= 1);
        assert_eq!(col.len().unwrap(), 1);
    }

    #[test]
    fn test_cosine_normalization() {
        let (_dir, col) = create_test_collection(3);

        // Add vectors that would have different magnitudes
        col.add(&[3.0, 0.0, 0.0], json!({}), Some("a"), None)
            .unwrap();
        col.add(&[1.0, 0.0, 0.0], json!({}), Some("b"), None)
            .unwrap();

        // Search should treat them as equal direction
        let results = col.search(&[1.0, 0.0, 0.0], 2, None).unwrap();

        // Both should have score close to 1.0 (same direction after normalization)
        assert!((results[0].score - 1.0).abs() < 0.01);
        assert!((results[1].score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_len() {
        let (_dir, col) = create_test_collection(3);

        assert_eq!(col.len().unwrap(), 0);
        assert!(col.is_empty().unwrap());

        col.add(&[1.0, 0.0, 0.0], json!({}), None, None).unwrap();
        assert_eq!(col.len().unwrap(), 1);
        assert!(!col.is_empty().unwrap());
    }

    #[test]
    fn test_pending_count() {
        let (_dir, col) = create_test_collection(3);

        assert_eq!(col.pending_count(), 0);

        col.add(&[1.0, 0.0, 0.0], json!({}), None, None).unwrap();
        assert_eq!(col.pending_count(), 1);

        col.add(&[0.0, 1.0, 0.0], json!({}), None, None).unwrap();
        assert_eq!(col.pending_count(), 2);

        col.flush_pending().unwrap();
        assert_eq!(col.pending_count(), 0);
    }

    #[test]
    fn test_pending_visible_in_search() {
        let (_dir, col) = create_test_collection(3);

        // Add without flushing
        col.add(&[1.0, 0.0, 0.0], json!({"name": "test"}), Some("id1"), None)
            .unwrap();

        // Should be visible immediately
        let results = col.search(&[1.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "id1");
        assert_eq!(results[0].metadata["name"], "test");
    }

    #[test]
    fn test_pending_visible_in_get() {
        let (_dir, col) = create_test_collection(3);

        // Add without flushing
        col.add(&[1.0, 0.0, 0.0], json!({"name": "test"}), Some("id1"), None)
            .unwrap();

        // Should be visible immediately
        let result = col.get("id1").unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata["name"], "test");
    }

    #[test]
    fn test_pending_visible_in_len() {
        let (_dir, col) = create_test_collection(3);

        assert_eq!(col.len().unwrap(), 0);

        col.add(&[1.0, 0.0, 0.0], json!({}), None, None).unwrap();
        assert_eq!(col.len().unwrap(), 1);

        col.add(&[0.0, 1.0, 0.0], json!({}), None, None).unwrap();
        assert_eq!(col.len().unwrap(), 2);
    }

    #[test]
    fn test_delete_pending() {
        let (_dir, col) = create_test_collection(3);

        col.add(&[1.0, 0.0, 0.0], json!({}), Some("id1"), None)
            .unwrap();
        assert_eq!(col.pending_count(), 1);
        assert_eq!(col.len().unwrap(), 1);

        // Delete from pending buffer
        let deleted = col.delete("id1").unwrap();
        assert!(deleted);
        assert_eq!(col.pending_count(), 0);
        assert_eq!(col.len().unwrap(), 0);
    }

    #[test]
    fn test_upsert_pending() {
        let (_dir, col) = create_test_collection(3);

        col.add(&[1.0, 0.0, 0.0], json!({"v": 1}), Some("id1"), None)
            .unwrap();
        assert_eq!(col.pending_count(), 1);

        // Upsert replaces the pending record
        col.upsert("id1", &[0.0, 1.0, 0.0], json!({"v": 2}), None)
            .unwrap();
        assert_eq!(col.pending_count(), 1); // Still 1 (replaced)

        let result = col.get("id1").unwrap().unwrap();
        assert_eq!(result.metadata["v"], 2);
    }

    #[test]
    fn test_write_config_immediate() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("test", 3, Metric::Cosine);
        let col = Collection::open_with_write_config(
            dir.path().join("col"),
            config,
            WriteConfig::immediate(),
        )
        .unwrap();

        // With immediate mode, writes should flush immediately
        col.add(&[1.0, 0.0, 0.0], json!({}), Some("id1"), None)
            .unwrap();

        // Pending count should be 0 after immediate flush
        assert_eq!(col.pending_count(), 0);
    }

    #[test]
    fn test_collection_with_hnsw() {
        use crate::search::HnswConfig;

        let dir = tempdir().unwrap();
        let hnsw_config = HnswConfig::with_m(4);
        let config = CollectionConfig::with_hnsw("test", 8, Metric::Cosine, hnsw_config);
        let col = Collection::open(dir.path().join("col"), config).unwrap();

        assert!(col.has_hnsw_index());
        // With deferred building, index isn't built until first search or sync
        assert!(col.hnsw_node_count().is_none());

        // Add vectors
        for i in 0..20 {
            let mut v = vec![0.0f32; 8];
            v[i % 8] = 1.0;
            col.add(&v, json!({"i": i}), Some(&format!("v{}", i)), None)
                .unwrap();
        }

        // Flush and build HNSW index
        col.sync().unwrap();

        assert_eq!(col.hnsw_node_count(), Some(20));

        // Search should work
        let query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = col.search(&query, 5, None).unwrap();

        assert!(!results.is_empty());
        // First result should be v0 or v8 or v16 (all have [1,0,0,0,0,0,0,0])
        let first_id = &results[0].id;
        assert!(first_id == "v0" || first_id == "v8" || first_id == "v16");
    }

    #[test]
    fn test_hnsw_persistence() {
        use crate::search::HnswConfig;

        let dir = tempdir().unwrap();
        let col_path = dir.path().join("col");

        // Create collection with HNSW
        {
            let hnsw_config = HnswConfig::with_m(4);
            let config = CollectionConfig::with_hnsw("test", 4, Metric::Cosine, hnsw_config);
            let col = Collection::open(&col_path, config).unwrap();

            for i in 0..10 {
                let mut v = vec![0.0f32; 4];
                v[i % 4] = 1.0;
                col.add(&v, json!({"i": i}), Some(&format!("v{}", i)), None)
                    .unwrap();
            }

            col.sync().unwrap();
            assert_eq!(col.hnsw_node_count(), Some(10));
        }

        // Reopen and verify HNSW was loaded
        {
            let hnsw_config = HnswConfig::with_m(4);
            let config = CollectionConfig::with_hnsw("test", 4, Metric::Cosine, hnsw_config);
            let col = Collection::open(&col_path, config).unwrap();

            assert!(col.has_hnsw_index());
            assert_eq!(col.hnsw_node_count(), Some(10));

            // Search should work
            let query = [1.0, 0.0, 0.0, 0.0];
            let results = col.search(&query, 5, None).unwrap();
            assert!(!results.is_empty());
        }
    }

    #[test]
    fn test_hnsw_delete() {
        use crate::search::HnswConfig;

        let dir = tempdir().unwrap();
        let hnsw_config = HnswConfig::with_m(4);
        let config = CollectionConfig::with_hnsw("test", 4, Metric::Cosine, hnsw_config);
        let col = Collection::open(dir.path().join("col"), config).unwrap();

        // Add and sync
        col.add(&[1.0, 0.0, 0.0, 0.0], json!({}), Some("v0"), None)
            .unwrap();
        col.add(&[0.0, 1.0, 0.0, 0.0], json!({}), Some("v1"), None)
            .unwrap();
        col.sync().unwrap();

        // Delete v0
        col.delete("v0").unwrap();

        // Search should not return deleted
        let query = [1.0, 0.0, 0.0, 0.0];
        let results = col.search(&query, 10, None).unwrap();

        for r in &results {
            assert_ne!(r.id, "v0", "Deleted record should not appear in search");
        }
    }
}
