//! Memory-mapped vector storage with fixed-slot allocation.
//!
//! # File Format
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Header (64 bytes):                                          │
//! │   [magic: u32]           # 0x41564543 ("AVEC")              │
//! │   [version: u16]         # Format version (currently 1)     │
//! │   [dimensions: u32]      # Vector dimensions                │
//! │   [slot_count: u64]      # Total allocated slots            │
//! │   [checksum: u32]        # CRC32 of header fields           │
//! │   [reserved: 42 bytes]   # Future use                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Slot 0: [f32 × dim]                                         │
//! │ Slot 1: [f32 × dim]                                         │
//! │ Slot 2: [TOMBSTONE]      # First f32 = NaN sentinel         │
//! │ ...                                                         │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use std::fs::{File, OpenOptions};
use std::path::Path;

use memmap2::MmapMut;

use crate::error::{AgentVecError, Result};

/// Magic number identifying AgentVec vector files: "AVEC" in ASCII.
const MAGIC: u32 = 0x4156_4543;

/// Current file format version.
const VERSION: u16 = 1;

/// Header size in bytes (fixed at 64 for alignment).
const HEADER_SIZE: usize = 64;

/// Initial file size when creating new vector storage.
/// We start with space for 1024 vectors to reduce early reallocations.
const INITIAL_CAPACITY: u64 = 1024;

/// Growth factor when expanding storage.
const GROWTH_FACTOR: f64 = 1.5;

/// NaN value used as tombstone marker.
/// Using a specific quiet NaN pattern for consistency.
const TOMBSTONE_MARKER: u32 = 0x7FC0_0000;

/// Memory-mapped vector storage.
///
/// Provides fixed-slot allocation for vectors with O(1) read/write.
/// Vectors are stored contiguously for cache-friendly sequential scans.
pub struct VectorStorage {
    /// The underlying file.
    file: File,

    /// Memory-mapped view of the file.
    mmap: MmapMut,

    /// Vector dimensions (fixed at creation).
    dimensions: usize,

    /// Number of allocated slots.
    slot_count: u64,

    /// Byte size of each slot.
    slot_size: usize,
}

/// File header structure.
///
/// Note: We don't use `#[repr(packed)]` because we manually serialize/deserialize
/// the header bytes anyway. This avoids alignment issues with packed struct references.
#[derive(Debug, Clone, Copy)]
struct Header {
    magic: u32,
    version: u16,
    dimensions: u32,
    slot_count: u64,
    #[allow(dead_code)] // Reserved for future integrity checks
    checksum: u32,
    // 42 bytes reserved for future use (in serialized format)
}

impl VectorStorage {
    /// Create a new vector storage file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the vectors.bin file
    /// * `dimensions` - Vector dimensions (fixed for lifetime of file)
    ///
    /// # Errors
    ///
    /// Returns error if file creation fails or dimensions are invalid.
    pub fn create(path: impl AsRef<Path>, dimensions: usize) -> Result<Self> {
        Self::validate_dimensions(dimensions)?;

        let path = path.as_ref();
        let slot_size = dimensions * std::mem::size_of::<f32>();

        // Create and size the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let initial_size = HEADER_SIZE as u64 + INITIAL_CAPACITY * slot_size as u64;
        file.set_len(initial_size)?;

        // Create mmap
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Write header
        let header = Header {
            magic: MAGIC,
            version: VERSION,
            dimensions: dimensions as u32,
            slot_count: 0,
            checksum: 0, // Will be computed
        };

        Self::write_header(&mut mmap, &header)?;

        Ok(Self {
            file,
            mmap,
            dimensions,
            slot_count: 0,
            slot_size,
        })
    }

    /// Open an existing vector storage file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the vectors.bin file
    ///
    /// # Errors
    ///
    /// Returns error if file doesn't exist, is corrupted, or has invalid format.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(AgentVecError::Corruption(
                "Vector file too small for header".into(),
            ));
        }

        // Read and validate header
        let header = Self::read_header(&mmap)?;

        if header.magic != MAGIC {
            return Err(AgentVecError::Corruption(format!(
                "Invalid magic number: expected 0x{:08X}, got 0x{:08X}",
                MAGIC, header.magic
            )));
        }

        if header.version != VERSION {
            return Err(AgentVecError::Corruption(format!(
                "Unsupported version: expected {}, got {}",
                VERSION, header.version
            )));
        }

        let dimensions = header.dimensions as usize;
        Self::validate_dimensions(dimensions)?;

        let slot_size = dimensions * std::mem::size_of::<f32>();

        Ok(Self {
            file,
            mmap,
            dimensions,
            slot_count: header.slot_count,
            slot_size,
        })
    }

    /// Validate dimensions are within acceptable range.
    fn validate_dimensions(dimensions: usize) -> Result<()> {
        if dimensions == 0 {
            return Err(AgentVecError::InvalidInput(
                "Dimensions must be greater than 0".into(),
            ));
        }

        if dimensions > crate::config::MAX_DIMENSIONS {
            return Err(AgentVecError::DimensionsTooLarge {
                max: crate::config::MAX_DIMENSIONS,
                got: dimensions,
            });
        }

        Ok(())
    }

    /// Get the vector dimensions.
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get the number of allocated slots.
    #[inline]
    pub fn slot_count(&self) -> u64 {
        self.slot_count
    }

    /// Get the byte offset for a slot.
    #[inline]
    fn slot_offset(&self, slot: u64) -> usize {
        HEADER_SIZE + (slot as usize * self.slot_size)
    }

    /// Allocate a new slot, growing the file if necessary.
    ///
    /// # Returns
    ///
    /// The slot index for the new vector.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be grown.
    pub fn allocate_slot(&mut self) -> Result<u64> {
        let slot = self.slot_count;
        let required_size = self.slot_offset(slot + 1);

        // Grow file if necessary
        if required_size > self.mmap.len() {
            self.grow(required_size)?;
        }

        self.slot_count += 1;
        self.update_slot_count()?;

        Ok(slot)
    }

    /// Grow the storage to accommodate at least `min_size` bytes.
    fn grow(&mut self, min_size: usize) -> Result<()> {
        let current_size = self.mmap.len();
        let new_size = std::cmp::max(
            min_size,
            (current_size as f64 * GROWTH_FACTOR) as usize,
        );

        // Flush and unmap before resize
        self.mmap.flush()?;

        // Resize file
        self.file.set_len(new_size as u64)?;

        // Remap
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };

        Ok(())
    }

    /// Write a vector to a slot.
    ///
    /// # Arguments
    ///
    /// * `slot` - Slot index
    /// * `vector` - Vector data (must match dimensions)
    ///
    /// # Errors
    ///
    /// Returns error if slot is out of bounds or vector dimensions don't match.
    pub fn write_slot(&mut self, slot: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(AgentVecError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        if slot >= self.slot_count {
            return Err(AgentVecError::NotFound(format!("Slot {} not allocated", slot)));
        }

        let offset = self.slot_offset(slot);
        let bytes = unsafe {
            std::slice::from_raw_parts(
                vector.as_ptr() as *const u8,
                vector.len() * std::mem::size_of::<f32>(),
            )
        };

        self.mmap[offset..offset + bytes.len()].copy_from_slice(bytes);

        Ok(())
    }

    /// Read a vector from a slot.
    ///
    /// # Arguments
    ///
    /// * `slot` - Slot index
    ///
    /// # Returns
    ///
    /// The vector data.
    ///
    /// # Errors
    ///
    /// Returns error if slot is out of bounds.
    pub fn read_slot(&self, slot: u64) -> Result<Vec<f32>> {
        if slot >= self.slot_count {
            return Err(AgentVecError::NotFound(format!("Slot {} not allocated", slot)));
        }

        let offset = self.slot_offset(slot);
        let bytes = &self.mmap[offset..offset + self.slot_size];

        let vector = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                self.dimensions,
            )
        };

        Ok(vector.to_vec())
    }

    /// Get a reference to a vector in a slot (zero-copy).
    ///
    /// # Arguments
    ///
    /// * `slot` - Slot index
    ///
    /// # Returns
    ///
    /// A slice reference to the vector data.
    ///
    /// # Errors
    ///
    /// Returns error if slot is out of bounds.
    pub fn read_slot_ref(&self, slot: u64) -> Result<&[f32]> {
        if slot >= self.slot_count {
            return Err(AgentVecError::NotFound(format!("Slot {} not allocated", slot)));
        }

        let offset = self.slot_offset(slot);
        let bytes = &self.mmap[offset..offset + self.slot_size];

        let vector = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                self.dimensions,
            )
        };

        Ok(vector)
    }

    /// Mark a slot as a tombstone (deleted).
    ///
    /// # Arguments
    ///
    /// * `slot` - Slot index
    ///
    /// # Errors
    ///
    /// Returns error if slot is out of bounds.
    pub fn mark_tombstone(&mut self, slot: u64) -> Result<()> {
        if slot >= self.slot_count {
            return Err(AgentVecError::NotFound(format!("Slot {} not allocated", slot)));
        }

        let offset = self.slot_offset(slot);

        // Write the NaN tombstone marker as the first f32
        let marker_bytes = TOMBSTONE_MARKER.to_le_bytes();
        self.mmap[offset..offset + 4].copy_from_slice(&marker_bytes);

        Ok(())
    }

    /// Check if a slot is marked as a tombstone.
    ///
    /// # Arguments
    ///
    /// * `slot` - Slot index
    ///
    /// # Returns
    ///
    /// `true` if the slot contains a tombstone marker.
    ///
    /// # Errors
    ///
    /// Returns error if slot is out of bounds.
    pub fn is_tombstone(&self, slot: u64) -> Result<bool> {
        if slot >= self.slot_count {
            return Err(AgentVecError::NotFound(format!("Slot {} not allocated", slot)));
        }

        let offset = self.slot_offset(slot);
        let bytes: [u8; 4] = self.mmap[offset..offset + 4].try_into().unwrap();
        let value = u32::from_le_bytes(bytes);

        // Check if it's our specific tombstone marker, or any NaN
        Ok(value == TOMBSTONE_MARKER || f32::from_bits(value).is_nan())
    }

    /// Check if a slot contains a valid (non-tombstone) vector.
    ///
    /// # Arguments
    ///
    /// * `slot` - Slot index
    ///
    /// # Returns
    ///
    /// `true` if the slot contains a valid vector.
    pub fn is_valid(&self, slot: u64) -> Result<bool> {
        self.is_tombstone(slot).map(|is_tomb| !is_tomb)
    }

    /// Flush all pending writes to disk.
    pub fn sync(&self) -> Result<()> {
        self.mmap.flush()?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Get the total size of the vector file in bytes.
    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }

    /// Get the size used by vectors (excluding reserved space).
    pub fn used_size(&self) -> u64 {
        HEADER_SIZE as u64 + self.slot_count * self.slot_size as u64
    }

    /// Write the header to the mmap.
    fn write_header(mmap: &mut MmapMut, header: &Header) -> Result<()> {
        let mut buf = [0u8; HEADER_SIZE];

        // Write fields
        buf[0..4].copy_from_slice(&header.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&header.version.to_le_bytes());
        buf[6..10].copy_from_slice(&header.dimensions.to_le_bytes());
        buf[10..18].copy_from_slice(&header.slot_count.to_le_bytes());

        // Compute checksum over the data fields (not including checksum itself)
        let checksum = Self::compute_checksum(&buf[0..18]);
        buf[18..22].copy_from_slice(&checksum.to_le_bytes());

        // Rest is reserved (zeros)
        mmap[0..HEADER_SIZE].copy_from_slice(&buf);

        Ok(())
    }

    /// Read the header from the mmap.
    fn read_header(mmap: &MmapMut) -> Result<Header> {
        if mmap.len() < HEADER_SIZE {
            return Err(AgentVecError::Corruption("File too small".into()));
        }

        let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        let version = u16::from_le_bytes(mmap[4..6].try_into().unwrap());
        let dimensions = u32::from_le_bytes(mmap[6..10].try_into().unwrap());
        let slot_count = u64::from_le_bytes(mmap[10..18].try_into().unwrap());
        let checksum = u32::from_le_bytes(mmap[18..22].try_into().unwrap());

        // Verify checksum
        let computed = Self::compute_checksum(&mmap[0..18]);
        if computed != checksum {
            return Err(AgentVecError::Corruption(format!(
                "Header checksum mismatch: expected 0x{:08X}, got 0x{:08X}",
                checksum, computed
            )));
        }

        Ok(Header {
            magic,
            version,
            dimensions,
            slot_count,
            checksum,
        })
    }

    /// Update the slot count in the header.
    fn update_slot_count(&mut self) -> Result<()> {
        // Write slot count
        self.mmap[10..18].copy_from_slice(&self.slot_count.to_le_bytes());

        // Recompute and write checksum
        let checksum = Self::compute_checksum(&self.mmap[0..18]);
        self.mmap[18..22].copy_from_slice(&checksum.to_le_bytes());

        Ok(())
    }

    /// Simple checksum function (CRC32 would be better, but this is simpler).
    fn compute_checksum(data: &[u8]) -> u32 {
        // Simple FNV-1a hash as checksum
        let mut hash: u32 = 0x811c_9dc5;
        for byte in data {
            hash ^= *byte as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
        hash
    }

    /// Advise the OS to preload vectors into memory.
    #[cfg(unix)]
    pub fn preload(&self) -> Result<()> {
        unsafe {
            libc::madvise(
                self.mmap.as_ptr() as *mut libc::c_void,
                self.mmap.len(),
                libc::MADV_WILLNEED,
            );
        }
        Ok(())
    }

    /// Advise the OS to preload vectors into memory (Windows no-op for now).
    #[cfg(windows)]
    pub fn preload(&self) -> Result<()> {
        // Windows doesn't have madvise, but we could use PrefetchVirtualMemory
        // For now, just touch the pages by reading through them
        // This is a no-op placeholder
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        // Create
        {
            let storage = VectorStorage::create(&path, 384).unwrap();
            assert_eq!(storage.dimensions(), 384);
            assert_eq!(storage.slot_count(), 0);
        }

        // Open
        {
            let storage = VectorStorage::open(&path).unwrap();
            assert_eq!(storage.dimensions(), 384);
            assert_eq!(storage.slot_count(), 0);
        }
    }

    #[test]
    fn test_allocate_and_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut storage = VectorStorage::create(&path, 3).unwrap();

        let slot = storage.allocate_slot().unwrap();
        assert_eq!(slot, 0);

        storage.write_slot(slot, &[1.0, 2.0, 3.0]).unwrap();
        let vec = storage.read_slot(slot).unwrap();
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_multiple_slots() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut storage = VectorStorage::create(&path, 3).unwrap();

        for i in 0..100 {
            let slot = storage.allocate_slot().unwrap();
            assert_eq!(slot, i as u64);
            storage.write_slot(slot, &[i as f32, 0.0, 0.0]).unwrap();
        }

        for i in 0..100 {
            let vec = storage.read_slot(i as u64).unwrap();
            assert_eq!(vec[0], i as f32);
        }
    }

    #[test]
    fn test_grow_storage() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut storage = VectorStorage::create(&path, 384).unwrap();
        let initial_size = storage.file_size();

        // Allocate more than initial capacity
        for _ in 0..2000 {
            storage.allocate_slot().unwrap();
        }

        assert!(storage.file_size() > initial_size);
        assert_eq!(storage.slot_count(), 2000);
    }

    #[test]
    fn test_tombstone() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut storage = VectorStorage::create(&path, 3).unwrap();

        let slot = storage.allocate_slot().unwrap();
        storage.write_slot(slot, &[1.0, 2.0, 3.0]).unwrap();

        assert!(!storage.is_tombstone(slot).unwrap());
        assert!(storage.is_valid(slot).unwrap());

        storage.mark_tombstone(slot).unwrap();

        assert!(storage.is_tombstone(slot).unwrap());
        assert!(!storage.is_valid(slot).unwrap());
    }

    #[test]
    fn test_dimension_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut storage = VectorStorage::create(&path, 3).unwrap();
        let slot = storage.allocate_slot().unwrap();

        let result = storage.write_slot(slot, &[1.0, 2.0]); // Wrong dimensions
        assert!(matches!(result, Err(AgentVecError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_invalid_dimensions() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let result = VectorStorage::create(&path, 0);
        assert!(matches!(result, Err(AgentVecError::InvalidInput(_))));

        let result = VectorStorage::create(&path, 100_000);
        assert!(matches!(result, Err(AgentVecError::DimensionsTooLarge { .. })));
    }

    #[test]
    fn test_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        // Write
        {
            let mut storage = VectorStorage::create(&path, 3).unwrap();
            let slot = storage.allocate_slot().unwrap();
            storage.write_slot(slot, &[1.0, 2.0, 3.0]).unwrap();
            storage.sync().unwrap();
        }

        // Read
        {
            let storage = VectorStorage::open(&path).unwrap();
            assert_eq!(storage.slot_count(), 1);
            let vec = storage.read_slot(0).unwrap();
            assert_eq!(vec, vec![1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_read_slot_ref() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let mut storage = VectorStorage::create(&path, 3).unwrap();
        let slot = storage.allocate_slot().unwrap();
        storage.write_slot(slot, &[1.0, 2.0, 3.0]).unwrap();

        let vec_ref = storage.read_slot_ref(slot).unwrap();
        assert_eq!(vec_ref, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_out_of_bounds() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vectors.bin");

        let storage = VectorStorage::create(&path, 3).unwrap();

        let result = storage.read_slot(0);
        assert!(matches!(result, Err(AgentVecError::NotFound(_))));
    }
}
