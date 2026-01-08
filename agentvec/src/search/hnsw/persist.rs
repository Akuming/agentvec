//! HNSW graph persistence.
//!
//! Uses bincode for efficient serialization with CRC32 checksum
//! for integrity verification.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{AgentVecError, Result};

use super::config::HnswConfig;
use super::graph::{HnswGraph, HnswNode};

/// Magic number for HNSW index files.
const HNSW_MAGIC: u32 = 0x484E5357; // "HNSW" in ASCII

/// Current version of the HNSW file format.
const HNSW_VERSION: u16 = 1;

/// Serializable representation of HNSW graph.
#[derive(Serialize, Deserialize)]
struct HnswPersist {
    /// Magic number for file identification.
    magic: u32,

    /// File format version.
    version: u16,

    /// Configuration used to build the graph.
    config: HnswConfig,

    /// Number of nodes in the graph.
    node_count: u32,

    /// Maximum layer in the graph.
    max_layer: u8,

    /// Entry point node ID.
    entry_point: Option<u32>,

    /// Slot offsets for each node.
    node_slots: Vec<u64>,

    /// Record IDs for each node.
    node_ids: Vec<String>,

    /// Maximum layer for each node.
    node_max_layers: Vec<u8>,

    /// Number of layers.
    num_layers: usize,

    /// CSR format: offsets into edges array for each layer.
    /// layer_offsets[layer][node] = start index in layer_edges[layer]
    layer_offsets: Vec<Vec<u32>>,

    /// CSR format: flat array of neighbor node IDs per layer.
    layer_edges: Vec<Vec<u32>>,
}

impl HnswGraph {
    /// Save the graph to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the index file
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if saving fails.
    pub fn save(&self, path: &Path) -> Result<()> {
        // Convert connections to CSR format for efficient serialization
        let mut layer_offsets = Vec::new();
        let mut layer_edges = Vec::new();

        for layer_conns in self.connections() {
            let mut offsets = vec![0u32];
            let mut edges = Vec::new();

            for node_neighbors in layer_conns {
                edges.extend(node_neighbors.iter().copied());
                offsets.push(edges.len() as u32);
            }

            layer_offsets.push(offsets);
            layer_edges.push(edges);
        }

        let persist = HnswPersist {
            magic: HNSW_MAGIC,
            version: HNSW_VERSION,
            config: self.config().clone(),
            node_count: self.node_count() as u32,
            max_layer: self.max_layer(),
            entry_point: self.entry_point(),
            node_slots: self.nodes().iter().map(|n| n.slot).collect(),
            node_ids: self.nodes().iter().map(|n| n.id.clone()).collect(),
            node_max_layers: self.nodes().iter().map(|n| n.max_layer).collect(),
            num_layers: layer_offsets.len(),
            layer_offsets,
            layer_edges,
        };

        // Serialize the data
        let data = bincode::serialize(&persist)
            .map_err(|e| AgentVecError::Serialization(e.to_string()))?;

        // Compute checksum
        let checksum = crc32fast::hash(&data);

        // Write to file
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&data)?;
        writer.write_all(&checksum.to_le_bytes())?;
        writer.flush()?;

        Ok(())
    }

    /// Load a graph from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the index file
    ///
    /// # Returns
    ///
    /// The loaded graph, or an error if loading fails.
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read all data
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        // Need at least 4 bytes for checksum
        if data.len() < 4 {
            return Err(AgentVecError::Corruption(
                "HNSW index file too small".to_string(),
            ));
        }

        // Extract and verify checksum
        let checksum_bytes: [u8; 4] = data[data.len() - 4..].try_into().unwrap();
        let stored_checksum = u32::from_le_bytes(checksum_bytes);
        let data = &data[..data.len() - 4];

        let computed_checksum = crc32fast::hash(data);
        if stored_checksum != computed_checksum {
            return Err(AgentVecError::Corruption(format!(
                "HNSW index checksum mismatch: expected {:08x}, got {:08x}",
                stored_checksum, computed_checksum
            )));
        }

        // Deserialize
        let persist: HnswPersist = bincode::deserialize(data)
            .map_err(|e| AgentVecError::Serialization(e.to_string()))?;

        // Verify magic and version
        if persist.magic != HNSW_MAGIC {
            return Err(AgentVecError::Corruption(format!(
                "Invalid HNSW magic number: {:08x}",
                persist.magic
            )));
        }

        if persist.version != HNSW_VERSION {
            return Err(AgentVecError::Corruption(format!(
                "Unsupported HNSW version: {} (expected {})",
                persist.version, HNSW_VERSION
            )));
        }

        // Reconstruct nodes
        let mut nodes = Vec::with_capacity(persist.node_count as usize);
        let mut slot_to_node = HashMap::with_capacity(persist.node_count as usize);

        for i in 0..persist.node_count as usize {
            let node = HnswNode {
                slot: persist.node_slots[i],
                id: persist.node_ids[i].clone(),
                max_layer: persist.node_max_layers[i],
            };
            slot_to_node.insert(node.slot, i as u32);
            nodes.push(node);
        }

        // Reconstruct connections from CSR format
        let mut connections = Vec::with_capacity(persist.num_layers);

        for layer_idx in 0..persist.num_layers {
            let offsets = &persist.layer_offsets[layer_idx];
            let edges = &persist.layer_edges[layer_idx];

            let mut layer_conns = Vec::with_capacity(persist.node_count as usize);

            for node_id in 0..persist.node_count as usize {
                let start = offsets[node_id] as usize;
                let end = offsets[node_id + 1] as usize;
                let neighbors: Vec<u32> = edges[start..end].to_vec();
                layer_conns.push(neighbors);
            }

            connections.push(layer_conns);
        }

        Ok(HnswGraph::restore(
            nodes,
            slot_to_node,
            connections,
            persist.entry_point,
            persist.max_layer,
            persist.config,
        ))
    }

    /// Check if an HNSW index file exists at the given path.
    pub fn exists(path: &Path) -> bool {
        path.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::VectorStorage;
    use tempfile::TempDir;

    fn create_test_graph() -> (TempDir, VectorStorage, HnswGraph) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");
        let mut storage = VectorStorage::create(&path, 4).unwrap();

        // Create some vectors
        for i in 0..5 {
            let slot = storage.allocate_slot().unwrap();
            let vec = vec![i as f32; 4];
            storage.write_slot(slot, &vec).unwrap();
        }

        let config = HnswConfig::with_m(4);
        let mut graph = HnswGraph::new(config);

        // Insert nodes
        for i in 0..5 {
            graph.insert(i as u64, format!("node{}", i), &storage, crate::config::Metric::Cosine);
        }

        (dir, storage, graph)
    }

    #[test]
    fn test_save_and_load() {
        let (dir, _storage, graph) = create_test_graph();
        let index_path = dir.path().join("hnsw.index");

        // Save
        graph.save(&index_path).unwrap();
        assert!(index_path.exists());

        // Load
        let loaded = HnswGraph::load(&index_path).unwrap();

        // Verify
        assert_eq!(loaded.node_count(), graph.node_count());
        assert_eq!(loaded.entry_point(), graph.entry_point());
        assert_eq!(loaded.max_layer(), graph.max_layer());

        // Verify nodes
        for i in 0..graph.node_count() {
            let orig = graph.get_node(i as u32).unwrap();
            let load = loaded.get_node(i as u32).unwrap();
            assert_eq!(orig.slot, load.slot);
            assert_eq!(orig.id, load.id);
            assert_eq!(orig.max_layer, load.max_layer);
        }

        // Verify connections at layer 0
        for i in 0..graph.node_count() {
            let orig_neighbors = graph.get_neighbors(0, i as u32);
            let load_neighbors = loaded.get_neighbors(0, i as u32);
            assert_eq!(orig_neighbors, load_neighbors);
        }
    }

    #[test]
    fn test_load_corrupted_checksum() {
        let (dir, _storage, graph) = create_test_graph();
        let index_path = dir.path().join("hnsw.index");

        // Save
        graph.save(&index_path).unwrap();

        // Corrupt the file
        let mut data = std::fs::read(&index_path).unwrap();
        if data.len() > 10 {
            data[10] ^= 0xFF; // Flip some bits
        }
        std::fs::write(&index_path, &data).unwrap();

        // Load should fail
        let result = HnswGraph::load(&index_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("checksum"));
    }

    #[test]
    fn test_empty_graph_persistence() {
        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("hnsw.index");

        let config = HnswConfig::default();
        let graph = HnswGraph::new(config);

        // Save empty graph
        graph.save(&index_path).unwrap();

        // Load
        let loaded = HnswGraph::load(&index_path).unwrap();

        assert!(loaded.is_empty());
        assert!(loaded.entry_point().is_none());
    }

    #[test]
    fn test_exists() {
        let dir = TempDir::new().unwrap();
        let index_path = dir.path().join("hnsw.index");

        assert!(!HnswGraph::exists(&index_path));

        let config = HnswConfig::default();
        let graph = HnswGraph::new(config);
        graph.save(&index_path).unwrap();

        assert!(HnswGraph::exists(&index_path));
    }
}
