//! HNSW graph data structures.

use std::collections::HashMap;

use rand::Rng;
use serde::{Deserialize, Serialize};

use super::config::HnswConfig;

/// A node in the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswNode {
    /// Slot offset in VectorStorage.
    pub slot: u64,

    /// Record ID for metadata lookups.
    pub id: String,

    /// Maximum layer this node exists in (0 = base layer only).
    pub max_layer: u8,
}

/// HNSW graph structure.
///
/// The graph consists of multiple layers, where layer 0 is the densest
/// and higher layers are sparser. Each node has connections to its
/// nearest neighbors at each layer it exists in.
#[derive(Debug)]
pub struct HnswGraph {
    /// All nodes indexed by internal node ID.
    pub(crate) nodes: Vec<HnswNode>,

    /// Mapping from slot offset to node ID for O(1) lookup.
    pub(crate) slot_to_node: HashMap<u64, u32>,

    /// Connections per layer: connections[layer][node_id] = neighbor node IDs.
    /// Layer 0 has M_max0 connections, other layers have M_max connections.
    pub(crate) connections: Vec<Vec<Vec<u32>>>,

    /// Entry point for search (highest layer node).
    pub(crate) entry_point: Option<u32>,

    /// Maximum layer currently in the graph.
    pub(crate) max_layer: u8,

    /// Configuration parameters.
    pub(crate) config: HnswConfig,
}

impl HnswGraph {
    /// Create a new empty HNSW graph.
    pub fn new(config: HnswConfig) -> Self {
        Self {
            nodes: Vec::new(),
            slot_to_node: HashMap::new(),
            connections: Vec::new(),
            entry_point: None,
            max_layer: 0,
            config,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the entry point node ID.
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Get the maximum layer in the graph.
    pub fn max_layer(&self) -> u8 {
        self.max_layer
    }

    /// Get a node by its internal ID.
    pub fn get_node(&self, node_id: u32) -> Option<&HnswNode> {
        self.nodes.get(node_id as usize)
    }

    /// Get a node ID by its slot offset.
    pub fn get_node_id_by_slot(&self, slot: u64) -> Option<u32> {
        self.slot_to_node.get(&slot).copied()
    }

    /// Get the neighbors of a node at a specific layer.
    pub fn get_neighbors(&self, layer: usize, node_id: u32) -> &[u32] {
        self.connections
            .get(layer)
            .and_then(|layer_conns| layer_conns.get(node_id as usize))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get mutable neighbors of a node at a specific layer.
    pub(crate) fn get_neighbors_mut(&mut self, layer: usize, node_id: u32) -> Option<&mut Vec<u32>> {
        self.connections
            .get_mut(layer)
            .and_then(|layer_conns| layer_conns.get_mut(node_id as usize))
    }

    /// Generate a random layer for a new node using exponential distribution.
    pub fn random_layer<R: Rng>(&self, rng: &mut R) -> u8 {
        let uniform: f64 = rng.gen();
        let layer = (-uniform.ln() * self.config.ml).floor() as u8;
        // Cap at a reasonable maximum to prevent memory issues
        layer.min(32)
    }

    /// Add a node to the graph (internal use - connections added separately).
    pub(crate) fn add_node(&mut self, slot: u64, id: String, max_layer: u8) -> u32 {
        let node_id = self.nodes.len() as u32;

        self.nodes.push(HnswNode {
            slot,
            id,
            max_layer,
        });
        self.slot_to_node.insert(slot, node_id);

        // Ensure connection storage exists for all layers up to max_layer
        while self.connections.len() <= max_layer as usize {
            self.connections.push(Vec::new());
        }

        // Ensure each layer has space for this node
        for layer in &mut self.connections {
            while layer.len() <= node_id as usize {
                layer.push(Vec::new());
            }
        }

        // Update entry point and max layer if this is the first node or has higher layer
        if self.entry_point.is_none() || max_layer > self.max_layer {
            self.entry_point = Some(node_id);
            self.max_layer = max_layer;
        }

        node_id
    }

    /// Set the neighbors of a node at a specific layer.
    pub(crate) fn set_neighbors(&mut self, layer: usize, node_id: u32, neighbors: Vec<u32>) {
        if let Some(layer_conns) = self.connections.get_mut(layer) {
            if let Some(node_neighbors) = layer_conns.get_mut(node_id as usize) {
                *node_neighbors = neighbors;
            }
        }
    }

    /// Add a neighbor connection (bidirectional).
    pub(crate) fn add_connection(&mut self, layer: usize, from: u32, to: u32) {
        if let Some(neighbors) = self.get_neighbors_mut(layer, from) {
            if !neighbors.contains(&to) {
                neighbors.push(to);
            }
        }
    }

    /// Remove a node from the graph by marking its slot as deleted.
    /// Returns true if the node existed.
    ///
    /// Note: This does not remove the node from the graph structure,
    /// it only removes the slot mapping. Deleted nodes are filtered
    /// during search.
    pub fn mark_deleted(&mut self, slot: u64) -> bool {
        self.slot_to_node.remove(&slot).is_some()
    }

    /// Check if a slot is still active (not deleted).
    pub fn is_slot_active(&self, slot: u64) -> bool {
        self.slot_to_node.contains_key(&slot)
    }

    /// Get the maximum number of connections for a layer.
    pub fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m_max0
        } else {
            self.config.m_max
        }
    }

    /// Get all nodes (for iteration/persistence).
    pub fn nodes(&self) -> &[HnswNode] {
        &self.nodes
    }

    /// Get all connections (for persistence).
    pub fn connections(&self) -> &[Vec<Vec<u32>>] {
        &self.connections
    }

    /// Restore graph state from loaded data (used by persistence).
    pub(crate) fn restore(
        nodes: Vec<HnswNode>,
        slot_to_node: HashMap<u64, u32>,
        connections: Vec<Vec<Vec<u32>>>,
        entry_point: Option<u32>,
        max_layer: u8,
        config: HnswConfig,
    ) -> Self {
        Self {
            nodes,
            slot_to_node,
            connections,
            entry_point,
            max_layer,
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_graph() {
        let config = HnswConfig::default();
        let graph = HnswGraph::new(config);

        assert!(graph.is_empty());
        assert_eq!(graph.node_count(), 0);
        assert!(graph.entry_point().is_none());
    }

    #[test]
    fn test_add_node() {
        let config = HnswConfig::default();
        let mut graph = HnswGraph::new(config);

        let node_id = graph.add_node(100, "test_id".to_string(), 2);

        assert_eq!(node_id, 0);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.entry_point(), Some(0));
        assert_eq!(graph.max_layer(), 2);

        let node = graph.get_node(0).unwrap();
        assert_eq!(node.slot, 100);
        assert_eq!(node.id, "test_id");
        assert_eq!(node.max_layer, 2);
    }

    #[test]
    fn test_get_node_by_slot() {
        let config = HnswConfig::default();
        let mut graph = HnswGraph::new(config);

        graph.add_node(100, "a".to_string(), 0);
        graph.add_node(200, "b".to_string(), 1);

        assert_eq!(graph.get_node_id_by_slot(100), Some(0));
        assert_eq!(graph.get_node_id_by_slot(200), Some(1));
        assert_eq!(graph.get_node_id_by_slot(300), None);
    }

    #[test]
    fn test_connections() {
        let config = HnswConfig::default();
        let mut graph = HnswGraph::new(config);

        graph.add_node(100, "a".to_string(), 1);
        graph.add_node(200, "b".to_string(), 1);
        graph.add_node(300, "c".to_string(), 0);

        // Add connections
        graph.set_neighbors(0, 0, vec![1, 2]);
        graph.set_neighbors(0, 1, vec![0, 2]);
        graph.set_neighbors(1, 0, vec![1]);
        graph.set_neighbors(1, 1, vec![0]);

        assert_eq!(graph.get_neighbors(0, 0), &[1, 2]);
        assert_eq!(graph.get_neighbors(0, 1), &[0, 2]);
        assert_eq!(graph.get_neighbors(1, 0), &[1]);
        assert!(graph.get_neighbors(0, 2).is_empty());  // node 2 only in layer 0
    }

    #[test]
    fn test_mark_deleted() {
        let config = HnswConfig::default();
        let mut graph = HnswGraph::new(config);

        graph.add_node(100, "a".to_string(), 0);

        assert!(graph.is_slot_active(100));
        assert!(graph.mark_deleted(100));
        assert!(!graph.is_slot_active(100));
        assert!(!graph.mark_deleted(100));  // Already deleted
    }

    #[test]
    fn test_random_layer() {
        let config = HnswConfig::default();
        let graph = HnswGraph::new(config);
        let mut rng = rand::thread_rng();

        // Generate many layers and verify distribution
        let mut layer_counts = vec![0usize; 10];
        for _ in 0..10000 {
            let layer = graph.random_layer(&mut rng) as usize;
            if layer < layer_counts.len() {
                layer_counts[layer] += 1;
            }
        }

        // Layer 0 should have most nodes
        assert!(layer_counts[0] > layer_counts[1]);
        assert!(layer_counts[1] > layer_counts[2]);
        // Most nodes should be in layer 0 (~63% with M=16)
        assert!(layer_counts[0] > 5000);
    }

    #[test]
    fn test_entry_point_updates() {
        let config = HnswConfig::default();
        let mut graph = HnswGraph::new(config);

        // First node becomes entry point
        graph.add_node(100, "a".to_string(), 1);
        assert_eq!(graph.entry_point(), Some(0));
        assert_eq!(graph.max_layer(), 1);

        // Lower layer node doesn't change entry point
        graph.add_node(200, "b".to_string(), 0);
        assert_eq!(graph.entry_point(), Some(0));
        assert_eq!(graph.max_layer(), 1);

        // Higher layer node becomes new entry point
        graph.add_node(300, "c".to_string(), 3);
        assert_eq!(graph.entry_point(), Some(2));
        assert_eq!(graph.max_layer(), 3);
    }
}
