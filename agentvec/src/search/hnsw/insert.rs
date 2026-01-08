//! HNSW insert algorithm implementation.

use std::cmp::Ordering;
use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::Metric;
use crate::search::distance::DistanceMetric;
use crate::storage::VectorStorage;

use super::graph::HnswGraph;

impl HnswGraph {
    /// Insert a new vector into the HNSW graph.
    ///
    /// # Arguments
    ///
    /// * `slot` - Slot offset in VectorStorage
    /// * `id` - Record ID for the vector
    /// * `vectors` - Vector storage containing the vector data
    /// * `metric` - Distance metric to use
    ///
    /// # Returns
    ///
    /// The internal node ID assigned to the new vector.
    pub fn insert(
        &mut self,
        slot: u64,
        id: String,
        vectors: &VectorStorage,
        metric: Metric,
    ) -> u32 {
        self.insert_with_ef(slot, id, vectors, metric, self.config.ef_construction)
    }

    /// Insert with custom ef_construction (for faster incremental inserts).
    ///
    /// Lower ef = faster but lower quality connections.
    /// For incremental inserts, ef=50-100 is usually sufficient.
    pub fn insert_with_ef(
        &mut self,
        slot: u64,
        id: String,
        vectors: &VectorStorage,
        metric: Metric,
        ef: usize,
    ) -> u32 {
        // Use RNG seeded from the slot for deterministic behavior
        let mut rng = StdRng::seed_from_u64(slot);

        // Assign random layer using exponential distribution
        let node_layer = self.random_layer(&mut rng);

        // Handle first node
        if self.entry_point.is_none() {
            return self.add_node(slot, id, node_layer);
        }

        // IMPORTANT: Capture entry point BEFORE add_node, because add_node
        // updates entry_point if the new node has a higher layer.
        let entry_point = self.entry_point.unwrap();

        // Add the node to the graph structure
        let node_id = self.add_node(slot, id, node_layer);

        // Get the query vector
        let query = match vectors.read_slot_ref(slot) {
            Ok(v) => v,
            Err(_) => return node_id, // Can't insert without vector
        };

        // Phase 1: Navigate from top layer down to node_layer + 1
        // Use greedy search to find the best entry point for each layer
        let mut current_entry = entry_point;

        for layer in (node_layer as usize + 1..=self.max_layer as usize).rev() {
            current_entry = self.search_layer_greedy(query, current_entry, layer, vectors, metric);
        }

        // Phase 2: Insert at each layer from node_layer down to 0
        for layer in (0..=node_layer as usize).rev() {
            // Find ef nearest neighbors at this layer
            let neighbors = self.search_layer_for_insert(
                query,
                current_entry,
                layer,
                ef,
                vectors,
                metric,
            );

            // Select the M best neighbors
            let m = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            let selected = self.select_neighbors(&neighbors, m, node_id);

            // Set the node's neighbors
            self.set_neighbors(layer, node_id, selected.clone());

            // Add bidirectional connections
            for &neighbor in &selected {
                self.add_connection(layer, neighbor, node_id);

                // Prune neighbor's connections if over limit
                let max_conn = self.max_connections(layer);
                let neighbor_conns = self.get_neighbors(layer, neighbor).to_vec();

                if neighbor_conns.len() > max_conn {
                    let pruned = self.prune_connections(
                        neighbor,
                        &neighbor_conns,
                        max_conn,
                        vectors,
                        metric,
                    );
                    self.set_neighbors(layer, neighbor, pruned);
                }
            }

            // Use the best neighbor as entry point for next layer
            if !neighbors.is_empty() {
                current_entry = neighbors[0].0;
            }
        }

        // Update entry point if new node has highest layer
        if node_layer > self.max_layer {
            // Already updated in add_node, but verify
            self.entry_point = Some(node_id);
            self.max_layer = node_layer;
        }

        node_id
    }

    /// Search a layer for candidates during insertion.
    /// Similar to search_layer_ef but returns all candidates found.
    fn search_layer_for_insert(
        &self,
        query: &[f32],
        entry: u32,
        layer: usize,
        ef: usize,
        vectors: &VectorStorage,
        metric: Metric,
    ) -> Vec<(u32, f32)> {
        let higher_is_better = metric.higher_is_better();
        let mut visited = HashSet::new();
        let mut candidates = Vec::new();
        let mut to_visit = Vec::new();

        // Start with entry point
        let entry_dist = self.compute_distance(query, entry, vectors, metric);
        to_visit.push((entry, entry_dist));
        visited.insert(entry);

        let mut worst_dist = entry_dist;

        while let Some((current, current_dist)) = to_visit.pop() {
            // Stop if current is worse than worst and we have enough candidates
            if candidates.len() >= ef
                && !Self::is_better(current_dist, worst_dist, higher_is_better)
            {
                continue;
            }

            // Add current to candidates
            candidates.push((current, current_dist));

            // Update worst distance
            if candidates.len() >= ef {
                candidates.sort_by(|a, b| {
                    if higher_is_better {
                        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                    } else {
                        a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
                    }
                });
                candidates.truncate(ef);
                worst_dist = candidates.last().unwrap().1;
            }

            // Explore neighbors
            for &neighbor in self.get_neighbors(layer, current) {
                if visited.insert(neighbor) {
                    let neighbor_dist = self.compute_distance(query, neighbor, vectors, metric);

                    // Only visit if promising
                    if candidates.len() < ef
                        || Self::is_better(neighbor_dist, worst_dist, higher_is_better)
                    {
                        to_visit.push((neighbor, neighbor_dist));
                    }
                }
            }

            // Sort to_visit to explore best candidates first
            to_visit.sort_by(|a, b| {
                if higher_is_better {
                    a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
                } else {
                    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                }
            });
        }

        // Final sort
        candidates.sort_by(|a, b| {
            if higher_is_better {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
            }
        });

        candidates
    }

    /// Select the M best neighbors from candidates.
    /// Uses simple distance-based selection.
    fn select_neighbors(
        &self,
        candidates: &[(u32, f32)],
        m: usize,
        exclude: u32,
    ) -> Vec<u32> {
        candidates
            .iter()
            .filter(|(id, _)| *id != exclude)
            .take(m)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Prune a node's connections to keep only the M best.
    fn prune_connections(
        &self,
        node_id: u32,
        neighbors: &[u32],
        m: usize,
        vectors: &VectorStorage,
        metric: Metric,
    ) -> Vec<u32> {
        let higher_is_better = metric.higher_is_better();
        let node = &self.nodes[node_id as usize];

        let node_vector = match vectors.read_slot_ref(node.slot) {
            Ok(v) => v,
            Err(_) => return neighbors.iter().take(m).copied().collect(),
        };

        // Score all neighbors
        let mut scored: Vec<_> = neighbors
            .iter()
            .filter_map(|&n| {
                let neighbor_slot = self.nodes[n as usize].slot;
                vectors
                    .read_slot_ref(neighbor_slot)
                    .ok()
                    .map(|v| (n, metric.compute(node_vector, v)))
            })
            .collect();

        // Sort by distance (best first)
        scored.sort_by(|a, b| {
            if higher_is_better {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
            }
        });

        // Keep top M
        scored.into_iter().take(m).map(|(n, _)| n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::hnsw::config::HnswConfig;
    use tempfile::TempDir;

    fn create_test_vectors(dim: usize, count: usize) -> (TempDir, VectorStorage) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");
        let mut storage = VectorStorage::create(&path, dim).unwrap();

        for i in 0..count {
            let slot = storage.allocate_slot().unwrap();
            let mut vec = vec![0.0f32; dim];
            // Create distinct vectors
            vec[i % dim] = 1.0;
            storage.write_slot(slot, &vec).unwrap();
        }

        (dir, storage)
    }

    #[test]
    fn test_insert_first_node() {
        let config = HnswConfig::default();
        let mut graph = HnswGraph::new(config);

        let (_dir, vectors) = create_test_vectors(4, 1);

        let node_id = graph.insert(0, "node0".to_string(), &vectors, Metric::Cosine);

        assert_eq!(node_id, 0);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.entry_point(), Some(0));
    }

    #[test]
    fn test_insert_multiple_nodes() {
        let config = HnswConfig::with_m(4);
        let mut graph = HnswGraph::new(config);

        let (_dir, vectors) = create_test_vectors(8, 10);

        for i in 0..10 {
            graph.insert(i as u64, format!("node{}", i), &vectors, Metric::Cosine);
        }

        assert_eq!(graph.node_count(), 10);
        assert!(graph.entry_point().is_some());

        // Check that nodes have connections at layer 0
        let mut total_connections = 0;
        for i in 0..10 {
            let neighbors = graph.get_neighbors(0, i);
            total_connections += neighbors.len();
        }
        // Should have some connections (bidirectional)
        assert!(total_connections > 0);
    }

    #[test]
    fn test_select_neighbors() {
        let config = HnswConfig::default();
        let graph = HnswGraph::new(config);

        let candidates = vec![(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)];

        let selected = graph.select_neighbors(&candidates, 2, 0);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], 1); // Best
        assert_eq!(selected[1], 2); // Second best
    }

    #[test]
    fn test_select_neighbors_excludes_self() {
        let config = HnswConfig::default();
        let graph = HnswGraph::new(config);

        let candidates = vec![(0, 1.0), (1, 0.9), (2, 0.8)];

        let selected = graph.select_neighbors(&candidates, 2, 0);
        assert_eq!(selected.len(), 2);
        assert!(!selected.contains(&0)); // Self excluded
    }
}
