//! HNSW search algorithm implementation.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::config::Metric;
use crate::search::distance::DistanceMetric;
use crate::storage::VectorStorage;

use super::graph::HnswGraph;

/// A candidate node during search, ordered by distance.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    node_id: u32,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node_id == other.node_id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // For min-heap: smaller distance = higher priority
        // Reverse comparison so BinaryHeap acts as min-heap
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Wrapper for max-heap behavior (worst candidates at top).
#[derive(Debug, Clone, Copy)]
struct MaxCandidate(Candidate);

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for MaxCandidate {}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // For max-heap: larger distance = higher priority
        self.0
            .distance
            .partial_cmp(&other.0.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl HnswGraph {
    /// Search for the k nearest neighbors of a query vector.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (should be normalized for cosine metric)
    /// * `k` - Number of nearest neighbors to return
    /// * `ef` - Size of dynamic candidate list (larger = better recall, slower)
    /// * `vectors` - Vector storage for reading vector data
    /// * `metric` - Distance metric to use
    /// * `deleted` - Set of deleted slot offsets to exclude
    ///
    /// # Returns
    ///
    /// Vector of (node_id, distance) pairs, sorted by distance (best first).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        vectors: &VectorStorage,
        metric: Metric,
        deleted: &HashSet<u64>,
    ) -> Vec<(u32, f32)> {
        if self.entry_point.is_none() || k == 0 {
            return Vec::new();
        }

        let entry_point = self.entry_point.unwrap();
        let ef = ef.max(k); // ef must be at least k

        // Start at the entry point
        let mut current = entry_point;

        // Phase 1: Greedy search from top layer down to layer 1
        // At each layer, find the closest node to use as entry for next layer
        for layer in (1..=self.max_layer as usize).rev() {
            current = self.search_layer_greedy(query, current, layer, vectors, metric);
        }

        // Phase 2: Beam search at layer 0 with ef candidates
        let candidates =
            self.search_layer_ef(query, current, 0, ef, vectors, metric, deleted);

        // Return top k results
        candidates.into_iter().take(k).collect()
    }

    /// Greedy search within a single layer.
    /// Returns the node closest to the query.
    pub(crate) fn search_layer_greedy(
        &self,
        query: &[f32],
        entry: u32,
        layer: usize,
        vectors: &VectorStorage,
        metric: Metric,
    ) -> u32 {
        let higher_is_better = metric.higher_is_better();
        let mut current = entry;
        let mut current_dist = self.compute_distance(query, current, vectors, metric);

        loop {
            let mut improved = false;

            for &neighbor in self.get_neighbors(layer, current) {
                let neighbor_dist = self.compute_distance(query, neighbor, vectors, metric);

                if Self::is_better(neighbor_dist, current_dist, higher_is_better) {
                    current = neighbor;
                    current_dist = neighbor_dist;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        current
    }

    /// Beam search within layer 0 using ef candidates.
    /// Returns candidates sorted by distance (best first).
    fn search_layer_ef(
        &self,
        query: &[f32],
        entry: u32,
        layer: usize,
        ef: usize,
        vectors: &VectorStorage,
        metric: Metric,
        deleted: &HashSet<u64>,
    ) -> Vec<(u32, f32)> {
        let higher_is_better = metric.higher_is_better();

        // Transform distance to "cost" that we always want to minimize
        // For L2: cost = distance (smaller is better)
        // For cosine/dot: cost = -distance (we negate so smaller cost = larger similarity)
        let to_cost = |dist: f32| -> f32 {
            if higher_is_better { -dist } else { dist }
        };
        let from_cost = |cost: f32| -> f32 {
            if higher_is_better { -cost } else { cost }
        };

        // Track visited nodes
        let mut visited = HashSet::new();

        // Min-heap of candidates to explore (lowest cost = best candidates first)
        let mut candidates = BinaryHeap::new();

        // Max-heap of results (highest cost = worst at top for efficient pruning)
        let mut results = BinaryHeap::new();

        // Initialize with entry point
        let entry_dist = self.compute_distance(query, entry, vectors, metric);
        let entry_cost = to_cost(entry_dist);
        candidates.push(Candidate {
            node_id: entry,
            distance: entry_cost,  // Store as cost
        });
        visited.insert(entry);

        // Add entry to results if not deleted
        let entry_slot = self.nodes[entry as usize].slot;
        if !deleted.contains(&entry_slot) {
            results.push(MaxCandidate(Candidate {
                node_id: entry,
                distance: entry_cost,
            }));
        }

        // Explore candidates
        while let Some(current) = candidates.pop() {
            // Stop if current candidate's cost is worse than the worst in results
            // (when results is full)
            if results.len() >= ef {
                let worst_cost = results.peek().unwrap().0.distance;
                if current.distance >= worst_cost {
                    break;
                }
            }

            // Explore all neighbors
            for &neighbor in self.get_neighbors(layer, current.node_id) {
                if !visited.insert(neighbor) {
                    continue; // Already visited
                }

                let neighbor_dist = self.compute_distance(query, neighbor, vectors, metric);
                let neighbor_cost = to_cost(neighbor_dist);

                // Check if this candidate is promising (lower cost = better)
                let dominated = results.len() >= ef && {
                    let worst_cost = results.peek().unwrap().0.distance;
                    neighbor_cost >= worst_cost
                };

                if !dominated {
                    candidates.push(Candidate {
                        node_id: neighbor,
                        distance: neighbor_cost,
                    });

                    // Add to results if not deleted
                    let neighbor_slot = self.nodes[neighbor as usize].slot;
                    if !deleted.contains(&neighbor_slot) {
                        results.push(MaxCandidate(Candidate {
                            node_id: neighbor,
                            distance: neighbor_cost,
                        }));

                        // Keep only ef best results (pop highest cost = worst)
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Extract and sort results (convert cost back to distance)
        let mut result_vec: Vec<_> = results
            .into_iter()
            .map(|mc| (mc.0.node_id, from_cost(mc.0.distance)))
            .collect();

        // Sort by distance (best first)
        result_vec.sort_by(|a, b| {
            if higher_is_better {
                b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
            }
        });

        result_vec
    }

    /// Compute distance between query and a node.
    pub(crate) fn compute_distance(
        &self,
        query: &[f32],
        node_id: u32,
        vectors: &VectorStorage,
        metric: Metric,
    ) -> f32 {
        let slot = self.nodes[node_id as usize].slot;
        match vectors.read_slot_ref(slot) {
            Ok(vector) => metric.compute(query, vector),
            Err(_) => {
                // Return worst possible score on error
                if metric.higher_is_better() {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            }
        }
    }

    /// Check if distance `a` is better than distance `b`.
    #[inline]
    pub(crate) fn is_better(a: f32, b: f32, higher_is_better: bool) -> bool {
        if higher_is_better {
            a > b
        } else {
            a < b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_ordering() {
        // Min-heap: smaller distance should have higher priority
        let c1 = Candidate {
            node_id: 1,
            distance: 0.5,
        };
        let c2 = Candidate {
            node_id: 2,
            distance: 0.3,
        };

        // c2 (0.3) should be "greater" for min-heap
        assert!(c2 > c1);
    }

    #[test]
    fn test_max_candidate_ordering() {
        // Max-heap: larger distance should have higher priority
        let c1 = MaxCandidate(Candidate {
            node_id: 1,
            distance: 0.5,
        });
        let c2 = MaxCandidate(Candidate {
            node_id: 2,
            distance: 0.3,
        });

        // c1 (0.5) should be "greater" for max-heap
        assert!(c1 > c2);
    }

    #[test]
    fn test_is_better() {
        // Higher is better (cosine, dot)
        assert!(HnswGraph::is_better(0.9, 0.8, true));
        assert!(!HnswGraph::is_better(0.7, 0.8, true));

        // Lower is better (L2)
        assert!(HnswGraph::is_better(0.1, 0.2, false));
        assert!(!HnswGraph::is_better(0.3, 0.2, false));
    }
}
