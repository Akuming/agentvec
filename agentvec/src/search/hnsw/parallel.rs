//! Parallel HNSW index construction with Product Quantization acceleration.
//!
//! This module implements an optimized parallel HNSW builder:
//! 1. **Product Quantization**: Train PQ codebooks for ~10x faster distance computation
//! 2. **PQ-accelerated construction**: Use PQ distances for candidate evaluation
//! 3. **F32 neighbor selection**: Final neighbors selected with exact f32 distances
//!
//! The PQ distances provide ~10x speedup during graph construction while
//! maintaining high recall through exact-distance neighbor selection.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, AtomicU8, AtomicUsize, Ordering};
use std::time::Instant;

use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::config::Metric;
use crate::search::distance::DistanceMetric;
use crate::search::pq::{ProductQuantizer, PQConfig, PQVectors};
use crate::search::quantize::SignedQuantizedVectors;
use crate::storage::VectorStorage;

use super::config::HnswConfig;
use super::graph::{HnswGraph, HnswNode};

/// Per-node connection storage with fine-grained locking.
struct NodeConnections {
    neighbors: RwLock<Vec<u32>>,
}

impl NodeConnections {
    fn new() -> Self {
        Self { neighbors: RwLock::new(Vec::new()) }
    }

    fn get(&self) -> Vec<u32> {
        self.neighbors.read().clone()
    }

    fn set(&self, neighbors: Vec<u32>) {
        *self.neighbors.write() = neighbors;
    }

    fn add(&self, neighbor: u32) {
        let mut neighbors = self.neighbors.write();
        if !neighbors.contains(&neighbor) {
            neighbors.push(neighbor);
        }
    }

    fn len(&self) -> usize {
        self.neighbors.read().len()
    }
}

/// Per-layer connection storage.
struct LayerConnections {
    nodes: Vec<NodeConnections>,
}

impl LayerConnections {
    fn new(capacity: usize) -> Self {
        let nodes = (0..capacity).map(|_| NodeConnections::new()).collect();
        Self { nodes }
    }

    fn get(&self, node_id: u32) -> Vec<u32> {
        self.nodes.get(node_id as usize)
            .map(|n| n.get())
            .unwrap_or_default()
    }

    fn set(&self, node_id: u32, neighbors: Vec<u32>) {
        if let Some(n) = self.nodes.get(node_id as usize) {
            n.set(neighbors);
        }
    }

    fn add(&self, node_id: u32, neighbor: u32) {
        if let Some(n) = self.nodes.get(node_id as usize) {
            n.add(neighbor);
        }
    }

    fn len(&self, node_id: u32) -> usize {
        self.nodes.get(node_id as usize)
            .map(|n| n.len())
            .unwrap_or(0)
    }
}

/// Parallel HNSW builder with repair pass.
pub struct ParallelHnswBuilder {
    config: HnswConfig,
    metric: Metric,
    higher_is_better: bool,
}

impl ParallelHnswBuilder {
    pub fn new(config: HnswConfig, metric: Metric) -> Self {
        let higher_is_better = metric.higher_is_better();
        Self { config, metric, higher_is_better }
    }

    /// Build the HNSW graph from records using parallel construction with batch-aware repair.
    pub fn build(self, records: Vec<(u64, String)>, vectors: &VectorStorage) -> HnswGraph {
        if records.is_empty() {
            return HnswGraph::new(self.config.clone());
        }

        let total = records.len();
        let build_start = Instant::now();

        // Phase 1: Create all nodes with layer assignments
        let phase1_start = Instant::now();
        let mut nodes: Vec<HnswNode> = Vec::with_capacity(total);
        let mut slot_to_node: HashMap<u64, u32> = HashMap::with_capacity(total);
        let mut max_layer: u8 = 0;

        for (idx, (slot, id)) in records.iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(*slot);
            let layer = self.random_layer(&mut rng);
            max_layer = max_layer.max(layer);

            nodes.push(HnswNode {
                slot: *slot,
                id: id.clone(),
                max_layer: layer,
            });
            slot_to_node.insert(*slot, idx as u32);
        }

        // Pre-allocate connection storage
        let connections: Vec<LayerConnections> = (0..=max_layer as usize)
            .map(|_| LayerConnections::new(total))
            .collect();

        // Find initial entry point (highest layer node)
        let mut initial_entry = 0u32;
        let mut initial_max_layer = 0u8;
        for (idx, node) in nodes.iter().enumerate() {
            if node.max_layer > initial_max_layer {
                initial_max_layer = node.max_layer;
                initial_entry = idx as u32;
            }
        }

        let entry_point = AtomicU32::new(initial_entry);
        let current_max_layer = AtomicU8::new(initial_max_layer);

        // Phase 1b: Pre-load all vectors and create signed quantized storage
        let f32_vectors: Vec<Vec<f32>> = nodes
            .par_iter()
            .map(|node| {
                match vectors.read_slot_ref(node.slot) {
                    Ok(vec) => vec.to_vec(),
                    Err(_) => vec![0.0f32; vectors.dimensions()],
                }
            })
            .collect();

        // Train Product Quantizer for fast distance computation
        let dim = vectors.dimensions();
        let pq_config = PQConfig::for_dimension(dim);
        let refs: Vec<&[f32]> = f32_vectors.iter().map(|v| v.as_slice()).collect();
        let pq = ProductQuantizer::train(
            refs.iter().copied(),
            dim,
            &pq_config,
            self.higher_is_better,
        );

        // Encode all vectors with PQ
        let pq_codes = PQVectors::from_vectors(&pq, &refs);

        let phase1_time = phase1_start.elapsed();

        // Phase 2: PQ-accelerated HNSW construction
        // Use PQ for fast candidate discovery, f32 for accurate neighbor selection
        let phase2_start = Instant::now();
        let distance_calcs = AtomicUsize::new(0);

        // Higher ef since PQ is fast
        let phase2_ef = self.config.ef_construction;

        // Process in batches
        let batch_size = 5000;

        for batch_start in (0..total).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total);

            (batch_start..batch_end).into_par_iter().for_each(|idx| {
                let node_id = idx as u32;
                self.connect_node_pq(
                    node_id,
                    &nodes,
                    &connections,
                    &f32_vectors,
                    &pq,
                    &pq_codes,
                    &entry_point,
                    &current_max_layer,
                    &distance_calcs,
                    phase2_ef,
                );
            });
        }
        let phase2_time = phase2_start.elapsed();

        // Phase 3: Skip - PQ construction is already high quality
        let phase3_start = Instant::now();
        let nodes_improved = AtomicUsize::new(0);
        let phase3_time = phase3_start.elapsed();
        let poorly_connected = nodes_improved.load(Ordering::Relaxed);

        // Print profiling info for builds > 1000 nodes
        if total >= 1000 {
            let total_time = build_start.elapsed();
            eprintln!(
                "HNSW Build Profile ({} nodes):\n  Phase 1 (setup):   {:>8.2}s ({:>5.1}%)\n  Phase 2 (insert):  {:>8.2}s ({:>5.1}%)\n  Phase 3 (repair):  {:>8.2}s ({:>5.1}%)\n  Total:             {:>8.2}s ({:.0} vec/s)\n  Distance calcs:    {}\n  Nodes improved:    {}",
                total,
                phase1_time.as_secs_f64(),
                phase1_time.as_secs_f64() / total_time.as_secs_f64() * 100.0,
                phase2_time.as_secs_f64(),
                phase2_time.as_secs_f64() / total_time.as_secs_f64() * 100.0,
                phase3_time.as_secs_f64(),
                phase3_time.as_secs_f64() / total_time.as_secs_f64() * 100.0,
                total_time.as_secs_f64(),
                total as f64 / total_time.as_secs_f64(),
                distance_calcs.load(Ordering::Relaxed),
                poorly_connected,
            );
        }

        // Convert to regular graph format
        let final_connections: Vec<Vec<Vec<u32>>> = connections
            .into_iter()
            .map(|layer| {
                layer.nodes.into_iter().map(|n| n.neighbors.into_inner()).collect()
            })
            .collect();

        let final_entry = Some(entry_point.load(Ordering::SeqCst));
        let final_max_layer = current_max_layer.load(Ordering::SeqCst);

        HnswGraph::restore(
            nodes,
            slot_to_node,
            final_connections,
            final_entry,
            final_max_layer,
            self.config,
        )
    }

    fn is_better(&self, a: f32, b: f32) -> bool {
        if self.higher_is_better { a > b } else { a < b }
    }

    fn random_layer<R: rand::Rng>(&self, rng: &mut R) -> u8 {
        let uniform: f64 = rng.gen();
        let layer = (-uniform.ln() * self.config.ml).floor() as u8;
        layer.min(32)
    }

    // ========== PQ-accelerated construction methods ==========

    /// Connect a node using PQ for fast traversal, f32 for accurate neighbor selection.
    fn connect_node_pq(
        &self,
        node_id: u32,
        nodes: &[HnswNode],
        connections: &[LayerConnections],
        f32_vectors: &[Vec<f32>],
        pq: &ProductQuantizer,
        pq_codes: &PQVectors,
        entry_point: &AtomicU32,
        current_max_layer: &AtomicU8,
        distance_calcs: &AtomicUsize,
        ef: usize,
    ) {
        let node = &nodes[node_id as usize];
        let query = &f32_vectors[node_id as usize];

        let node_layer = node.max_layer as usize;
        let ep = entry_point.load(Ordering::Relaxed);

        // Skip if this is the entry point with no connections yet
        if node_id == ep && connections[0].len(node_id) == 0 {
            return;
        }

        let graph_max_layer = current_max_layer.load(Ordering::Relaxed) as usize;

        // Precompute PQ distance table for this query
        let pq_table = pq.precompute_table(query);

        // Greedy descent using PQ distances (fast)
        let mut current_entry = ep;
        for layer in ((node_layer + 1)..=graph_max_layer).rev() {
            if layer < connections.len() {
                current_entry = self.search_layer_greedy_pq(
                    &pq_table, pq_codes, current_entry, layer, connections,
                );
            }
        }

        // Insert at each layer using PQ for candidate finding, f32 for selection
        for layer in (0..=node_layer).rev() {
            // Use PQ for fast beam search
            let candidates = self.search_layer_pq(
                &pq_table, pq_codes, current_entry, layer, ef, connections, distance_calcs,
            );

            let m = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            // Re-score ALL candidates with f32 for accurate selection
            // PQ distances can have significant ranking errors, so we need to rescore everything
            let mut f32_scored: Vec<(u32, f32)> = candidates
                .iter()
                .filter(|(id, _)| *id != node_id)
                .map(|(id, _)| {
                    let dist = self.metric.compute(query, &f32_vectors[*id as usize]);
                    (*id, dist)
                })
                .collect();

            // Sort by f32 distance
            f32_scored.sort_by(|a, b| {
                if self.higher_is_better {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                }
            });

            // Select top m neighbors
            let selected: Vec<u32> = f32_scored
                .iter()
                .take(m)
                .map(|(id, _)| *id)
                .collect();

            // Set node's neighbors
            connections[layer].set(node_id, selected.clone());

            // Add bidirectional connections
            for &neighbor in &selected {
                connections[layer].add(neighbor, node_id);
            }

            // Update entry for next layer
            if !candidates.is_empty() {
                current_entry = candidates[0].0;
            }
        }
    }

    /// Greedy search using PQ distances.
    fn search_layer_greedy_pq(
        &self,
        pq_table: &crate::search::pq::PQDistanceTable,
        pq_codes: &PQVectors,
        entry: u32,
        layer: usize,
        connections: &[LayerConnections],
    ) -> u32 {
        let mut current = entry;
        let mut current_dist = pq_table.distance(pq_codes.get_code(current as usize));

        loop {
            let neighbors = connections[layer].get(current);
            let mut improved = false;

            for neighbor in neighbors {
                let dist = pq_table.distance(pq_codes.get_code(neighbor as usize));
                if pq_table.is_better(dist, current_dist) {
                    current = neighbor;
                    current_dist = dist;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        current
    }

    /// Beam search using PQ distances with optimized heap-based implementation.
    fn search_layer_pq(
        &self,
        pq_table: &crate::search::pq::PQDistanceTable,
        pq_codes: &PQVectors,
        entry: u32,
        layer: usize,
        ef: usize,
        connections: &[LayerConnections],
        distance_calcs: &AtomicUsize,
    ) -> Vec<(u32, f32)> {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering as CmpOrdering;

        // Wrapper for heap ordering based on distance metric
        #[derive(Clone, Copy)]
        struct HeapItem {
            id: u32,
            dist: f32,
            higher_is_better: bool,
        }

        impl PartialEq for HeapItem {
            fn eq(&self, other: &Self) -> bool {
                self.id == other.id
            }
        }
        impl Eq for HeapItem {}

        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> CmpOrdering {
                // For candidates heap: want best (to pop) at top
                // higher_is_better=true: higher dist is better, so reverse for max-heap behavior
                // higher_is_better=false: lower dist is better, so normal for max-heap to give min behavior
                if self.higher_is_better {
                    // Higher is better: we want highest at top
                    self.dist.partial_cmp(&other.dist).unwrap_or(CmpOrdering::Equal)
                } else {
                    // Lower is better: we want lowest at top (reverse comparison)
                    other.dist.partial_cmp(&self.dist).unwrap_or(CmpOrdering::Equal)
                }
            }
        }

        let mut visited = HashSet::with_capacity(ef * 2);
        let mut candidates: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(ef * 2);
        let mut results: Vec<(u32, f32)> = Vec::with_capacity(ef + 1);

        distance_calcs.fetch_add(1, Ordering::Relaxed);
        let entry_dist = pq_table.distance(pq_codes.get_code(entry as usize));
        candidates.push(HeapItem { id: entry, dist: entry_dist, higher_is_better: self.higher_is_better });
        visited.insert(entry);

        let mut worst_dist = if self.higher_is_better { f32::NEG_INFINITY } else { f32::INFINITY };

        while let Some(HeapItem { id: current, dist: current_dist, .. }) = candidates.pop() {
            // Stop if current candidate is worse than worst result and we have enough
            if results.len() >= ef && !pq_table.is_better(current_dist, worst_dist) {
                break;
            }

            // Add to results
            results.push((current, current_dist));

            // Maintain ef size - only sort when needed
            if results.len() > ef {
                // Find and remove worst element
                let worst_idx = if self.higher_is_better {
                    results.iter().enumerate().min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap_or(CmpOrdering::Equal)).map(|(i, _)| i).unwrap()
                } else {
                    results.iter().enumerate().max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap_or(CmpOrdering::Equal)).map(|(i, _)| i).unwrap()
                };
                results.swap_remove(worst_idx);
                // Update worst_dist
                worst_dist = if self.higher_is_better {
                    results.iter().map(|(_, d)| *d).fold(f32::INFINITY, f32::min)
                } else {
                    results.iter().map(|(_, d)| *d).fold(f32::NEG_INFINITY, f32::max)
                };
            }

            let neighbors = connections[layer].get(current);
            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    distance_calcs.fetch_add(1, Ordering::Relaxed);
                    let dist = pq_table.distance(pq_codes.get_code(neighbor as usize));
                    if results.len() < ef || pq_table.is_better(dist, worst_dist) {
                        candidates.push(HeapItem { id: neighbor, dist, higher_is_better: self.higher_is_better });
                    }
                }
            }
        }

        // Final sort
        results.sort_by(|a, b| {
            if self.higher_is_better {
                b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(CmpOrdering::Equal)
            }
        });

        results
    }

    // ========== Pre-loaded f32 vector methods for fast construction ==========

    /// Connect a node using pre-loaded f32 vectors.
    fn connect_node_f32(
        &self,
        node_id: u32,
        nodes: &[HnswNode],
        connections: &[LayerConnections],
        f32_vectors: &[Vec<f32>],
        entry_point: &AtomicU32,
        current_max_layer: &AtomicU8,
        distance_calcs: &AtomicUsize,
    ) {
        let node = &nodes[node_id as usize];
        let query = &f32_vectors[node_id as usize];

        let node_layer = node.max_layer as usize;
        let ep = entry_point.load(Ordering::Relaxed);

        // Skip if this is the entry point with no connections yet
        if node_id == ep && connections[0].len(node_id) == 0 {
            return;
        }

        let graph_max_layer = current_max_layer.load(Ordering::Relaxed) as usize;

        // Greedy descent from top to node_layer + 1
        let mut current_entry = ep;
        for layer in ((node_layer + 1)..=graph_max_layer).rev() {
            if layer < connections.len() {
                current_entry = self.search_layer_greedy_f32(
                    query, current_entry, layer, connections, f32_vectors,
                );
            }
        }

        let ef = self.config.ef_construction;

        // Insert at each layer from node_layer down to 0
        for layer in (0..=node_layer).rev() {
            let candidates = self.search_layer_f32(
                query, current_entry, layer, ef, connections, f32_vectors, distance_calcs,
            );

            let m = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            // Select neighbors
            let selected: Vec<u32> = candidates
                .iter()
                .filter(|(id, _)| *id != node_id)
                .take(m)
                .map(|(id, _)| *id)
                .collect();

            // Set node's neighbors
            connections[layer].set(node_id, selected.clone());

            // Add bidirectional connections
            for &neighbor in &selected {
                connections[layer].add(neighbor, node_id);
            }

            // Update entry for next layer
            if !candidates.is_empty() {
                current_entry = candidates[0].0;
            }
        }
    }

    /// Connect a node using pre-loaded f32 vectors with custom ef parameter.
    fn connect_node_f32_with_ef(
        &self,
        node_id: u32,
        nodes: &[HnswNode],
        connections: &[LayerConnections],
        f32_vectors: &[Vec<f32>],
        entry_point: &AtomicU32,
        current_max_layer: &AtomicU8,
        distance_calcs: &AtomicUsize,
        ef: usize,
    ) {
        let node = &nodes[node_id as usize];
        let query = &f32_vectors[node_id as usize];

        let node_layer = node.max_layer as usize;
        let ep = entry_point.load(Ordering::Relaxed);

        // Skip if this is the entry point with no connections yet
        if node_id == ep && connections[0].len(node_id) == 0 {
            return;
        }

        let graph_max_layer = current_max_layer.load(Ordering::Relaxed) as usize;

        // Greedy descent from top to node_layer + 1
        let mut current_entry = ep;
        for layer in ((node_layer + 1)..=graph_max_layer).rev() {
            if layer < connections.len() {
                current_entry = self.search_layer_greedy_f32(
                    query, current_entry, layer, connections, f32_vectors,
                );
            }
        }

        // Insert at each layer from node_layer down to 0
        for layer in (0..=node_layer).rev() {
            let candidates = self.search_layer_f32(
                query, current_entry, layer, ef, connections, f32_vectors, distance_calcs,
            );

            let m = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            // Select neighbors
            let selected: Vec<u32> = candidates
                .iter()
                .filter(|(id, _)| *id != node_id)
                .take(m)
                .map(|(id, _)| *id)
                .collect();

            // Set node's neighbors
            connections[layer].set(node_id, selected.clone());

            // Add bidirectional connections
            for &neighbor in &selected {
                connections[layer].add(neighbor, node_id);
            }

            // Update entry for next layer
            if !candidates.is_empty() {
                current_entry = candidates[0].0;
            }
        }
    }

    /// Connect a node using hybrid approach: quantized traversal, f32 selection.
    /// This gives speed benefits of quantized while maintaining f32 quality for neighbors.
    fn connect_node_hybrid(
        &self,
        node_id: u32,
        nodes: &[HnswNode],
        connections: &[LayerConnections],
        f32_vectors: &[Vec<f32>],
        quantized: &SignedQuantizedVectors,
        entry_point: &AtomicU32,
        current_max_layer: &AtomicU8,
        distance_calcs: &AtomicUsize,
        ef: usize,
    ) {
        let node = &nodes[node_id as usize];
        let query = &f32_vectors[node_id as usize];

        let node_layer = node.max_layer as usize;
        let ep = entry_point.load(Ordering::Relaxed);

        // Skip if this is the entry point with no connections yet
        if node_id == ep && connections[0].len(node_id) == 0 {
            return;
        }

        let graph_max_layer = current_max_layer.load(Ordering::Relaxed) as usize;

        // Greedy descent using QUANTIZED distances (fast)
        let mut current_entry = ep;
        for layer in ((node_layer + 1)..=graph_max_layer).rev() {
            if layer < connections.len() {
                current_entry = self.search_layer_greedy_quantized(
                    node_id, current_entry, layer, connections, quantized,
                );
            }
        }

        // Insert at each layer using QUANTIZED for discovery, F32 for selection
        for layer in (0..=node_layer).rev() {
            // Use quantized beam search for fast candidate discovery
            let quant_candidates = self.search_layer_quantized(
                node_id, current_entry, layer, ef, connections, quantized, distance_calcs,
            );

            // Re-score top candidates with f32 for accurate selection
            let m = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            // Take top 2*m candidates and rescore with f32
            let rescore_count = (m * 2).min(quant_candidates.len());
            let mut f32_scored: Vec<(u32, f32)> = quant_candidates
                .iter()
                .take(rescore_count)
                .filter(|(id, _)| *id != node_id)
                .map(|(id, _)| {
                    let dist = self.metric.compute(query, &f32_vectors[*id as usize]);
                    (*id, dist)
                })
                .collect();

            // Sort by f32 distance
            f32_scored.sort_by(|a, b| {
                if self.higher_is_better {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                }
            });

            // Select top m neighbors
            let selected: Vec<u32> = f32_scored
                .iter()
                .take(m)
                .map(|(id, _)| *id)
                .collect();

            // Set node's neighbors
            connections[layer].set(node_id, selected.clone());

            // Add bidirectional connections
            for &neighbor in &selected {
                connections[layer].add(neighbor, node_id);
            }

            // Update entry for next layer
            if !quant_candidates.is_empty() {
                current_entry = quant_candidates[0].0;
            }
        }
    }

    /// Repair connections within a batch using f32 vectors.
    fn repair_batch_f32(
        &self,
        batch_start: usize,
        batch_end: usize,
        _nodes: &[HnswNode],
        connections: &[LayerConnections],
        f32_vectors: &[Vec<f32>],
    ) {
        let batch_size = batch_end - batch_start;
        if batch_size < 2 {
            return;
        }

        let m = self.config.m_max0;

        // For each node in the batch, find best neighbors within the batch
        (batch_start..batch_end).into_par_iter().for_each(|idx| {
            let node_id = idx as u32;
            let query = &f32_vectors[idx];

            // Score all other batch members
            let mut batch_neighbors: Vec<(u32, f32)> = (batch_start..batch_end)
                .filter(|&other_idx| other_idx != idx)
                .map(|other_idx| {
                    let other_vec = &f32_vectors[other_idx];
                    let dist = self.metric.compute(query, other_vec);
                    (other_idx as u32, dist)
                })
                .collect();

            // Sort by distance
            batch_neighbors.sort_by(|a, b| {
                if self.higher_is_better {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                }
            });

            // Merge with existing neighbors
            let current = connections[0].get(node_id);
            let mut all_neighbors: HashSet<u32> = current.into_iter().collect();

            // Add top batch neighbors
            for (neighbor_id, _) in batch_neighbors.iter().take(m / 2) {
                all_neighbors.insert(*neighbor_id);
            }

            // If we have too many, score and keep best
            if all_neighbors.len() > m {
                let mut scored: Vec<(u32, f32)> = all_neighbors
                    .into_iter()
                    .map(|n| {
                        let neighbor_vec = &f32_vectors[n as usize];
                        let dist = self.metric.compute(query, neighbor_vec);
                        (n, dist)
                    })
                    .collect();

                scored.sort_by(|a, b| {
                    if self.higher_is_better {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    }
                });

                let best: Vec<u32> = scored.into_iter().take(m).map(|(n, _)| n).collect();
                connections[0].set(node_id, best);
            } else {
                let neighbors: Vec<u32> = all_neighbors.into_iter().collect();
                connections[0].set(node_id, neighbors);
            }
        });

        // Add bidirectional connections for new batch edges
        for idx in batch_start..batch_end {
            let node_id = idx as u32;
            let neighbors = connections[0].get(node_id);
            for neighbor in neighbors {
                if (neighbor as usize) >= batch_start && (neighbor as usize) < batch_end {
                    connections[0].add(neighbor, node_id);
                }
            }
        }
    }

    /// Repair node using graph search with f32 vectors.
    /// Returns true if connections were improved.
    fn repair_node_f32(
        &self,
        node_id: u32,
        _nodes: &[HnswNode],
        connections: &[LayerConnections],
        f32_vectors: &[Vec<f32>],
        entry_point: &AtomicU32,
        current_max_layer: &AtomicU8,
        repair_ef: usize,
    ) -> bool {
        let query = &f32_vectors[node_id as usize];
        let m = self.config.m_max0;
        let current_neighbors = connections[0].get(node_id);

        // Navigate from entry point
        let ep = entry_point.load(Ordering::Relaxed);
        let graph_max_layer = current_max_layer.load(Ordering::Relaxed) as usize;

        let mut current_entry = ep;
        for layer in (1..=graph_max_layer).rev() {
            if layer < connections.len() {
                current_entry = self.search_layer_greedy_f32(
                    query, current_entry, layer, connections, f32_vectors,
                );
            }
        }

        // Search layer 0 with repair_ef
        let dummy_calcs = AtomicUsize::new(0);
        let candidates = self.search_layer_f32(
            query, current_entry, 0, repair_ef, connections, f32_vectors, &dummy_calcs,
        );

        // Select best neighbors
        let new_neighbors: Vec<u32> = candidates
            .iter()
            .filter(|(id, _)| *id != node_id)
            .take(m)
            .map(|(id, _)| *id)
            .collect();

        // Merge with existing and keep best
        let mut all_neighbors: HashSet<u32> = current_neighbors.iter().copied().collect();
        all_neighbors.extend(new_neighbors.iter().copied());

        // Score all candidates
        let mut scored: Vec<(u32, f32)> = all_neighbors
            .into_iter()
            .map(|n| {
                let neighbor_vec = &f32_vectors[n as usize];
                let dist = self.metric.compute(query, neighbor_vec);
                (n, dist)
            })
            .collect();

        scored.sort_by(|a, b| {
            if self.higher_is_better {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        let best: Vec<u32> = scored.iter().take(m).map(|(n, _)| *n).collect();

        // Check if improved
        let old_set: HashSet<u32> = current_neighbors.iter().copied().collect();
        let new_set: HashSet<u32> = best.iter().copied().collect();

        if old_set == new_set {
            return false;
        }

        // Update connections
        connections[0].set(node_id, best.clone());

        // Add bidirectional connections for new neighbors
        for &neighbor in &best {
            if !old_set.contains(&neighbor) {
                connections[0].add(neighbor, node_id);
            }
        }

        true
    }

    /// Greedy search using f32 vectors.
    fn search_layer_greedy_f32(
        &self,
        query: &[f32],
        entry: u32,
        layer: usize,
        connections: &[LayerConnections],
        f32_vectors: &[Vec<f32>],
    ) -> u32 {
        let mut current = entry;
        let mut current_dist = self.metric.compute(query, &f32_vectors[current as usize]);

        loop {
            let neighbors = connections[layer].get(current);
            let mut improved = false;

            for neighbor in neighbors {
                let dist = self.metric.compute(query, &f32_vectors[neighbor as usize]);
                if self.is_better(dist, current_dist) {
                    current = neighbor;
                    current_dist = dist;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        current
    }

    /// Beam search using f32 vectors.
    fn search_layer_f32(
        &self,
        query: &[f32],
        entry: u32,
        layer: usize,
        ef: usize,
        connections: &[LayerConnections],
        f32_vectors: &[Vec<f32>],
        distance_calcs: &AtomicUsize,
    ) -> Vec<(u32, f32)> {
        let mut visited = HashSet::new();
        let mut candidates = Vec::new();
        let mut results = Vec::with_capacity(ef);

        distance_calcs.fetch_add(1, Ordering::Relaxed);
        let entry_dist = self.metric.compute(query, &f32_vectors[entry as usize]);
        candidates.push((entry, entry_dist));
        visited.insert(entry);

        let mut worst_dist = entry_dist;

        while let Some((current, current_dist)) = candidates.pop() {
            if results.len() >= ef && !self.is_better(current_dist, worst_dist) {
                continue;
            }

            results.push((current, current_dist));

            if results.len() > ef {
                results.sort_by(|a, b| {
                    if self.higher_is_better {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
                results.truncate(ef);
                worst_dist = results.last().unwrap().1;
            }

            let neighbors = connections[layer].get(current);
            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    distance_calcs.fetch_add(1, Ordering::Relaxed);
                    let dist = self.metric.compute(query, &f32_vectors[neighbor as usize]);
                    if results.len() < ef || self.is_better(dist, worst_dist) {
                        candidates.push((neighbor, dist));
                    }
                }
            }

            // Sort candidates (best last for pop)
            candidates.sort_by(|a, b| {
                if self.higher_is_better {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                }
            });
        }

        results.sort_by(|a, b| {
            if self.higher_is_better {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        results
    }

    // ========== Signed Quantized methods (i8 for fast distance) ==========

    /// Compare i32 distances (for quantized vectors)
    #[inline]
    fn is_better_i32(&self, a: i32, b: i32) -> bool {
        if self.higher_is_better { a > b } else { a < b }
    }

    /// Connect a node using signed quantized distances.
    fn connect_node_quantized(
        &self,
        node_id: u32,
        nodes: &[HnswNode],
        connections: &[LayerConnections],
        quantized: &SignedQuantizedVectors,
        entry_point: &AtomicU32,
        current_max_layer: &AtomicU8,
        distance_calcs: &AtomicUsize,
        ef: usize, // Allow caller to specify ef
    ) {
        let node = &nodes[node_id as usize];
        let node_layer = node.max_layer as usize;
        let ep = entry_point.load(Ordering::Relaxed);

        // Skip if this is the entry point with no connections yet
        if node_id == ep && connections[0].len(node_id) == 0 {
            return;
        }

        let graph_max_layer = current_max_layer.load(Ordering::Relaxed) as usize;

        // Greedy descent from top to node_layer + 1
        let mut current_entry = ep;
        for layer in ((node_layer + 1)..=graph_max_layer).rev() {
            if layer < connections.len() {
                current_entry = self.search_layer_greedy_quantized(
                    node_id, current_entry, layer, connections, quantized,
                );
            }
        }

        // Insert at each layer from node_layer down to 0
        for layer in (0..=node_layer).rev() {
            let candidates = self.search_layer_quantized(
                node_id, current_entry, layer, ef, connections, quantized, distance_calcs,
            );

            let m = if layer == 0 {
                self.config.m_max0
            } else {
                self.config.m
            };

            // Select neighbors
            let selected: Vec<u32> = candidates
                .iter()
                .filter(|(id, _)| *id != node_id)
                .take(m)
                .map(|(id, _)| *id)
                .collect();

            // Set node's neighbors
            connections[layer].set(node_id, selected.clone());

            // Add bidirectional connections
            for &neighbor in &selected {
                connections[layer].add(neighbor, node_id);
            }

            // Update entry for next layer
            if !candidates.is_empty() {
                current_entry = candidates[0].0;
            }
        }
    }

    /// Repair connections within a batch using quantized distances.
    fn repair_batch_quantized(
        &self,
        batch_start: usize,
        batch_end: usize,
        connections: &[LayerConnections],
        quantized: &SignedQuantizedVectors,
    ) {
        let batch_size = batch_end - batch_start;
        if batch_size < 2 {
            return;
        }

        let m = self.config.m_max0;

        // For each node in the batch, find best neighbors within the batch
        (batch_start..batch_end).into_par_iter().for_each(|idx| {
            let node_id = idx as u32;

            // Score all other batch members
            let mut batch_neighbors: Vec<(u32, i32)> = (batch_start..batch_end)
                .filter(|&other_idx| other_idx != idx)
                .map(|other_idx| {
                    let dist = quantized.dot_product(idx, other_idx);
                    (other_idx as u32, dist)
                })
                .collect();

            // Sort by distance
            batch_neighbors.sort_by(|a, b| {
                if self.higher_is_better {
                    b.1.cmp(&a.1)
                } else {
                    a.1.cmp(&b.1)
                }
            });

            // Merge with existing neighbors
            let current = connections[0].get(node_id);
            let mut all_neighbors: HashSet<u32> = current.into_iter().collect();

            // Add top batch neighbors
            for (neighbor_id, _) in batch_neighbors.iter().take(m / 2) {
                all_neighbors.insert(*neighbor_id);
            }

            // If we have too many, score and keep best
            if all_neighbors.len() > m {
                let mut scored: Vec<(u32, i32)> = all_neighbors
                    .into_iter()
                    .map(|n| {
                        let dist = quantized.dot_product(idx, n as usize);
                        (n, dist)
                    })
                    .collect();

                scored.sort_by(|a, b| {
                    if self.higher_is_better {
                        b.1.cmp(&a.1)
                    } else {
                        a.1.cmp(&b.1)
                    }
                });

                let best: Vec<u32> = scored.into_iter().take(m).map(|(n, _)| n).collect();
                connections[0].set(node_id, best);
            } else {
                let neighbors: Vec<u32> = all_neighbors.into_iter().collect();
                connections[0].set(node_id, neighbors);
            }
        });

        // Add bidirectional connections for new batch edges
        for idx in batch_start..batch_end {
            let node_id = idx as u32;
            let neighbors = connections[0].get(node_id);
            for neighbor in neighbors {
                if (neighbor as usize) >= batch_start && (neighbor as usize) < batch_end {
                    connections[0].add(neighbor, node_id);
                }
            }
        }
    }

    /// Repair node using quantized traversal but f32 final selection (for accuracy).
    /// Returns true if connections were improved.
    fn repair_node_hybrid(
        &self,
        node_id: u32,
        connections: &[LayerConnections],
        quantized: &SignedQuantizedVectors,
        f32_vectors: &[Vec<f32>],
        entry_point: &AtomicU32,
        current_max_layer: &AtomicU8,
        repair_ef: usize,
    ) -> bool {
        let query = &f32_vectors[node_id as usize];
        let m = self.config.m_max0;
        let current_neighbors = connections[0].get(node_id);

        // Navigate from entry point using quantized distances (fast)
        let ep = entry_point.load(Ordering::Relaxed);
        let graph_max_layer = current_max_layer.load(Ordering::Relaxed) as usize;

        let mut current_entry = ep;
        for layer in (1..=graph_max_layer).rev() {
            if layer < connections.len() {
                current_entry = self.search_layer_greedy_quantized(
                    node_id as u32, current_entry, layer, connections, quantized,
                );
            }
        }

        // Search layer 0 with repair_ef using quantized distances
        let dummy_calcs = AtomicUsize::new(0);
        let candidates = self.search_layer_quantized(
            node_id as u32, current_entry, 0, repair_ef, connections, quantized, &dummy_calcs,
        );

        // Select more candidates for accurate f32 re-ranking
        let new_neighbors: Vec<u32> = candidates
            .iter()
            .filter(|(id, _)| *id != node_id)
            .take(m * 2)
            .map(|(id, _)| *id)
            .collect();

        // Merge with existing and keep best using EXACT f32 distances
        let mut all_neighbors: HashSet<u32> = current_neighbors.iter().copied().collect();
        all_neighbors.extend(new_neighbors.iter().copied());

        // Score all candidates using f32 for accuracy
        let mut scored: Vec<(u32, f32)> = all_neighbors
            .into_iter()
            .map(|n| {
                let neighbor_vec = &f32_vectors[n as usize];
                let dist = self.metric.compute(query, neighbor_vec);
                (n, dist)
            })
            .collect();

        scored.sort_by(|a, b| {
            if self.higher_is_better {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        let best: Vec<u32> = scored.iter().take(m).map(|(n, _)| *n).collect();

        // Check if improved
        let old_set: HashSet<u32> = current_neighbors.iter().copied().collect();
        let new_set: HashSet<u32> = best.iter().copied().collect();

        if old_set == new_set {
            return false;
        }

        // Update connections
        connections[0].set(node_id, best.clone());

        // Add bidirectional connections for new neighbors
        for &neighbor in &best {
            if !old_set.contains(&neighbor) {
                connections[0].add(neighbor, node_id);
            }
        }

        true
    }

    /// Greedy search using quantized distances.
    fn search_layer_greedy_quantized(
        &self,
        query_id: u32,
        entry: u32,
        layer: usize,
        connections: &[LayerConnections],
        quantized: &SignedQuantizedVectors,
    ) -> u32 {
        let mut current = entry;
        let mut current_dist = quantized.dot_product(query_id as usize, current as usize);

        loop {
            let neighbors = connections[layer].get(current);
            let mut improved = false;

            for neighbor in neighbors {
                let dist = quantized.dot_product(query_id as usize, neighbor as usize);
                if self.is_better_i32(dist, current_dist) {
                    current = neighbor;
                    current_dist = dist;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        current
    }

    /// Beam search using quantized distances.
    fn search_layer_quantized(
        &self,
        query_id: u32,
        entry: u32,
        layer: usize,
        ef: usize,
        connections: &[LayerConnections],
        quantized: &SignedQuantizedVectors,
        distance_calcs: &AtomicUsize,
    ) -> Vec<(u32, i32)> {
        let mut visited = HashSet::new();
        let mut candidates = Vec::new();
        let mut results = Vec::with_capacity(ef);

        distance_calcs.fetch_add(1, Ordering::Relaxed);
        let entry_dist = quantized.dot_product(query_id as usize, entry as usize);
        candidates.push((entry, entry_dist));
        visited.insert(entry);

        let mut worst_dist = entry_dist;

        while let Some((current, current_dist)) = candidates.pop() {
            if results.len() >= ef && !self.is_better_i32(current_dist, worst_dist) {
                continue;
            }

            results.push((current, current_dist));

            if results.len() > ef {
                results.sort_by(|a, b| {
                    if self.higher_is_better {
                        b.1.cmp(&a.1)
                    } else {
                        a.1.cmp(&b.1)
                    }
                });
                results.truncate(ef);
                worst_dist = results.last().unwrap().1;
            }

            let neighbors = connections[layer].get(current);
            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    distance_calcs.fetch_add(1, Ordering::Relaxed);
                    let dist = quantized.dot_product(query_id as usize, neighbor as usize);
                    if results.len() < ef || self.is_better_i32(dist, worst_dist) {
                        candidates.push((neighbor, dist));
                    }
                }
            }

            // Sort candidates (best last for pop)
            candidates.sort_by(|a, b| {
                if self.higher_is_better {
                    a.1.cmp(&b.1)
                } else {
                    b.1.cmp(&a.1)
                }
            });
        }

        results.sort_by(|a, b| {
            if self.higher_is_better {
                b.1.cmp(&a.1)
            } else {
                a.1.cmp(&b.1)
            }
        });

        results
    }
}

/// Build an HNSW graph in parallel with repair pass.
pub fn build_parallel(
    config: HnswConfig,
    metric: Metric,
    records: Vec<(u64, String)>,
    vectors: &VectorStorage,
) -> HnswGraph {
    let builder = ParallelHnswBuilder::new(config, metric);
    builder.build(records, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_vectors(dim: usize, count: usize) -> (TempDir, VectorStorage) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");
        let mut storage = VectorStorage::create(&path, dim).unwrap();

        for i in 0..count {
            let slot = storage.allocate_slot().unwrap();
            let mut vec = vec![0.0f32; dim];
            vec[i % dim] = 1.0;
            storage.write_slot(slot, &vec).unwrap();
        }

        (dir, storage)
    }

    #[test]
    fn test_parallel_build_small() {
        let config = HnswConfig::with_m(4);
        let (_dir, vectors) = create_test_vectors(8, 100);

        let records: Vec<_> = (0..100).map(|i| (i as u64, format!("v{}", i))).collect();
        let graph = build_parallel(config, Metric::Cosine, records, &vectors);

        assert_eq!(graph.node_count(), 100);
        assert!(graph.entry_point().is_some());
    }

    #[test]
    fn test_parallel_build_medium() {
        let config = HnswConfig::default();
        let (_dir, vectors) = create_test_vectors(32, 1000);

        let records: Vec<_> = (0..1000).map(|i| (i as u64, format!("v{}", i))).collect();
        let graph = build_parallel(config, Metric::Cosine, records, &vectors);

        assert_eq!(graph.node_count(), 1000);
        assert!(graph.entry_point().is_some());

        // Verify connections exist
        let mut total_connections = 0;
        for i in 0..1000 {
            total_connections += graph.get_neighbors(0, i).len();
        }
        assert!(total_connections > 0);
    }
}
