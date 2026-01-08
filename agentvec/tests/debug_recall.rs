//! Debug test to understand low recall issue

use agentvec::{Collection, CollectionConfig, HnswConfig, Metric, WriteConfig};
use serde_json::json;
use tempfile::tempdir;

fn generate_vectors(count: usize, dim: usize, seed: usize) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..count)
        .map(|i| {
            let mut vec: Vec<f32> = (0..dim)
                .map(|j| {
                    let mut hasher = DefaultHasher::new();
                    (seed * 1000000 + i * dim + j).hash(&mut hasher);
                    let h = hasher.finish();
                    ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
                })
                .collect();

            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                vec.iter_mut().for_each(|x| *x /= norm);
            }
            vec
        })
        .collect()
}

#[test]
fn debug_exact_vs_hnsw_detailed() {
    let dir = tempdir().unwrap();
    let num_vectors = 1000;
    let dimensions = 384;

    println!("\n=== Detailed Exact vs HNSW Comparison ===\n");

    let vectors = generate_vectors(num_vectors, dimensions, 42);

    // Create collections
    let exact_config = CollectionConfig::new("exact", dimensions, Metric::Cosine);
    let exact_col = Collection::open(dir.path().join("exact"), exact_config).unwrap();

    let hnsw_config = HnswConfig::default();
    let hnsw_col_config = CollectionConfig::with_hnsw("hnsw", dimensions, Metric::Cosine, hnsw_config);
    let hnsw_col = Collection::open(dir.path().join("hnsw"), hnsw_col_config).unwrap();

    // Insert with consistent IDs
    for (i, v) in vectors.iter().enumerate() {
        exact_col.add(v, json!({"i": i}), Some(&format!("v{}", i)), None).unwrap();
        hnsw_col.add(v, json!({"i": i}), Some(&format!("v{}", i)), None).unwrap();
    }
    exact_col.sync().unwrap();
    hnsw_col.sync().unwrap();

    println!("Inserted {} vectors", num_vectors);
    println!("HNSW has {} nodes", hnsw_col.hnsw_node_count().unwrap_or(0));

    // Test with query = v0
    let query = &vectors[0];

    let exact_results = exact_col.search(query, 10, None).unwrap();
    let hnsw_results = hnsw_col.search(query, 10, None).unwrap();

    println!("\nQuery: v0 (searching for itself and similar)");
    println!("\nExact top-10:");
    for r in &exact_results {
        println!("  {} : {:.4}", r.id, r.score);
    }

    println!("\nHNSW top-10:");
    for r in &hnsw_results {
        println!("  {} : {:.4}", r.id, r.score);
    }

    // Check overlap
    let exact_ids: std::collections::HashSet<_> = exact_results.iter().map(|r| &r.id).collect();
    let hnsw_ids: std::collections::HashSet<_> = hnsw_results.iter().map(|r| &r.id).collect();
    let overlap = exact_ids.intersection(&hnsw_ids).count();

    println!("\nOverlap: {}/10", overlap);

    // Check if the scores are at least in the right ballpark
    let exact_best = exact_results[0].score;
    let hnsw_best = hnsw_results[0].score;
    println!("\nBest scores: Exact={:.4}, HNSW={:.4}", exact_best, hnsw_best);

    // If HNSW is finding v0 as best, it's working correctly
    let hnsw_found_v0 = hnsw_results[0].id == "v0";
    println!("HNSW found v0 as best: {}", hnsw_found_v0);

    assert!(hnsw_found_v0, "HNSW should find the query vector itself");
    assert!(overlap >= 5, "Should have at least 50% overlap, got {}/10", overlap);
}

#[test]
fn debug_hnsw_graph_quality() {
    use agentvec::search::hnsw::HnswGraph;

    let dir = tempdir().unwrap();
    let num_vectors = 100;
    let dimensions = 384;

    println!("\n=== HNSW Graph Quality Check ===\n");

    let vectors = generate_vectors(num_vectors, dimensions, 42);

    // Create HNSW graph directly
    let hnsw_config = HnswConfig::default();
    let vectors_path = dir.path().join("vectors.bin");
    let mut storage = agentvec::storage::VectorStorage::create(&vectors_path, dimensions).unwrap();

    let mut graph = HnswGraph::new(hnsw_config);

    // Insert vectors
    for (i, v) in vectors.iter().enumerate() {
        let slot = storage.allocate_slot().unwrap();
        storage.write_slot(slot, v).unwrap();
        graph.insert(slot, format!("v{}", i), &storage, Metric::Cosine);
    }

    println!("Graph stats:");
    println!("  Nodes: {}", graph.node_count());
    println!("  Entry point: {:?}", graph.entry_point());
    println!("  Max layer: {}", graph.max_layer());

    // Check connectivity at layer 0
    let mut total_neighbors = 0;
    let mut min_neighbors = usize::MAX;
    let mut max_neighbors = 0;
    for i in 0..graph.node_count() {
        let neighbors = graph.get_neighbors(0, i as u32);
        total_neighbors += neighbors.len();
        min_neighbors = min_neighbors.min(neighbors.len());
        max_neighbors = max_neighbors.max(neighbors.len());
    }
    let avg_neighbors = total_neighbors as f64 / graph.node_count() as f64;

    println!("\nLayer 0 connectivity:");
    println!("  Min neighbors: {}", min_neighbors);
    println!("  Max neighbors: {}", max_neighbors);
    println!("  Avg neighbors: {:.1}", avg_neighbors);

    // Test search
    let query = &vectors[0];
    let deleted = std::collections::HashSet::new();
    let results = graph.search(query, 10, 50, &storage, Metric::Cosine, &deleted);

    println!("\nSearch for v0:");
    for (node_id, score) in &results {
        let node = graph.get_node(*node_id).unwrap();
        println!("  {} : {:.4}", node.id, score);
    }

    // v0 should be first with score ~1.0
    assert!(!results.is_empty(), "Should have results");
    let (first_node, first_score) = &results[0];
    let first_id = &graph.get_node(*first_node).unwrap().id;
    println!("\nFirst result: {} with score {:.4}", first_id, first_score);

    assert_eq!(first_id, "v0", "First result should be v0");
    assert!(*first_score > 0.99, "Score should be ~1.0, got {}", first_score);
}
