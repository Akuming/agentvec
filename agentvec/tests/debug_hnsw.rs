//! Debug test for HNSW recall issue

use agentvec::{Collection, CollectionConfig, HnswConfig, Metric};
use serde_json::json;
use tempfile::tempdir;

#[test]
fn debug_hnsw_recall() {
    use agentvec::search::hnsw::{HnswGraph, HnswIndex};

    let dir = tempdir().unwrap();

    // Create HNSW graph directly for maximum debugging
    let hnsw_config = HnswConfig::with_m(4);
    let vectors_path = dir.path().join("vectors.bin");
    let mut storage = agentvec::storage::VectorStorage::create(&vectors_path, 4).unwrap();

    let mut graph = HnswGraph::new(hnsw_config);

    // Add 5 simple vectors
    let test_vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0], // v0 - query target
        vec![0.9, 0.1, 0.0, 0.0], // v1 - similar to v0
        vec![0.8, 0.2, 0.0, 0.0], // v2 - similar to v0
        vec![0.0, 1.0, 0.0, 0.0], // v3 - orthogonal
        vec![0.0, 0.0, 1.0, 0.0], // v4 - orthogonal
    ];

    for (i, v) in test_vectors.iter().enumerate() {
        // Normalize the vector
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = v.iter().map(|x| x / norm).collect();

        let slot = storage.allocate_slot().unwrap();
        storage.write_slot(slot, &normalized).unwrap();
        let node_id = graph.insert(slot, format!("v{}", i), &storage, Metric::Cosine);
        println!("Inserted v{} at slot {} -> node_id {}", i, slot, node_id);
    }

    println!("\nGraph has {} nodes, entry_point={:?}, max_layer={}",
        graph.node_count(), graph.entry_point(), graph.max_layer());

    // Print graph connectivity
    println!("\nGraph connectivity at layer 0:");
    for i in 0..graph.node_count() {
        let neighbors = graph.get_neighbors(0, i as u32);
        let node = graph.get_node(i as u32).unwrap();
        println!("  Node {} ({}): neighbors = {:?}", i, node.id, neighbors);
    }

    // Do a manual search using the graph
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let deleted = std::collections::HashSet::new();
    let results = graph.search(&query, 5, 100, &storage, Metric::Cosine, &deleted);

    println!("\nQuery: [1.0, 0.0, 0.0, 0.0]");
    println!("Search results (node_id, score):");
    for (node_id, score) in &results {
        let node = graph.get_node(*node_id).unwrap();
        println!("  node {} ({}): {:.4}", node_id, node.id, score);
    }

    // Check what v0's actual score is
    let v0_vec = storage.read_slot_ref(0).unwrap();
    let cosine = agentvec::search::distance::dot_product(&query, v0_vec);
    println!("\nManual cosine(query, v0) = {:.4}", cosine);

    // First result should be v0
    assert!(!results.is_empty(), "Should have results");
    let (first_node, first_score) = &results[0];
    let first_node_info = graph.get_node(*first_node).unwrap();
    println!("\nFirst result: node {} ({}) with score {:.4}", first_node, first_node_info.id, first_score);

    assert!(*first_score > 0.8, "First result should have high score, got {}", first_score);
}

#[test]
fn debug_exact_vs_hnsw() {
    let dir = tempdir().unwrap();

    // Create two collections - one exact, one HNSW
    let exact_config = CollectionConfig::new("exact", 4, Metric::Cosine);
    let exact_col = Collection::open(dir.path().join("exact"), exact_config).unwrap();

    let hnsw_config = HnswConfig::with_m(16);
    let hnsw_col_config = CollectionConfig::with_hnsw("hnsw", 4, Metric::Cosine, hnsw_config);
    let hnsw_col = Collection::open(dir.path().join("hnsw"), hnsw_col_config).unwrap();

    // Add same 5 vectors to both
    let vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.8, 0.2, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];

    for (i, v) in vectors.iter().enumerate() {
        exact_col.add(v, json!({"i": i}), Some(&format!("v{}", i)), None).unwrap();
        hnsw_col.add(v, json!({"i": i}), Some(&format!("v{}", i)), None).unwrap();
    }
    exact_col.sync().unwrap();
    hnsw_col.sync().unwrap();

    let query = vec![1.0, 0.0, 0.0, 0.0];

    let exact_results = exact_col.search(&query, 3, None).unwrap();
    let hnsw_results = hnsw_col.search(&query, 3, None).unwrap();

    println!("\n");
    println!("Query: [1.0, 0.0, 0.0, 0.0]");
    println!("\nExact search results:");
    for r in &exact_results {
        println!("  {} : {:.4}", r.id, r.score);
    }

    println!("\nHNSW search results:");
    for r in &hnsw_results {
        println!("  {} : {:.4}", r.id, r.score);
    }

    // Check if results match
    let exact_ids: Vec<&str> = exact_results.iter().map(|r| r.id.as_str()).collect();
    let hnsw_ids: Vec<&str> = hnsw_results.iter().map(|r| r.id.as_str()).collect();

    println!("\nExact IDs: {:?}", exact_ids);
    println!("HNSW IDs: {:?}", hnsw_ids);

    // At least the top result should match
    assert_eq!(exact_ids[0], hnsw_ids[0],
        "Top result should match. Exact: {:?}, HNSW: {:?}", exact_ids, hnsw_ids);
}
