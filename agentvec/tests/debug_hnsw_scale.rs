//! Debug test for HNSW at scale

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
fn debug_hnsw_at_1000_vectors() {
    let dir = tempdir().unwrap();

    let hnsw_config = HnswConfig::high_recall();
    let config = CollectionConfig::with_hnsw("test", 128, Metric::Cosine, hnsw_config);
    let col = Collection::open_with_write_config(
        dir.path().join("test"),
        config,
        WriteConfig::throughput(),
    ).unwrap();

    let vectors = generate_vectors(1000, 128, 42);

    println!("\nInserting 1000 vectors using add()...");
    for (i, v) in vectors.iter().enumerate() {
        col.add(v, json!({"i": i}), Some(&format!("v{}", i)), None).unwrap();
    }
    col.sync().unwrap();

    println!("Collection has {} records", col.len().unwrap());
    println!("Has HNSW index: {}", col.has_hnsw_index());
    if col.has_hnsw_index() {
        println!("HNSW node count: {:?}", col.hnsw_node_count());
    }

    // Search
    let query = &vectors[0];
    let results = col.search(query, 10, None).unwrap();

    println!("\nSearch for v0:");
    println!("Got {} results", results.len());
    for r in &results {
        println!("  {} : {:.4}", r.id, r.score);
    }

    assert!(!results.is_empty(), "Should have results");
    assert_eq!(results[0].id, "v0", "First result should be v0");
}

#[test]
fn debug_hnsw_batch_insert() {
    let dir = tempdir().unwrap();

    let hnsw_config = HnswConfig::high_recall();
    let config = CollectionConfig::with_hnsw("test", 128, Metric::Cosine, hnsw_config);
    let col = Collection::open_with_write_config(
        dir.path().join("test"),
        config,
        WriteConfig::throughput(),
    ).unwrap();

    let vectors = generate_vectors(1000, 128, 42);

    println!("\nInserting 1000 vectors using add_batch()...");
    let batch_vecs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let batch_metas: Vec<_> = (0..1000).map(|i| json!({"i": i})).collect();
    let batch_ids: Vec<String> = (0..1000).map(|i| format!("v{}", i)).collect();
    let batch_id_refs: Vec<&str> = batch_ids.iter().map(|s| s.as_str()).collect();

    col.add_batch(&batch_vecs, &batch_metas, Some(&batch_id_refs), None).unwrap();
    col.sync().unwrap();

    println!("Collection has {} records", col.len().unwrap());
    println!("Has HNSW index: {}", col.has_hnsw_index());
    if col.has_hnsw_index() {
        println!("HNSW node count: {:?}", col.hnsw_node_count());
    }

    // Search
    let query = &vectors[0];
    let results = col.search(query, 10, None).unwrap();

    println!("\nSearch for v0:");
    println!("Got {} results", results.len());
    for r in &results {
        println!("  {} : {:.4}", r.id, r.score);
    }

    assert!(!results.is_empty(), "Should have results");
    assert_eq!(results[0].id, "v0", "First result should be v0");
}

#[test]
fn debug_hnsw_10k_batch() {
    let dir = tempdir().unwrap();

    let hnsw_config = HnswConfig::high_recall();
    let config = CollectionConfig::with_hnsw("test", 384, Metric::Cosine, hnsw_config);
    let col = Collection::open_with_write_config(
        dir.path().join("test"),
        config,
        WriteConfig::throughput(),
    ).unwrap();

    let vectors = generate_vectors(10000, 384, 42);

    println!("\nInserting 10000 vectors in batches of 1000...");
    for batch_start in (0..10000).step_by(1000) {
        let batch_end = batch_start + 1000;
        let batch_vecs: Vec<&[f32]> = vectors[batch_start..batch_end]
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let batch_metas: Vec<_> = (batch_start..batch_end).map(|i| json!({"i": i})).collect();

        col.add_batch(&batch_vecs, &batch_metas, None, None).unwrap();
        println!("  Inserted batch {}-{}", batch_start, batch_end);
    }
    col.sync().unwrap();

    println!("\nCollection has {} records", col.len().unwrap());
    println!("Has HNSW index: {}", col.has_hnsw_index());
    if col.has_hnsw_index() {
        println!("HNSW node count: {:?}", col.hnsw_node_count());
    }

    // Search
    let query = &vectors[0];
    let results = col.search(query, 10, None).unwrap();

    println!("\nSearch for vector 0:");
    println!("Got {} results", results.len());
    for r in &results {
        println!("  {} : {:.4}", r.id, r.score);
    }

    // Verify we get the correct first result
    assert!(!results.is_empty(), "Should have results");
    assert!(results[0].score > 0.99, "First result should have high score (got {})", results[0].score);
}
