//! Debug test comparing in-dataset queries vs random queries

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
fn compare_in_dataset_vs_random_queries() {
    let dir = tempdir().unwrap();
    let num_vectors = 5000;
    let dimensions = 384;
    let k = 10;

    println!("\n=== In-Dataset vs Random Query Comparison ===\n");

    let vectors = generate_vectors(num_vectors, dimensions, 42);
    let random_queries = generate_vectors(100, dimensions, 123); // Different seed

    // Create collections
    let exact_config = CollectionConfig::new("exact", dimensions, Metric::Cosine);
    let exact_col = Collection::open(dir.path().join("exact"), exact_config).unwrap();

    let hnsw_config = HnswConfig::default();
    let hnsw_col_config = CollectionConfig::with_hnsw("hnsw", dimensions, Metric::Cosine, hnsw_config);
    let hnsw_col = Collection::open(dir.path().join("hnsw"), hnsw_col_config).unwrap();

    // Batch insert
    println!("Inserting {} vectors...", num_vectors);
    for batch_start in (0..num_vectors).step_by(1000) {
        let batch_end = (batch_start + 1000).min(num_vectors);
        let batch_vecs: Vec<&[f32]> = vectors[batch_start..batch_end]
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let batch_metas: Vec<_> = (batch_start..batch_end).map(|i| json!({"i": i})).collect();
        let batch_ids: Vec<String> = (batch_start..batch_end).map(|i| format!("v{}", i)).collect();
        let batch_id_refs: Vec<&str> = batch_ids.iter().map(|s| s.as_str()).collect();

        exact_col.add_batch(&batch_vecs, &batch_metas, Some(&batch_id_refs), None).unwrap();
        hnsw_col.add_batch(&batch_vecs, &batch_metas, Some(&batch_id_refs), None).unwrap();
    }
    exact_col.sync().unwrap();
    hnsw_col.sync().unwrap();

    println!("HNSW has {} nodes", hnsw_col.hnsw_node_count().unwrap_or(0));

    // Test 1: In-dataset queries (use first 50 vectors as queries)
    println!("\n--- Test 1: In-Dataset Queries ---");
    let mut in_dataset_recall = 0.0;
    for i in 0..50 {
        let query = &vectors[i];
        let exact_results = exact_col.search(query, k, None).unwrap();
        let hnsw_results = hnsw_col.search(query, k, None).unwrap();

        let exact_ids: std::collections::HashSet<_> = exact_results.iter().map(|r| &r.id).collect();
        let hnsw_ids: std::collections::HashSet<_> = hnsw_results.iter().map(|r| &r.id).collect();
        let overlap = exact_ids.intersection(&hnsw_ids).count();
        in_dataset_recall += overlap as f64 / k as f64;
    }
    in_dataset_recall /= 50.0;
    println!("Average recall: {:.1}%", in_dataset_recall * 100.0);

    // Test 2: Random queries (completely different vectors)
    println!("\n--- Test 2: Random Queries ---");
    let mut random_recall = 0.0;
    for i in 0..50 {
        let query = &random_queries[i];
        let exact_results = exact_col.search(query, k, None).unwrap();
        let hnsw_results = hnsw_col.search(query, k, None).unwrap();

        let exact_ids: std::collections::HashSet<_> = exact_results.iter().map(|r| &r.id).collect();
        let hnsw_ids: std::collections::HashSet<_> = hnsw_results.iter().map(|r| &r.id).collect();
        let overlap = exact_ids.intersection(&hnsw_ids).count();
        random_recall += overlap as f64 / k as f64;
    }
    random_recall /= 50.0;
    println!("Average recall: {:.1}%", random_recall * 100.0);

    // Show detailed comparison for first random query
    println!("\n--- Detailed: First Random Query ---");
    let query = &random_queries[0];
    let exact_results = exact_col.search(query, k, None).unwrap();
    let hnsw_results = hnsw_col.search(query, k, None).unwrap();

    println!("Exact top-10:");
    for r in &exact_results {
        println!("  {} : {:.4}", r.id, r.score);
    }

    println!("\nHNSW top-10:");
    for r in &hnsw_results {
        println!("  {} : {:.4}", r.id, r.score);
    }

    let exact_ids: std::collections::HashSet<_> = exact_results.iter().map(|r| &r.id).collect();
    let hnsw_ids: std::collections::HashSet<_> = hnsw_results.iter().map(|r| &r.id).collect();
    let overlap = exact_ids.intersection(&hnsw_ids).count();
    println!("\nOverlap: {}/{}", overlap, k);

    println!("\n=== Summary ===");
    println!("In-dataset query recall: {:.1}%", in_dataset_recall * 100.0);
    println!("Random query recall: {:.1}%", random_recall * 100.0);
}
