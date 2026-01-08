//! Benchmark comparing incremental insertion vs full rebuild

use std::time::Instant;

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
fn benchmark_incremental_insertion() {
    let dir = tempdir().unwrap();
    let initial_vectors = 50_000;
    let additional_vectors = 10_000;
    let dimensions = 384;

    println!("\n=== Incremental Insertion Benchmark ===\n");
    println!("Initial: {} vectors, Additional: {} vectors", initial_vectors, additional_vectors);

    // Generate all vectors upfront
    let all_vectors = generate_vectors(initial_vectors + additional_vectors, dimensions, 42);

    // Create collection with HNSW
    let hnsw_config = HnswConfig::default();
    let col_config = CollectionConfig::with_hnsw("test", dimensions, Metric::Cosine, hnsw_config);
    let col = Collection::open_with_write_config(
        dir.path().join("col"),
        col_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Insert initial vectors
    println!("Inserting initial {} vectors...", initial_vectors);
    let insert_start = Instant::now();
    for batch_start in (0..initial_vectors).step_by(5000) {
        let batch_end = (batch_start + 5000).min(initial_vectors);
        let batch_vecs: Vec<&[f32]> = all_vectors[batch_start..batch_end]
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let batch_metas: Vec<_> = (batch_start..batch_end).map(|i| json!({"i": i})).collect();
        let batch_ids: Vec<String> = (batch_start..batch_end).map(|i| format!("v{}", i)).collect();
        let batch_id_refs: Vec<&str> = batch_ids.iter().map(|s| s.as_str()).collect();
        col.add_batch(&batch_vecs, &batch_metas, Some(&batch_id_refs), None).unwrap();
    }

    // Build initial index
    println!("Building initial HNSW index...");
    let build_start = Instant::now();
    col.sync().unwrap();
    let initial_build_time = build_start.elapsed();
    let total_insert_time = insert_start.elapsed();

    println!("  Initial build: {:.2}s ({:.0} vec/s)",
        initial_build_time.as_secs_f64(),
        initial_vectors as f64 / initial_build_time.as_secs_f64());

    // Verify initial index
    let query = &all_vectors[0];
    let results = col.search(query, 10, None).unwrap();
    assert!(!results.is_empty(), "Search should return results");
    println!("  Initial search working: {} results", results.len());

    // Now add additional vectors
    println!("\nAdding {} additional vectors...", additional_vectors);
    let add_start = Instant::now();
    for batch_start in (initial_vectors..initial_vectors + additional_vectors).step_by(5000) {
        let batch_end = (batch_start + 5000).min(initial_vectors + additional_vectors);
        let batch_vecs: Vec<&[f32]> = all_vectors[batch_start..batch_end]
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let batch_metas: Vec<_> = (batch_start..batch_end).map(|i| json!({"i": i})).collect();
        let batch_ids: Vec<String> = (batch_start..batch_end).map(|i| format!("v{}", i)).collect();
        let batch_id_refs: Vec<&str> = batch_ids.iter().map(|s| s.as_str()).collect();
        col.add_batch(&batch_vecs, &batch_metas, Some(&batch_id_refs), None).unwrap();
    }

    // Sync with incremental insertion
    println!("Syncing with incremental insertion...");
    let incremental_start = Instant::now();
    col.sync().unwrap();
    let incremental_time = incremental_start.elapsed();
    let add_total_time = add_start.elapsed();

    println!("  Incremental update: {:.2}s ({:.0} vec/s)",
        incremental_time.as_secs_f64(),
        additional_vectors as f64 / incremental_time.as_secs_f64());

    // Verify search still works
    let results = col.search(query, 10, None).unwrap();
    assert!(!results.is_empty(), "Search should return results after incremental update");

    // Calculate comparison metrics
    let full_rebuild_estimate = initial_build_time.as_secs_f64() *
        ((initial_vectors + additional_vectors) as f64 / initial_vectors as f64).powf(1.5);

    println!("\n=== Results ===");
    println!("Initial {} vectors:", initial_vectors);
    println!("  Build time: {:.2}s ({:.0} vec/s)",
        initial_build_time.as_secs_f64(),
        initial_vectors as f64 / initial_build_time.as_secs_f64());

    println!("\nAdditional {} vectors:", additional_vectors);
    println!("  Incremental time: {:.2}s ({:.0} vec/s)",
        incremental_time.as_secs_f64(),
        additional_vectors as f64 / incremental_time.as_secs_f64());
    println!("  Est. full rebuild: {:.2}s", full_rebuild_estimate);
    println!("  Speedup: {:.1}x", full_rebuild_estimate / incremental_time.as_secs_f64());

    // The incremental insertion should be much faster than full rebuild
    // For 10K additional vectors, incremental should be < 10s
    assert!(
        incremental_time.as_secs_f64() < 30.0,
        "Incremental insertion should complete in < 30s, took {:.2}s",
        incremental_time.as_secs_f64()
    );
}

#[test]
fn test_incremental_recall() {
    let dir = tempdir().unwrap();
    let initial_vectors = 10_000;
    let additional_vectors = 1_000;
    let dimensions = 128;
    let k = 10;
    let num_queries = 50;

    println!("\n=== Incremental Insertion Recall Test ===\n");

    let all_vectors = generate_vectors(initial_vectors + additional_vectors, dimensions, 42);

    // Create exact search collection for ground truth
    let exact_config = CollectionConfig::new("exact", dimensions, Metric::Cosine);
    let exact_col = Collection::open_with_write_config(
        dir.path().join("exact"),
        exact_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Create HNSW collection
    let hnsw_config = HnswConfig::default();
    let hnsw_col_config = CollectionConfig::with_hnsw("hnsw", dimensions, Metric::Cosine, hnsw_config);
    let hnsw_col = Collection::open_with_write_config(
        dir.path().join("hnsw"),
        hnsw_col_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Insert initial vectors into both
    for batch_start in (0..initial_vectors).step_by(1000) {
        let batch_end = (batch_start + 1000).min(initial_vectors);
        let batch_vecs: Vec<&[f32]> = all_vectors[batch_start..batch_end]
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

    // Add additional vectors (triggers incremental insertion)
    for batch_start in (initial_vectors..initial_vectors + additional_vectors).step_by(1000) {
        let batch_end = (batch_start + 1000).min(initial_vectors + additional_vectors);
        let batch_vecs: Vec<&[f32]> = all_vectors[batch_start..batch_end]
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

    // Test recall using queries from the additional vectors (harder test)
    let queries: Vec<_> = all_vectors[initial_vectors..initial_vectors + num_queries].iter().collect();

    let mut total_recall = 0.0;
    for query in &queries {
        let exact_results = exact_col.search(query, k, None).unwrap();
        let hnsw_results = hnsw_col.search(query, k, None).unwrap();

        let exact_ids: std::collections::HashSet<_> = exact_results.iter().map(|r| &r.id).collect();
        let hnsw_ids: std::collections::HashSet<_> = hnsw_results.iter().map(|r| &r.id).collect();
        let intersection = exact_ids.intersection(&hnsw_ids).count();
        let recall = if exact_ids.is_empty() { 1.0 } else { intersection as f64 / exact_ids.len() as f64 };
        total_recall += recall;
    }
    let avg_recall = total_recall / queries.len() as f64;

    println!("Recall after incremental insertion: {:.1}%", avg_recall * 100.0);

    assert!(
        avg_recall >= 0.85,
        "Recall should be >= 85% after incremental insertion, got {:.1}%",
        avg_recall * 100.0
    );
}
