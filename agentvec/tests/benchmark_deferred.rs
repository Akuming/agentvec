//! Benchmark showing deferred HNSW building benefits
//!
//! This demonstrates that inserts are now fast (like exact search),
//! with HNSW building done once at the end.

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
fn benchmark_deferred_building() {
    let dir = tempdir().unwrap();
    let num_vectors = 10_000;
    let dimensions = 384;
    let k = 10;

    println!("\n=== Deferred HNSW Building Benchmark ===\n");

    let vectors = generate_vectors(num_vectors, dimensions, 42);
    let queries: Vec<Vec<f32>> = vectors.iter().take(100).cloned().collect();

    // Create HNSW collection
    let hnsw_config = HnswConfig::default();
    let hnsw_col_config = CollectionConfig::with_hnsw("hnsw", dimensions, Metric::Cosine, hnsw_config);
    let hnsw_col = Collection::open_with_write_config(
        dir.path().join("hnsw"),
        hnsw_col_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Phase 1: Fast batch inserts (WITHOUT HNSW overhead)
    println!("Phase 1: Batch inserts (deferred HNSW)...");
    let insert_start = Instant::now();
    for batch_start in (0..num_vectors).step_by(1000) {
        let batch_end = (batch_start + 1000).min(num_vectors);
        let batch_vecs: Vec<&[f32]> = vectors[batch_start..batch_end]
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let batch_metas: Vec<_> = (batch_start..batch_end).map(|i| json!({"i": i})).collect();
        let batch_ids: Vec<String> = (batch_start..batch_end).map(|i| format!("v{}", i)).collect();
        let batch_id_refs: Vec<&str> = batch_ids.iter().map(|s| s.as_str()).collect();
        hnsw_col.add_batch(&batch_vecs, &batch_metas, Some(&batch_id_refs), None).unwrap();
    }
    let insert_time = insert_start.elapsed();
    println!("  Insert time: {:.2}s ({:.0} vec/s)",
        insert_time.as_secs_f64(),
        num_vectors as f64 / insert_time.as_secs_f64());

    // Verify HNSW is not built yet
    assert!(hnsw_col.hnsw_is_dirty(), "HNSW should be marked dirty");
    println!("  HNSW index is dirty (not built yet)");

    // Phase 2: HNSW build (during sync or first search)
    println!("\nPhase 2: HNSW index build (during sync)...");
    let build_start = Instant::now();
    hnsw_col.sync().unwrap();
    let build_time = build_start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f64());
    println!("  HNSW node count: {}", hnsw_col.hnsw_node_count().unwrap_or(0));

    // Phase 3: Fast searches
    println!("\nPhase 3: Search performance...");
    let search_start = Instant::now();
    let mut results = Vec::with_capacity(100);
    for q in &queries {
        results.push(hnsw_col.search(q, k, None).unwrap());
    }
    let search_time = search_start.elapsed();
    let latency_ms = search_time.as_secs_f64() * 1000.0 / 100.0;
    println!("  Search latency: {:.2}ms/query ({:.0} QPS)",
        latency_ms,
        1000.0 / latency_ms);

    // Summary
    println!("\n=== Summary ===");
    println!("Insert rate:       {:.0} vec/s (excluding HNSW build)",
        num_vectors as f64 / insert_time.as_secs_f64());
    println!("HNSW build:        {:.2}s (one-time cost)", build_time.as_secs_f64());
    println!("Search latency:    {:.2}ms", latency_ms);
    println!("Total throughput:  {:.0} vec/s (including build)",
        num_vectors as f64 / (insert_time.as_secs_f64() + build_time.as_secs_f64()));

    // PRD check
    println!("\nPRD Targets:");
    let insert_pass = num_vectors as f64 / insert_time.as_secs_f64() > 10_000.0;
    let search_pass = latency_ms < 10.0;
    println!("  Insert >10K vec/s: {}", if insert_pass { "PASS" } else { "FAIL" });
    println!("  Search <10ms:      {}", if search_pass { "PASS" } else { "FAIL" });
}
