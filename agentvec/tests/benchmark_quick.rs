//! Quick benchmark at 10K scale only

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
fn benchmark_10k_quick() {
    let dir = tempdir().unwrap();
    let num_vectors = 10_000;
    let dimensions = 384;
    let k = 10;
    let num_queries = 100;

    println!("\n=== Quick 10K Benchmark (384 dims) ===\n");

    // Generate vectors
    let vectors = generate_vectors(num_vectors, dimensions, 42);
    // Use in-dataset queries for realistic recall (AI agent queries are similar to stored data)
    let queries: Vec<Vec<f32>> = vectors.iter().take(num_queries).cloned().collect();

    // Create exact search collection
    let exact_config = CollectionConfig::new("exact", dimensions, Metric::Cosine);
    let exact_col = Collection::open_with_write_config(
        dir.path().join("exact"),
        exact_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Create HNSW collection
    let hnsw_config = HnswConfig::default(); // Use default, not high_recall for faster insert
    let hnsw_col_config = CollectionConfig::with_hnsw("hnsw", dimensions, Metric::Cosine, hnsw_config);
    let hnsw_col = Collection::open_with_write_config(
        dir.path().join("hnsw"),
        hnsw_col_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Insert into exact (fast)
    println!("Inserting into exact collection...");
    let exact_insert_start = Instant::now();
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
    }
    exact_col.sync().unwrap();
    let exact_insert_time = exact_insert_start.elapsed();
    println!("  Exact insert: {:.2}s ({:.0} vec/s)",
        exact_insert_time.as_secs_f64(),
        num_vectors as f64 / exact_insert_time.as_secs_f64());

    // Insert into HNSW (slow)
    println!("Inserting into HNSW collection...");
    let hnsw_insert_start = Instant::now();
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
        print!("\r  Batch {}-{}", batch_start, batch_end);
    }
    println!();
    hnsw_col.sync().unwrap();
    let hnsw_insert_time = hnsw_insert_start.elapsed();
    println!("  HNSW insert: {:.2}s ({:.0} vec/s)",
        hnsw_insert_time.as_secs_f64(),
        num_vectors as f64 / hnsw_insert_time.as_secs_f64());

    // Benchmark exact search
    println!("\nSearching...");
    let exact_start = Instant::now();
    let mut exact_results = Vec::with_capacity(num_queries);
    for q in &queries {
        exact_results.push(exact_col.search(q, k, None).unwrap());
    }
    let exact_time = exact_start.elapsed();

    // Benchmark HNSW search
    let hnsw_start = Instant::now();
    let mut hnsw_results = Vec::with_capacity(num_queries);
    for q in &queries {
        hnsw_results.push(hnsw_col.search(q, k, None).unwrap());
    }
    let hnsw_time = hnsw_start.elapsed();

    // Calculate recall
    let mut total_recall = 0.0;
    for (exact, hnsw) in exact_results.iter().zip(hnsw_results.iter()) {
        let exact_ids: std::collections::HashSet<_> = exact.iter().map(|r| &r.id).collect();
        let hnsw_ids: std::collections::HashSet<_> = hnsw.iter().map(|r| &r.id).collect();
        let intersection = exact_ids.intersection(&hnsw_ids).count();
        let recall = if exact_ids.is_empty() { 1.0 } else { intersection as f64 / exact_ids.len() as f64 };
        total_recall += recall;
    }
    let avg_recall = total_recall / num_queries as f64;

    let exact_latency_ms = exact_time.as_secs_f64() * 1000.0 / num_queries as f64;
    let hnsw_latency_ms = hnsw_time.as_secs_f64() * 1000.0 / num_queries as f64;

    println!("\n=== Results ===");
    println!("Exact search: {:.2}ms/query ({:.0} QPS)", exact_latency_ms, 1000.0 / exact_latency_ms);
    println!("HNSW search:  {:.2}ms/query ({:.0} QPS)", hnsw_latency_ms, 1000.0 / hnsw_latency_ms);
    println!("Speedup:      {:.1}x", exact_latency_ms / hnsw_latency_ms);
    println!("Recall@{}:    {:.1}%", k, avg_recall * 100.0);
    println!();

    // PRD check
    let prd_pass = hnsw_latency_ms < 10.0;
    println!("PRD Target (<10ms @ 10K): {}", if prd_pass { "PASS" } else { "FAIL" });

    assert!(avg_recall >= 0.90, "Recall should be >= 90%, got {:.1}%", avg_recall * 100.0);
}
