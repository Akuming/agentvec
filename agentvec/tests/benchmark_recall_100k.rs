//! Focused benchmark testing recall at 100K scale with improved config

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
fn benchmark_recall_100k() {
    let dir = tempdir().unwrap();
    let num_vectors = 250_000;
    let dimensions = 384;
    let k = 10;
    let num_queries = 100;

    println!("\n=== 250K Recall Benchmark ===\n");

    // Generate vectors
    println!("Generating {} vectors...", num_vectors);
    let gen_start = Instant::now();
    let vectors = generate_vectors(num_vectors, dimensions, 42);
    let queries: Vec<Vec<f32>> = vectors.iter().take(num_queries).cloned().collect();
    println!("Generated in {:.2}s\n", gen_start.elapsed().as_secs_f64());

    // Create exact search collection for ground truth
    println!("Creating exact search collection...");
    let exact_config = CollectionConfig::new("exact", dimensions, Metric::Cosine);
    let exact_col = Collection::open_with_write_config(
        dir.path().join("exact"),
        exact_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Create HNSW collection with PQ-optimized config (higher ef for PQ approximation)
    println!("Creating HNSW collection with PQ-optimized config...");
    let hnsw_config = HnswConfig::pq_optimized();
    println!("  M={}, ef_construction={}, ef_search={}",
        hnsw_config.m, hnsw_config.ef_construction, hnsw_config.ef_search);

    let hnsw_col_config = CollectionConfig::with_hnsw("hnsw", dimensions, Metric::Cosine, hnsw_config);
    let hnsw_col = Collection::open_with_write_config(
        dir.path().join("hnsw"),
        hnsw_col_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Insert vectors
    println!("\nInserting {} vectors...", num_vectors);
    let insert_start = Instant::now();
    for batch_start in (0..num_vectors).step_by(5000) {
        let batch_end = (batch_start + 5000).min(num_vectors);
        let batch_vecs: Vec<&[f32]> = vectors[batch_start..batch_end]
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let batch_metas: Vec<_> = (batch_start..batch_end).map(|i| json!({"i": i})).collect();
        let batch_ids: Vec<String> = (batch_start..batch_end).map(|i| format!("v{}", i)).collect();
        let batch_id_refs: Vec<&str> = batch_ids.iter().map(|s| s.as_str()).collect();

        exact_col.add_batch(&batch_vecs, &batch_metas, Some(&batch_id_refs), None).unwrap();
        hnsw_col.add_batch(&batch_vecs, &batch_metas, Some(&batch_id_refs), None).unwrap();

        if (batch_end % 20000) == 0 || batch_end == num_vectors {
            print!("\r  Inserted {}/{}", batch_end, num_vectors);
        }
    }
    println!();

    // Sync both
    println!("Syncing exact collection...");
    exact_col.sync().unwrap();

    println!("Building HNSW index (this takes a while)...");
    let build_start = Instant::now();
    hnsw_col.sync().unwrap();
    let build_time = build_start.elapsed();
    println!("  HNSW build: {:.1}s", build_time.as_secs_f64());

    let insert_time = insert_start.elapsed();
    println!("Total insert+build: {:.1}s\n", insert_time.as_secs_f64());

    // Benchmark exact search
    println!("Running {} exact search queries...", num_queries);
    let exact_start = Instant::now();
    let mut exact_results = Vec::with_capacity(num_queries);
    for q in &queries {
        exact_results.push(exact_col.search(q, k, None).unwrap());
    }
    let exact_time = exact_start.elapsed();
    let exact_latency_ms = exact_time.as_secs_f64() * 1000.0 / num_queries as f64;

    // Benchmark HNSW search
    println!("Running {} HNSW search queries...", num_queries);
    let hnsw_start = Instant::now();
    let mut hnsw_results = Vec::with_capacity(num_queries);
    for q in &queries {
        hnsw_results.push(hnsw_col.search(q, k, None).unwrap());
    }
    let hnsw_time = hnsw_start.elapsed();
    let hnsw_latency_ms = hnsw_time.as_secs_f64() * 1000.0 / num_queries as f64;

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

    // Results
    println!("\n=== Results @ 250K vectors ===");
    println!("Exact search:  {:.2}ms/query", exact_latency_ms);
    println!("HNSW search:   {:.2}ms/query", hnsw_latency_ms);
    println!("Speedup:       {:.1}x", exact_latency_ms / hnsw_latency_ms);
    println!("Recall@{}:     {:.1}%", k, avg_recall * 100.0);
    println!();

    // PRD compliance
    let latency_pass = hnsw_latency_ms < 50.0;
    let recall_pass = avg_recall >= 0.95;

    println!("PRD Compliance:");
    println!("  Latency <50ms: {} ({:.2}ms)", if latency_pass { "PASS" } else { "FAIL" }, hnsw_latency_ms);
    println!("  Recall >=95%:  {} ({:.1}%)", if recall_pass { "PASS" } else { "FAIL" }, avg_recall * 100.0);
    println!();

    if latency_pass && recall_pass {
        println!("Result: ALL TARGETS MET");
    } else {
        println!("Result: SOME TARGETS NOT MET");
    }

    assert!(avg_recall >= 0.90, "Recall should be >= 90%, got {:.1}%", avg_recall * 100.0);
}
