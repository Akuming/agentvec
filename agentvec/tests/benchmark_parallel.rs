//! Benchmark comparing sequential vs parallel HNSW construction

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
fn benchmark_parallel_construction_10k() {
    let dir = tempdir().unwrap();
    let num_vectors = 10_000;
    let dimensions = 384;
    let k = 10;

    println!("\n=== Parallel Construction Benchmark (10K vectors, 384 dims) ===\n");
    println!("CPU threads: {}", rayon::current_num_threads());

    let vectors = generate_vectors(num_vectors, dimensions, 42);
    let queries: Vec<Vec<f32>> = vectors.iter().take(100).cloned().collect();

    // Create HNSW collection with production config
    let hnsw_config = HnswConfig::high_recall();
    println!("Config: M={}, ef_construction={}, ef_search={}",
        hnsw_config.m, hnsw_config.ef_construction, hnsw_config.ef_search);

    let hnsw_col_config = CollectionConfig::with_hnsw("hnsw", dimensions, Metric::Cosine, hnsw_config);
    let hnsw_col = Collection::open_with_write_config(
        dir.path().join("hnsw"),
        hnsw_col_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Phase 1: Fast batch inserts
    println!("\nPhase 1: Batch inserts...");
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

    // Phase 2: Parallel HNSW build
    println!("\nPhase 2: Parallel HNSW build (during sync)...");
    let build_start = Instant::now();
    hnsw_col.sync().unwrap();
    let build_time = build_start.elapsed();
    println!("  Build time: {:.2}s ({:.0} vec/s)",
        build_time.as_secs_f64(),
        num_vectors as f64 / build_time.as_secs_f64());

    // Phase 3: Search
    println!("\nPhase 3: Search performance...");
    let search_start = Instant::now();
    let mut results = Vec::with_capacity(100);
    for q in &queries {
        results.push(hnsw_col.search(q, k, None).unwrap());
    }
    let search_time = search_start.elapsed();
    let latency_ms = search_time.as_secs_f64() * 1000.0 / 100.0;
    println!("  Search latency: {:.2}ms/query", latency_ms);

    // Summary
    let total_time = insert_time.as_secs_f64() + build_time.as_secs_f64();
    println!("\n=== Summary ===");
    println!("Insert rate:     {:.0} vec/s", num_vectors as f64 / insert_time.as_secs_f64());
    println!("Build rate:      {:.0} vec/s", num_vectors as f64 / build_time.as_secs_f64());
    println!("Total time:      {:.2}s", total_time);
    println!("Total rate:      {:.0} vec/s", num_vectors as f64 / total_time);
    println!("Search latency:  {:.2}ms", latency_ms);
    println!("HNSW nodes:      {}", hnsw_col.hnsw_node_count().unwrap_or(0));
}

#[test]
fn benchmark_parallel_construction_50k() {
    let dir = tempdir().unwrap();
    let num_vectors = 50_000;
    let dimensions = 384;
    let k = 10;

    println!("\n=== Parallel Construction Benchmark (50K vectors, 384 dims) ===\n");
    println!("CPU threads: {}", rayon::current_num_threads());

    let vectors = generate_vectors(num_vectors, dimensions, 42);
    let queries: Vec<Vec<f32>> = vectors.iter().take(100).cloned().collect();

    // Create exact collection for recall comparison
    let exact_config = CollectionConfig::new("exact", dimensions, Metric::Cosine);
    let exact_col = Collection::open_with_write_config(
        dir.path().join("exact"),
        exact_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Create HNSW collection
    let hnsw_config = HnswConfig::high_recall();
    let hnsw_col_config = CollectionConfig::with_hnsw("hnsw", dimensions, Metric::Cosine, hnsw_config);
    let hnsw_col = Collection::open_with_write_config(
        dir.path().join("hnsw"),
        hnsw_col_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Insert into both
    println!("Inserting {} vectors...", num_vectors);
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
        print!("\r  {}/{}", batch_end, num_vectors);
    }
    println!();

    exact_col.sync().unwrap();

    println!("Building HNSW index (parallel)...");
    let build_start = Instant::now();
    hnsw_col.sync().unwrap();
    let build_time = build_start.elapsed();
    println!("  Build time: {:.2}s ({:.0} vec/s)",
        build_time.as_secs_f64(),
        num_vectors as f64 / build_time.as_secs_f64());

    // Benchmark searches
    println!("\nSearching...");
    let exact_start = Instant::now();
    let mut exact_results = Vec::with_capacity(100);
    for q in &queries {
        exact_results.push(exact_col.search(q, k, None).unwrap());
    }
    let exact_time = exact_start.elapsed();

    let hnsw_start = Instant::now();
    let mut hnsw_results = Vec::with_capacity(100);
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
        total_recall += intersection as f64 / k as f64;
    }
    let avg_recall = total_recall / 100.0;

    let exact_latency = exact_time.as_secs_f64() * 1000.0 / 100.0;
    let hnsw_latency = hnsw_time.as_secs_f64() * 1000.0 / 100.0;

    println!("\n=== Results @ 50K ===");
    println!("Build rate:     {:.0} vec/s", num_vectors as f64 / build_time.as_secs_f64());
    println!("Exact latency:  {:.2}ms", exact_latency);
    println!("HNSW latency:   {:.2}ms", hnsw_latency);
    println!("Speedup:        {:.1}x", exact_latency / hnsw_latency);
    println!("Recall@{}:      {:.1}%", k, avg_recall * 100.0);
}
