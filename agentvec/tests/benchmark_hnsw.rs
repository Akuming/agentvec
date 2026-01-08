//! Benchmark comparing HNSW approximate search vs exact search.

use std::time::Instant;

use agentvec::{Collection, CollectionConfig, HnswConfig, Metric, WriteConfig};
use serde_json::json;
use tempfile::tempdir;

fn separator() -> String {
    "=".repeat(80)
}

fn line() -> String {
    "-".repeat(80)
}

/// Generate random vectors for benchmarking.
fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..count)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let mut hasher = DefaultHasher::new();
                    (i * dim + j).hash(&mut hasher);
                    let h = hasher.finish();
                    // Convert to float in range [-1, 1]
                    ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
                })
                .collect()
        })
        .collect()
}

/// Normalize a vector to unit length.
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Benchmark result for a single run.
#[derive(Debug, Clone)]
struct BenchResult {
    num_vectors: usize,
    dimensions: usize,
    num_queries: usize,
    k: usize,

    // Timing
    insert_time_ms: f64,
    exact_search_time_ms: f64,
    hnsw_search_time_ms: f64,

    // Per-query metrics
    exact_per_query_us: f64,
    hnsw_per_query_us: f64,
    speedup: f64,

    // Quality (recall)
    recall_at_k: f64,
}

fn run_benchmark(num_vectors: usize, dimensions: usize, k: usize, num_queries: usize) -> BenchResult {
    let dir = tempdir().unwrap();

    // Generate vectors
    let vectors: Vec<Vec<f32>> = generate_vectors(num_vectors, dimensions)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    let queries: Vec<Vec<f32>> = generate_vectors(num_queries, dimensions)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    // Create exact search collection
    let exact_config = CollectionConfig::new("exact", dimensions, Metric::Cosine);
    let exact_col = Collection::open_with_write_config(
        dir.path().join("exact"),
        exact_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Create HNSW collection with high recall settings
    let hnsw_config = HnswConfig::high_recall(); // M=32, ef_construction=400, ef_search=100
    let hnsw_col_config = CollectionConfig::with_hnsw("hnsw", dimensions, Metric::Cosine, hnsw_config);
    let hnsw_col = Collection::open_with_write_config(
        dir.path().join("hnsw"),
        hnsw_col_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Insert vectors into both collections
    let insert_start = Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        exact_col.add(v, json!({"i": i}), Some(&format!("v{}", i)), None).unwrap();
        hnsw_col.add(v, json!({"i": i}), Some(&format!("v{}", i)), None).unwrap();
    }
    exact_col.sync().unwrap();
    hnsw_col.sync().unwrap();
    let insert_time = insert_start.elapsed();

    // Benchmark exact search
    let exact_start = Instant::now();
    let mut exact_results = Vec::with_capacity(num_queries);
    for q in &queries {
        let results = exact_col.search(q, k, None).unwrap();
        exact_results.push(results);
    }
    let exact_time = exact_start.elapsed();

    // Benchmark HNSW search
    let hnsw_start = Instant::now();
    let mut hnsw_results = Vec::with_capacity(num_queries);
    for q in &queries {
        let results = hnsw_col.search(q, k, None).unwrap();
        hnsw_results.push(results);
    }
    let hnsw_time = hnsw_start.elapsed();

    // Calculate recall (how many of the exact top-k are in HNSW top-k)
    let mut total_recall = 0.0;
    for (exact, hnsw) in exact_results.iter().zip(hnsw_results.iter()) {
        let exact_ids: std::collections::HashSet<_> = exact.iter().map(|r| &r.id).collect();
        let hnsw_ids: std::collections::HashSet<_> = hnsw.iter().map(|r| &r.id).collect();
        let intersection = exact_ids.intersection(&hnsw_ids).count();
        let recall = if exact_ids.is_empty() {
            1.0
        } else {
            intersection as f64 / exact_ids.len() as f64
        };
        total_recall += recall;
    }
    let avg_recall = total_recall / num_queries as f64;

    let exact_per_query_us = exact_time.as_micros() as f64 / num_queries as f64;
    let hnsw_per_query_us = hnsw_time.as_micros() as f64 / num_queries as f64;

    BenchResult {
        num_vectors,
        dimensions,
        num_queries,
        k,
        insert_time_ms: insert_time.as_millis() as f64,
        exact_search_time_ms: exact_time.as_millis() as f64,
        hnsw_search_time_ms: hnsw_time.as_millis() as f64,
        exact_per_query_us,
        hnsw_per_query_us,
        speedup: exact_per_query_us / hnsw_per_query_us,
        recall_at_k: avg_recall,
    }
}

#[test]
fn benchmark_hnsw_vs_exact() {
    println!("\n");
    println!("{}", separator());
    println!("HNSW vs Exact Search Benchmark");
    println!("{}", separator());
    println!();

    let dimensions = 128;
    let k = 10;
    let num_queries = 100;

    // Test at different scales
    let scales = [1_000, 5_000, 10_000, 25_000];

    println!("{:<12} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "Vectors", "Exact (us)", "HNSW (us)", "Speedup", "Recall@10", "Status");
    println!("{}", line());

    for &num_vectors in &scales {
        let result = run_benchmark(num_vectors, dimensions, k, num_queries);

        let status = if result.recall_at_k >= 0.95 { "✓" } else { "⚠" };

        println!("{:<12} {:>12.1} {:>12.1} {:>10.1}x {:>9.1}% {:>10}",
            format!("{}", num_vectors),
            result.exact_per_query_us,
            result.hnsw_per_query_us,
            result.speedup,
            result.recall_at_k * 100.0,
            status);
    }

    println!();
    println!("Config: dim={}, k={}, queries={}", dimensions, k, num_queries);
    println!("HNSW: M=16, ef_construction=200, ef_search=50");
    println!();
}

#[test]
fn benchmark_hnsw_scaling() {
    println!("\n");
    println!("{}", separator());
    println!("HNSW Search Time Scaling");
    println!("{}", separator());
    println!();

    let dimensions = 384; // Common embedding size
    let k = 10;
    let num_queries = 50;

    let scales = [1_000, 2_500, 5_000, 10_000, 20_000];

    println!("{:<12} {:>15} {:>15} {:>12}",
        "Vectors", "HNSW (us/q)", "Insert (ms)", "Recall@10");
    println!("{}", "-".repeat(60));

    for &num_vectors in &scales {
        let result = run_benchmark(num_vectors, dimensions, k, num_queries);

        println!("{:<12} {:>15.1} {:>15.0} {:>11.1}%",
            format!("{}", num_vectors),
            result.hnsw_per_query_us,
            result.insert_time_ms,
            result.recall_at_k * 100.0);
    }

    println!();
    println!("Config: dim={}, k={}, queries={}", dimensions, k, num_queries);
    println!();
}

#[test]
fn benchmark_detailed_comparison() {
    println!("\n");
    println!("{}", separator());
    println!("Detailed HNSW vs Exact Comparison (10K vectors, 384 dim)");
    println!("{}", separator());
    println!();

    let num_vectors = 10_000;
    let dimensions = 384;
    let num_queries = 100;

    for k in [1, 5, 10, 20, 50] {
        let result = run_benchmark(num_vectors, dimensions, k, num_queries);

        println!("k={:<3}: Exact {:>8.1}µs | HNSW {:>8.1}µs | {:>5.1}x faster | {:.1}% recall",
            k,
            result.exact_per_query_us,
            result.hnsw_per_query_us,
            result.speedup,
            result.recall_at_k * 100.0);
    }

    println!();
}
