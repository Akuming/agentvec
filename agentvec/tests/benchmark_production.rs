//! Production-scale benchmark for HNSW vs Exact search.
//! Tests at 100K and 1M vectors with 384 dimensions (PRD target).
//!
//! PRD Targets:
//! - < 10 ms @ 10K vectors
//! - < 50 ms @ 100K vectors
//! - < 50 ms @ 1M vectors (Phase 4 milestone)

use std::time::Instant;

use agentvec::{Collection, CollectionConfig, HnswConfig, Metric, WriteConfig};
use serde_json::json;
use tempfile::tempdir;

/// Generate deterministic pseudo-random vectors for benchmarking.
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

            // Normalize for cosine similarity
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                vec.iter_mut().for_each(|x| *x /= norm);
            }
            vec
        })
        .collect()
}

#[derive(Debug, Clone)]
struct BenchmarkResult {
    num_vectors: usize,
    dimensions: usize,
    k: usize,
    num_queries: usize,

    // Insert performance
    insert_time_secs: f64,
    inserts_per_sec: f64,

    // Search performance
    exact_latency_ms: f64,
    hnsw_latency_ms: f64,
    speedup: f64,

    // Throughput
    exact_qps: f64,
    hnsw_qps: f64,

    // Quality
    recall_at_k: f64,

    // PRD compliance
    meets_prd_target: bool,
    prd_target_ms: f64,
}

fn run_production_benchmark(
    num_vectors: usize,
    dimensions: usize,
    k: usize,
    num_queries: usize,
) -> BenchmarkResult {
    let dir = tempdir().unwrap();

    // Generate vectors
    println!("  Generating {} vectors of {} dimensions...", num_vectors, dimensions);
    let gen_start = Instant::now();
    let vectors = generate_vectors(num_vectors, dimensions, 42);
    // Use in-dataset queries (first num_queries vectors) for realistic recall measurement
    // Real AI agent queries are similar to stored embeddings, not random vectors
    let queries: Vec<Vec<f32>> = vectors.iter().take(num_queries).cloned().collect();
    println!("  Generated in {:.2}s", gen_start.elapsed().as_secs_f64());

    // Create exact search collection
    let exact_config = CollectionConfig::new("exact", dimensions, Metric::Cosine);
    let exact_col = Collection::open_with_write_config(
        dir.path().join("exact"),
        exact_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Create HNSW collection with high recall settings
    let hnsw_config = HnswConfig::high_recall();
    let hnsw_col_config = CollectionConfig::with_hnsw("hnsw", dimensions, Metric::Cosine, hnsw_config);
    let hnsw_col = Collection::open_with_write_config(
        dir.path().join("hnsw"),
        hnsw_col_config,
        WriteConfig::throughput(),
    ).unwrap();

    // Insert vectors
    println!("  Inserting {} vectors...", num_vectors);
    let insert_start = Instant::now();

    // Use batch insert for speed - use consistent IDs for both collections
    let batch_size = 1000;
    for batch_start in (0..num_vectors).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_vectors);
        let batch_vecs: Vec<&[f32]> = vectors[batch_start..batch_end]
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let batch_metas: Vec<_> = (batch_start..batch_end)
            .map(|i| json!({"i": i}))
            .collect();

        // Use consistent IDs so we can compare recall
        let batch_ids: Vec<String> = (batch_start..batch_end)
            .map(|i| format!("v{}", i))
            .collect();
        let batch_id_refs: Vec<&str> = batch_ids.iter().map(|s| s.as_str()).collect();

        exact_col.add_batch(&batch_vecs, &batch_metas, Some(&batch_id_refs), None).unwrap();
        hnsw_col.add_batch(&batch_vecs, &batch_metas, Some(&batch_id_refs), None).unwrap();

        if (batch_end % 10000) == 0 || batch_end == num_vectors {
            print!("\r  Inserted {}/{}", batch_end, num_vectors);
        }
    }
    println!();

    exact_col.sync().unwrap();
    hnsw_col.sync().unwrap();
    let insert_time = insert_start.elapsed();

    println!("  Insert complete in {:.2}s ({:.0} vectors/sec)",
        insert_time.as_secs_f64(),
        num_vectors as f64 / insert_time.as_secs_f64());

    // Benchmark exact search
    println!("  Running {} exact search queries...", num_queries);
    let exact_start = Instant::now();
    let mut exact_results = Vec::with_capacity(num_queries);
    for q in &queries {
        let results = exact_col.search(q, k, None).unwrap();
        exact_results.push(results);
    }
    let exact_time = exact_start.elapsed();
    let exact_latency_ms = exact_time.as_secs_f64() * 1000.0 / num_queries as f64;

    // Benchmark HNSW search
    println!("  Running {} HNSW search queries...", num_queries);
    let hnsw_start = Instant::now();
    let mut hnsw_results = Vec::with_capacity(num_queries);
    for q in &queries {
        let results = hnsw_col.search(q, k, None).unwrap();
        hnsw_results.push(results);
    }
    let hnsw_time = hnsw_start.elapsed();
    let hnsw_latency_ms = hnsw_time.as_secs_f64() * 1000.0 / num_queries as f64;

    // Calculate recall
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

    // Determine PRD target
    let prd_target_ms = if num_vectors <= 10_000 {
        10.0
    } else if num_vectors <= 100_000 {
        50.0
    } else {
        50.0 // 1M target is also 50ms
    };

    let meets_prd = hnsw_latency_ms <= prd_target_ms;

    BenchmarkResult {
        num_vectors,
        dimensions,
        k,
        num_queries,
        insert_time_secs: insert_time.as_secs_f64(),
        inserts_per_sec: num_vectors as f64 / insert_time.as_secs_f64(),
        exact_latency_ms,
        hnsw_latency_ms,
        speedup: exact_latency_ms / hnsw_latency_ms,
        exact_qps: 1000.0 / exact_latency_ms,
        hnsw_qps: 1000.0 / hnsw_latency_ms,
        recall_at_k: avg_recall,
        meets_prd_target: meets_prd,
        prd_target_ms,
    }
}

fn print_separator() {
    println!("{}", "=".repeat(90));
}

fn print_line() {
    println!("{}", "-".repeat(90));
}

#[test]
fn benchmark_production_scale() {
    println!("\n");
    print_separator();
    println!("PRODUCTION-SCALE BENCHMARK (PRD Compliance Test)");
    println!("AgentVec HNSW vs Exact Search");
    print_separator();
    println!();

    let dimensions = 384; // Sentence transformer embedding size (PRD: episodic memory)
    let k = 10;
    let num_queries = 100;

    // Test at production scales
    let scales = [
        (10_000, "10K - PRD Target: <10ms"),
        (50_000, "50K - Interpolated"),
        (100_000, "100K - PRD Target: <50ms"),
    ];

    println!("{:<12} {:>12} {:>12} {:>10} {:>10} {:>8} {:>12}",
        "Scale", "Exact (ms)", "HNSW (ms)", "Speedup", "Recall", "QPS", "PRD Status");
    print_line();

    let mut results = Vec::new();

    for (num_vectors, description) in scales {
        println!("\n[{}]", description);

        let result = run_production_benchmark(num_vectors, dimensions, k, num_queries);

        let status = if result.meets_prd_target {
            format!("PASS (<{}ms)", result.prd_target_ms as i32)
        } else {
            format!("FAIL (>{}ms)", result.prd_target_ms as i32)
        };

        println!("{:<12} {:>12.2} {:>12.2} {:>10.1}x {:>9.1}% {:>8.0} {:>12}",
            format!("{}", num_vectors),
            result.exact_latency_ms,
            result.hnsw_latency_ms,
            result.speedup,
            result.recall_at_k * 100.0,
            result.hnsw_qps,
            status);

        results.push(result);
    }

    println!();
    print_separator();
    println!("SUMMARY");
    print_separator();
    println!();

    println!("Configuration:");
    println!("  Dimensions: {} (sentence transformer)", dimensions);
    println!("  k: {}", k);
    println!("  Queries: {}", num_queries);
    println!("  HNSW: M=32, ef_construction=400, ef_search=100 (high_recall preset)");
    println!();

    println!("Insert Performance:");
    for r in &results {
        println!("  {}K vectors: {:.0} inserts/sec ({:.1}s total)",
            r.num_vectors / 1000,
            r.inserts_per_sec,
            r.insert_time_secs);
    }
    println!();

    println!("PRD Compliance:");
    for r in &results {
        let status = if r.meets_prd_target { "PASS" } else { "FAIL" };
        println!("  {}K: {:.2}ms (target: <{}ms) - {}",
            r.num_vectors / 1000,
            r.hnsw_latency_ms,
            r.prd_target_ms as i32,
            status);
    }
    println!();

    // Check if all tests pass
    let all_pass = results.iter().all(|r| r.meets_prd_target);
    let all_high_recall = results.iter().all(|r| r.recall_at_k >= 0.95);

    if all_pass && all_high_recall {
        println!("Result: ALL PRD TARGETS MET with >95% recall");
    } else {
        if !all_pass {
            println!("Result: SOME LATENCY TARGETS NOT MET - optimization needed");
        }
        if !all_high_recall {
            println!("Result: RECALL BELOW 95% - quality issue");
        }
    }
    println!();
}

#[test]
#[ignore] // This test takes a long time, run explicitly
fn benchmark_one_million() {
    println!("\n");
    print_separator();
    println!("1M VECTOR BENCHMARK (Phase 4 PRD Milestone)");
    println!("Target: <50ms search latency");
    print_separator();
    println!();

    let dimensions = 384;
    let k = 10;
    let num_queries = 50; // Fewer queries for 1M scale

    let result = run_production_benchmark(1_000_000, dimensions, k, num_queries);

    println!();
    println!("Results at 1M vectors:");
    println!("  HNSW Latency: {:.2}ms", result.hnsw_latency_ms);
    println!("  Exact Latency: {:.2}ms", result.exact_latency_ms);
    println!("  Speedup: {:.1}x", result.speedup);
    println!("  QPS: {:.0}", result.hnsw_qps);
    println!("  Recall@{}: {:.1}%", k, result.recall_at_k * 100.0);
    println!("  Insert Rate: {:.0} vectors/sec", result.inserts_per_sec);
    println!();

    let status = if result.meets_prd_target { "PASS" } else { "FAIL" };
    println!("PRD Phase 4 Target (<50ms): {}", status);
    println!();
}
