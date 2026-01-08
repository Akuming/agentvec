//! SIMD performance verification benchmark

use std::time::Instant;

use agentvec::search::distance::{dot_product, dot_product_scalar, l2_squared, l2_squared_scalar};

fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..count)
        .map(|i| {
            let mut vec: Vec<f32> = (0..dim)
                .map(|j| {
                    let mut hasher = DefaultHasher::new();
                    (i * dim + j).hash(&mut hasher);
                    let h = hasher.finish();
                    ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
                })
                .collect();

            // Normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                vec.iter_mut().for_each(|x| *x /= norm);
            }
            vec
        })
        .collect()
}

#[test]
fn benchmark_simd_distance() {
    println!("\n=== SIMD Distance Computation Benchmark ===\n");

    let dim = 384;
    let num_vectors = 10_000;
    let num_ops = 1_000_000;

    println!("Generating {} vectors of dimension {}...", num_vectors, dim);
    let vectors = generate_vectors(num_vectors, dim);

    // Warmup
    for i in 0..1000 {
        let _ = dot_product(&vectors[i % num_vectors], &vectors[(i + 1) % num_vectors]);
    }

    // Benchmark SIMD dot product
    println!("\nBenchmarking dot_product (SIMD auto-dispatch)...");
    let start = Instant::now();
    let mut sum = 0.0f32;
    for i in 0..num_ops {
        sum += dot_product(&vectors[i % num_vectors], &vectors[(i + 1) % num_vectors]);
    }
    let simd_time = start.elapsed();
    println!("  {} ops in {:.2}s = {:.2}M ops/sec",
        num_ops, simd_time.as_secs_f64(),
        num_ops as f64 / simd_time.as_secs_f64() / 1_000_000.0);
    println!("  (sum = {} to prevent optimization)", sum);

    // Benchmark scalar dot product
    println!("\nBenchmarking dot_product_scalar...");
    let start = Instant::now();
    let mut sum = 0.0f32;
    for i in 0..num_ops {
        sum += dot_product_scalar(&vectors[i % num_vectors], &vectors[(i + 1) % num_vectors]);
    }
    let scalar_time = start.elapsed();
    println!("  {} ops in {:.2}s = {:.2}M ops/sec",
        num_ops, scalar_time.as_secs_f64(),
        num_ops as f64 / scalar_time.as_secs_f64() / 1_000_000.0);
    println!("  (sum = {} to prevent optimization)", sum);

    let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
    println!("\nSIMD speedup: {:.2}x", speedup);

    // Benchmark L2 squared
    println!("\n--- L2 Squared ---");

    println!("Benchmarking l2_squared (SIMD)...");
    let start = Instant::now();
    let mut sum = 0.0f32;
    for i in 0..num_ops {
        sum += l2_squared(&vectors[i % num_vectors], &vectors[(i + 1) % num_vectors]);
    }
    let simd_l2_time = start.elapsed();
    println!("  {} ops in {:.2}s = {:.2}M ops/sec",
        num_ops, simd_l2_time.as_secs_f64(),
        num_ops as f64 / simd_l2_time.as_secs_f64() / 1_000_000.0);

    println!("Benchmarking l2_squared_scalar...");
    let start = Instant::now();
    let mut sum = 0.0f32;
    for i in 0..num_ops {
        sum += l2_squared_scalar(&vectors[i % num_vectors], &vectors[(i + 1) % num_vectors]);
    }
    let scalar_l2_time = start.elapsed();
    println!("  {} ops in {:.2}s = {:.2}M ops/sec",
        num_ops, scalar_l2_time.as_secs_f64(),
        num_ops as f64 / scalar_l2_time.as_secs_f64() / 1_000_000.0);

    let l2_speedup = scalar_l2_time.as_secs_f64() / simd_l2_time.as_secs_f64();
    println!("\nL2 SIMD speedup: {:.2}x", l2_speedup);

    // Summary
    println!("\n=== Summary ===");
    println!("Dot product:  SIMD {:.1}x faster than scalar", speedup);
    println!("L2 squared:   SIMD {:.1}x faster than scalar", l2_speedup);

    if speedup < 2.0 {
        println!("\nWARNING: SIMD speedup is low. Check if AVX2 is enabled.");
        println!("Expected 4-8x speedup with AVX2.");
    }

    // Theoretical throughput
    let ops_per_sec = num_ops as f64 / simd_time.as_secs_f64();
    let flops_per_op = dim as f64 * 2.0; // multiply + add
    let gflops = ops_per_sec * flops_per_op / 1e9;
    println!("\nTheoretical throughput: {:.1} GFLOPS", gflops);
}
