//! Benchmark for scalar quantization performance

use std::sync::Arc;
use std::time::Instant;

use agentvec::search::{dot_product, ScalarQuantizer, QuantizedVectors};

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
fn benchmark_quantized_distance() {
    println!("\n=== Scalar Quantization Benchmark ===\n");

    let dim = 384;
    let num_vectors = 10_000;
    let num_ops = 1_000_000;

    println!("Generating {} vectors of dimension {}...", num_vectors, dim);
    let vectors = generate_vectors(num_vectors, dim);

    // Fit quantizer
    println!("Fitting quantizer...");
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let quantizer = Arc::new(ScalarQuantizer::fit(refs.iter().copied()));

    // Pre-quantize all vectors
    println!("Quantizing vectors...");
    let quantized: Vec<Vec<u8>> = vectors.iter()
        .map(|v| quantizer.quantize(v))
        .collect();

    // Create quantized storage
    let storage = QuantizedVectors::from_vectors(
        quantizer.clone(),
        refs.iter().copied(),
    );

    // Warmup
    for i in 0..1000 {
        let _ = dot_product(&vectors[i % num_vectors], &vectors[(i + 1) % num_vectors]);
        let _ = quantizer.dot_product_quantized(&quantized[i % num_vectors], &quantized[(i + 1) % num_vectors]);
    }

    // Benchmark f32 dot product
    println!("\nBenchmarking f32 dot_product...");
    let start = Instant::now();
    let mut sum = 0.0f32;
    for i in 0..num_ops {
        sum += dot_product(&vectors[i % num_vectors], &vectors[(i + 1) % num_vectors]);
    }
    let f32_time = start.elapsed();
    println!("  {} ops in {:.2}s = {:.2}M ops/sec",
        num_ops, f32_time.as_secs_f64(),
        num_ops as f64 / f32_time.as_secs_f64() / 1_000_000.0);
    println!("  (sum = {} to prevent optimization)", sum);

    // Benchmark quantized dot product
    println!("\nBenchmarking quantized dot_product...");
    let start = Instant::now();
    let mut sum = 0i32;
    for i in 0..num_ops {
        sum = sum.wrapping_add(quantizer.dot_product_quantized(
            &quantized[i % num_vectors],
            &quantized[(i + 1) % num_vectors],
        ));
    }
    let quant_time = start.elapsed();
    println!("  {} ops in {:.2}s = {:.2}M ops/sec",
        num_ops, quant_time.as_secs_f64(),
        num_ops as f64 / quant_time.as_secs_f64() / 1_000_000.0);
    println!("  (sum = {} to prevent optimization)", sum);

    // Benchmark using QueryTables (precomputed lookup)
    println!("\nBenchmarking QueryTables lookup...");
    let tables = quantizer.precompute_query_tables(&vectors[0]);
    let start = Instant::now();
    let mut sum = 0.0f32;
    for i in 0..num_ops {
        sum += tables.dot_product(&quantized[i % num_vectors]);
    }
    let table_time = start.elapsed();
    println!("  {} ops in {:.2}s = {:.2}M ops/sec",
        num_ops, table_time.as_secs_f64(),
        num_ops as f64 / table_time.as_secs_f64() / 1_000_000.0);
    println!("  (sum = {} to prevent optimization)", sum);

    // Memory comparison
    let f32_bytes = num_vectors * dim * 4;
    let quant_bytes = num_vectors * dim;
    println!("\n=== Memory Usage ===");
    println!("f32 vectors:  {} MB", f32_bytes / 1_000_000);
    println!("u8 vectors:   {} MB", quant_bytes / 1_000_000);
    println!("Compression:  {}x", f32_bytes as f64 / quant_bytes as f64);

    // Summary
    let speedup = f32_time.as_secs_f64() / quant_time.as_secs_f64();
    let table_speedup = f32_time.as_secs_f64() / table_time.as_secs_f64();

    println!("\n=== Summary ===");
    println!("Quantized dot:  {:.1}x faster than f32", speedup);
    println!("QueryTables:    {:.1}x faster than f32", table_speedup);
    println!("Memory:         4x reduction");
}

#[test]
fn benchmark_quantized_l2() {
    println!("\n=== L2 Squared Quantization Benchmark ===\n");

    let dim = 384;
    let num_vectors = 10_000;
    let num_ops = 500_000;

    println!("Generating {} vectors of dimension {}...", num_vectors, dim);
    let vectors = generate_vectors(num_vectors, dim);

    // Fit quantizer
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let quantizer = ScalarQuantizer::fit(refs.iter().copied());

    // Pre-quantize all vectors
    let quantized: Vec<Vec<u8>> = vectors.iter()
        .map(|v| quantizer.quantize(v))
        .collect();

    // Benchmark f32 L2
    println!("Benchmarking f32 l2_squared...");
    let start = Instant::now();
    let mut sum = 0.0f32;
    for i in 0..num_ops {
        sum += agentvec::search::l2_squared(&vectors[i % num_vectors], &vectors[(i + 1) % num_vectors]);
    }
    let f32_time = start.elapsed();
    println!("  {} ops in {:.2}s = {:.2}M ops/sec",
        num_ops, f32_time.as_secs_f64(),
        num_ops as f64 / f32_time.as_secs_f64() / 1_000_000.0);

    // Benchmark quantized L2
    println!("Benchmarking quantized l2_squared...");
    let start = Instant::now();
    let mut sum = 0i32;
    for i in 0..num_ops {
        sum = sum.wrapping_add(quantizer.l2_squared_quantized(
            &quantized[i % num_vectors],
            &quantized[(i + 1) % num_vectors],
        ));
    }
    let quant_time = start.elapsed();
    println!("  {} ops in {:.2}s = {:.2}M ops/sec",
        num_ops, quant_time.as_secs_f64(),
        num_ops as f64 / quant_time.as_secs_f64() / 1_000_000.0);

    let speedup = f32_time.as_secs_f64() / quant_time.as_secs_f64();
    println!("\nSpeedup: {:.1}x", speedup);

    // Prevent optimization
    assert!(sum != 0 || f32_time.as_secs_f64() > 0.0);
}
