//! Benchmarks for distance calculation functions.
//!
//! Tests SIMD-optimized implementations across different vector dimensions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;

use agentvec::search::distance::{dot_product, l2_squared, normalize_l2};

/// Generate a random vector of the given dimension.
fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Generate a normalized random vector (for cosine similarity).
fn random_normalized_vector(dim: usize) -> Vec<f32> {
    normalize_l2(&random_vector(dim))
}

/// Cosine similarity (dot product of normalized vectors).
#[inline]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    dot_product(a, b)
}

/// L2 distance (sqrt of l2_squared).
#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// Benchmark dot product for various dimensions.
fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    // Common embedding dimensions
    for dim in [128, 256, 384, 512, 768, 1024, 1536, 3072] {
        group.throughput(Throughput::Elements(dim as u64));

        let a = random_vector(dim);
        let b = random_vector(dim);

        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bencher, _| {
            bencher.iter(|| dot_product(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

/// Benchmark L2 distance for various dimensions.
fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_distance");

    for dim in [128, 256, 384, 512, 768, 1024, 1536, 3072] {
        group.throughput(Throughput::Elements(dim as u64));

        let a = random_vector(dim);
        let b = random_vector(dim);

        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bencher, _| {
            bencher.iter(|| l2_distance(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

/// Benchmark cosine similarity for various dimensions.
fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in [128, 256, 384, 512, 768, 1024, 1536, 3072] {
        group.throughput(Throughput::Elements(dim as u64));

        // Cosine similarity uses normalized vectors
        let a = random_normalized_vector(dim);
        let b = random_normalized_vector(dim);

        group.bench_with_input(BenchmarkId::new("simd", dim), &dim, |bencher, _| {
            bencher.iter(|| cosine_similarity(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

/// Benchmark batch distance calculations (simulating search over N vectors).
fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_search");

    let dim = 384; // Common dimension for sentence embeddings

    for n_vectors in [100, 1000, 10000] {
        // For dot product, use raw vectors
        let query_raw = random_vector(dim);
        let vectors_raw: Vec<Vec<f32>> = (0..n_vectors).map(|_| random_vector(dim)).collect();

        // For cosine, use normalized vectors
        let query_norm = random_normalized_vector(dim);
        let vectors_norm: Vec<Vec<f32>> = (0..n_vectors)
            .map(|_| random_normalized_vector(dim))
            .collect();

        group.throughput(Throughput::Elements(n_vectors as u64));

        group.bench_with_input(
            BenchmarkId::new("dot_product", n_vectors),
            &n_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    let mut scores = Vec::with_capacity(vectors_raw.len());
                    for v in &vectors_raw {
                        scores.push(dot_product(black_box(&query_raw), black_box(v)));
                    }
                    scores
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cosine", n_vectors),
            &n_vectors,
            |bencher, _| {
                bencher.iter(|| {
                    let mut scores = Vec::with_capacity(vectors_norm.len());
                    for v in &vectors_norm {
                        scores.push(cosine_similarity(black_box(&query_norm), black_box(v)));
                    }
                    scores
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dot_product,
    bench_l2_distance,
    bench_cosine_similarity,
    bench_batch_distances
);
criterion_main!(benches);
