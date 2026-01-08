//! Benchmarks for Collection operations.
//!
//! Tests add, search, get, delete, and batch operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use serde_json::json;
use tempfile::TempDir;

use agentvec::{AgentVec, Metric};

/// Generate a random vector of the given dimension.
fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Generate a normalized random vector (for cosine similarity).
fn random_normalized_vector(dim: usize) -> Vec<f32> {
    let mut v = random_vector(dim);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Benchmark single vector add operation.
fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_add");

    for dim in [128, 384, 768, 1536] {
        let temp_dir = TempDir::new().unwrap();
        let db = AgentVec::open(temp_dir.path()).unwrap();
        let collection = db.collection("bench", dim, Metric::Cosine).unwrap();

        group.throughput(Throughput::Elements(1));

        group.bench_with_input(BenchmarkId::new("single", dim), &dim, |bencher, &d| {
            bencher.iter(|| {
                let v = random_normalized_vector(d);
                collection
                    .add(black_box(&v), json!({"key": "value"}), None, None)
                    .unwrap()
            })
        });
    }

    group.finish();
}

/// Benchmark batch add operation.
fn bench_add_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_add_batch");
    group.sample_size(50); // Reduce samples since we create fresh collections

    let dim = 384;

    for batch_size in [10, 100, 1000] {
        // Prepare vectors once (reused across iterations)
        let vectors: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| random_normalized_vector(dim))
            .collect();
        let vec_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let metadatas: Vec<serde_json::Value> =
            (0..batch_size).map(|i| json!({"index": i})).collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |bencher, _| {
                // Create fresh collection for each iteration to avoid accumulation
                bencher.iter_custom(|iters| {
                    let mut total = std::time::Duration::ZERO;
                    for _ in 0..iters {
                        let temp_dir = TempDir::new().unwrap();
                        let db = AgentVec::open(temp_dir.path()).unwrap();
                        let collection = db.collection("bench", dim, Metric::Cosine).unwrap();

                        let start = std::time::Instant::now();
                        collection
                            .add_batch(
                                black_box(&vec_refs),
                                black_box(&metadatas),
                                None,
                                None,
                            )
                            .unwrap();
                        total += start.elapsed();
                    }
                    total
                })
            },
        );
    }

    group.finish();
}

/// Benchmark search operation at different collection sizes.
fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_search");
    group.sample_size(50); // Reduce samples for slower benchmarks

    let dim = 384;

    for n_vectors in [100, 1000, 10000] {
        let temp_dir = TempDir::new().unwrap();
        let db = AgentVec::open(temp_dir.path()).unwrap();
        let collection = db.collection("bench", dim, Metric::Cosine).unwrap();

        // Pre-populate collection
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|_| random_normalized_vector(dim))
            .collect();
        let vec_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let metadatas: Vec<serde_json::Value> =
            (0..n_vectors).map(|i| json!({"index": i})).collect();
        collection
            .add_batch(&vec_refs, &metadatas, None, None)
            .unwrap();

        // Preload vectors for fair comparison
        collection.preload().unwrap();

        let query = random_normalized_vector(dim);

        group.throughput(Throughput::Elements(n_vectors as u64));

        // Search with k=10
        group.bench_with_input(
            BenchmarkId::new("k10", n_vectors),
            &n_vectors,
            |bencher, _| {
                bencher.iter(|| collection.search(black_box(&query), 10, None).unwrap())
            },
        );

        // Search with k=100
        if n_vectors >= 100 {
            group.bench_with_input(
                BenchmarkId::new("k100", n_vectors),
                &n_vectors,
                |bencher, _| {
                    bencher.iter(|| collection.search(black_box(&query), 100, None).unwrap())
                },
            );
        }
    }

    group.finish();
}

/// Benchmark get by ID operation.
fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_get");

    let dim = 384;
    let n_vectors = 10000;

    let temp_dir = TempDir::new().unwrap();
    let db = AgentVec::open(temp_dir.path()).unwrap();
    let collection = db.collection("bench", dim, Metric::Cosine).unwrap();

    // Pre-populate collection
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| random_normalized_vector(dim))
        .collect();
    let vec_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let metadatas: Vec<serde_json::Value> = (0..n_vectors).map(|i| json!({"index": i})).collect();
    let ids = collection
        .add_batch(&vec_refs, &metadatas, None, None)
        .unwrap();

    // Pick some random IDs to get
    let mut rng = rand::thread_rng();
    let test_ids: Vec<&str> = (0..100)
        .map(|_| ids[rng.gen_range(0..ids.len())].as_str())
        .collect();

    group.bench_function("by_id", |bencher| {
        let mut i = 0;
        bencher.iter(|| {
            let id = test_ids[i % test_ids.len()];
            i += 1;
            collection.get(black_box(id)).unwrap()
        })
    });

    group.finish();
}

/// Benchmark delete operation.
fn bench_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_delete");
    group.sample_size(50);

    let dim = 384;

    let temp_dir = TempDir::new().unwrap();
    let db = AgentVec::open(temp_dir.path()).unwrap();
    let collection = db.collection("bench", dim, Metric::Cosine).unwrap();

    // Pre-populate with many vectors we can delete
    let n_vectors = 10000;
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| random_normalized_vector(dim))
        .collect();
    let vec_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let metadatas: Vec<serde_json::Value> = (0..n_vectors).map(|i| json!({"index": i})).collect();
    let ids = collection
        .add_batch(&vec_refs, &metadatas, None, None)
        .unwrap();

    let mut id_iter = ids.into_iter();

    group.bench_function("single", |bencher| {
        bencher.iter(|| {
            if let Some(id) = id_iter.next() {
                collection.delete(black_box(&id)).unwrap()
            } else {
                false
            }
        })
    });

    group.finish();
}

/// Benchmark upsert operation.
fn bench_upsert(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_upsert");

    let dim = 384;

    let temp_dir = TempDir::new().unwrap();
    let db = AgentVec::open(temp_dir.path()).unwrap();
    let collection = db.collection("bench", dim, Metric::Cosine).unwrap();

    // Pre-populate with some vectors
    let n_existing = 1000;
    let ids: Vec<String> = (0..n_existing).map(|i| format!("vec_{i}")).collect();
    for id in &ids {
        let v = random_normalized_vector(dim);
        collection.add(&v, json!({"id": id}), Some(id), None).unwrap();
    }

    let mut i = 0;

    group.bench_function("existing_id", |bencher| {
        bencher.iter(|| {
            let id = &ids[i % ids.len()];
            let v = random_normalized_vector(dim);
            i += 1;
            collection
                .upsert(black_box(id), black_box(&v), json!({"updated": true}), None)
                .unwrap()
        })
    });

    group.finish();
}

/// Benchmark compact operation.
fn bench_compact(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_compact");
    group.sample_size(20); // Compact is slow, reduce samples

    let dim = 384;
    let n_vectors = 5000;

    let temp_dir = TempDir::new().unwrap();
    let db = AgentVec::open(temp_dir.path()).unwrap();
    let collection = db.collection("bench", dim, Metric::Cosine).unwrap();

    // Pre-populate and delete half to create tombstones
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|_| random_normalized_vector(dim))
        .collect();
    let vec_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let metadatas: Vec<serde_json::Value> = (0..n_vectors).map(|i| json!({"index": i})).collect();
    let ids = collection
        .add_batch(&vec_refs, &metadatas, None, None)
        .unwrap();

    // Delete every other record
    for (i, id) in ids.iter().enumerate() {
        if i % 2 == 0 {
            collection.delete(id).unwrap();
        }
    }

    group.bench_function("with_tombstones", |bencher| {
        bencher.iter(|| collection.compact().unwrap())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_add_batch,
    bench_search,
    bench_get,
    bench_delete,
    bench_upsert,
    bench_compact
);
criterion_main!(benches);
