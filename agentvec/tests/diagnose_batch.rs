//! Diagnostic test to identify batch performance bottlenecks.
//!
//! Run with: cargo test -p agentvec --release diagnose_batch -- --nocapture

use agentvec::{AgentVec, Metric};
use serde_json::json;
use std::time::Instant;
use tempfile::tempdir;

fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = ((seed.wrapping_add(i as u64))
                .wrapping_mul(1103515245)
                .wrapping_add(12345)) as f32;
            (x / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn diagnose_batch_performance() {
    println!("\n======== Batch Performance Diagnosis ========\n");

    let dir = tempdir().unwrap();
    let db = AgentVec::open(dir.path()).unwrap();
    let col = db.collection("bench", 384, Metric::Cosine).unwrap();

    // Test different batch sizes
    for &batch_size in &[10, 100, 1000] {
        println!("--- Batch size: {} ---", batch_size);

        // Prepare data
        let vectors: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| random_vector(384, i as u64))
            .collect();
        let vec_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let metadatas: Vec<serde_json::Value> = (0..batch_size)
            .map(|i| json!({"index": i}))
            .collect();

        // Time add_batch
        let start = Instant::now();
        let _ids = col.add_batch(&vec_refs, &metadatas, None, None).unwrap();
        let elapsed = start.elapsed();

        println!(
            "  Total: {:?} | Per-vector: {:?} | Vectors/sec: {:.0}",
            elapsed,
            elapsed / batch_size as u32,
            batch_size as f64 / elapsed.as_secs_f64()
        );
    }

    println!("\n--- Comparing single adds ---");

    // Fresh collection for single adds
    let col2 = db.collection("bench2", 384, Metric::Cosine).unwrap();

    let num_single = 20;
    let start = Instant::now();
    for i in 0..num_single {
        let v = random_vector(384, 10000 + i as u64);
        col2.add(&v, json!({"i": i}), None, None).unwrap();
    }
    let elapsed = start.elapsed();
    println!(
        "  {} single adds: {:?} | Per-add: {:?}",
        num_single,
        elapsed,
        elapsed / num_single
    );

    println!("\n--- Sync timing ---");

    let sync_start = Instant::now();
    col.sync().unwrap();
    println!("  col.sync(): {:?}", sync_start.elapsed());

    let sync_start = Instant::now();
    db.sync().unwrap();
    println!("  db.sync(): {:?}", sync_start.elapsed());

    println!("\n======== Analysis ========");
    println!("If batch per-vector time >> 100µs: likely redb transaction overhead");
    println!("If single add >> batch per-vector: confirms transaction batching helps");
    println!("Single add does 2 redb commits vs batch does 2 total commits");
}

#[test]
fn diagnose_component_timing() {
    println!("\n======== Component Timing Breakdown ========\n");

    let dir = tempdir().unwrap();
    let db = AgentVec::open(dir.path()).unwrap();
    let col = db.collection("timing", 384, Metric::Cosine).unwrap();

    let batch_size = 100;

    // 1. Time vector normalization
    let vectors: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| random_vector(384, i as u64))
        .collect();

    let norm_start = Instant::now();
    let _normalized: Vec<Vec<f32>> = vectors
        .iter()
        .map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter().map(|x| x / norm).collect()
            } else {
                v.clone()
            }
        })
        .collect();
    println!("1. Normalize {} vectors: {:?}", batch_size, norm_start.elapsed());

    // 2. Time metadata JSON preparation
    let json_start = Instant::now();
    let _metadatas: Vec<serde_json::Value> = (0..batch_size)
        .map(|i| json!({"index": i, "type": "test", "data": "some metadata string"}))
        .collect();
    println!("2. Create {} JSON metadatas: {:?}", batch_size, json_start.elapsed());

    // 3. Time UUID generation
    let uuid_start = Instant::now();
    let _uuids: Vec<String> = (0..batch_size)
        .map(|_| uuid::Uuid::new_v4().to_string())
        .collect();
    println!("3. Generate {} UUIDs: {:?}", batch_size, uuid_start.elapsed());

    // 4. Time pure mmap write (after collection has grown)
    // First, grow the collection
    let grow_vectors: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| random_vector(384, 5000 + i as u64))
        .collect();
    let grow_refs: Vec<&[f32]> = grow_vectors.iter().map(|v| v.as_slice()).collect();
    let grow_metas: Vec<serde_json::Value> = (0..batch_size).map(|i| json!({"i": i})).collect();
    col.add_batch(&grow_refs, &grow_metas, None, None).unwrap();

    println!("4. (Collection pre-grown with {} vectors)", batch_size);

    // 5. Do another batch and time
    let batch2_start = Instant::now();
    let vecs2: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| random_vector(384, 9000 + i as u64))
        .collect();
    let refs2: Vec<&[f32]> = vecs2.iter().map(|v| v.as_slice()).collect();
    let metas2: Vec<serde_json::Value> = (0..batch_size).map(|i| json!({"i": i})).collect();
    col.add_batch(&refs2, &metas2, None, None).unwrap();
    println!(
        "5. Second batch of {} (pre-grown file): {:?}",
        batch_size,
        batch2_start.elapsed()
    );

    println!("\n======== Expected Bottlenecks ========");
    println!("- If normalize is slow: SIMD could help more");
    println!("- If batch time >> (normalize + json + uuid): it's I/O (redb commits, mmap sync)");
    println!("- redb commit includes fsync which is ~5-15ms on Windows");
}
