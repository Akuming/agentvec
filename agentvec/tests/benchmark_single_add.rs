//! Quick benchmark to measure single-add performance with write coalescing.

use agentvec::{AgentVec, Metric, WriteConfig, Collection, CollectionConfig};
use rand::Rng;
use serde_json::json;
use tempfile::TempDir;

fn random_normalized_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[test]
fn benchmark_single_add_coalesced() {
    let temp_dir = TempDir::new().unwrap();
    let db = AgentVec::open(temp_dir.path()).unwrap();
    let collection = db.collection("bench", 384, Metric::Cosine).unwrap();

    let n = 100;
    let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_normalized_vector(384)).collect();

    println!("\n======== COALESCED SINGLE ADD (default) ========");

    let start = std::time::Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        collection.add(v, json!({"index": i}), None, None).unwrap();
    }
    let elapsed_adds = start.elapsed();

    // Now flush
    let flush_start = std::time::Instant::now();
    collection.sync().unwrap();
    let flush_elapsed = flush_start.elapsed();

    let total = elapsed_adds + flush_elapsed;

    println!("Adds only: {:?} ({:.2} µs/add)", elapsed_adds, elapsed_adds.as_micros() as f64 / n as f64);
    println!("Flush:     {:?}", flush_elapsed);
    println!("Total:     {:?} ({:.2} µs/add)", total, total.as_micros() as f64 / n as f64);
    println!("Throughput: {:.0} adds/sec", n as f64 / total.as_secs_f64());
}

#[test]
fn benchmark_single_add_immediate() {
    let temp_dir = TempDir::new().unwrap();
    let config = CollectionConfig::new("bench", 384, Metric::Cosine);
    let collection = Collection::open_with_write_config(
        temp_dir.path().join("col"),
        config,
        WriteConfig::immediate(),
    ).unwrap();

    let n = 100;
    let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_normalized_vector(384)).collect();

    println!("\n======== IMMEDIATE SINGLE ADD (old behavior) ========");

    let start = std::time::Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        collection.add(v, json!({"index": i}), None, None).unwrap();
    }
    let elapsed = start.elapsed();

    println!("Total:     {:?} ({:.2} ms/add)", elapsed, elapsed.as_millis() as f64 / n as f64);
    println!("Throughput: {:.0} adds/sec", n as f64 / elapsed.as_secs_f64());
}

#[test]
fn compare_add_modes() {
    let n = 50; // Enough to see the difference
    let dim = 384;
    let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_normalized_vector(dim)).collect();

    println!("\n======== ADD MODE COMPARISON ({} vectors, dim={}) ========\n", n, dim);

    // Coalesced mode
    let temp_dir1 = TempDir::new().unwrap();
    let db1 = AgentVec::open(temp_dir1.path()).unwrap();
    let col1 = db1.collection("bench", dim, Metric::Cosine).unwrap();

    let start1 = std::time::Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        col1.add(v, json!({"i": i}), None, None).unwrap();
    }
    col1.sync().unwrap();
    let coalesced_time = start1.elapsed();

    // Immediate mode
    let temp_dir2 = TempDir::new().unwrap();
    let config2 = CollectionConfig::new("bench", dim, Metric::Cosine);
    let col2 = Collection::open_with_write_config(
        temp_dir2.path().join("col"),
        config2,
        WriteConfig::immediate(),
    ).unwrap();

    let start2 = std::time::Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        col2.add(v, json!({"i": i}), None, None).unwrap();
    }
    let immediate_time = start2.elapsed();

    // Batch add for comparison
    let temp_dir3 = TempDir::new().unwrap();
    let db3 = AgentVec::open(temp_dir3.path()).unwrap();
    let col3 = db3.collection("bench", dim, Metric::Cosine).unwrap();

    let vec_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let metadatas: Vec<serde_json::Value> = (0..n).map(|i| json!({"i": i})).collect();

    let start3 = std::time::Instant::now();
    col3.add_batch(&vec_refs, &metadatas, None, None).unwrap();
    let batch_time = start3.elapsed();

    println!("Coalesced adds: {:?} ({:.1} µs/add)", coalesced_time, coalesced_time.as_micros() as f64 / n as f64);
    println!("Immediate adds: {:?} ({:.1} ms/add)", immediate_time, immediate_time.as_millis() as f64 / n as f64);
    println!("Batch add:      {:?} ({:.1} µs/add)", batch_time, batch_time.as_micros() as f64 / n as f64);
    println!();
    println!("Speedup (coalesced vs immediate): {:.1}x", immediate_time.as_secs_f64() / coalesced_time.as_secs_f64());
    println!("Speedup (batch vs immediate):     {:.1}x", immediate_time.as_secs_f64() / batch_time.as_secs_f64());
}
