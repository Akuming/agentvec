//! Comprehensive test suite for AgentVec
//!
//! This module contains thorough tests covering:
//! - Integration tests (full workflows)
//! - Edge cases (empty DBs, boundary conditions)
//! - Search correctness (recall validation)
//! - Concurrent operations (stress tests)
//! - Export/import round-trips
//! - HNSW index behavior
//! - Filter correctness

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use agentvec::{
    AgentVec, Collection, CollectionConfig, Filter, HnswConfig, Metric, WriteConfig,
};
use serde_json::json;
use tempfile::tempdir;

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate deterministic test vectors
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

/// Calculate recall between exact and approximate results
fn calculate_recall(exact: &[String], approximate: &[String]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let exact_set: HashSet<_> = exact.iter().collect();
    let approx_set: HashSet<_> = approximate.iter().collect();
    let intersection = exact_set.intersection(&approx_set).count();
    intersection as f64 / exact.len() as f64
}

// ============================================================================
// Integration Tests - Full Workflows
// ============================================================================

#[test]
fn test_agent_memory_workflow() {
    // Simulates a realistic AI agent memory workflow:
    // 1. Create database with multiple collections (episodic, semantic, working)
    // 2. Add memories to each collection
    // 3. Search across collections
    // 4. Update and delete memories
    // 5. Verify persistence

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("agent.avdb");

    // Phase 1: Create database and collections
    {
        let db = AgentVec::open(&db_path).unwrap();

        // Episodic memory (conversation history)
        let episodic = db.collection("episodic", 128, Metric::Cosine).unwrap();

        // Semantic memory (facts and knowledge)
        let semantic = db.collection("semantic", 128, Metric::Cosine).unwrap();

        // Working memory (short-term context) with TTL
        let working = db.collection("working", 128, Metric::Cosine).unwrap();

        // Add episodic memories
        let vectors = generate_vectors(50, 128, 42);
        for (i, v) in vectors.iter().enumerate() {
            episodic.add(
                v,
                json!({
                    "turn": i,
                    "speaker": if i % 2 == 0 { "user" } else { "assistant" },
                    "content": format!("message_{}", i)
                }),
                Some(&format!("ep_{}", i)),
                None,
            ).unwrap();
        }

        // Add semantic memories
        let semantic_vectors = generate_vectors(30, 128, 123);
        let categories = ["fact", "rule", "preference"];
        for (i, v) in semantic_vectors.iter().enumerate() {
            semantic.add(
                v,
                json!({
                    "category": categories[i % 3],
                    "confidence": 0.5 + (i as f64 * 0.01),
                    "content": format!("knowledge_{}", i)
                }),
                Some(&format!("sem_{}", i)),
                None,
            ).unwrap();
        }

        // Add working memory with TTL (60 seconds)
        let working_vectors = generate_vectors(10, 128, 456);
        for (i, v) in working_vectors.iter().enumerate() {
            working.add(
                v,
                json!({
                    "type": "context",
                    "content": format!("context_{}", i)
                }),
                Some(&format!("work_{}", i)),
                Some(60), // 60 second TTL
            ).unwrap();
        }

        // Sync all collections
        db.sync().unwrap();

        // Verify counts
        assert_eq!(episodic.len().unwrap(), 50);
        assert_eq!(semantic.len().unwrap(), 30);
        assert_eq!(working.len().unwrap(), 10);
    }

    // Phase 2: Reopen and query
    {
        let db = AgentVec::open(&db_path).unwrap();
        let episodic = db.get_collection("episodic").unwrap();
        let semantic = db.get_collection("semantic").unwrap();

        // Search episodic memories
        let query = generate_vectors(1, 128, 42)[0].clone();
        let results = episodic.search(&query, 5, None).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, "ep_0"); // Should find itself first

        // Search with filter
        let filter = Filter::new().eq("speaker", "user");
        let filtered = episodic.search(&query, 10, Some(filter)).unwrap();
        for r in &filtered {
            assert_eq!(r.metadata["speaker"], "user");
        }

        // Search semantic with range filter
        let sem_query = generate_vectors(1, 128, 123)[0].clone();
        let filter = Filter::from_json(&json!({
            "confidence": {"$gte": 0.6, "$lt": 0.9}
        }));
        let sem_results = semantic.search(&sem_query, 10, Some(filter)).unwrap();
        for r in &sem_results {
            let conf = r.metadata["confidence"].as_f64().unwrap();
            assert!(conf >= 0.6 && conf < 0.9);
        }
    }

    // Phase 3: Update and delete
    {
        let db = AgentVec::open(&db_path).unwrap();
        let episodic = db.get_collection("episodic").unwrap();

        // Update a memory
        let new_vec = generate_vectors(1, 128, 999)[0].clone();
        episodic.upsert("ep_0", &new_vec, json!({"turn": 0, "speaker": "user", "updated": true}), None).unwrap();

        let updated = episodic.get("ep_0").unwrap().unwrap();
        assert_eq!(updated.metadata["updated"], true);

        // Delete some memories
        episodic.delete("ep_1").unwrap();
        episodic.delete("ep_2").unwrap();

        assert!(episodic.get("ep_1").unwrap().is_none());
        assert_eq!(episodic.len().unwrap(), 48); // 50 - 2 = 48

        episodic.sync().unwrap();
    }

    // Phase 4: Final verification
    {
        let db = AgentVec::open(&db_path).unwrap();
        let episodic = db.get_collection("episodic").unwrap();

        assert_eq!(episodic.len().unwrap(), 48);

        let updated = episodic.get("ep_0").unwrap().unwrap();
        assert_eq!(updated.metadata["updated"], true);
    }
}

#[test]
fn test_full_crud_cycle() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("crud", 64, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    // Create
    let vectors = generate_vectors(100, 64, 42);
    let ids: Vec<String> = (0..100).map(|i| col.add(
        &vectors[i],
        json!({"idx": i}),
        Some(&format!("id_{}", i)),
        None
    ).unwrap()).collect();

    assert_eq!(col.len().unwrap(), 100);

    // Read
    for (i, id) in ids.iter().enumerate() {
        let record = col.get(id).unwrap();
        assert!(record.is_some());
        assert_eq!(record.unwrap().metadata["idx"], i);
    }

    // Update (using upsert)
    for i in 0..10 {
        let new_vec = generate_vectors(1, 64, 1000 + i)[0].clone();
        col.upsert(&ids[i], &new_vec, json!({"idx": i, "updated": true}), None).unwrap();
    }

    // Verify updates
    for i in 0..10 {
        let record = col.get(&ids[i]).unwrap().unwrap();
        assert_eq!(record.metadata["updated"], true);
    }
    assert_eq!(col.len().unwrap(), 100); // Count should be same

    // Delete
    for i in 50..60 {
        col.delete(&ids[i]).unwrap();
    }
    assert_eq!(col.len().unwrap(), 90);

    // Verify deletes
    for i in 50..60 {
        assert!(col.get(&ids[i]).unwrap().is_none());
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_empty_collection_operations() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("empty", 32, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    // Empty collection checks
    assert!(col.is_empty().unwrap());
    assert_eq!(col.len().unwrap(), 0);

    // Search on empty collection
    let query = vec![1.0; 32];
    let results = col.search(&query, 10, None).unwrap();
    assert!(results.is_empty());

    // Delete non-existent
    let deleted = col.delete("nonexistent").unwrap();
    assert!(!deleted);

    // Get non-existent
    let record = col.get("nonexistent").unwrap();
    assert!(record.is_none());

    // Sync on empty collection (should not error)
    col.sync().unwrap();
}

#[test]
fn test_single_record_collection() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("single", 16, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let vec = vec![1.0; 16];
    col.add(&vec, json!({"only": true}), Some("single"), None).unwrap();

    assert_eq!(col.len().unwrap(), 1);

    // Search should find the single record
    let results = col.search(&vec, 10, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "single");

    // Delete and verify empty
    col.delete("single").unwrap();
    assert!(col.is_empty().unwrap());
}

#[test]
fn test_dimension_boundary_cases() {
    let dir = tempdir().unwrap();

    // Minimum dimension (1)
    let config1 = CollectionConfig::new("dim1", 1, Metric::Cosine);
    let col1 = Collection::open(dir.path().join("col1"), config1).unwrap();
    col1.add(&[1.0], json!({}), None, None).unwrap();
    let results = col1.search(&[1.0], 1, None).unwrap();
    assert_eq!(results.len(), 1);

    // Large dimension (4096 - typical for some embedding models)
    let config2 = CollectionConfig::new("dim4096", 4096, Metric::Cosine);
    let col2 = Collection::open(dir.path().join("col4096"), config2).unwrap();
    let large_vec: Vec<f32> = (0..4096).map(|i| i as f32 / 4096.0).collect();
    col2.add(&large_vec, json!({}), Some("large"), None).unwrap();

    let results = col2.search(&large_vec, 1, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "large");
}

#[test]
fn test_special_characters_in_ids() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("special", 8, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let _base_vec = vec![1.0; 8];

    // Test various special characters
    let special_ids = vec![
        "id with spaces",
        "id-with-dashes",
        "id_with_underscores",
        "id.with.dots",
        "id:with:colons",
        "id/with/slashes",
        "id@with@at",
        "id#with#hash",
        "émojis🎉",
        "中文字符",
        "очень длинный идентификатор",
    ];

    for (i, special_id) in special_ids.iter().enumerate() {
        let mut v = vec![0.0f32; 8];
        v[i % 8] = 1.0;
        col.add(&v, json!({"idx": i}), Some(special_id), None).unwrap();
    }

    col.sync().unwrap();

    // Verify all can be retrieved
    for special_id in &special_ids {
        let record = col.get(special_id).unwrap();
        assert!(record.is_some(), "Failed to get: {}", special_id);
    }
}

#[test]
fn test_metadata_edge_cases() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("meta", 4, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let vec = vec![1.0, 0.0, 0.0, 0.0];

    // Empty metadata
    col.add(&vec, json!({}), Some("empty_meta"), None).unwrap();

    // Deeply nested metadata
    col.add(&vec, json!({
        "level1": {
            "level2": {
                "level3": {
                    "value": 42
                }
            }
        }
    }), Some("nested"), None).unwrap();

    // Array metadata
    col.add(&vec, json!({
        "tags": ["a", "b", "c"],
        "scores": [1.0, 2.0, 3.0]
    }), Some("arrays"), None).unwrap();

    // Large string value
    let large_string = "x".repeat(10000);
    col.add(&vec, json!({"large": large_string}), Some("large_str"), None).unwrap();

    // Null values
    col.add(&vec, json!({"nullable": null}), Some("null_val"), None).unwrap();

    col.sync().unwrap();

    // Verify retrieval
    assert!(col.get("empty_meta").unwrap().is_some());

    let nested = col.get("nested").unwrap().unwrap();
    assert_eq!(nested.metadata["level1"]["level2"]["level3"]["value"], 42);

    let arrays = col.get("arrays").unwrap().unwrap();
    assert_eq!(arrays.metadata["tags"].as_array().unwrap().len(), 3);

    let large = col.get("large_str").unwrap().unwrap();
    assert_eq!(large.metadata["large"].as_str().unwrap().len(), 10000);
}

#[test]
fn test_ttl_expiration() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("ttl", 4, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let vec = vec![1.0; 4];

    // Add record with TTL=0 (immediate expiry)
    col.add(&vec, json!({}), Some("immediate"), Some(0)).unwrap();

    // Add record with TTL=3 seconds (gives enough buffer for test overhead)
    col.add(&vec, json!({}), Some("short"), Some(3)).unwrap();

    // Add record with no TTL (permanent)
    col.add(&vec, json!({}), Some("permanent"), None).unwrap();

    // Wait briefly for immediate expiration
    thread::sleep(Duration::from_millis(100));

    // Immediate should be expired
    assert!(col.get("immediate").unwrap().is_none());

    // Short should still exist (only 100ms passed out of 3s)
    assert!(col.get("short").unwrap().is_some());

    // Permanent should exist
    assert!(col.get("permanent").unwrap().is_some());

    // Wait for short TTL to expire (wait 4 seconds total to be safe)
    thread::sleep(Duration::from_secs(4));

    // Now short should be expired
    assert!(col.get("short").unwrap().is_none());

    // Permanent should still exist
    assert!(col.get("permanent").unwrap().is_some());

    // Search should exclude expired
    let results = col.search(&vec, 10, None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "permanent");
}

// ============================================================================
// Search Correctness Tests
// ============================================================================

#[test]
fn test_exact_search_correctness() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("exact", 64, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let num_vectors = 500;
    let vectors = generate_vectors(num_vectors, 64, 42);

    // Insert all vectors
    for (i, v) in vectors.iter().enumerate() {
        col.add(v, json!({"idx": i}), Some(&format!("v{}", i)), None).unwrap();
    }
    col.sync().unwrap();

    // For each of the first 20 vectors, verify search finds itself first
    for i in 0..20 {
        let results = col.search(&vectors[i], 1, None).unwrap();
        assert_eq!(results.len(), 1, "Query {} should return 1 result", i);
        assert_eq!(results[0].id, format!("v{}", i),
            "Query {} should find itself (got {})", i, results[0].id);
        assert!(results[0].score > 0.99,
            "Query {} score should be ~1.0 (got {})", i, results[0].score);
    }
}

#[test]
fn test_hnsw_recall() {
    // Test that HNSW achieves acceptable recall compared to exact search
    let dir = tempdir().unwrap();

    let num_vectors = 1000;
    let dimensions = 128;
    let k = 10;
    let num_queries = 50;

    let vectors = generate_vectors(num_vectors, dimensions, 42);
    let queries: Vec<Vec<f32>> = vectors.iter().take(num_queries).cloned().collect();

    // Create exact search collection
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

    // Insert into both collections
    for (i, v) in vectors.iter().enumerate() {
        exact_col.add(v, json!({"i": i}), Some(&format!("v{}", i)), None).unwrap();
        hnsw_col.add(v, json!({"i": i}), Some(&format!("v{}", i)), None).unwrap();
    }
    exact_col.sync().unwrap();
    hnsw_col.sync().unwrap();

    // Run queries and calculate recall
    let mut total_recall = 0.0;
    for query in &queries {
        let exact_results = exact_col.search(query, k, None).unwrap();
        let hnsw_results = hnsw_col.search(query, k, None).unwrap();

        let exact_ids: Vec<String> = exact_results.iter().map(|r| r.id.clone()).collect();
        let hnsw_ids: Vec<String> = hnsw_results.iter().map(|r| r.id.clone()).collect();

        total_recall += calculate_recall(&exact_ids, &hnsw_ids);
    }

    let avg_recall = total_recall / num_queries as f64;
    assert!(avg_recall >= 0.90,
        "HNSW recall should be >= 90%, got {:.1}%", avg_recall * 100.0);
}

#[test]
fn test_distance_metrics() {
    let dir = tempdir().unwrap();

    // Test all three metrics
    for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
        let config = CollectionConfig::new(&format!("{:?}", metric), 4, metric);
        let col = Collection::open(dir.path().join(format!("{:?}", metric)), config).unwrap();

        // Add orthogonal vectors
        col.add(&[1.0, 0.0, 0.0, 0.0], json!({}), Some("x"), None).unwrap();
        col.add(&[0.0, 1.0, 0.0, 0.0], json!({}), Some("y"), None).unwrap();
        col.add(&[0.0, 0.0, 1.0, 0.0], json!({}), Some("z"), None).unwrap();
        col.add(&[0.0, 0.0, 0.0, 1.0], json!({}), Some("w"), None).unwrap();
        col.sync().unwrap();

        // Search for [1,0,0,0] - should find "x" first
        let results = col.search(&[1.0, 0.0, 0.0, 0.0], 4, None).unwrap();
        assert_eq!(results[0].id, "x", "{:?}: First result should be 'x'", metric);

        // Search for [0.5, 0.5, 0, 0] - should find "x" or "y" first (equally close)
        let diagonal = [0.5f32.sqrt(), 0.5f32.sqrt(), 0.0, 0.0];
        let results = col.search(&diagonal, 2, None).unwrap();
        let top_ids: HashSet<_> = results.iter().take(2).map(|r| r.id.as_str()).collect();
        assert!(top_ids.contains("x") || top_ids.contains("y"),
            "{:?}: Diagonal query should find x or y in top 2", metric);
    }
}

// ============================================================================
// Filter Correctness Tests
// ============================================================================

#[test]
fn test_filter_with_search() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("filter", 32, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let vectors = generate_vectors(100, 32, 42);

    // Add vectors with various metadata
    let filter_categories = ["A", "B", "C"];
    for (i, v) in vectors.iter().enumerate() {
        col.add(v, json!({
            "category": filter_categories[i % 3],
            "priority": i % 5,
            "active": i % 2 == 0,
            "score": i as f64 * 0.1
        }), Some(&format!("v{}", i)), None).unwrap();
    }
    col.sync().unwrap();

    // Test equality filter
    let filter = Filter::new().eq("category", "A");
    let results = col.search(&vectors[0], 50, Some(filter)).unwrap();
    for r in &results {
        assert_eq!(r.metadata["category"], "A");
    }

    // Test range filter
    let filter = Filter::from_json(&json!({
        "score": {"$gte": 5.0, "$lt": 8.0}
    }));
    let results = col.search(&vectors[0], 50, Some(filter)).unwrap();
    for r in &results {
        let score = r.metadata["score"].as_f64().unwrap();
        assert!(score >= 5.0 && score < 8.0);
    }

    // Test combined filters
    let filter = Filter::new()
        .eq("active", true)
        .gte("priority", 2);
    let results = col.search(&vectors[0], 50, Some(filter)).unwrap();
    for r in &results {
        assert_eq!(r.metadata["active"], true);
        assert!(r.metadata["priority"].as_i64().unwrap() >= 2);
    }

    // Test $in filter
    let filter = Filter::new()
        .is_in("category", vec![json!("A"), json!("B")]);
    let results = col.search(&vectors[0], 50, Some(filter)).unwrap();
    for r in &results {
        let cat = r.metadata["category"].as_str().unwrap();
        assert!(cat == "A" || cat == "B");
    }

    // Test $nin filter
    let filter = Filter::new()
        .not_in("category", vec![json!("C")]);
    let results = col.search(&vectors[0], 50, Some(filter)).unwrap();
    for r in &results {
        assert_ne!(r.metadata["category"], "C");
    }
}

// ============================================================================
// Concurrent Operation Tests
// ============================================================================

#[test]
fn test_concurrent_reads() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("concurrent", 32, Metric::Cosine);
    let col = Arc::new(Collection::open(dir.path().join("col"), config).unwrap());

    // Pre-populate with data
    let vectors = generate_vectors(100, 32, 42);
    for (i, v) in vectors.iter().enumerate() {
        col.add(v, json!({"idx": i}), Some(&format!("v{}", i)), None).unwrap();
    }
    col.sync().unwrap();

    // Spawn multiple reader threads
    let num_readers = 8;
    let reads_per_thread = 100;
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_readers)
        .map(|_| {
            let col = Arc::clone(&col);
            let vectors = vectors.clone();
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                for i in 0..reads_per_thread {
                    let query = &vectors[i % vectors.len()];
                    let results = col.search(query, 5, None).unwrap();
                    if results.len() == 5 {
                        success_count.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let total_reads = num_readers * reads_per_thread;
    let successes = success_count.load(Ordering::SeqCst);
    assert_eq!(successes, total_reads,
        "All concurrent reads should succeed ({}/{})", successes, total_reads);
}

#[test]
fn test_concurrent_writes_single_collection() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("concurrent", 16, Metric::Cosine);
    let col = Arc::new(Collection::open_with_write_config(
        dir.path().join("col"),
        config,
        WriteConfig::throughput(),
    ).unwrap());

    let num_writers = 4;
    let writes_per_thread = 50;

    let handles: Vec<_> = (0..num_writers)
        .map(|writer_id| {
            let col = Arc::clone(&col);

            thread::spawn(move || {
                for i in 0..writes_per_thread {
                    let mut v = vec![0.0f32; 16];
                    v[(writer_id * writes_per_thread + i) % 16] = 1.0;

                    col.add(
                        &v,
                        json!({"writer": writer_id, "idx": i}),
                        Some(&format!("w{}_i{}", writer_id, i)),
                        None,
                    ).unwrap();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    col.sync().unwrap();

    let expected_count = num_writers * writes_per_thread;
    assert_eq!(col.len().unwrap(), expected_count,
        "All concurrent writes should succeed");
}

#[test]
fn test_concurrent_read_write() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("rw", 16, Metric::Cosine);
    let col = Arc::new(Collection::open(dir.path().join("col"), config).unwrap());

    // Pre-populate
    let vectors = generate_vectors(50, 16, 42);
    for (i, v) in vectors.iter().enumerate() {
        col.add(v, json!({"idx": i}), Some(&format!("init_{}", i)), None).unwrap();
    }
    col.sync().unwrap();

    let read_count = Arc::new(AtomicUsize::new(0));
    let write_count = Arc::new(AtomicUsize::new(0));

    // Reader thread
    let col_read = Arc::clone(&col);
    let read_count_clone = Arc::clone(&read_count);
    let vectors_clone = vectors.clone();
    let reader = thread::spawn(move || {
        for _ in 0..200 {
            let query = &vectors_clone[0];
            let _ = col_read.search(query, 5, None);
            read_count_clone.fetch_add(1, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(1));
        }
    });

    // Writer thread
    let col_write = Arc::clone(&col);
    let write_count_clone = Arc::clone(&write_count);
    let writer = thread::spawn(move || {
        for i in 0..100 {
            let v: Vec<f32> = (0..16).map(|j| ((i + j) as f32).sin()).collect();
            col_write.add(&v, json!({"new": i}), Some(&format!("new_{}", i)), None).unwrap();
            write_count_clone.fetch_add(1, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(2));
        }
    });

    reader.join().unwrap();
    writer.join().unwrap();

    assert_eq!(read_count.load(Ordering::SeqCst), 200);
    assert_eq!(write_count.load(Ordering::SeqCst), 100);

    // Verify final state
    col.sync().unwrap();
    assert_eq!(col.len().unwrap(), 150); // 50 initial + 100 new
}

// ============================================================================
// Export/Import Tests
// ============================================================================

#[test]
fn test_export_import_roundtrip() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("export", 64, Metric::Cosine);
    let col = Collection::open(dir.path().join("original"), config.clone()).unwrap();

    let vectors = generate_vectors(100, 64, 42);

    // Add records with various metadata
    for (i, v) in vectors.iter().enumerate() {
        col.add(v, json!({
            "idx": i,
            "name": format!("record_{}", i),
            "nested": {"value": i * 10}
        }), Some(&format!("rec_{}", i)), None).unwrap();
    }
    col.sync().unwrap();

    // Export to file
    let export_path = dir.path().join("export.ndjson");
    let exported = col.export_to_file(&export_path).unwrap();
    assert_eq!(exported, 100);

    // Import to new collection
    let new_config = CollectionConfig::new("import", 64, Metric::Cosine);
    let new_col = Collection::open(dir.path().join("imported"), new_config).unwrap();
    let stats = new_col.import_from_file(&export_path).unwrap();

    assert_eq!(stats.imported, 100);
    assert_eq!(stats.failed, 0);
    assert_eq!(new_col.len().unwrap(), 100);

    // Verify data integrity
    for i in 0..100 {
        let id = format!("rec_{}", i);

        let orig = col.get(&id).unwrap().unwrap();
        let imported = new_col.get(&id).unwrap().unwrap();

        assert_eq!(orig.metadata["idx"], imported.metadata["idx"]);
        assert_eq!(orig.metadata["name"], imported.metadata["name"]);
        assert_eq!(orig.metadata["nested"]["value"], imported.metadata["nested"]["value"]);
    }

    // Verify search works on imported collection
    let results = new_col.search(&vectors[0], 5, None).unwrap();
    assert_eq!(results.len(), 5);
    assert_eq!(results[0].id, "rec_0");
}

#[test]
fn test_export_import_with_ttl() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("ttl", 8, Metric::Cosine);
    let col = Collection::open(dir.path().join("original"), config.clone()).unwrap();

    let vec = vec![1.0; 8];

    // Add records with different TTLs
    col.add(&vec, json!({"type": "permanent"}), Some("perm"), None).unwrap();
    col.add(&vec, json!({"type": "ttl"}), Some("ttl"), Some(3600)).unwrap(); // 1 hour TTL
    col.sync().unwrap();

    // Export
    let export_path = dir.path().join("export.ndjson");
    col.export_to_file(&export_path).unwrap();

    // Import
    let new_col = Collection::open(dir.path().join("imported"), config).unwrap();
    new_col.import_from_file(&export_path).unwrap();

    // Both should exist in imported collection
    assert!(new_col.get("perm").unwrap().is_some());
    assert!(new_col.get("ttl").unwrap().is_some());
}

// ============================================================================
// HNSW Specific Tests
// ============================================================================

#[test]
fn test_hnsw_incremental_build() {
    let dir = tempdir().unwrap();
    let hnsw_config = HnswConfig::with_m(8);
    let config = CollectionConfig::with_hnsw("hnsw", 64, Metric::Cosine, hnsw_config);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let vectors = generate_vectors(100, 64, 42);

    // Add first batch
    for (i, v) in vectors.iter().take(50).enumerate() {
        col.add(v, json!({"batch": 1, "idx": i}), Some(&format!("v{}", i)), None).unwrap();
    }
    col.sync().unwrap();

    // Verify HNSW index is built
    assert!(col.has_hnsw_index());
    assert_eq!(col.hnsw_node_count(), Some(50));

    // Add second batch
    for (i, v) in vectors.iter().skip(50).enumerate() {
        col.add(v, json!({"batch": 2, "idx": i + 50}), Some(&format!("v{}", i + 50)), None).unwrap();
    }
    col.sync().unwrap();

    // Verify incremental update
    assert_eq!(col.hnsw_node_count(), Some(100));

    // Search should work correctly
    let results = col.search(&vectors[0], 10, None).unwrap();
    assert_eq!(results.len(), 10);
    assert_eq!(results[0].id, "v0");
}

#[test]
fn test_hnsw_persistence_and_reload() {
    let dir = tempdir().unwrap();
    let col_path = dir.path().join("col");

    // Create and populate
    {
        let hnsw_config = HnswConfig::with_m(8);
        let config = CollectionConfig::with_hnsw("hnsw", 32, Metric::Cosine, hnsw_config);
        let col = Collection::open(&col_path, config).unwrap();

        let vectors = generate_vectors(50, 32, 42);
        for (i, v) in vectors.iter().enumerate() {
            col.add(v, json!({"idx": i}), Some(&format!("v{}", i)), None).unwrap();
        }
        col.sync().unwrap();

        assert_eq!(col.hnsw_node_count(), Some(50));
    }

    // Reopen and verify HNSW is loaded
    {
        let hnsw_config = HnswConfig::with_m(8);
        let config = CollectionConfig::with_hnsw("hnsw", 32, Metric::Cosine, hnsw_config);
        let col = Collection::open(&col_path, config).unwrap();

        assert!(col.has_hnsw_index());
        assert_eq!(col.hnsw_node_count(), Some(50));

        // Search should work without rebuilding
        let vectors = generate_vectors(50, 32, 42);
        let results = col.search(&vectors[0], 5, None).unwrap();
        assert!(!results.is_empty());
    }
}

#[test]
fn test_hnsw_delete_handling() {
    let dir = tempdir().unwrap();
    let hnsw_config = HnswConfig::with_m(8);
    let config = CollectionConfig::with_hnsw("hnsw", 16, Metric::Cosine, hnsw_config);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let vectors = generate_vectors(20, 16, 42);
    for (i, v) in vectors.iter().enumerate() {
        col.add(v, json!({}), Some(&format!("v{}", i)), None).unwrap();
    }
    col.sync().unwrap();

    // Delete some vectors
    col.delete("v0").unwrap();
    col.delete("v5").unwrap();
    col.delete("v10").unwrap();

    // Search should not return deleted
    let results = col.search(&vectors[0], 20, None).unwrap();
    let result_ids: HashSet<_> = results.iter().map(|r| r.id.as_str()).collect();

    assert!(!result_ids.contains("v0"));
    assert!(!result_ids.contains("v5"));
    assert!(!result_ids.contains("v10"));
    assert_eq!(results.len(), 17); // 20 - 3 deleted
}

// ============================================================================
// Batch Operations Tests
// ============================================================================

#[test]
fn test_batch_add() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("batch", 32, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let vectors = generate_vectors(1000, 32, 42);
    let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let metadatas: Vec<_> = (0..1000).map(|i| json!({"idx": i})).collect();
    let ids: Vec<String> = (0..1000).map(|i| format!("v{}", i)).collect();
    let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();

    let result_ids = col.add_batch(&vector_refs, &metadatas, Some(&id_refs), None).unwrap();

    assert_eq!(result_ids.len(), 1000);
    assert_eq!(col.len().unwrap(), 1000);

    // Verify some records
    for i in [0, 100, 500, 999] {
        let record = col.get(&format!("v{}", i)).unwrap().unwrap();
        assert_eq!(record.metadata["idx"], i);
    }
}

#[test]
fn test_batch_add_with_ttl() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("batch_ttl", 8, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let vectors = generate_vectors(10, 8, 42);
    let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let metadatas: Vec<_> = (0..10).map(|i| json!({"idx": i})).collect();

    // Mix of TTLs: some expire, some don't
    let ttls: Vec<Option<u64>> = (0..10).map(|i| {
        if i < 5 { Some(0) } else { None }  // First 5 expire immediately
    }).collect();

    col.add_batch(&vector_refs, &metadatas, None, Some(&ttls)).unwrap();

    thread::sleep(Duration::from_millis(100));

    // Only non-expired should remain searchable
    let results = col.search(&vectors[0], 10, None).unwrap();
    assert_eq!(results.len(), 5);
}

// ============================================================================
// Compaction Tests
// ============================================================================

#[test]
fn test_compact_removes_expired_and_deleted() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("compact", 8, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let vec = vec![1.0; 8];

    // Add records: some permanent, some with TTL, some to delete
    for i in 0..10 {
        let ttl = if i < 3 { Some(0) } else { None }; // First 3 expire immediately
        col.add(&vec, json!({"idx": i}), Some(&format!("v{}", i)), ttl).unwrap();
    }
    col.sync().unwrap();

    // Delete some
    col.delete("v5").unwrap();
    col.delete("v6").unwrap();

    thread::sleep(Duration::from_millis(100));

    // Compact
    let stats = col.compact().unwrap();

    // Should have removed expired (3) + deleted (2) = 5
    assert!(stats.expired_removed >= 3 || stats.tombstones_removed >= 2);

    // Only 5 should remain: v3, v4, v7, v8, v9
    assert_eq!(col.len().unwrap(), 5);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_dimension_mismatch_errors() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("dim", 8, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    // Wrong dimension on add
    let wrong_vec = vec![1.0; 16];
    let result = col.add(&wrong_vec, json!({}), None, None);
    assert!(result.is_err());

    // Wrong dimension on search
    let result = col.search(&wrong_vec, 5, None);
    assert!(result.is_err());
}

#[test]
fn test_database_collection_errors() {
    let dir = tempdir().unwrap();
    let db = AgentVec::open(dir.path().join("db.avdb")).unwrap();

    // Create collection
    db.collection("test", 32, Metric::Cosine).unwrap();

    // Try to open with different dimensions - should error
    let result = db.collection("test", 64, Metric::Cosine);
    assert!(result.is_err());

    // Try to open with different metric - should error
    let result = db.collection("test", 32, Metric::L2);
    assert!(result.is_err());

    // Get non-existent collection - should error
    let result = db.get_collection("nonexistent");
    assert!(result.is_err());
}

// ============================================================================
// Performance Smoke Tests
// ============================================================================

#[test]
fn test_performance_10k_insert_search() {
    use std::time::Instant;

    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("perf", 128, Metric::Cosine);
    let col = Collection::open_with_write_config(
        dir.path().join("col"),
        config,
        WriteConfig::throughput(),
    ).unwrap();

    let num_vectors = 10_000;
    let vectors = generate_vectors(num_vectors, 128, 42);

    // Measure insert time
    let insert_start = Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        col.add(v, json!({"i": i}), Some(&format!("v{}", i)), None).unwrap();
    }
    col.sync().unwrap();
    let insert_time = insert_start.elapsed();

    let insert_rate = num_vectors as f64 / insert_time.as_secs_f64();
    println!("Insert rate: {:.0} vec/s", insert_rate);
    // Relaxed assertion: just ensure we can insert at a reasonable rate
    assert!(insert_rate > 500.0, "Insert should be > 500 vec/s (got {:.0})", insert_rate);

    // Measure search time
    let num_queries = 100;
    let search_start = Instant::now();
    for i in 0..num_queries {
        col.search(&vectors[i], 10, None).unwrap();
    }
    let search_time = search_start.elapsed();

    let latency_ms = search_time.as_secs_f64() * 1000.0 / num_queries as f64;
    println!("Search latency: {:.2}ms/query", latency_ms);
    // Relaxed assertion for CI/test environments: allow up to 200ms
    // Production benchmarks should use dedicated benchmark tests
    assert!(latency_ms < 200.0, "Search should complete in reasonable time (got {:.2}ms)", latency_ms);

    // Verify search correctness
    let results = col.search(&vectors[0], 10, None).unwrap();
    assert_eq!(results.len(), 10);
    assert_eq!(results[0].id, "v0", "First result should be the query vector itself");
}

#[test]
fn test_get_with_vector() {
    let dir = tempdir().unwrap();
    let config = CollectionConfig::new("get_vec", 8, Metric::Cosine);
    let col = Collection::open(dir.path().join("col"), config).unwrap();

    let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    col.add(&vec, json!({"test": true}), Some("test_id"), None).unwrap();
    col.sync().unwrap();

    // Get with vector
    let result = col.get_with_vector("test_id").unwrap();
    assert!(result.is_some());

    let full_record = result.unwrap();
    assert_eq!(full_record.id, "test_id");
    assert_eq!(full_record.metadata["test"], true);

    // Vector should be normalized (for cosine metric)
    let norm: f32 = full_record.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Vector should be normalized");
}
