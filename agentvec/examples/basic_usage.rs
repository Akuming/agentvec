//! Basic AgentVec usage example
//!
//! Run with: cargo run --example basic_usage

use agentvec::{AgentVec, Filter, Metric};
use serde_json::json;

fn main() -> agentvec::Result<()> {
    // Create a temporary directory for this example
    let temp_dir = std::env::temp_dir().join("agentvec_example");
    let _ = std::fs::remove_dir_all(&temp_dir); // Clean up any previous run

    println!("Opening database at {:?}", temp_dir);

    // Open or create a database
    let db = AgentVec::open(&temp_dir)?;

    // Create a collection for storing memories
    // - 384 dimensions (typical for small embedding models)
    // - Cosine similarity (best for text embeddings)
    let memories = db.collection("memories", 384, Metric::Cosine)?;

    println!("Created collection: {}", memories.name());
    println!("Dimensions: {}", memories.dimensions());
    println!("Metric: {:?}", memories.metric());

    // Generate some sample embeddings (in real usage, use an embedding model)
    let embedding1 = vec![0.1_f32; 384];
    let embedding2 = vec![0.2_f32; 384];
    let embedding3 = vec![-0.1_f32; 384];

    // Add vectors with metadata
    let id1 = memories.add(
        &embedding1,
        json!({
            "type": "conversation",
            "user": "alice",
            "message": "Hello, how are you?"
        }),
        None,  // Auto-generate ID
        None,  // No TTL (permanent)
    )?;
    println!("Added record: {}", id1);

    // Add with custom ID
    let id2 = memories.add(
        &embedding2,
        json!({
            "type": "conversation",
            "user": "bob",
            "message": "I'm working on a project"
        }),
        Some("conv_002"),  // Custom ID
        Some(3600),        // TTL: 1 hour
    )?;
    println!("Added record: {}", id2);

    // Upsert (insert or update)
    memories.upsert(
        "conv_003",
        &embedding3,
        json!({
            "type": "note",
            "user": "alice",
            "message": "Remember to follow up"
        }),
        None,
    )?;
    println!("Upserted record: conv_003");

    // Check collection size
    println!("Collection size: {} records", memories.len()?);

    // Search for similar vectors
    let query = vec![0.15_f32; 384];
    let results = memories.search(&query, 10, None)?;

    println!("\nSearch results (top 10):");
    for result in &results {
        println!(
            "  {} (score: {:.4}): {}",
            result.id,
            result.score,
            result.metadata
        );
    }

    // Search with filter
    let filter = Filter::new().eq("user", "alice");
    let filtered_results = memories.search(&query, 10, Some(filter))?;

    println!("\nFiltered results (user = alice):");
    for result in &filtered_results {
        println!(
            "  {} (score: {:.4}): {}",
            result.id,
            result.score,
            result.metadata
        );
    }

    // Get a specific record by ID
    if let Some(record) = memories.get("conv_003")? {
        println!("\nRetrieved record conv_003:");
        println!("  ID: {}", record.id);
        println!("  Metadata: {}", record.metadata);
    }

    // Delete a record
    let deleted = memories.delete(&id1)?;
    println!("\nDeleted {}: {}", id1, deleted);

    // Compact to remove expired/deleted records
    let stats = memories.compact()?;
    println!(
        "Compact stats: {} expired, {} tombstones removed",
        stats.expired_removed, stats.tombstones_removed
    );

    // Sync to disk (ensures durability)
    db.sync()?;
    println!("Database synced to disk");

    // Cleanup
    std::fs::remove_dir_all(&temp_dir)?;
    println!("\nExample completed successfully!");

    Ok(())
}
