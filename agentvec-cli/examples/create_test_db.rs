//! Test script for CLI - creates test data
use agentvec::{AgentVec, Metric};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = AgentVec::open("./test_cli_db")?;

    // Create collection
    let col = db.collection("memories", 4, Metric::Cosine)?;

    // Add vectors
    col.add(&[1.0, 0.0, 0.0, 0.0], json!({"type": "user", "name": "alice"}), Some("id1"), None)?;
    col.add(&[0.0, 1.0, 0.0, 0.0], json!({"type": "user", "name": "bob"}), Some("id2"), None)?;
    col.add(&[0.9, 0.1, 0.0, 0.0], json!({"type": "assistant", "name": "claude"}), Some("id3"), None)?;
    col.add(&[0.5, 0.5, 0.0, 0.0], json!({"type": "system", "name": "system"}), Some("id4"), None)?;

    col.sync()?;

    println!("Created test database with {} vectors", col.len()?);
    Ok(())
}
