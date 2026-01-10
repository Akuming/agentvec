//! AI Agent Memory Patterns with AgentVec
//!
//! This example demonstrates how to use AgentVec for different types
//! of AI agent memory: working memory, episodic memory, and semantic memory.
//!
//! Run with: cargo run --example agent_memory

use agentvec::{AgentVec, Filter, Metric};
use serde_json::json;

fn main() -> agentvec::Result<()> {
    let temp_dir = std::env::temp_dir().join("agent_memory_example");
    let _ = std::fs::remove_dir_all(&temp_dir);

    println!("=== AI Agent Memory Example ===\n");

    let db = AgentVec::open(&temp_dir)?;

    // =========================================================================
    // Working Memory (Short-term, high turnover)
    // =========================================================================
    // Use for: Current conversation context, temporary state
    // TTL: Minutes to hours
    println!("--- Working Memory ---");

    let working = db.collection("working", 384, Metric::Cosine)?;

    // Store current conversation turns with short TTL
    let turn1 = vec![0.1_f32; 384]; // Embedding for "What's the weather?"
    working.add(
        &turn1,
        json!({
            "role": "user",
            "content": "What's the weather like today?",
            "turn": 1
        }),
        None,
        Some(1800), // 30 minute TTL
    )?;

    let turn2 = vec![0.12_f32; 384]; // Embedding for weather response
    working.add(
        &turn2,
        json!({
            "role": "assistant",
            "content": "It's sunny and 72°F today.",
            "turn": 2
        }),
        None,
        Some(1800),
    )?;

    println!("Working memory: {} items", working.len()?);

    // =========================================================================
    // Episodic Memory (Medium-term, experiences)
    // =========================================================================
    // Use for: Past conversations, user interactions, events
    // TTL: Hours to days
    println!("\n--- Episodic Memory ---");

    let episodic = db.collection("episodic", 384, Metric::Cosine)?;

    // Store conversation summaries
    let conv_summary = vec![0.2_f32; 384];
    episodic.add(
        &conv_summary,
        json!({
            "type": "conversation",
            "user_id": "user_123",
            "summary": "User asked about weather and project deadlines",
            "sentiment": "neutral",
            "timestamp": "2024-01-15T10:30:00Z"
        }),
        None,
        Some(86400 * 7), // 7 day TTL
    )?;

    // Store user preferences learned from interactions
    let pref_embedding = vec![0.25_f32; 384];
    episodic.add(
        &pref_embedding,
        json!({
            "type": "preference",
            "user_id": "user_123",
            "preference": "prefers concise responses",
            "confidence": 0.8
        }),
        None,
        Some(86400 * 30), // 30 day TTL
    )?;

    println!("Episodic memory: {} items", episodic.len()?);

    // =========================================================================
    // Semantic Memory (Long-term, knowledge)
    // =========================================================================
    // Use for: Facts, documentation, permanent knowledge
    // TTL: None (permanent) or very long
    println!("\n--- Semantic Memory ---");

    let semantic = db.collection("semantic", 1536, Metric::Cosine)?; // Larger model

    // Store factual knowledge (no TTL - permanent)
    let fact1 = vec![0.3_f32; 1536];
    semantic.add(
        &fact1,
        json!({
            "type": "fact",
            "domain": "company",
            "content": "The company was founded in 2020",
            "source": "about_page"
        }),
        Some("fact_founding"),
        None, // No TTL - permanent
    )?;

    let fact2 = vec![0.35_f32; 1536];
    semantic.add(
        &fact2,
        json!({
            "type": "procedure",
            "domain": "support",
            "content": "To reset password: go to settings > security > reset",
            "source": "help_docs"
        }),
        Some("proc_password_reset"),
        None,
    )?;

    println!("Semantic memory: {} items", semantic.len()?);

    // =========================================================================
    // Memory Retrieval Patterns
    // =========================================================================
    println!("\n--- Memory Retrieval ---");

    // 1. Get relevant context for current query
    let query_embedding = vec![0.11_f32; 384];

    // Search working memory first (most recent context)
    let working_results = working.search(&query_embedding, 3, None)?;
    println!("Recent context from working memory:");
    for r in &working_results {
        println!("  - {}: {}", r.id, r.metadata["content"]);
    }

    // 2. Get user-specific memories
    let user_filter = Filter::new().eq("user_id", "user_123");
    let user_memories = episodic.search(&query_embedding, 5, Some(user_filter))?;
    println!("\nUser-specific memories:");
    for r in &user_memories {
        if let Some(summary) = r.metadata.get("summary") {
            println!("  - {}", summary);
        }
        if let Some(pref) = r.metadata.get("preference") {
            println!("  - Preference: {}", pref);
        }
    }

    // 3. Get domain-specific knowledge
    let domain_query = vec![0.32_f32; 1536];
    let domain_filter = Filter::new().eq("domain", "support");
    let knowledge = semantic.search(&domain_query, 3, Some(domain_filter))?;
    println!("\nSupport knowledge:");
    for r in &knowledge {
        println!("  - {}", r.metadata["content"]);
    }

    // =========================================================================
    // Memory Maintenance
    // =========================================================================
    println!("\n--- Memory Maintenance ---");

    // Update a memory (upsert)
    let updated_embedding = vec![0.26_f32; 384];
    episodic.upsert(
        "user_123_preferences",
        &updated_embedding,
        json!({
            "type": "preference",
            "user_id": "user_123",
            "preference": "prefers concise responses with examples",
            "confidence": 0.9,
            "updated_at": "2024-01-15T11:00:00Z"
        }),
        Some(86400 * 30),
    )?;
    println!("Updated user preferences");

    // Compact expired memories
    let working_stats = working.compact()?;
    let episodic_stats = episodic.compact()?;
    println!(
        "Compacted: {} working, {} episodic expired",
        working_stats.expired_removed, episodic_stats.expired_removed
    );

    // Sync all changes
    db.sync()?;
    println!("All memories synced to disk");

    // =========================================================================
    // Memory Statistics
    // =========================================================================
    println!("\n--- Memory Statistics ---");
    println!("Working memory:  {} records, {} bytes",
        working.len()?, working.vectors_size_bytes());
    println!("Episodic memory: {} records, {} bytes",
        episodic.len()?, episodic.vectors_size_bytes());
    println!("Semantic memory: {} records, {} bytes",
        semantic.len()?, semantic.vectors_size_bytes());

    // Cleanup
    std::fs::remove_dir_all(&temp_dir)?;
    println!("\nExample completed!");

    Ok(())
}
