//! Advanced Filtering Example
//!
//! This example demonstrates all the filter operators available in AgentVec.
//!
//! Run with: cargo run --example filtering

use agentvec::{AgentVec, Filter, Metric};
use serde_json::json;

fn main() -> agentvec::Result<()> {
    let temp_dir = std::env::temp_dir().join("filtering_example");
    let _ = std::fs::remove_dir_all(&temp_dir);

    println!("=== Filtering Example ===\n");

    let db = AgentVec::open(&temp_dir)?;
    let collection = db.collection("documents", 384, Metric::Cosine)?;

    // Add sample documents with various metadata
    let docs = vec![
        (vec![0.1_f32; 384], json!({
            "title": "Introduction to Rust",
            "category": "programming",
            "difficulty": "beginner",
            "views": 1500,
            "rating": 4.5,
            "tags": ["rust", "programming", "tutorial"],
            "published": true
        })),
        (vec![0.2_f32; 384], json!({
            "title": "Advanced Rust Patterns",
            "category": "programming",
            "difficulty": "advanced",
            "views": 800,
            "rating": 4.8,
            "tags": ["rust", "patterns", "advanced"],
            "published": true
        })),
        (vec![0.3_f32; 384], json!({
            "title": "Python for Data Science",
            "category": "data-science",
            "difficulty": "intermediate",
            "views": 2500,
            "rating": 4.2,
            "tags": ["python", "data-science", "ml"],
            "published": true
        })),
        (vec![0.4_f32; 384], json!({
            "title": "Machine Learning Basics",
            "category": "data-science",
            "difficulty": "beginner",
            "views": 3000,
            "rating": 4.6,
            "tags": ["ml", "ai", "beginner"],
            "published": true
        })),
        (vec![0.5_f32; 384], json!({
            "title": "Draft: New Features",
            "category": "programming",
            "difficulty": "intermediate",
            "views": 50,
            "rating": 0.0,
            "tags": ["draft"],
            "published": false
        })),
    ];

    for (i, (embedding, metadata)) in docs.iter().enumerate() {
        collection.add(embedding, metadata.clone(), Some(&format!("doc_{}", i)), None)?;
    }
    println!("Added {} documents\n", collection.len()?);

    let query = vec![0.15_f32; 384];

    // =========================================================================
    // Equality Filters
    // =========================================================================
    println!("--- Equality Filters ---");

    // Simple equality
    let filter = Filter::new().eq("category", "programming");
    let results = collection.search(&query, 10, Some(filter))?;
    println!("category = 'programming': {} results", results.len());
    for r in &results {
        println!("  - {}", r.metadata["title"]);
    }

    // Explicit $eq operator
    let filter = Filter::new().eq("difficulty", "beginner");
    let results = collection.search(&query, 10, Some(filter))?;
    println!("\ndifficulty = 'beginner': {} results", results.len());
    for r in &results {
        println!("  - {}", r.metadata["title"]);
    }

    // Boolean equality
    let filter = Filter::new().eq("published", true);
    let results = collection.search(&query, 10, Some(filter))?;
    println!("\npublished = true: {} results", results.len());

    // =========================================================================
    // Not Equal Filter
    // =========================================================================
    println!("\n--- Not Equal Filter ---");

    let filter = Filter::new().ne("category", "data-science");
    let results = collection.search(&query, 10, Some(filter))?;
    println!("category != 'data-science': {} results", results.len());
    for r in &results {
        println!("  - {} ({})", r.metadata["title"], r.metadata["category"]);
    }

    // =========================================================================
    // Comparison Filters
    // =========================================================================
    println!("\n--- Comparison Filters ---");

    // Greater than
    let filter = Filter::new().gt("views", 1000);
    let results = collection.search(&query, 10, Some(filter))?;
    println!("views > 1000: {} results", results.len());
    for r in &results {
        println!("  - {} ({} views)", r.metadata["title"], r.metadata["views"]);
    }

    // Greater than or equal
    let filter = Filter::new().gte("rating", 4.5);
    let results = collection.search(&query, 10, Some(filter))?;
    println!("\nrating >= 4.5: {} results", results.len());
    for r in &results {
        println!("  - {} (rating: {})", r.metadata["title"], r.metadata["rating"]);
    }

    // Less than
    let filter = Filter::new().lt("views", 1000);
    let results = collection.search(&query, 10, Some(filter))?;
    println!("\nviews < 1000: {} results", results.len());
    for r in &results {
        println!("  - {} ({} views)", r.metadata["title"], r.metadata["views"]);
    }

    // Less than or equal
    let filter = Filter::new().lte("rating", 4.2);
    let results = collection.search(&query, 10, Some(filter))?;
    println!("\nrating <= 4.2: {} results", results.len());
    for r in &results {
        println!("  - {} (rating: {})", r.metadata["title"], r.metadata["rating"]);
    }

    // =========================================================================
    // Set Operators
    // =========================================================================
    println!("\n--- Set Operators ---");

    // $in - matches any value in the set
    let filter = Filter::new().is_in("difficulty", vec![json!("beginner"), json!("intermediate")]);
    let results = collection.search(&query, 10, Some(filter))?;
    println!("difficulty IN ['beginner', 'intermediate']: {} results", results.len());
    for r in &results {
        println!("  - {} ({})", r.metadata["title"], r.metadata["difficulty"]);
    }

    // $nin - matches none of the values in the set
    let filter = Filter::new().not_in("difficulty", vec![json!("advanced")]);
    let results = collection.search(&query, 10, Some(filter))?;
    println!("\ndifficulty NOT IN ['advanced']: {} results", results.len());
    for r in &results {
        println!("  - {} ({})", r.metadata["title"], r.metadata["difficulty"]);
    }

    // =========================================================================
    // Combined Filters (AND semantics)
    // =========================================================================
    println!("\n--- Combined Filters (AND) ---");

    // Multiple conditions
    let filter = Filter::new()
        .eq("category", "programming")
        .eq("published", true)
        .gte("rating", 4.0);
    let results = collection.search(&query, 10, Some(filter))?;
    println!("category='programming' AND published=true AND rating>=4.0: {} results", results.len());
    for r in &results {
        println!("  - {} (rating: {})", r.metadata["title"], r.metadata["rating"]);
    }

    // Range query (between)
    let filter = Filter::new()
        .gte("views", 500)
        .lte("views", 2000);
    let results = collection.search(&query, 10, Some(filter))?;
    println!("\nviews BETWEEN 500 AND 2000: {} results", results.len());
    for r in &results {
        println!("  - {} ({} views)", r.metadata["title"], r.metadata["views"]);
    }

    // Complex filter
    let filter = Filter::new()
        .is_in("category", vec![json!("programming"), json!("data-science")])
        .ne("difficulty", "advanced")
        .gt("views", 100)
        .eq("published", true);
    let results = collection.search(&query, 10, Some(filter))?;
    println!("\nComplex filter: {} results", results.len());
    for r in &results {
        println!("  - {} ({}, {} views)",
            r.metadata["title"],
            r.metadata["difficulty"],
            r.metadata["views"]
        );
    }

    // =========================================================================
    // Filter from JSON
    // =========================================================================
    println!("\n--- Filter from JSON ---");

    // Filters can also be built from JSON (useful for APIs)
    let json_filter = json!({
        "category": "programming",
        "views": {"$gt": 500}
    });
    let filter = Filter::from_json(&json_filter);
    let results = collection.search(&query, 10, Some(filter))?;
    println!("From JSON filter: {} results", results.len());
    for r in &results {
        println!("  - {}", r.metadata["title"]);
    }

    // Cleanup
    std::fs::remove_dir_all(&temp_dir)?;
    println!("\nExample completed!");

    Ok(())
}
