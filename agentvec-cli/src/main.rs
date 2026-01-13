//! AgentVec CLI - Command-line interface for AgentVec vector database.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use comfy_table::{presets::UTF8_FULL, Table};
use std::path::PathBuf;

use agentvec::{AgentVec, Filter, Metric};

#[derive(Parser)]
#[command(name = "agentvec")]
#[command(author, version, about = "Command-line interface for AgentVec vector database", long_about = None)]
struct Cli {
    /// Path to the AgentVec database directory
    #[arg(short, long, default_value = ".")]
    path: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show database information
    Info,

    /// List all collections
    Collections,

    /// Show collection statistics
    Stats {
        /// Collection name
        collection: String,
    },

    /// Search for similar vectors
    Search {
        /// Collection name
        collection: String,

        /// Query vector as comma-separated floats (e.g., "0.1,0.2,0.3")
        #[arg(short, long)]
        query: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value = "10")]
        limit: usize,

        /// Filter as JSON (e.g., '{"type": "user"}')
        #[arg(short, long)]
        filter: Option<String>,
    },

    /// Get a vector by ID
    Get {
        /// Collection name
        collection: String,

        /// Vector ID
        id: String,
    },

    /// Compact a collection (remove tombstones)
    Compact {
        /// Collection name
        collection: String,
    },

    /// Export a collection to NDJSON
    Export {
        /// Collection name
        collection: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Import vectors from NDJSON
    Import {
        /// Collection name
        collection: String,

        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Vector dimensions (required for new collections)
        #[arg(short, long)]
        dimensions: Option<usize>,

        /// Distance metric (cosine, dot, l2)
        #[arg(short, long, default_value = "cosine")]
        metric: String,
    },

    /// Validate database integrity
    Validate {
        /// Collection name (optional, validates all if not specified)
        collection: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info => cmd_info(&cli.path),
        Commands::Collections => cmd_collections(&cli.path),
        Commands::Stats { collection } => cmd_stats(&cli.path, &collection),
        Commands::Search {
            collection,
            query,
            limit,
            filter,
        } => cmd_search(&cli.path, &collection, &query, limit, filter),
        Commands::Get { collection, id } => cmd_get(&cli.path, &collection, &id),
        Commands::Compact { collection } => cmd_compact(&cli.path, &collection),
        Commands::Export { collection, output } => cmd_export(&cli.path, &collection, &output),
        Commands::Import {
            collection,
            input,
            dimensions,
            metric,
        } => cmd_import(&cli.path, &collection, &input, dimensions, &metric),
        Commands::Validate { collection } => cmd_validate(&cli.path, collection.as_deref()),
    }
}

fn cmd_info(path: &PathBuf) -> Result<()> {
    let db = AgentVec::open(path).context("Failed to open database")?;
    let collections = db.collections().context("Failed to list collections")?;

    println!("{}", "AgentVec Database".bold().cyan());
    println!("{}: {}", "Path".bold(), path.display());
    println!("{}: {}", "Collections".bold(), collections.len());

    // Show recovery stats
    let recovery = db.recovery_stats();
    if recovery.had_recovery() {
        println!(
            "{}: {} promoted, {} rolled back",
            "Recovery".bold(),
            recovery.promoted,
            recovery.rolled_back
        );
    }

    if !collections.is_empty() {
        println!("\n{}", "Collections:".bold());
        for name in collections {
            let collection = db.get_collection(&name)?;
            let len = collection.len()?;
            let dims = collection.dimensions();
            let metric = format_metric(collection.metric());
            println!(
                "  {} ({} vectors, {}D, {})",
                name.green(),
                len,
                dims,
                metric
            );
        }
    }

    Ok(())
}

fn cmd_collections(path: &PathBuf) -> Result<()> {
    let db = AgentVec::open(path).context("Failed to open database")?;
    let collections = db.collections().context("Failed to list collections")?;

    if collections.is_empty() {
        println!("{}", "No collections found.".yellow());
        return Ok(());
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Name", "Vectors", "Dimensions", "Metric"]);

    for name in collections {
        let collection = db.get_collection(&name)?;
        table.add_row(vec![
            name,
            collection.len()?.to_string(),
            collection.dimensions().to_string(),
            format_metric(collection.metric()),
        ]);
    }

    println!("{table}");
    Ok(())
}

fn cmd_stats(path: &PathBuf, collection_name: &str) -> Result<()> {
    let db = AgentVec::open(path).context("Failed to open database")?;
    let collection = db.get_collection(collection_name)?;

    println!("{} {}", "Collection:".bold(), collection_name.green());
    println!("{}: {}", "Vectors".bold(), collection.len()?);
    println!("{}: {}", "Dimensions".bold(), collection.dimensions());
    println!("{}: {}", "Metric".bold(), format_metric(collection.metric()));
    println!(
        "{}: {} bytes",
        "Vector storage".bold(),
        collection.vectors_size_bytes()
    );
    println!("{}: {}", "Pending writes".bold(), collection.pending_count());

    if collection.has_hnsw_index() {
        println!(
            "{}: {}",
            "HNSW index".bold(),
            if let Some(count) = collection.hnsw_node_count() {
                format!("{} nodes", count)
            } else {
                "not built yet".to_string()
            }
        );
    }

    Ok(())
}

fn cmd_search(
    path: &PathBuf,
    collection_name: &str,
    query_str: &str,
    limit: usize,
    filter: Option<String>,
) -> Result<()> {
    let db = AgentVec::open(path).context("Failed to open database")?;
    let collection = db.get_collection(collection_name)?;

    // Parse query vector
    let query: Vec<f32> = query_str
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("Invalid query vector format. Use comma-separated floats.")?;

    if query.len() != collection.dimensions() {
        anyhow::bail!(
            "Query dimensions ({}) don't match collection dimensions ({})",
            query.len(),
            collection.dimensions()
        );
    }

    // Parse filter if provided
    let filter_obj: Option<Filter> = if let Some(f) = filter {
        let filter_value: serde_json::Value =
            serde_json::from_str(&f).context("Invalid filter JSON")?;
        Some(parse_filter(&filter_value)?)
    } else {
        None
    };

    let results = collection.search(&query, limit, filter_obj)?;

    if results.is_empty() {
        println!("{}", "No results found.".yellow());
        return Ok(());
    }

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["Rank", "ID", "Score", "Metadata"]);

    for (i, result) in results.iter().enumerate() {
        let metadata_str = serde_json::to_string(&result.metadata).unwrap_or_default();

        table.add_row(vec![
            (i + 1).to_string(),
            truncate_id(&result.id),
            format!("{:.6}", result.score),
            truncate_string(&metadata_str, 40),
        ]);
    }

    println!("{table}");
    println!("\n{} {}", "Total results:".bold(), results.len());
    Ok(())
}

fn cmd_get(path: &PathBuf, collection_name: &str, id: &str) -> Result<()> {
    let db = AgentVec::open(path).context("Failed to open database")?;
    let collection = db.get_collection(collection_name)?;

    match collection.get_with_vector(id)? {
        Some(record) => {
            println!("{}: {}", "ID".bold(), record.id);
            println!(
                "{}: {:?}",
                "Vector".bold(),
                &record.vector[..record.vector.len().min(10)]
            );
            if record.vector.len() > 10 {
                println!("  ... ({} dimensions total)", record.vector.len());
            }
            if !record.metadata.is_null() {
                println!(
                    "{}: {}",
                    "Metadata".bold(),
                    serde_json::to_string_pretty(&record.metadata)?
                );
            }
        }
        None => {
            println!("{}", format!("Vector with ID '{}' not found.", id).yellow());
        }
    }

    Ok(())
}

fn cmd_compact(path: &PathBuf, collection_name: &str) -> Result<()> {
    let db = AgentVec::open(path).context("Failed to open database")?;
    let collection = db.get_collection(collection_name)?;

    println!("Compacting collection '{}'...", collection_name);
    let stats = collection.compact()?;

    println!(
        "{} Removed {} expired, {} tombstones in {}ms (freed {} bytes)",
        "Done!".green(),
        stats.expired_removed,
        stats.tombstones_removed,
        stats.duration_ms,
        stats.bytes_freed
    );

    Ok(())
}

fn cmd_export(path: &PathBuf, collection_name: &str, output: &PathBuf) -> Result<()> {
    let db = AgentVec::open(path).context("Failed to open database")?;
    let collection = db.get_collection(collection_name)?;

    println!("Exporting collection '{}'...", collection_name);

    let count = collection.export_to_file(output)?;

    println!(
        "{} Exported {} records to {}",
        "Done!".green(),
        count,
        output.display()
    );

    Ok(())
}

fn cmd_import(
    path: &PathBuf,
    collection_name: &str,
    input: &PathBuf,
    dimensions: Option<usize>,
    metric_str: &str,
) -> Result<()> {
    let db = AgentVec::open(path).context("Failed to open database")?;

    println!("Importing to collection '{}'...", collection_name);

    // Try to get existing collection or create new one
    let collection = match db.get_collection(collection_name) {
        Ok(col) => col,
        Err(_) => {
            // Collection doesn't exist, need dimensions to create it
            let dims = dimensions.context(
                "Collection doesn't exist. Please specify --dimensions to create it.",
            )?;
            let metric = parse_metric(metric_str)?;
            db.collection(collection_name, dims, metric)?
        }
    };

    let stats = collection.import_from_file(input)?;

    println!(
        "{} Imported {} records ({} failed) in {}ms",
        "Done!".green(),
        stats.imported,
        stats.failed,
        stats.duration_ms
    );

    Ok(())
}

fn cmd_validate(path: &PathBuf, collection_name: Option<&str>) -> Result<()> {
    let db = AgentVec::open(path).context("Failed to open database")?;

    let collections: Vec<String> = match collection_name {
        Some(name) => vec![name.to_string()],
        None => db.collections()?,
    };

    if collections.is_empty() {
        println!("{}", "No collections to validate.".yellow());
        return Ok(());
    }

    let mut all_valid = true;

    for name in &collections {
        print!("Validating '{}'... ", name);

        match db.get_collection(name) {
            Ok(collection) => {
                // Basic validation: try to count records
                match collection.len() {
                    Ok(count) => {
                        println!(
                            "{} ({} records, {} pending)",
                            "OK".green(),
                            count,
                            collection.pending_count()
                        );
                    }
                    Err(e) => {
                        println!("{}: {}", "ERROR".red(), e);
                        all_valid = false;
                    }
                }
            }
            Err(e) => {
                println!("{}: {}", "FAILED".red(), e);
                all_valid = false;
            }
        }
    }

    if all_valid {
        println!("\n{}", "All collections are valid.".green().bold());
    } else {
        println!("\n{}", "Some collections have issues.".red().bold());
    }

    Ok(())
}

// Helper functions

fn format_metric(metric: Metric) -> String {
    match metric {
        Metric::Cosine => "cosine".to_string(),
        Metric::Dot => "dot".to_string(),
        Metric::L2 => "l2".to_string(),
    }
}

fn parse_metric(s: &str) -> Result<Metric> {
    match s.to_lowercase().as_str() {
        "cosine" => Ok(Metric::Cosine),
        "dot" => Ok(Metric::Dot),
        "l2" => Ok(Metric::L2),
        _ => anyhow::bail!("Invalid metric '{}'. Use: cosine, dot, or l2", s),
    }
}

fn truncate_id(id: &str) -> String {
    if id.len() > 12 {
        format!("{}...", &id[..12])
    } else {
        id.to_string()
    }
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s.to_string()
    }
}

/// Parse a JSON filter value into a Filter.
fn parse_filter(value: &serde_json::Value) -> Result<Filter> {
    let mut filter = Filter::new();

    if let serde_json::Value::Object(obj) = value {
        for (key, val) in obj {
            match val {
                serde_json::Value::String(s) => {
                    filter = filter.eq(key.clone(), s.as_str());
                }
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        filter = filter.eq(key.clone(), i);
                    } else if let Some(f) = n.as_f64() {
                        filter = filter.eq(key.clone(), f);
                    }
                }
                serde_json::Value::Bool(b) => {
                    filter = filter.eq(key.clone(), *b);
                }
                _ => {
                    anyhow::bail!("Unsupported filter value type for key '{}'", key);
                }
            }
        }
    } else {
        anyhow::bail!("Filter must be a JSON object");
    }

    Ok(filter)
}
