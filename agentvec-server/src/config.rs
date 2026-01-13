use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(name = "agentvec-server")]
#[command(about = "HTTP server for AgentVec vector database")]
pub struct ServerConfig {
    /// Path to the AgentVec database directory
    #[arg(short, long, default_value = "./agentvec.avdb", env = "AGENTVEC_DB_PATH")]
    pub db_path: PathBuf,

    /// Host address to bind to
    #[arg(short = 'H', long, default_value = "0.0.0.0", env = "AGENTVEC_HOST")]
    pub host: String,

    /// Port to listen on
    #[arg(short, long, default_value = "8080", env = "AGENTVEC_PORT")]
    pub port: u16,

    /// Enable CORS for all origins
    #[arg(long, default_value = "false", env = "AGENTVEC_CORS")]
    pub cors: bool,

    /// Request timeout in seconds
    #[arg(long, default_value = "30", env = "AGENTVEC_TIMEOUT")]
    pub timeout: u64,
}

impl ServerConfig {
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}
