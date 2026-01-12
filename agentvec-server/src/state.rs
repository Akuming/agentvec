use std::sync::Arc;
use agentvec::AgentVec;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<AgentVec>,
}

impl AppState {
    pub fn new(db: AgentVec) -> Self {
        Self { db: Arc::new(db) }
    }
}
