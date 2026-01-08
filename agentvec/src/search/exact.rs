//! Brute-force exact search implementation.
//!
//! Scans all vectors and returns the top-k matches.
//! Uses parallel iteration for large collections.
//! This is efficient for collections up to ~100K vectors.

use rayon::prelude::*;

use crate::config::Metric;
use crate::filter::Filter;
use crate::search::distance::DistanceMetric;
use crate::search::{SearchResult, TopK};
use crate::storage::{MetadataStorage, Record, VectorStorage};
use crate::error::Result;

/// Threshold for using parallel search (number of candidates).
/// Below this, sequential is faster due to thread overhead.
const PARALLEL_THRESHOLD: usize = 500;

/// Brute-force exact search engine.
pub struct ExactSearch;

impl ExactSearch {
    /// Search for the top-k nearest neighbors.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector
    /// * `k` - Number of results to return
    /// * `filter` - Optional metadata filter
    /// * `metric` - Distance metric to use
    /// * `vectors` - Vector storage
    /// * `metadata` - Metadata storage
    ///
    /// # Returns
    ///
    /// A vector of search results, sorted by score.
    pub fn search(
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        metric: Metric,
        vectors: &VectorStorage,
        metadata: &MetadataStorage,
    ) -> Result<Vec<SearchResult>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        // Calculate over-fetch limit for filtered queries
        let fetch_limit = filter
            .map(|f| k * f.over_fetch_multiplier())
            .unwrap_or(k);

        let higher_is_better = metric.higher_is_better();
        let mut topk = TopK::new(fetch_limit, higher_is_better);

        // Collect candidate records with their slots
        let mut candidates: Vec<(u64, Record)> = Vec::new();

        let txn = metadata.begin_read()?;
        metadata.iter_records(&txn, |record| {
            // Skip non-active records
            if !record.is_active() {
                return Ok(true);
            }

            // Apply metadata filter if present
            if let Some(f) = filter {
                if !f.matches(&record.metadata()) {
                    return Ok(true);
                }
            }

            candidates.push((record.slot_offset, record));
            Ok(true)
        })?;

        // Score candidates - use parallel processing for large collections
        let scores: Vec<(f32, usize)> = if candidates.len() >= PARALLEL_THRESHOLD {
            // Parallel scoring for large collections
            candidates
                .par_iter()
                .enumerate()
                .filter_map(|(idx, (slot, _record))| {
                    // Check if slot is valid (not tombstone)
                    if !vectors.is_valid(*slot).unwrap_or(false) {
                        return None;
                    }

                    // Get the vector
                    let vector = vectors.read_slot_ref(*slot).ok()?;

                    // Compute score
                    let score = metric.compute(query, vector);

                    Some((score, idx))
                })
                .collect()
        } else {
            // Sequential scoring for small collections (avoid thread overhead)
            candidates
                .iter()
                .enumerate()
                .filter_map(|(idx, (slot, _record))| {
                    if !vectors.is_valid(*slot).unwrap_or(false) {
                        return None;
                    }
                    let vector = vectors.read_slot_ref(*slot).ok()?;
                    let score = metric.compute(query, vector);
                    Some((score, idx))
                })
                .collect()
        };

        // Add all scores to top-k
        for (score, idx) in scores {
            topk.push(score, idx);
        }

        // Convert to search results
        let results: Vec<SearchResult> = topk
            .into_results()
            .into_iter()
            .take(k) // Take only k, even if we over-fetched
            .map(|(score, idx)| {
                let (_, record) = &candidates[idx];
                SearchResult::new(&record.id, score, record.metadata())
            })
            .collect();

        Ok(results)
    }

    /// Search with pre-collected records (for use by Collection).
    ///
    /// This variant takes a slice of records and their vectors directly,
    /// avoiding re-reading from storage during search.
    pub fn search_with_data(
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        metric: Metric,
        records: &[(Record, &[f32])],
    ) -> Vec<SearchResult> {
        if k == 0 {
            return Vec::new();
        }

        let higher_is_better = metric.higher_is_better();
        let mut topk = TopK::new(k, higher_is_better);

        // Score records - use parallel for large collections
        let scores: Vec<(f32, usize)> = if records.len() >= PARALLEL_THRESHOLD {
            records
                .par_iter()
                .enumerate()
                .filter_map(|(idx, (record, vector))| {
                    // Skip non-active records
                    if !record.is_active() {
                        return None;
                    }

                    // Apply metadata filter if present
                    if let Some(f) = filter {
                        if !f.matches(&record.metadata()) {
                            return None;
                        }
                    }

                    // Compute score
                    let score = metric.compute(query, vector);
                    Some((score, idx))
                })
                .collect()
        } else {
            records
                .iter()
                .enumerate()
                .filter_map(|(idx, (record, vector))| {
                    if !record.is_active() {
                        return None;
                    }
                    if let Some(f) = filter {
                        if !f.matches(&record.metadata()) {
                            return None;
                        }
                    }
                    let score = metric.compute(query, vector);
                    Some((score, idx))
                })
                .collect()
        };

        // Add to top-k
        for (score, idx) in scores {
            topk.push(score, idx);
        }

        // Convert to search results
        topk.into_results()
            .into_iter()
            .map(|(score, idx)| {
                let (record, _) = &records[idx];
                SearchResult::new(&record.id, score, record.metadata())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::distance::normalize_l2;
    use crate::recovery::RecordStatus;
    use serde_json::json;

    fn make_record(id: &str, metadata: serde_json::Value) -> Record {
        Record::new(id, 0, metadata, None, RecordStatus::Active)
    }

    #[test]
    fn test_search_with_data_basic() {
        let v1 = normalize_l2(&[1.0, 0.0, 0.0]);
        let v2 = normalize_l2(&[0.0, 1.0, 0.0]);
        let v3 = normalize_l2(&[1.0, 1.0, 0.0]);

        let records: Vec<(Record, &[f32])> = vec![
            (make_record("id1", json!({"name": "v1"})), v1.as_slice()),
            (make_record("id2", json!({"name": "v2"})), v2.as_slice()),
            (make_record("id3", json!({"name": "v3"})), v3.as_slice()),
        ];

        let query = normalize_l2(&[1.0, 0.0, 0.0]);
        let results = ExactSearch::search_with_data(&query, 2, None, Metric::Cosine, &records);

        assert_eq!(results.len(), 2);
        // v1 should be closest (identical direction)
        assert_eq!(results[0].id, "id1");
        assert!((results[0].score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_search_with_filter() {
        let v1 = normalize_l2(&[1.0, 0.0, 0.0]);
        let v2 = normalize_l2(&[0.9, 0.1, 0.0]);
        let v3 = normalize_l2(&[0.8, 0.2, 0.0]);

        let records: Vec<(Record, &[f32])> = vec![
            (make_record("id1", json!({"type": "a"})), v1.as_slice()),
            (make_record("id2", json!({"type": "b"})), v2.as_slice()),
            (make_record("id3", json!({"type": "a"})), v3.as_slice()),
        ];

        let filter = Filter::new().eq("type", "a");
        let query = normalize_l2(&[1.0, 0.0, 0.0]);
        let results = ExactSearch::search_with_data(&query, 10, Some(&filter), Metric::Cosine, &records);

        // Should only return type="a" records
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.metadata["type"], "a");
        }
    }

    #[test]
    fn test_search_k_zero() {
        let v1 = normalize_l2(&[1.0, 0.0, 0.0]);
        let records: Vec<(Record, &[f32])> = vec![
            (make_record("id1", json!({})), v1.as_slice()),
        ];

        let query = normalize_l2(&[1.0, 0.0, 0.0]);
        let results = ExactSearch::search_with_data(&query, 0, None, Metric::Cosine, &records);

        assert!(results.is_empty());
    }

    #[test]
    fn test_search_l2_metric() {
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let v3 = [2.0, 0.0, 0.0];

        let records: Vec<(Record, &[f32])> = vec![
            (make_record("id1", json!({})), &v1),
            (make_record("id2", json!({})), &v2),
            (make_record("id3", json!({})), &v3),
        ];

        let query = [1.0, 0.0, 0.0];
        let results = ExactSearch::search_with_data(&query, 2, None, Metric::L2, &records);

        assert_eq!(results.len(), 2);
        // v1 should be closest (distance 0)
        assert_eq!(results[0].id, "id1");
        assert!((results[0].score - 0.0).abs() < 0.001);
        // v3 should be second (distance 1.0 squared = 1.0)
        assert_eq!(results[1].id, "id3");
        assert!((results[1].score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_search_skips_expired() {
        let v1 = normalize_l2(&[1.0, 0.0, 0.0]);
        let v2 = normalize_l2(&[0.9, 0.1, 0.0]);

        let mut expired_record = make_record("id1", json!({}));
        expired_record.expires_at = Some(0); // Expired

        let records: Vec<(Record, &[f32])> = vec![
            (expired_record, v1.as_slice()),
            (make_record("id2", json!({})), v2.as_slice()),
        ];

        let query = normalize_l2(&[1.0, 0.0, 0.0]);
        let results = ExactSearch::search_with_data(&query, 10, None, Metric::Cosine, &records);

        // Should only return non-expired record
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "id2");
    }

    #[test]
    fn test_search_skips_tombstone() {
        let v1 = normalize_l2(&[1.0, 0.0, 0.0]);
        let v2 = normalize_l2(&[0.9, 0.1, 0.0]);

        let mut tombstone = make_record("id1", json!({}));
        tombstone.status = RecordStatus::Tombstone;

        let records: Vec<(Record, &[f32])> = vec![
            (tombstone, v1.as_slice()),
            (make_record("id2", json!({})), v2.as_slice()),
        ];

        let query = normalize_l2(&[1.0, 0.0, 0.0]);
        let results = ExactSearch::search_with_data(&query, 10, None, Metric::Cosine, &records);

        // Should only return active record
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "id2");
    }

    #[test]
    fn test_search_ordering() {
        // Create vectors with known similarity ordering
        let query_norm = normalize_l2(&[1.0, 0.0, 0.0]);

        let v1 = normalize_l2(&[0.5, 0.5, 0.0]);   // cos ~ 0.707
        let v2 = normalize_l2(&[0.9, 0.1, 0.0]);   // cos ~ 0.994
        let v3 = normalize_l2(&[0.1, 0.9, 0.0]);   // cos ~ 0.111
        let v4 = normalize_l2(&[1.0, 0.0, 0.0]);   // cos = 1.0

        let records: Vec<(Record, &[f32])> = vec![
            (make_record("id1", json!({})), v1.as_slice()),
            (make_record("id2", json!({})), v2.as_slice()),
            (make_record("id3", json!({})), v3.as_slice()),
            (make_record("id4", json!({})), v4.as_slice()),
        ];

        let results = ExactSearch::search_with_data(&query_norm, 4, None, Metric::Cosine, &records);

        assert_eq!(results.len(), 4);
        // Verify ordering: id4, id2, id1, id3
        assert_eq!(results[0].id, "id4"); // cos = 1.0
        assert_eq!(results[1].id, "id2"); // cos ~ 0.994
        assert_eq!(results[2].id, "id1"); // cos ~ 0.707
        assert_eq!(results[3].id, "id3"); // cos ~ 0.111
    }
}
