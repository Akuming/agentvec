//! Metadata filtering for search operations.

use serde_json::Value as JsonValue;

/// A filter for metadata matching during search.
///
/// Filters use AND semantics - all conditions must match.
///
/// # Example
///
/// ```
/// use agentvec::Filter;
///
/// let filter = Filter::new()
///     .eq("user", "alice")
///     .eq("type", "conversation");
/// ```
#[derive(Debug, Clone, Default)]
pub struct Filter {
    conditions: Vec<(String, JsonValue)>,
}

impl Filter {
    /// Create a new empty filter.
    ///
    /// An empty filter matches all records.
    #[must_use]
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Add an equality condition.
    ///
    /// The record's metadata must have a field with the given name
    /// and the value must equal the provided value.
    ///
    /// # Arguments
    ///
    /// * `field` - The metadata field name
    /// * `value` - The expected value
    ///
    /// # Returns
    ///
    /// Self for method chaining.
    #[must_use]
    pub fn eq(mut self, field: impl Into<String>, value: impl Into<JsonValue>) -> Self {
        self.conditions.push((field.into(), value.into()));
        self
    }

    /// Check if the filter matches the given metadata.
    ///
    /// All conditions must match (AND semantics).
    ///
    /// # Arguments
    ///
    /// * `metadata` - The metadata JSON object to check
    ///
    /// # Returns
    ///
    /// `true` if all conditions match, `false` otherwise.
    #[must_use]
    pub fn matches(&self, metadata: &JsonValue) -> bool {
        self.conditions.iter().all(|(field, expected)| {
            metadata.get(field).map_or(false, |actual| actual == expected)
        })
    }

    /// Returns true if the filter has no conditions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Returns the number of conditions in the filter.
    #[must_use]
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    /// Calculate the over-fetch multiplier for this filter.
    ///
    /// When filtering, we fetch more candidates than `k` to ensure
    /// we have enough matches after filtering. This returns a
    /// reasonable multiplier based on the number of conditions.
    #[must_use]
    pub fn over_fetch_multiplier(&self) -> usize {
        if self.conditions.is_empty() {
            1
        } else {
            // Heuristic: 10x for any filter, could be tuned
            10
        }
    }

    /// Create a filter from a JSON object.
    ///
    /// Each key-value pair becomes an equality condition.
    ///
    /// # Arguments
    ///
    /// * `json` - A JSON object where each key-value pair becomes a condition
    ///
    /// # Example
    ///
    /// ```
    /// use agentvec::Filter;
    /// use serde_json::json;
    ///
    /// let filter = Filter::from_json(&json!({"user": "alice", "type": "chat"}));
    /// ```
    #[must_use]
    pub fn from_json(json: &JsonValue) -> Self {
        let mut filter = Filter::new();
        if let Some(obj) = json.as_object() {
            for (key, val) in obj {
                filter = filter.eq(key.clone(), val.clone());
            }
        }
        filter
    }
}

/// Create a filter from a JSON object.
///
/// Each key-value pair becomes an equality condition.
impl From<JsonValue> for Filter {
    fn from(value: JsonValue) -> Self {
        let mut filter = Filter::new();
        if let Some(obj) = value.as_object() {
            for (key, val) in obj {
                filter = filter.eq(key.clone(), val.clone());
            }
        }
        filter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_empty_filter_matches_all() {
        let filter = Filter::new();
        assert!(filter.matches(&json!({})));
        assert!(filter.matches(&json!({"a": 1})));
        assert!(filter.matches(&json!({"a": 1, "b": "test"})));
    }

    #[test]
    fn test_single_condition() {
        let filter = Filter::new().eq("user", "alice");

        assert!(filter.matches(&json!({"user": "alice"})));
        assert!(filter.matches(&json!({"user": "alice", "extra": 123})));
        assert!(!filter.matches(&json!({"user": "bob"})));
        assert!(!filter.matches(&json!({})));
    }

    #[test]
    fn test_multiple_conditions_and() {
        let filter = Filter::new()
            .eq("user", "alice")
            .eq("type", "conversation");

        assert!(filter.matches(&json!({"user": "alice", "type": "conversation"})));
        assert!(filter.matches(&json!({"user": "alice", "type": "conversation", "id": 1})));
        assert!(!filter.matches(&json!({"user": "alice"}))); // Missing type
        assert!(!filter.matches(&json!({"type": "conversation"}))); // Missing user
        assert!(!filter.matches(&json!({"user": "bob", "type": "conversation"}))); // Wrong user
    }

    #[test]
    fn test_numeric_values() {
        let filter = Filter::new().eq("count", 42);

        assert!(filter.matches(&json!({"count": 42})));
        assert!(!filter.matches(&json!({"count": 43})));
        assert!(!filter.matches(&json!({"count": "42"}))); // String, not number
    }

    #[test]
    fn test_boolean_values() {
        let filter = Filter::new().eq("active", true);

        assert!(filter.matches(&json!({"active": true})));
        assert!(!filter.matches(&json!({"active": false})));
    }

    #[test]
    fn test_null_values() {
        let filter = Filter::new().eq("deleted_at", JsonValue::Null);

        assert!(filter.matches(&json!({"deleted_at": null})));
        assert!(!filter.matches(&json!({"deleted_at": "2024-01-01"})));
        assert!(!filter.matches(&json!({}))); // Field missing is not the same as null
    }

    #[test]
    fn test_from_json() {
        let filter: Filter = json!({"user": "alice", "type": "chat"}).into();

        assert_eq!(filter.len(), 2);
        assert!(filter.matches(&json!({"user": "alice", "type": "chat"})));
    }

    #[test]
    fn test_over_fetch_multiplier() {
        assert_eq!(Filter::new().over_fetch_multiplier(), 1);
        assert_eq!(Filter::new().eq("a", 1).over_fetch_multiplier(), 10);
    }

    #[test]
    fn test_is_empty() {
        assert!(Filter::new().is_empty());
        assert!(!Filter::new().eq("a", 1).is_empty());
    }
}
