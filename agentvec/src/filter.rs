//! Metadata filtering for search operations.
//!
//! Supports MongoDB-like query operators for flexible filtering.
//!
//! # Operators
//!
//! - `$eq` - Equality (default when value is not an object with operators)
//! - `$ne` - Not equal
//! - `$gt` - Greater than
//! - `$gte` - Greater than or equal
//! - `$lt` - Less than
//! - `$lte` - Less than or equal
//! - `$in` - Value is in array
//! - `$nin` - Value is not in array
//!
//! # Examples
//!
//! ```
//! use agentvec::Filter;
//! use serde_json::json;
//!
//! // Simple equality
//! let filter = Filter::new().eq("user", "alice");
//!
//! // From JSON with operators
//! let filter = Filter::from_json(&json!({
//!     "user": "alice",
//!     "age": {"$gte": 18, "$lt": 65},
//!     "status": {"$in": ["active", "pending"]}
//! }));
//! ```

use serde_json::Value as JsonValue;

/// Comparison operators for filter conditions.
#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    /// Equality: field == value
    Eq(JsonValue),
    /// Not equal: field != value
    Ne(JsonValue),
    /// Greater than: field > value (numeric/string comparison)
    Gt(JsonValue),
    /// Greater than or equal: field >= value
    Gte(JsonValue),
    /// Less than: field < value
    Lt(JsonValue),
    /// Less than or equal: field <= value
    Lte(JsonValue),
    /// Value is in the given array
    In(Vec<JsonValue>),
    /// Value is not in the given array
    Nin(Vec<JsonValue>),
}

/// A condition on a single field, which may have multiple operators.
#[derive(Debug, Clone)]
pub struct FieldCondition {
    pub field: String,
    pub operators: Vec<Operator>,
}

impl FieldCondition {
    /// Create a new field condition with a single operator.
    pub fn new(field: impl Into<String>, operator: Operator) -> Self {
        Self {
            field: field.into(),
            operators: vec![operator],
        }
    }

    /// Add another operator to this condition (AND semantics).
    pub fn and(mut self, operator: Operator) -> Self {
        self.operators.push(operator);
        self
    }

    /// Check if the condition matches the given metadata value.
    pub fn matches(&self, metadata: &JsonValue) -> bool {
        let actual = match metadata.get(&self.field) {
            Some(v) => v,
            None => return false, // Field doesn't exist
        };

        // All operators must match (AND semantics)
        self.operators.iter().all(|op| op.matches(actual))
    }
}

impl Operator {
    /// Check if this operator matches the given value.
    pub fn matches(&self, actual: &JsonValue) -> bool {
        match self {
            Operator::Eq(expected) => actual == expected,
            Operator::Ne(expected) => actual != expected,
            Operator::Gt(expected) => compare_values(actual, expected) == Some(std::cmp::Ordering::Greater),
            Operator::Gte(expected) => {
                matches!(compare_values(actual, expected), Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal))
            }
            Operator::Lt(expected) => compare_values(actual, expected) == Some(std::cmp::Ordering::Less),
            Operator::Lte(expected) => {
                matches!(compare_values(actual, expected), Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal))
            }
            Operator::In(values) => values.iter().any(|v| actual == v),
            Operator::Nin(values) => !values.iter().any(|v| actual == v),
        }
    }
}

/// Compare two JSON values.
///
/// Returns `None` if the values are not comparable (different types or non-comparable types).
/// Supports comparison for:
/// - Numbers (compared as f64)
/// - Strings (lexicographic comparison)
fn compare_values(a: &JsonValue, b: &JsonValue) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (JsonValue::Number(a), JsonValue::Number(b)) => {
            let a = a.as_f64()?;
            let b = b.as_f64()?;
            a.partial_cmp(&b)
        }
        (JsonValue::String(a), JsonValue::String(b)) => Some(a.cmp(b)),
        _ => None, // Types don't match or aren't comparable
    }
}

/// A filter for metadata matching during search.
///
/// Filters use AND semantics - all conditions must match.
///
/// # Example
///
/// ```
/// use agentvec::Filter;
/// use serde_json::json;
///
/// // Builder API
/// let filter = Filter::new()
///     .eq("user", "alice")
///     .gte("age", 18)
///     .lt("age", 65);
///
/// // From JSON with operators
/// let filter = Filter::from_json(&json!({
///     "user": "alice",
///     "age": {"$gte": 18, "$lt": 65},
///     "status": {"$in": ["active", "pending"]}
/// }));
/// ```
#[derive(Debug, Clone, Default)]
pub struct Filter {
    conditions: Vec<FieldCondition>,
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
    #[must_use]
    pub fn eq(mut self, field: impl Into<String>, value: impl Into<JsonValue>) -> Self {
        self.conditions.push(FieldCondition::new(field, Operator::Eq(value.into())));
        self
    }

    /// Add a not-equal condition.
    #[must_use]
    pub fn ne(mut self, field: impl Into<String>, value: impl Into<JsonValue>) -> Self {
        self.conditions.push(FieldCondition::new(field, Operator::Ne(value.into())));
        self
    }

    /// Add a greater-than condition.
    #[must_use]
    pub fn gt(mut self, field: impl Into<String>, value: impl Into<JsonValue>) -> Self {
        self.conditions.push(FieldCondition::new(field, Operator::Gt(value.into())));
        self
    }

    /// Add a greater-than-or-equal condition.
    #[must_use]
    pub fn gte(mut self, field: impl Into<String>, value: impl Into<JsonValue>) -> Self {
        self.conditions.push(FieldCondition::new(field, Operator::Gte(value.into())));
        self
    }

    /// Add a less-than condition.
    #[must_use]
    pub fn lt(mut self, field: impl Into<String>, value: impl Into<JsonValue>) -> Self {
        self.conditions.push(FieldCondition::new(field, Operator::Lt(value.into())));
        self
    }

    /// Add a less-than-or-equal condition.
    #[must_use]
    pub fn lte(mut self, field: impl Into<String>, value: impl Into<JsonValue>) -> Self {
        self.conditions.push(FieldCondition::new(field, Operator::Lte(value.into())));
        self
    }

    /// Add an "in" condition (value must be one of the given values).
    #[must_use]
    pub fn is_in(mut self, field: impl Into<String>, values: Vec<JsonValue>) -> Self {
        self.conditions.push(FieldCondition::new(field, Operator::In(values)));
        self
    }

    /// Add a "not in" condition (value must not be any of the given values).
    #[must_use]
    pub fn not_in(mut self, field: impl Into<String>, values: Vec<JsonValue>) -> Self {
        self.conditions.push(FieldCondition::new(field, Operator::Nin(values)));
        self
    }

    /// Check if the filter matches the given metadata.
    ///
    /// All conditions must match (AND semantics).
    #[must_use]
    pub fn matches(&self, metadata: &JsonValue) -> bool {
        self.conditions.iter().all(|cond| cond.matches(metadata))
    }

    /// Returns true if the filter has no conditions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }

    /// Returns the number of field conditions in the filter.
    #[must_use]
    pub fn len(&self) -> usize {
        self.conditions.len()
    }

    /// Calculate the over-fetch multiplier for this filter.
    ///
    /// When filtering, we fetch more candidates than `k` to ensure
    /// we have enough matches after filtering.
    #[must_use]
    pub fn over_fetch_multiplier(&self) -> usize {
        if self.conditions.is_empty() {
            1
        } else {
            // Heuristic: 10x for any filter
            10
        }
    }

    /// Create a filter from a JSON object.
    ///
    /// Supports MongoDB-like operator syntax:
    /// - Simple equality: `{"field": "value"}`
    /// - Operators: `{"field": {"$gt": 10, "$lte": 100}}`
    /// - Set membership: `{"field": {"$in": [1, 2, 3]}}`
    ///
    /// # Example
    ///
    /// ```
    /// use agentvec::Filter;
    /// use serde_json::json;
    ///
    /// let filter = Filter::from_json(&json!({
    ///     "user": "alice",
    ///     "age": {"$gte": 18, "$lt": 65},
    ///     "status": {"$in": ["active", "pending"]}
    /// }));
    /// ```
    #[must_use]
    pub fn from_json(json: &JsonValue) -> Self {
        let mut filter = Filter::new();

        if let Some(obj) = json.as_object() {
            for (field, value) in obj {
                if let Some(condition) = parse_field_condition(field, value) {
                    filter.conditions.push(condition);
                }
            }
        }

        filter
    }
}

/// Parse a field condition from a JSON value.
///
/// If the value is an object with operator keys ($gt, $lt, etc.), parse as operators.
/// Otherwise, treat as equality condition.
fn parse_field_condition(field: &str, value: &JsonValue) -> Option<FieldCondition> {
    if let Some(obj) = value.as_object() {
        // Check if this looks like an operator object
        let has_operators = obj.keys().any(|k| k.starts_with('$'));

        if has_operators {
            let mut operators = Vec::new();

            for (key, val) in obj {
                match key.as_str() {
                    "$eq" => operators.push(Operator::Eq(val.clone())),
                    "$ne" => operators.push(Operator::Ne(val.clone())),
                    "$gt" => operators.push(Operator::Gt(val.clone())),
                    "$gte" => operators.push(Operator::Gte(val.clone())),
                    "$lt" => operators.push(Operator::Lt(val.clone())),
                    "$lte" => operators.push(Operator::Lte(val.clone())),
                    "$in" => {
                        if let Some(arr) = val.as_array() {
                            operators.push(Operator::In(arr.clone()));
                        }
                    }
                    "$nin" => {
                        if let Some(arr) = val.as_array() {
                            operators.push(Operator::Nin(arr.clone()));
                        }
                    }
                    _ => {
                        // Unknown operator, ignore
                    }
                }
            }

            if operators.is_empty() {
                return None;
            }

            return Some(FieldCondition {
                field: field.to_string(),
                operators,
            });
        }
    }

    // Not an operator object, treat as equality
    Some(FieldCondition::new(field, Operator::Eq(value.clone())))
}

/// Create a filter from a JSON object.
impl From<JsonValue> for Filter {
    fn from(value: JsonValue) -> Self {
        Filter::from_json(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ========== Basic equality tests (existing functionality) ==========

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
    fn test_from_json_simple() {
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

    // ========== New operator tests ==========

    #[test]
    fn test_ne_operator() {
        let filter = Filter::new().ne("status", "deleted");

        assert!(filter.matches(&json!({"status": "active"})));
        assert!(filter.matches(&json!({"status": "pending"})));
        assert!(!filter.matches(&json!({"status": "deleted"})));
        assert!(!filter.matches(&json!({}))); // Field must exist
    }

    #[test]
    fn test_gt_operator_numbers() {
        let filter = Filter::new().gt("age", 18);

        assert!(filter.matches(&json!({"age": 19})));
        assert!(filter.matches(&json!({"age": 100})));
        assert!(!filter.matches(&json!({"age": 18}))); // Not greater
        assert!(!filter.matches(&json!({"age": 17})));
        assert!(!filter.matches(&json!({"age": "20"}))); // String, not comparable
    }

    #[test]
    fn test_gte_operator_numbers() {
        let filter = Filter::new().gte("age", 18);

        assert!(filter.matches(&json!({"age": 18})));
        assert!(filter.matches(&json!({"age": 19})));
        assert!(!filter.matches(&json!({"age": 17})));
    }

    #[test]
    fn test_lt_operator_numbers() {
        let filter = Filter::new().lt("age", 65);

        assert!(filter.matches(&json!({"age": 64})));
        assert!(filter.matches(&json!({"age": 18})));
        assert!(!filter.matches(&json!({"age": 65}))); // Not less
        assert!(!filter.matches(&json!({"age": 100})));
    }

    #[test]
    fn test_lte_operator_numbers() {
        let filter = Filter::new().lte("age", 65);

        assert!(filter.matches(&json!({"age": 65})));
        assert!(filter.matches(&json!({"age": 64})));
        assert!(!filter.matches(&json!({"age": 66})));
    }

    #[test]
    fn test_range_query() {
        // age >= 18 AND age < 65
        let filter = Filter::new().gte("age", 18).lt("age", 65);

        assert!(filter.matches(&json!({"age": 18})));
        assert!(filter.matches(&json!({"age": 30})));
        assert!(filter.matches(&json!({"age": 64})));
        assert!(!filter.matches(&json!({"age": 17})));
        assert!(!filter.matches(&json!({"age": 65})));
    }

    #[test]
    fn test_string_comparison() {
        let filter = Filter::new().gte("name", "M");

        assert!(filter.matches(&json!({"name": "Mike"})));
        assert!(filter.matches(&json!({"name": "Zoe"})));
        assert!(!filter.matches(&json!({"name": "Alice"})));
        assert!(!filter.matches(&json!({"name": "Bob"})));
    }

    #[test]
    fn test_in_operator() {
        let filter = Filter::new().is_in("status", vec![json!("active"), json!("pending")]);

        assert!(filter.matches(&json!({"status": "active"})));
        assert!(filter.matches(&json!({"status": "pending"})));
        assert!(!filter.matches(&json!({"status": "deleted"})));
        assert!(!filter.matches(&json!({"status": "archived"})));
    }

    #[test]
    fn test_nin_operator() {
        let filter = Filter::new().not_in("status", vec![json!("deleted"), json!("archived")]);

        assert!(filter.matches(&json!({"status": "active"})));
        assert!(filter.matches(&json!({"status": "pending"})));
        assert!(!filter.matches(&json!({"status": "deleted"})));
        assert!(!filter.matches(&json!({"status": "archived"})));
    }

    #[test]
    fn test_in_with_numbers() {
        let filter = Filter::new().is_in("priority", vec![json!(1), json!(2), json!(3)]);

        assert!(filter.matches(&json!({"priority": 1})));
        assert!(filter.matches(&json!({"priority": 2})));
        assert!(!filter.matches(&json!({"priority": 4})));
        assert!(!filter.matches(&json!({"priority": "1"}))); // String not in number list
    }

    // ========== JSON operator syntax tests ==========

    #[test]
    fn test_from_json_with_gt() {
        let filter = Filter::from_json(&json!({
            "age": {"$gt": 18}
        }));

        assert!(filter.matches(&json!({"age": 19})));
        assert!(!filter.matches(&json!({"age": 18})));
    }

    #[test]
    fn test_from_json_with_range() {
        let filter = Filter::from_json(&json!({
            "age": {"$gte": 18, "$lt": 65}
        }));

        assert!(filter.matches(&json!({"age": 18})));
        assert!(filter.matches(&json!({"age": 30})));
        assert!(!filter.matches(&json!({"age": 17})));
        assert!(!filter.matches(&json!({"age": 65})));
    }

    #[test]
    fn test_from_json_with_in() {
        let filter = Filter::from_json(&json!({
            "status": {"$in": ["active", "pending"]}
        }));

        assert!(filter.matches(&json!({"status": "active"})));
        assert!(filter.matches(&json!({"status": "pending"})));
        assert!(!filter.matches(&json!({"status": "deleted"})));
    }

    #[test]
    fn test_from_json_with_nin() {
        let filter = Filter::from_json(&json!({
            "status": {"$nin": ["deleted", "archived"]}
        }));

        assert!(filter.matches(&json!({"status": "active"})));
        assert!(!filter.matches(&json!({"status": "deleted"})));
    }

    #[test]
    fn test_from_json_complex() {
        let filter = Filter::from_json(&json!({
            "user": "alice",
            "age": {"$gte": 18, "$lt": 65},
            "status": {"$in": ["active", "pending"]},
            "role": {"$ne": "guest"}
        }));

        assert!(filter.matches(&json!({
            "user": "alice",
            "age": 30,
            "status": "active",
            "role": "admin"
        })));

        assert!(!filter.matches(&json!({
            "user": "alice",
            "age": 30,
            "status": "active",
            "role": "guest"  // Wrong role
        })));

        assert!(!filter.matches(&json!({
            "user": "bob",  // Wrong user
            "age": 30,
            "status": "active",
            "role": "admin"
        })));
    }

    #[test]
    fn test_from_json_mixed_operators_and_equality() {
        let filter = Filter::from_json(&json!({
            "type": "user",  // Simple equality
            "score": {"$gte": 100}  // Operator
        }));

        assert!(filter.matches(&json!({"type": "user", "score": 100})));
        assert!(filter.matches(&json!({"type": "user", "score": 200})));
        assert!(!filter.matches(&json!({"type": "user", "score": 50})));
        assert!(!filter.matches(&json!({"type": "admin", "score": 200})));
    }

    #[test]
    fn test_from_json_explicit_eq() {
        let filter = Filter::from_json(&json!({
            "status": {"$eq": "active"}
        }));

        assert!(filter.matches(&json!({"status": "active"})));
        assert!(!filter.matches(&json!({"status": "inactive"})));
    }

    #[test]
    fn test_nested_object_equality() {
        // A value that looks like operators but isn't (no $ prefix)
        let filter = Filter::from_json(&json!({
            "config": {"enabled": true, "level": 5}
        }));

        assert!(filter.matches(&json!({
            "config": {"enabled": true, "level": 5}
        })));
        assert!(!filter.matches(&json!({
            "config": {"enabled": false, "level": 5}
        })));
    }

    #[test]
    fn test_float_comparison() {
        let filter = Filter::new().gt("score", 0.5);

        assert!(filter.matches(&json!({"score": 0.75})));
        assert!(filter.matches(&json!({"score": 1.0})));
        assert!(!filter.matches(&json!({"score": 0.5})));
        assert!(!filter.matches(&json!({"score": 0.25})));
    }

    #[test]
    fn test_comparison_type_mismatch() {
        // Comparing number to string should not match for ordering operators
        let filter = Filter::new().gt("value", 10);

        assert!(!filter.matches(&json!({"value": "20"}))); // String "20" not > number 10
        assert!(!filter.matches(&json!({"value": null})));
        assert!(!filter.matches(&json!({"value": true})));
    }
}
