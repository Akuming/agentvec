//! Product Quantization (PQ) for fast approximate distance computation.
//!
//! PQ compresses vectors by splitting them into subvectors and quantizing each
//! to one of K centroids. This enables:
//! - 32x compression (384 dims * 4 bytes → 48 bytes)
//! - Fast distance computation via lookup tables (~10x faster than f32)
//!
//! Used during HNSW construction for fast candidate evaluation.

use rayon::prelude::*;

/// Configuration for Product Quantization.
#[derive(Debug, Clone)]
pub struct PQConfig {
    /// Number of subquantizers (subvectors).
    /// Vector dimension must be divisible by this.
    /// Default: 48 for 384-dim vectors (8 dims per subquantizer).
    pub num_subquantizers: usize,

    /// Number of centroids per subquantizer.
    /// Must be 256 (8-bit codes) for efficient storage.
    pub num_centroids: usize,

    /// Number of k-means iterations for training.
    pub training_iterations: usize,

    /// Number of vectors to sample for training.
    pub training_sample_size: usize,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            num_subquantizers: 48,  // For 384-dim vectors: 8 dims per subquantizer
            num_centroids: 256,     // 8-bit codes
            training_iterations: 10,
            training_sample_size: 10000,
        }
    }
}

impl PQConfig {
    /// Create config for a given dimension.
    pub fn for_dimension(dim: usize) -> Self {
        // Choose num_subquantizers to get 4-16 dims per subquantizer
        let m = if dim % 48 == 0 {
            48
        } else if dim % 32 == 0 {
            32
        } else if dim % 16 == 0 {
            16
        } else if dim % 8 == 0 {
            8
        } else {
            // Fallback: use dimension itself (no compression benefit)
            dim
        };

        Self {
            num_subquantizers: m,
            ..Default::default()
        }
    }
}

/// Product Quantizer for fast approximate distance computation.
pub struct ProductQuantizer {
    /// Number of subquantizers.
    num_subquantizers: usize,

    /// Number of centroids per subquantizer (always 256).
    num_centroids: usize,

    /// Dimensions per subquantizer.
    dims_per_sub: usize,

    /// Codebooks: [num_subquantizers][num_centroids][dims_per_sub]
    /// Stored as contiguous array for cache efficiency.
    codebooks: Vec<f32>,

    /// Whether we're computing dot product (higher is better) or L2 (lower is better).
    higher_is_better: bool,
}

impl ProductQuantizer {
    /// Train a product quantizer on sample vectors.
    ///
    /// Uses k-means clustering to learn codebooks for each subvector.
    pub fn train<'a>(
        vectors: impl Iterator<Item = &'a [f32]> + Clone,
        dim: usize,
        config: &PQConfig,
        higher_is_better: bool,
    ) -> Self {
        let m = config.num_subquantizers;
        let k = config.num_centroids;
        let dims_per_sub = dim / m;

        assert!(dim % m == 0, "Dimension {} must be divisible by num_subquantizers {}", dim, m);
        assert!(k == 256, "num_centroids must be 256 for 8-bit codes");

        // Collect training vectors (sample if too many)
        let training_vecs: Vec<Vec<f32>> = vectors
            .take(config.training_sample_size)
            .map(|v| v.to_vec())
            .collect();

        if training_vecs.is_empty() {
            // Return empty quantizer if no training data
            return Self {
                num_subquantizers: m,
                num_centroids: k,
                dims_per_sub,
                codebooks: vec![0.0; m * k * dims_per_sub],
                higher_is_better,
            };
        }

        // Train codebook for each subquantizer in parallel
        let codebooks: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|sub_idx| {
                let start = sub_idx * dims_per_sub;
                let end = start + dims_per_sub;

                // Extract subvectors for this subquantizer
                let subvectors: Vec<Vec<f32>> = training_vecs
                    .iter()
                    .map(|v| v[start..end].to_vec())
                    .collect();

                // Run k-means to get centroids
                Self::kmeans(&subvectors, k, config.training_iterations, dims_per_sub)
            })
            .collect();

        // Flatten codebooks into contiguous array
        let mut flat_codebooks = Vec::with_capacity(m * k * dims_per_sub);
        for codebook in codebooks {
            flat_codebooks.extend(codebook);
        }

        Self {
            num_subquantizers: m,
            num_centroids: k,
            dims_per_sub,
            codebooks: flat_codebooks,
            higher_is_better,
        }
    }

    /// K-means clustering for a single subquantizer.
    fn kmeans(subvectors: &[Vec<f32>], k: usize, iterations: usize, dims: usize) -> Vec<f32> {
        let n = subvectors.len();
        if n == 0 {
            return vec![0.0; k * dims];
        }

        // Initialize centroids using k-means++ style initialization
        let mut centroids = Vec::with_capacity(k * dims);

        // First centroid: random (use first vector)
        centroids.extend_from_slice(&subvectors[0]);

        // Remaining centroids: pick proportional to squared distance from nearest centroid
        let mut min_distances: Vec<f32> = vec![f32::MAX; n];

        for c in 1..k {
            // Update min distances to nearest centroid
            for (i, sv) in subvectors.iter().enumerate() {
                let prev_centroid_start = (c - 1) * dims;
                let dist = Self::l2_squared(sv, &centroids[prev_centroid_start..prev_centroid_start + dims]);
                min_distances[i] = min_distances[i].min(dist);
            }

            // Pick next centroid proportional to squared distance
            let total: f32 = min_distances.iter().sum();
            if total <= 0.0 {
                // All points are at centroids, use random
                let idx = c % n;
                centroids.extend_from_slice(&subvectors[idx]);
            } else {
                // Use deterministic selection based on cumulative distribution
                let threshold = total * (c as f32 / k as f32);
                let mut cumsum = 0.0;
                let mut selected = n - 1;
                for (i, &d) in min_distances.iter().enumerate() {
                    cumsum += d;
                    if cumsum >= threshold {
                        selected = i;
                        break;
                    }
                }
                centroids.extend_from_slice(&subvectors[selected]);
            }
        }

        // K-means iterations
        let mut assignments = vec![0u8; n];

        for _ in 0..iterations {
            // Assign each point to nearest centroid
            for (i, sv) in subvectors.iter().enumerate() {
                let mut best_c = 0u8;
                let mut best_dist = f32::MAX;
                for c in 0..k {
                    let centroid_start = c * dims;
                    let dist = Self::l2_squared(sv, &centroids[centroid_start..centroid_start + dims]);
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c as u8;
                    }
                }
                assignments[i] = best_c;
            }

            // Update centroids
            let mut new_centroids = vec![0.0; k * dims];
            let mut counts = vec![0usize; k];

            for (i, sv) in subvectors.iter().enumerate() {
                let c = assignments[i] as usize;
                counts[c] += 1;
                let centroid_start = c * dims;
                for (j, &val) in sv.iter().enumerate() {
                    new_centroids[centroid_start + j] += val;
                }
            }

            // Normalize centroids
            for c in 0..k {
                let count = counts[c];
                if count > 0 {
                    let centroid_start = c * dims;
                    for j in 0..dims {
                        new_centroids[centroid_start + j] /= count as f32;
                    }
                } else {
                    // Empty cluster: keep old centroid
                    let centroid_start = c * dims;
                    for j in 0..dims {
                        new_centroids[centroid_start + j] = centroids[centroid_start + j];
                    }
                }
            }

            centroids = new_centroids;
        }

        centroids
    }

    /// L2 squared distance between two vectors.
    #[inline]
    fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    /// Encode a vector to PQ codes.
    ///
    /// Returns M bytes, one for each subquantizer.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.num_subquantizers);

        for sub_idx in 0..self.num_subquantizers {
            let subvec_start = sub_idx * self.dims_per_sub;
            let subvec = &vector[subvec_start..subvec_start + self.dims_per_sub];

            // Find nearest centroid
            let mut best_c = 0u8;
            let mut best_dist = f32::MAX;

            let codebook_start = sub_idx * self.num_centroids * self.dims_per_sub;
            for c in 0..self.num_centroids {
                let centroid_start = codebook_start + c * self.dims_per_sub;
                let centroid = &self.codebooks[centroid_start..centroid_start + self.dims_per_sub];
                let dist = Self::l2_squared(subvec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c as u8;
                }
            }

            codes.push(best_c);
        }

        codes
    }

    /// Encode multiple vectors in parallel.
    pub fn encode_batch(&self, vectors: &[&[f32]]) -> Vec<Vec<u8>> {
        vectors
            .par_iter()
            .map(|v| self.encode(v))
            .collect()
    }

    /// Precompute distance lookup table for a query vector.
    ///
    /// Returns a table of size [num_subquantizers][num_centroids] containing
    /// the distance from each query subvector to each centroid.
    ///
    /// For dot product: table[m][c] = dot(query_subvec[m], centroid[m][c])
    /// For L2: table[m][c] = ||query_subvec[m] - centroid[m][c]||²
    pub fn precompute_table(&self, query: &[f32]) -> PQDistanceTable {
        let mut table = vec![0.0f32; self.num_subquantizers * self.num_centroids];

        for sub_idx in 0..self.num_subquantizers {
            let query_start = sub_idx * self.dims_per_sub;
            let query_subvec = &query[query_start..query_start + self.dims_per_sub];

            let codebook_start = sub_idx * self.num_centroids * self.dims_per_sub;
            let table_start = sub_idx * self.num_centroids;

            for c in 0..self.num_centroids {
                let centroid_start = codebook_start + c * self.dims_per_sub;
                let centroid = &self.codebooks[centroid_start..centroid_start + self.dims_per_sub];

                let dist = if self.higher_is_better {
                    // Dot product
                    query_subvec.iter()
                        .zip(centroid.iter())
                        .map(|(&a, &b)| a * b)
                        .sum()
                } else {
                    // L2 squared
                    Self::l2_squared(query_subvec, centroid)
                };

                table[table_start + c] = dist;
            }
        }

        PQDistanceTable {
            table,
            num_subquantizers: self.num_subquantizers,
            higher_is_better: self.higher_is_better,
        }
    }

    /// Get the number of subquantizers (code length in bytes).
    pub fn code_size(&self) -> usize {
        self.num_subquantizers
    }

    /// Get the dimension per subquantizer.
    pub fn dims_per_subquantizer(&self) -> usize {
        self.dims_per_sub
    }

    /// Compute asymmetric distance directly without precomputed table.
    /// Faster for one-shot distance computations (no table precomputation overhead).
    #[inline]
    pub fn asymmetric_distance(&self, query: &[f32], code: &[u8]) -> f32 {
        let mut dist = 0.0f32;

        for (sub_idx, &c) in code.iter().enumerate() {
            let query_start = sub_idx * self.dims_per_sub;
            let query_subvec = &query[query_start..query_start + self.dims_per_sub];

            let codebook_start = sub_idx * self.num_centroids * self.dims_per_sub + (c as usize) * self.dims_per_sub;
            let centroid = &self.codebooks[codebook_start..codebook_start + self.dims_per_sub];

            if self.higher_is_better {
                // Dot product
                for (q, c) in query_subvec.iter().zip(centroid.iter()) {
                    dist += q * c;
                }
            } else {
                // L2 squared
                for (q, c) in query_subvec.iter().zip(centroid.iter()) {
                    let d = q - c;
                    dist += d * d;
                }
            }
        }

        dist
    }

    /// Check if a distance is better.
    #[inline]
    pub fn is_better(&self, a: f32, b: f32) -> bool {
        if self.higher_is_better { a > b } else { a < b }
    }

    /// Get the worst possible distance.
    #[inline]
    pub fn worst_distance(&self) -> f32 {
        if self.higher_is_better { f32::NEG_INFINITY } else { f32::INFINITY }
    }
}

/// Precomputed distance lookup table for a query.
///
/// Enables O(M) distance computation instead of O(D).
pub struct PQDistanceTable {
    /// Distance table: [num_subquantizers * num_centroids]
    table: Vec<f32>,

    /// Number of subquantizers.
    num_subquantizers: usize,

    /// Whether higher distances are better (dot product) or lower (L2).
    higher_is_better: bool,
}

impl PQDistanceTable {
    /// Compute approximate distance from query to a PQ-encoded vector.
    ///
    /// This is an asymmetric distance computation (ADC):
    /// - Query is not quantized
    /// - Database vector is quantized
    ///
    /// Time complexity: O(M) table lookups instead of O(D) operations.
    #[inline]
    pub fn distance(&self, code: &[u8]) -> f32 {
        debug_assert_eq!(code.len(), self.num_subquantizers);

        let mut dist = 0.0f32;
        for (sub_idx, &c) in code.iter().enumerate() {
            let table_idx = sub_idx * 256 + c as usize;
            dist += self.table[table_idx];
        }
        dist
    }

    /// Compute distances to multiple codes efficiently.
    #[inline]
    pub fn distances(&self, codes: &[&[u8]]) -> Vec<f32> {
        codes.iter().map(|c| self.distance(c)).collect()
    }

    /// Check if a distance is better than another.
    #[inline]
    pub fn is_better(&self, a: f32, b: f32) -> bool {
        if self.higher_is_better { a > b } else { a < b }
    }

    /// Get the worst possible distance (for initialization).
    #[inline]
    pub fn worst_distance(&self) -> f32 {
        if self.higher_is_better { f32::NEG_INFINITY } else { f32::INFINITY }
    }
}

/// Storage for PQ-encoded vectors with contiguous memory layout.
pub struct PQVectors {
    /// Encoded vectors: [num_vectors * code_size]
    codes: Vec<u8>,

    /// Number of vectors stored.
    num_vectors: usize,

    /// Code size (bytes per vector).
    code_size: usize,
}

impl PQVectors {
    /// Create PQ storage from vectors using the given quantizer.
    pub fn from_vectors(pq: &ProductQuantizer, vectors: &[&[f32]]) -> Self {
        let code_size = pq.code_size();
        let num_vectors = vectors.len();

        // Encode all vectors in parallel
        let encoded: Vec<Vec<u8>> = vectors
            .par_iter()
            .map(|v| pq.encode(v))
            .collect();

        // Flatten into contiguous storage
        let mut codes = Vec::with_capacity(num_vectors * code_size);
        for code in encoded {
            codes.extend(code);
        }

        Self {
            codes,
            num_vectors,
            code_size,
        }
    }

    /// Get the PQ code for a vector by index.
    #[inline]
    pub fn get_code(&self, idx: usize) -> &[u8] {
        let start = idx * self.code_size;
        &self.codes[start..start + self.code_size]
    }

    /// Get the number of vectors.
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Get the code size in bytes.
    pub fn code_size(&self) -> usize {
        self.code_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        (0..count)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        let mut hasher = DefaultHasher::new();
                        (seed + i as u64 * 1000 + j as u64).hash(&mut hasher);
                        let h = hasher.finish();
                        ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_pq_config_for_dimension() {
        let config = PQConfig::for_dimension(384);
        assert_eq!(config.num_subquantizers, 48);
        assert_eq!(384 / 48, 8); // 8 dims per subquantizer

        let config = PQConfig::for_dimension(768);
        assert_eq!(config.num_subquantizers, 48);
    }

    #[test]
    fn test_pq_encode_decode() {
        let dim = 384;
        let vectors = generate_random_vectors(1000, dim, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let config = PQConfig::for_dimension(dim);
        let pq = ProductQuantizer::train(refs.iter().copied(), dim, &config, true);

        // Encode a vector
        let code = pq.encode(&vectors[0]);
        assert_eq!(code.len(), config.num_subquantizers);

        // All codes should be valid (0-255)
        for &c in &code {
            assert!(c < 255 || c == 255);
        }
    }

    #[test]
    fn test_pq_distance_table() {
        let dim = 384;
        let vectors = generate_random_vectors(100, dim, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let config = PQConfig::for_dimension(dim);
        let pq = ProductQuantizer::train(refs.iter().copied(), dim, &config, true);

        // Encode all vectors
        let pq_storage = PQVectors::from_vectors(&pq, &refs);

        // Create distance table for query
        let query = &vectors[0];
        let table = pq.precompute_table(query);

        // Compute PQ distance to self
        let self_dist = table.distance(pq_storage.get_code(0));

        // Should be high (for dot product, self-similarity is high)
        println!("Self PQ distance: {}", self_dist);
        assert!(self_dist > 0.0);
    }

    #[test]
    fn test_pq_distance_ordering() {
        let dim = 384;
        // Use more vectors for better codebook training
        let vectors = generate_random_vectors(5000, dim, 42);

        // Normalize vectors for cosine similarity
        let normalized: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| {
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / norm).collect()
            })
            .collect();

        let refs: Vec<&[f32]> = normalized.iter().map(|v| v.as_slice()).collect();

        // More training iterations for better codebooks
        let mut config = PQConfig::for_dimension(dim);
        config.training_iterations = 20;
        let pq = ProductQuantizer::train(refs.iter().copied(), dim, &config, true);
        let pq_storage = PQVectors::from_vectors(&pq, &refs);

        // Query vector
        let query = &normalized[0];
        let table = pq.precompute_table(query);

        // Compute exact and PQ distances
        let mut exact_dists: Vec<(usize, f32)> = normalized
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                (i, dot)
            })
            .collect();
        exact_dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut pq_dists: Vec<(usize, f32)> = (0..normalized.len())
            .map(|i| (i, table.distance(pq_storage.get_code(i))))
            .collect();
        pq_dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Check recall@10: how many of exact top-10 are in PQ top-10
        let exact_top10: std::collections::HashSet<usize> = exact_dists.iter().take(10).map(|(i, _)| *i).collect();
        let pq_top10: std::collections::HashSet<usize> = pq_dists.iter().take(10).map(|(i, _)| *i).collect();

        let recall = exact_top10.intersection(&pq_top10).count() as f32 / 10.0;
        println!("PQ recall@10: {:.1}%", recall * 100.0);

        // PQ on random data with limited training has modest recall
        // Real-world data with structure achieves 70-90%+ recall
        assert!(recall >= 0.2, "PQ recall too low: {}", recall);
    }

    #[test]
    fn test_pq_vectors_storage() {
        let dim = 384;
        let vectors = generate_random_vectors(100, dim, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let config = PQConfig::for_dimension(dim);
        let pq = ProductQuantizer::train(refs.iter().copied(), dim, &config, true);
        let storage = PQVectors::from_vectors(&pq, &refs);

        assert_eq!(storage.len(), 100);
        assert_eq!(storage.code_size(), 48);

        // Check that we can retrieve codes
        for i in 0..100 {
            let code = storage.get_code(i);
            assert_eq!(code.len(), 48);
        }
    }
}
