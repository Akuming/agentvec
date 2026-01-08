//! Scalar quantization for vector compression.
//!
//! Compresses f32 vectors to u8 (4x memory reduction) while maintaining
//! approximate distance relationships. Uses SIMD-accelerated integer
//! operations for fast distance computation.
//!
//! # Algorithm
//!
//! 1. Calibrate: Compute per-dimension min/max from training vectors
//! 2. Quantize: Map each f32 to 0-255 range using linear scaling
//! 3. Distance: Use integer dot product (SIMD) for approximate distance
//! 4. Rerank: Use original f32 vectors for final top-k ranking

use std::sync::Arc;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Scalar quantizer that compresses f32 vectors to u8.
///
/// Uses per-dimension scaling to map f32 values to the 0-255 range.
/// The quantized representation enables 4x memory reduction and
/// faster SIMD-accelerated distance computation.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    /// Number of dimensions
    dimensions: usize,

    /// Per-dimension minimum values (for dequantization)
    mins: Vec<f32>,

    /// Per-dimension scale factors: (max - min) / 255.0
    scales: Vec<f32>,

    /// Inverse scales for quantization: 255.0 / (max - min)
    inv_scales: Vec<f32>,
}

impl ScalarQuantizer {
    /// Create a new quantizer calibrated on the given vectors.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Training vectors to compute min/max ranges
    ///
    /// # Returns
    ///
    /// A calibrated ScalarQuantizer ready to quantize/dequantize vectors.
    pub fn fit<I, V>(vectors: I) -> Self
    where
        I: Iterator<Item = V>,
        V: AsRef<[f32]>,
    {
        let mut peekable = vectors.peekable();

        // Get dimensions from first vector
        let first = peekable.peek().expect("Need at least one vector to fit");
        let dimensions = first.as_ref().len();

        let mut mins = vec![f32::MAX; dimensions];
        let mut maxs = vec![f32::MIN; dimensions];

        // Find per-dimension min/max
        for vec in peekable {
            let v = vec.as_ref();
            for (i, &val) in v.iter().enumerate() {
                mins[i] = mins[i].min(val);
                maxs[i] = maxs[i].max(val);
            }
        }

        // Compute scales with epsilon to avoid division by zero
        let scales: Vec<f32> = mins.iter().zip(&maxs)
            .map(|(&min, &max)| {
                let range = max - min;
                if range < f32::EPSILON {
                    1.0 // Default scale for constant dimensions
                } else {
                    range / 255.0
                }
            })
            .collect();

        let inv_scales: Vec<f32> = scales.iter()
            .map(|&s| if s < f32::EPSILON { 0.0 } else { 1.0 / s })
            .collect();

        Self {
            dimensions,
            mins,
            scales,
            inv_scales,
        }
    }

    /// Create a quantizer with pre-computed calibration parameters.
    pub fn with_params(mins: Vec<f32>, scales: Vec<f32>) -> Self {
        let dimensions = mins.len();
        let inv_scales: Vec<f32> = scales.iter()
            .map(|&s| if s < f32::EPSILON { 0.0 } else { 1.0 / s })
            .collect();

        Self {
            dimensions,
            mins,
            scales,
            inv_scales,
        }
    }

    /// Get the number of dimensions.
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Quantize a f32 vector to u8.
    ///
    /// # Arguments
    ///
    /// * `vector` - Input f32 vector
    ///
    /// # Returns
    ///
    /// Quantized u8 vector of the same length.
    #[inline]
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        debug_assert_eq!(vector.len(), self.dimensions);

        vector.iter().enumerate()
            .map(|(i, &val)| {
                let normalized = (val - self.mins[i]) * self.inv_scales[i];
                normalized.clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    /// Quantize a f32 vector into an existing buffer.
    #[inline]
    pub fn quantize_into(&self, vector: &[f32], output: &mut [u8]) {
        debug_assert_eq!(vector.len(), self.dimensions);
        debug_assert_eq!(output.len(), self.dimensions);

        for (i, &val) in vector.iter().enumerate() {
            let normalized = (val - self.mins[i]) * self.inv_scales[i];
            output[i] = normalized.clamp(0.0, 255.0) as u8;
        }
    }

    /// Dequantize a u8 vector back to f32 (approximate).
    ///
    /// Note: This is lossy - the original values cannot be perfectly recovered.
    #[inline]
    pub fn dequantize(&self, quantized: &[u8]) -> Vec<f32> {
        debug_assert_eq!(quantized.len(), self.dimensions);

        quantized.iter().enumerate()
            .map(|(i, &q)| q as f32 * self.scales[i] + self.mins[i])
            .collect()
    }

    /// Compute approximate dot product between two quantized vectors.
    ///
    /// Uses integer arithmetic for speed. The result is NOT the true dot product
    /// but is monotonically related to it for ranking purposes.
    #[inline]
    pub fn dot_product_quantized(&self, a: &[u8], b: &[u8]) -> i32 {
        debug_assert_eq!(a.len(), self.dimensions);
        debug_assert_eq!(b.len(), self.dimensions);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { dot_product_u8_avx2(a, b) };
            }
        }

        dot_product_u8_scalar(a, b)
    }

    /// Compute approximate L2 squared distance between two quantized vectors.
    #[inline]
    pub fn l2_squared_quantized(&self, a: &[u8], b: &[u8]) -> i32 {
        debug_assert_eq!(a.len(), self.dimensions);
        debug_assert_eq!(b.len(), self.dimensions);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { l2_squared_u8_avx2(a, b) };
            }
        }

        l2_squared_u8_scalar(a, b)
    }

    /// Precompute lookup tables for a query vector.
    ///
    /// For cosine/dot product with quantized vectors, we can precompute:
    /// table[i][q] = query[i] * dequantize(q, i)
    ///
    /// This allows O(1) distance contribution lookup per dimension.
    #[inline]
    pub fn precompute_query_tables(&self, query: &[f32]) -> QueryTables {
        debug_assert_eq!(query.len(), self.dimensions);

        let mut tables = vec![0.0f32; self.dimensions * 256];

        for i in 0..self.dimensions {
            let q_val = query[i];
            let min = self.mins[i];
            let scale = self.scales[i];

            for code in 0..256 {
                let reconstructed = code as f32 * scale + min;
                tables[i * 256 + code] = q_val * reconstructed;
            }
        }

        QueryTables {
            tables,
            dimensions: self.dimensions,
        }
    }
}

/// Precomputed lookup tables for fast query-to-quantized distance.
#[derive(Debug, Clone)]
pub struct QueryTables {
    /// Lookup tables: [dim][code] -> contribution to dot product
    tables: Vec<f32>,
    dimensions: usize,
}

impl QueryTables {
    /// Compute dot product with a quantized vector using table lookup.
    #[inline]
    pub fn dot_product(&self, quantized: &[u8]) -> f32 {
        debug_assert_eq!(quantized.len(), self.dimensions);

        let mut sum = 0.0f32;
        for (i, &code) in quantized.iter().enumerate() {
            sum += self.tables[i * 256 + code as usize];
        }
        sum
    }
}

// ============ Scalar Implementations ============

/// Scalar integer dot product.
#[inline]
fn dot_product_u8_scalar(a: &[u8], b: &[u8]) -> i32 {
    let mut sum = 0i32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x as i32 * y as i32;
    }
    sum
}

/// Scalar integer L2 squared.
#[inline]
fn l2_squared_u8_scalar(a: &[u8], b: &[u8]) -> i32 {
    let mut sum = 0i32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = x as i32 - y as i32;
        sum += diff * diff;
    }
    sum
}

// ============ SIMD Implementations ============

/// AVX2 integer dot product for u8 vectors.
///
/// Uses _mm256_maddubs_epi16 and _mm256_madd_epi16 for efficient
/// 8-bit multiply-accumulate.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_u8_avx2(a: &[u8], b: &[u8]) -> i32 {
    let n = a.len();
    let chunks = n / 32;
    let remainder = n % 32;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // Accumulator for 8 i32 values
    let mut acc = _mm256_setzero_si256();

    for i in 0..chunks {
        let offset = i * 32;

        // Load 32 bytes from each vector
        let va = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);
        let vb = _mm256_loadu_si256(b_ptr.add(offset) as *const __m256i);

        // Multiply pairs of u8 values and add horizontally to i16
        // Note: _mm256_maddubs_epi16 treats first arg as unsigned, second as signed
        // We need both unsigned, so we use a workaround with _mm256_maddubs_epi16
        // Actually, for u8*u8 we need to be careful about overflow

        // Split into low and high 128-bit parts for safer handling
        let va_lo = _mm256_unpacklo_epi8(va, _mm256_setzero_si256());
        let va_hi = _mm256_unpackhi_epi8(va, _mm256_setzero_si256());
        let vb_lo = _mm256_unpacklo_epi8(vb, _mm256_setzero_si256());
        let vb_hi = _mm256_unpackhi_epi8(vb, _mm256_setzero_si256());

        // Now we have 16-bit values, multiply and accumulate
        let prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
        let prod_hi = _mm256_madd_epi16(va_hi, vb_hi);

        acc = _mm256_add_epi32(acc, prod_lo);
        acc = _mm256_add_epi32(acc, prod_hi);
    }

    // Horizontal sum of 8 i32 values
    let sum128 = _mm_add_epi32(
        _mm256_extracti128_si256(acc, 0),
        _mm256_extracti128_si256(acc, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    let mut result = _mm_cvtsi128_si32(sum32);

    // Handle remainder
    let base = chunks * 32;
    for i in 0..remainder {
        result += a[base + i] as i32 * b[base + i] as i32;
    }

    result
}

/// AVX2 integer L2 squared for u8 vectors.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_squared_u8_avx2(a: &[u8], b: &[u8]) -> i32 {
    let n = a.len();
    let chunks = n / 32;
    let remainder = n % 32;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc = _mm256_setzero_si256();

    for i in 0..chunks {
        let offset = i * 32;

        let va = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);
        let vb = _mm256_loadu_si256(b_ptr.add(offset) as *const __m256i);

        // Compute absolute difference using SAD (sum of absolute differences)
        // But we need squared differences, so we unpack to 16-bit first
        let va_lo = _mm256_unpacklo_epi8(va, _mm256_setzero_si256());
        let va_hi = _mm256_unpackhi_epi8(va, _mm256_setzero_si256());
        let vb_lo = _mm256_unpacklo_epi8(vb, _mm256_setzero_si256());
        let vb_hi = _mm256_unpackhi_epi8(vb, _mm256_setzero_si256());

        // Compute differences (as i16)
        let diff_lo = _mm256_sub_epi16(va_lo, vb_lo);
        let diff_hi = _mm256_sub_epi16(va_hi, vb_hi);

        // Square and accumulate using madd with self
        let sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
        let sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);

        acc = _mm256_add_epi32(acc, sq_lo);
        acc = _mm256_add_epi32(acc, sq_hi);
    }

    // Horizontal sum
    let sum128 = _mm_add_epi32(
        _mm256_extracti128_si256(acc, 0),
        _mm256_extracti128_si256(acc, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    let mut result = _mm_cvtsi128_si32(sum32);

    // Handle remainder
    let base = chunks * 32;
    for i in 0..remainder {
        let diff = a[base + i] as i32 - b[base + i] as i32;
        result += diff * diff;
    }

    result
}

// ============ Quantized Vector Storage ============

/// Storage for quantized vectors with their original indices.
#[derive(Debug)]
pub struct QuantizedVectors {
    /// The quantizer used
    quantizer: Arc<ScalarQuantizer>,

    /// Quantized vectors stored contiguously
    data: Vec<u8>,

    /// Number of vectors
    count: usize,
}

impl QuantizedVectors {
    /// Create a new quantized vector storage.
    pub fn new(quantizer: Arc<ScalarQuantizer>) -> Self {
        Self {
            quantizer,
            data: Vec::new(),
            count: 0,
        }
    }

    /// Create from existing vectors.
    pub fn from_vectors<'a, I>(quantizer: Arc<ScalarQuantizer>, vectors: I) -> Self
    where
        I: Iterator<Item = &'a [f32]>,
    {
        let dim = quantizer.dimensions();
        let mut data = Vec::new();
        let mut count = 0;

        for vec in vectors {
            let start = data.len();
            data.resize(start + dim, 0);
            quantizer.quantize_into(vec, &mut data[start..start + dim]);
            count += 1;
        }

        Self {
            quantizer,
            data,
            count,
        }
    }

    /// Add a vector.
    pub fn push(&mut self, vector: &[f32]) {
        let dim = self.quantizer.dimensions();
        let start = self.data.len();
        self.data.resize(start + dim, 0);
        self.quantizer.quantize_into(vector, &mut self.data[start..start + dim]);
        self.count += 1;
    }

    /// Get a quantized vector by index.
    #[inline]
    pub fn get(&self, index: usize) -> &[u8] {
        let dim = self.quantizer.dimensions();
        let start = index * dim;
        &self.data[start..start + dim]
    }

    /// Get the number of vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the quantizer.
    #[inline]
    pub fn quantizer(&self) -> &ScalarQuantizer {
        &self.quantizer
    }

    /// Compute dot product between a query and a stored vector.
    #[inline]
    pub fn dot_product(&self, query_quantized: &[u8], index: usize) -> i32 {
        self.quantizer.dot_product_quantized(query_quantized, self.get(index))
    }

    /// Compute L2 squared between a query and a stored vector.
    #[inline]
    pub fn l2_squared(&self, query_quantized: &[u8], index: usize) -> i32 {
        self.quantizer.l2_squared_quantized(query_quantized, self.get(index))
    }
}

// ============ Signed Scalar Quantization (for dot product) ============

/// Signed scalar quantizer for dot product / cosine similarity.
///
/// Maps f32 values to i8 [-127, 127], preserving sign information.
/// This is essential for dot product where negative values indicate
/// opposite directions.
///
/// For normalized vectors in [-1, 1]:
/// - value * 127 gives the quantized i8 value
/// - dot product ordering is preserved
#[derive(Debug, Clone)]
pub struct SignedQuantizer {
    /// Number of dimensions
    dimensions: usize,
    /// Scale factor (typically 127.0 for normalized vectors)
    scale: f32,
}

impl SignedQuantizer {
    /// Create a quantizer for normalized vectors (values in [-1, 1]).
    pub fn for_normalized(dimensions: usize) -> Self {
        Self {
            dimensions,
            scale: 127.0,
        }
    }

    /// Create with custom scale factor.
    pub fn with_scale(dimensions: usize, scale: f32) -> Self {
        Self { dimensions, scale }
    }

    /// Get dimensions.
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Quantize f32 vector to i8.
    #[inline]
    pub fn quantize(&self, vector: &[f32]) -> Vec<i8> {
        debug_assert_eq!(vector.len(), self.dimensions);
        vector.iter()
            .map(|&v| (v * self.scale).clamp(-127.0, 127.0) as i8)
            .collect()
    }

    /// Quantize into existing buffer.
    #[inline]
    pub fn quantize_into(&self, vector: &[f32], output: &mut [i8]) {
        debug_assert_eq!(vector.len(), self.dimensions);
        debug_assert_eq!(output.len(), self.dimensions);
        for (i, &v) in vector.iter().enumerate() {
            output[i] = (v * self.scale).clamp(-127.0, 127.0) as i8;
        }
    }

    /// Compute dot product between two signed quantized vectors.
    /// Higher values = more similar (for cosine/dot product).
    #[inline]
    pub fn dot_product(&self, a: &[i8], b: &[i8]) -> i32 {
        debug_assert_eq!(a.len(), self.dimensions);
        debug_assert_eq!(b.len(), self.dimensions);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { dot_product_i8_avx2(a, b) };
            }
        }

        dot_product_i8_scalar(a, b)
    }
}

/// Scalar i8 dot product.
#[inline]
fn dot_product_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut sum = 0i32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x as i32 * y as i32;
    }
    sum
}

/// AVX2 signed i8 dot product.
///
/// Uses pmaddubsw which treats first arg as unsigned, second as signed.
/// We handle this by computing abs values and tracking signs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len();
    let chunks = n / 32;
    let remainder = n % 32;

    let a_ptr = a.as_ptr() as *const u8;
    let b_ptr = b.as_ptr() as *const u8;

    // For signed multiplication, we use a different approach:
    // Unpack to i16, multiply, and accumulate to i32
    let mut acc = _mm256_setzero_si256();

    for i in 0..chunks {
        let offset = i * 32;

        // Load 32 bytes as i8
        let va = _mm256_loadu_si256(a_ptr.add(offset) as *const __m256i);
        let vb = _mm256_loadu_si256(b_ptr.add(offset) as *const __m256i);

        // Sign-extend i8 to i16 (low and high halves)
        // unpacklo/hi with sign extension
        let va_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 0));
        let va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        let vb_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 0));
        let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));

        // Multiply i16 pairs and add adjacent pairs to i32
        let prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
        let prod_hi = _mm256_madd_epi16(va_hi, vb_hi);

        acc = _mm256_add_epi32(acc, prod_lo);
        acc = _mm256_add_epi32(acc, prod_hi);
    }

    // Horizontal sum of 8 i32 values
    let sum128 = _mm_add_epi32(
        _mm256_extracti128_si256(acc, 0),
        _mm256_extracti128_si256(acc, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    let mut result = _mm_cvtsi128_si32(sum32);

    // Handle remainder
    let base = chunks * 32;
    for i in 0..remainder {
        result += a[base + i] as i32 * b[base + i] as i32;
    }

    result
}

/// Contiguous storage for signed quantized vectors.
#[derive(Debug)]
pub struct SignedQuantizedVectors {
    /// The quantizer
    dimensions: usize,
    scale: f32,
    /// Quantized data stored contiguously
    data: Vec<i8>,
    /// Number of vectors
    count: usize,
}

impl SignedQuantizedVectors {
    /// Create empty storage.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            scale: 127.0,
            data: Vec::new(),
            count: 0,
        }
    }

    /// Create from f32 vectors (parallel quantization).
    pub fn from_f32_vectors(vectors: &[Vec<f32>]) -> Self {
        use rayon::prelude::*;

        if vectors.is_empty() {
            return Self::new(0);
        }

        let dimensions = vectors[0].len();
        let scale = 127.0f32;
        let count = vectors.len();

        // Allocate contiguous buffer
        let mut data = vec![0i8; count * dimensions];

        // Parallel quantization
        data.par_chunks_mut(dimensions)
            .zip(vectors.par_iter())
            .for_each(|(output, input)| {
                for (i, &v) in input.iter().enumerate() {
                    output[i] = (v * scale).clamp(-127.0, 127.0) as i8;
                }
            });

        Self {
            dimensions,
            scale,
            data,
            count,
        }
    }

    /// Get quantized vector by index.
    #[inline]
    pub fn get(&self, index: usize) -> &[i8] {
        let start = index * self.dimensions;
        &self.data[start..start + self.dimensions]
    }

    /// Get number of vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Compute dot product between two stored vectors.
    #[inline]
    pub fn dot_product(&self, a_idx: usize, b_idx: usize) -> i32 {
        let a = self.get(a_idx);
        let b = self.get(b_idx);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { dot_product_i8_avx2(a, b) };
            }
        }

        dot_product_i8_scalar(a, b)
    }

    /// Compute dot product between a query and stored vector.
    #[inline]
    pub fn dot_product_with_query(&self, query: &[i8], index: usize) -> i32 {
        let stored = self.get(index);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { dot_product_i8_avx2(query, stored) };
            }
        }

        dot_product_i8_scalar(query, stored)
    }

    /// Get dimensions.
    #[inline]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Quantize a single f32 vector.
    #[inline]
    pub fn quantize(&self, vector: &[f32]) -> Vec<i8> {
        vector.iter()
            .map(|&v| (v * self.scale).clamp(-127.0, 127.0) as i8)
            .collect()
    }
}

// ============ Tests ============

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        (0..count)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        let mut hasher = DefaultHasher::new();
                        (i * dim + j).hash(&mut hasher);
                        let h = hasher.finish();
                        ((h as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_quantizer_fit() {
        let vectors = generate_vectors(100, 384);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let quantizer = ScalarQuantizer::fit(refs.into_iter());

        assert_eq!(quantizer.dimensions(), 384);
    }

    #[test]
    fn test_quantize_dequantize() {
        let vectors = generate_vectors(100, 384);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let quantizer = ScalarQuantizer::fit(refs.into_iter());

        // Quantize and dequantize
        let original = &vectors[0];
        let quantized = quantizer.quantize(original);
        let reconstructed = quantizer.dequantize(&quantized);

        assert_eq!(quantized.len(), 384);
        assert_eq!(reconstructed.len(), 384);

        // Check reconstruction error is reasonable (within quantization error)
        let mut max_error = 0.0f32;
        for (o, r) in original.iter().zip(&reconstructed) {
            max_error = max_error.max((o - r).abs());
        }

        // Max error should be at most scale/2 per dimension
        // With range [-1, 1], scale = 2/255 ≈ 0.008
        assert!(max_error < 0.02, "Max reconstruction error: {}", max_error);
    }

    #[test]
    fn test_dot_product_quantized() {
        let vectors = generate_vectors(100, 384);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let quantizer = ScalarQuantizer::fit(refs.into_iter());

        let a = quantizer.quantize(&vectors[0]);
        let b = quantizer.quantize(&vectors[1]);

        let dot = quantizer.dot_product_quantized(&a, &b);

        // Just verify it returns a reasonable value
        assert!(dot >= 0, "Dot product should be non-negative for [0,255] values");
    }

    #[test]
    fn test_dot_product_simd_vs_scalar() {
        let vectors = generate_vectors(100, 384);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let quantizer = ScalarQuantizer::fit(refs.into_iter());

        let a = quantizer.quantize(&vectors[0]);
        let b = quantizer.quantize(&vectors[1]);

        let scalar = dot_product_u8_scalar(&a, &b);
        let simd = quantizer.dot_product_quantized(&a, &b);

        assert_eq!(scalar, simd, "SIMD and scalar should match");
    }

    #[test]
    fn test_l2_squared_simd_vs_scalar() {
        let vectors = generate_vectors(100, 384);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let quantizer = ScalarQuantizer::fit(refs.into_iter());

        let a = quantizer.quantize(&vectors[0]);
        let b = quantizer.quantize(&vectors[1]);

        let scalar = l2_squared_u8_scalar(&a, &b);
        let simd = quantizer.l2_squared_quantized(&a, &b);

        assert_eq!(scalar, simd, "SIMD and scalar should match");
    }

    #[test]
    fn test_query_tables() {
        let vectors = generate_vectors(100, 384);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let quantizer = ScalarQuantizer::fit(refs.into_iter());

        let query = &vectors[0];
        let tables = quantizer.precompute_query_tables(query);

        let quantized = quantizer.quantize(&vectors[1]);

        // Compare table lookup vs direct computation
        let table_dot = tables.dot_product(&quantized);

        // Direct: dequantize and compute
        let dequantized = quantizer.dequantize(&quantized);
        let direct_dot: f32 = query.iter().zip(&dequantized)
            .map(|(a, b)| a * b)
            .sum();

        // Should be very close
        let error = (table_dot - direct_dot).abs();
        assert!(error < 0.01, "Table lookup error: {}", error);
    }

    #[test]
    fn test_quantized_vectors_storage() {
        let vectors = generate_vectors(100, 384);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let quantizer = Arc::new(ScalarQuantizer::fit(refs.iter().copied()));
        let storage = QuantizedVectors::from_vectors(
            quantizer.clone(),
            refs.iter().copied(),
        );

        assert_eq!(storage.len(), 100);
        assert_eq!(storage.get(0).len(), 384);

        // Verify stored data matches direct quantization
        let expected = quantizer.quantize(&vectors[50]);
        assert_eq!(storage.get(50), expected.as_slice());
    }

    #[test]
    fn test_quantized_distance_correlation() {
        // Test that quantized distances correlate with exact distances
        // (i.e., if exact(a,b) > exact(a,c), then quantized(a,b) > quantized(a,c) most of the time)
        let dim = 128;

        // Create vectors with clear differences
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = a.iter().map(|&x| x * 0.9).collect(); // Similar to a
        let c: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).cos()).collect(); // Different

        let vectors = vec![a.as_slice(), b.as_slice(), c.as_slice()];
        let quantizer = ScalarQuantizer::fit(vectors.iter().copied());

        // Exact dot products
        let exact_ab: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let exact_ac: f32 = a.iter().zip(&c).map(|(x, y)| x * y).sum();

        // Quantized dot products
        let qa = quantizer.quantize(&a);
        let qb = quantizer.quantize(&b);
        let qc = quantizer.quantize(&c);

        let quant_ab = quantizer.dot_product_quantized(&qa, &qb);
        let quant_ac = quantizer.dot_product_quantized(&qa, &qc);

        // b should be more similar to a than c is
        assert!(exact_ab > exact_ac, "Exact: a-b should be more similar than a-c");

        // Same ordering should hold for quantized
        // Note: quantized values are always positive, so we compare magnitudes
        assert!(quant_ab > quant_ac, "Quantized: a-b ({}) should be more similar than a-c ({})", quant_ab, quant_ac);
    }

    #[test]
    fn test_signed_quantization_preserves_ordering() {
        // Test that signed quantization preserves dot product ordering
        // This is the key property for cosine similarity
        let dim = 384;

        // Create normalized vectors
        let mut a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut b: Vec<f32> = a.iter().map(|&x| x * 0.9 + 0.1).collect();
        let mut c: Vec<f32> = (0..dim).map(|i| -(i as f32 * 0.1).sin()).collect(); // Opposite direction

        // Normalize
        fn normalize(v: &mut [f32]) {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                v.iter_mut().for_each(|x| *x /= norm);
            }
        }
        normalize(&mut a);
        normalize(&mut b);
        normalize(&mut c);

        // Exact dot products
        let exact_ab: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let exact_ac: f32 = a.iter().zip(&c).map(|(x, y)| x * y).sum();

        // b should be more similar to a than c (c is opposite direction)
        assert!(exact_ab > exact_ac, "Exact: a·b ({}) should be > a·c ({})", exact_ab, exact_ac);
        assert!(exact_ac < 0.0, "a·c should be negative (opposite directions)");

        // Signed quantization
        let quantizer = SignedQuantizer::for_normalized(dim);
        let qa = quantizer.quantize(&a);
        let qb = quantizer.quantize(&b);
        let qc = quantizer.quantize(&c);

        let quant_ab = quantizer.dot_product(&qa, &qb);
        let quant_ac = quantizer.dot_product(&qa, &qc);

        // Signed quantization should preserve ordering INCLUDING negative values
        assert!(quant_ab > quant_ac, "Signed quantized: a·b ({}) should be > a·c ({})", quant_ab, quant_ac);
        assert!(quant_ac < 0, "Signed quantized a·c should be negative: {}", quant_ac);
    }

    #[test]
    fn test_signed_quantized_vectors_storage() {
        let dim = 384;
        let count = 100;

        // Generate normalized vectors
        let vectors: Vec<Vec<f32>> = (0..count)
            .map(|i| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|j| ((i * dim + j) as f32 * 0.01).sin())
                    .collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect();

        let storage = SignedQuantizedVectors::from_f32_vectors(&vectors);

        assert_eq!(storage.len(), 100);
        assert_eq!(storage.dimensions(), 384);

        // Test dot product between stored vectors
        let dot_01 = storage.dot_product(0, 1);
        let dot_00 = storage.dot_product(0, 0); // Self dot product should be highest

        assert!(dot_00 > dot_01, "Self dot product should be highest");
        assert!(dot_00 > 0, "Self dot product should be positive");
    }

    #[test]
    fn test_signed_simd_vs_scalar() {
        let dim = 384;
        let quantizer = SignedQuantizer::for_normalized(dim);

        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();

        let qa = quantizer.quantize(&a);
        let qb = quantizer.quantize(&b);

        let scalar = dot_product_i8_scalar(&qa, &qb);
        let simd = quantizer.dot_product(&qa, &qb);

        assert_eq!(scalar, simd, "SIMD and scalar should match for signed i8");
    }
}
