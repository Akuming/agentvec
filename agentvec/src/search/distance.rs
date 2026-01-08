//! Distance and similarity functions for vector search.
//!
//! Provides optimized implementations using:
//! - AVX2 (256-bit SIMD) for x86_64 when available
//! - SSE (128-bit SIMD) for x86_64 fallback
//! - NEON (128-bit SIMD) for ARM64
//! - Scalar fallback for all other platforms

use crate::config::Metric;

// ============ CPU Feature Detection ============

/// Check if AVX2 + FMA is available at runtime (x86_64 only).
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2_fma() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// Check if SSE is available at runtime (x86_64 only).
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_sse() -> bool {
    is_x86_feature_detected!("sse")
}

// ============ Trait Definition ============

/// Trait for distance/similarity computation.
pub trait DistanceMetric {
    /// Compute the distance/similarity between two vectors.
    ///
    /// For similarity metrics (cosine, dot): higher is more similar.
    /// For distance metrics (L2): lower is more similar.
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;

    /// Returns true if higher scores indicate more similar vectors.
    fn higher_is_better(&self) -> bool;
}

impl DistanceMetric for Metric {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Cosine => dot_product(a, b), // Assumes normalized vectors
            Metric::Dot => dot_product(a, b),
            Metric::L2 => l2_squared(a, b),
        }
    }

    fn higher_is_better(&self) -> bool {
        match self {
            Metric::Cosine => true,
            Metric::Dot => true,
            Metric::L2 => false,
        }
    }
}

// ============ Dot Product ============

/// Compute the dot product of two vectors.
///
/// Automatically selects the fastest available implementation:
/// - AVX2 on x86_64 with AVX2 support
/// - SSE on x86_64 without AVX2
/// - NEON on ARM64
/// - Scalar fallback otherwise
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as a)
///
/// # Returns
///
/// The dot product: sum of element-wise products.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector lengths must match");

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: We've verified AVX2+FMA is available
            return unsafe { dot_product_avx2(a, b) };
        }
        if has_sse() {
            // SAFETY: We've verified SSE is available
            return unsafe { dot_product_sse(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return unsafe { dot_product_neon(a, b) };
    }

    // Fallback to scalar
    dot_product_scalar(a, b)
}

/// Scalar dot product implementation.
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    // Unrolled loop for better performance
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let idx = i * 4;
        sum0 += a[idx] * b[idx];
        sum1 += a[idx + 1] * b[idx + 1];
        sum2 += a[idx + 2] * b[idx + 2];
        sum3 += a[idx + 3] * b[idx + 3];
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        sum0 += a[base + i] * b[base + i];
    }

    sum0 + sum1 + sum2 + sum3
}

/// AVX2 dot product (processes 8 floats at a time).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    // Accumulator for 8 floats
    let mut sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        // Fused multiply-add: sum += va * vb
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of the 8 floats in sum
    // sum = [a0, a1, a2, a3, a4, a5, a6, a7]
    let high = _mm256_extractf128_ps(sum, 1); // [a4, a5, a6, a7]
    let low = _mm256_castps256_ps128(sum); // [a0, a1, a2, a3]
    let sum128 = _mm_add_ps(low, high); // [a0+a4, a1+a5, a2+a6, a3+a7]

    // Now reduce 4 floats to 1
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder with scalar
    let base = chunks * 8;
    for i in 0..remainder {
        result += a[base + i] * b[base + i];
    }

    result
}

/// SSE dot product (processes 4 floats at a time).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = _mm_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a_ptr.add(offset));
        let vb = _mm_loadu_ps(b_ptr.add(offset));
        let prod = _mm_mul_ps(va, vb);
        sum = _mm_add_ps(sum, prod);
    }

    // Horizontal sum of 4 floats
    let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        result += a[base + i] * b[base + i];
    }

    result
}

/// NEON dot product for ARM64 (processes 4 floats at a time).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }

    // Horizontal sum
    let mut result = vaddvq_f32(sum);

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        result += a[base + i] * b[base + i];
    }

    result
}

// ============ L2 Squared Distance ============

/// Compute the squared L2 (Euclidean) distance between two vectors.
///
/// Automatically selects the fastest available implementation.
///
/// Returns the sum of squared differences. Take sqrt() for actual Euclidean distance,
/// but squared is fine for comparison (monotonic).
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as a)
///
/// # Returns
///
/// The squared L2 distance: sum of (a[i] - b[i])^2.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector lengths must match");

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { l2_squared_avx2(a, b) };
        }
        if has_sse() {
            return unsafe { l2_squared_sse(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { l2_squared_neon(a, b) };
    }

    l2_squared_scalar(a, b)
}

/// Scalar L2 squared implementation.
#[inline]
pub fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let idx = i * 4;
        let d0 = a[idx] - b[idx];
        let d1 = a[idx + 1] - b[idx + 1];
        let d2 = a[idx + 2] - b[idx + 2];
        let d3 = a[idx + 3] - b[idx + 3];
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
    }

    let base = chunks * 4;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum0 += d * d;
    }

    sum0 + sum1 + sum2 + sum3
}

/// AVX2 L2 squared (processes 8 floats at a time).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        let diff = _mm256_sub_ps(va, vb);
        // diff^2 and accumulate
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    let high = _mm256_extractf128_ps(sum, 1);
    let low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(low, high);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder
    let base = chunks * 8;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        result += d * d;
    }

    result
}

/// SSE L2 squared (processes 4 floats at a time).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn l2_squared_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = _mm_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a_ptr.add(offset));
        let vb = _mm_loadu_ps(b_ptr.add(offset));
        let diff = _mm_sub_ps(va, vb);
        let sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, sq);
    }

    // Horizontal sum
    let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = _mm_cvtss_f32(sum32);

    let base = chunks * 4;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        result += d * d;
    }

    result
}

/// NEON L2 squared for ARM64.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
    }

    let mut result = vaddvq_f32(sum);

    let base = chunks * 4;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        result += d * d;
    }

    result
}

// ============ L2 Norm and Normalization ============

/// Compute the L2 norm (magnitude) of a vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { l2_norm_avx2(v) };
        }
    }

    l2_norm_scalar(v)
}

/// Scalar L2 norm implementation.
#[inline]
pub fn l2_norm_scalar(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in v {
        sum += x * x;
    }
    sum.sqrt()
}

/// AVX2 L2 norm.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn l2_norm_avx2(v: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = v.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum = _mm256_setzero_ps();
    let v_ptr = v.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let vv = _mm256_loadu_ps(v_ptr.add(offset));
        sum = _mm256_fmadd_ps(vv, vv, sum);
    }

    // Horizontal sum
    let high = _mm256_extractf128_ps(sum, 1);
    let low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(low, high);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result = _mm_cvtss_f32(sum32);

    let base = chunks * 8;
    for i in 0..remainder {
        result += v[base + i] * v[base + i];
    }

    result.sqrt()
}

/// Normalize a vector to unit length (L2 normalization).
///
/// # Arguments
///
/// * `v` - Input vector
///
/// # Returns
///
/// A new vector with L2 norm of 1.0.
/// If the input has zero magnitude, returns a zero vector.
#[inline]
pub fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm < f32::EPSILON {
        return vec![0.0; v.len()];
    }

    let inv_norm = 1.0 / norm;
    v.iter().map(|&x| x * inv_norm).collect()
}

/// Normalize a vector in place.
///
/// # Arguments
///
/// * `v` - Vector to normalize
///
/// # Returns
///
/// The original norm (useful for checking if normalization was meaningful).
#[inline]
pub fn normalize_l2_inplace(v: &mut [f32]) -> f32 {
    let norm = l2_norm(v);
    if norm < f32::EPSILON {
        return 0.0;
    }

    let inv_norm = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
    norm
}

/// Check if a vector is approximately normalized (L2 norm close to 1.0).
#[inline]
pub fn is_normalized(v: &[f32], tolerance: f32) -> bool {
    let norm = l2_norm(v);
    (norm - 1.0).abs() < tolerance
}

// ============ Tests ============

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_zeros() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        let result = dot_product(&a, &b);
        assert!((result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let result = dot_product(&a, &b);
        assert!((result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_parallel() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let result = dot_product(&a, &b);
        assert!((result - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_large() {
        // Test with realistic embedding size
        let a: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..384).map(|i| (383 - i) as f32 * 0.01).collect();

        let result = dot_product(&a, &b);

        // Verify against simple loop
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 0.1);
    }

    #[test]
    fn test_dot_product_simd_vs_scalar() {
        // Ensure SIMD and scalar produce same results
        let a: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32).cos()).collect();

        let scalar = dot_product_scalar(&a, &b);
        let auto = dot_product(&a, &b);

        assert!(
            (scalar - auto).abs() < 0.001,
            "SIMD and scalar mismatch: scalar={}, auto={}",
            scalar,
            auto
        );
    }

    #[test]
    fn test_l2_squared() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
        let result = l2_squared(&a, &b);
        assert!((result - 27.0).abs() < EPSILON);
    }

    #[test]
    fn test_l2_squared_same() {
        let a = [1.0, 2.0, 3.0];
        let result = l2_squared(&a, &a);
        assert!((result - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_l2_squared_simd_vs_scalar() {
        let a: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32).cos()).collect();

        let scalar = l2_squared_scalar(&a, &b);
        let auto = l2_squared(&a, &b);

        assert!(
            (scalar - auto).abs() < 0.001,
            "SIMD and scalar mismatch: scalar={}, auto={}",
            scalar,
            auto
        );
    }

    #[test]
    fn test_l2_norm() {
        let v = [3.0, 4.0]; // 3-4-5 triangle
        let result = l2_norm(&v);
        assert!((result - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_l2() {
        let v = [3.0, 4.0];
        let normalized = normalize_l2(&v);

        // Verify magnitude is 1
        let norm = l2_norm(&normalized);
        assert!((norm - 1.0).abs() < EPSILON);

        // Verify direction preserved
        assert!((normalized[0] - 0.6).abs() < EPSILON);
        assert!((normalized[1] - 0.8).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_l2_zero_vector() {
        let v = [0.0, 0.0, 0.0];
        let normalized = normalize_l2(&v);
        assert_eq!(normalized, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_l2_already_normalized() {
        let v = [1.0, 0.0, 0.0];
        let normalized = normalize_l2(&v);
        assert!((normalized[0] - 1.0).abs() < EPSILON);
        assert!((normalized[1] - 0.0).abs() < EPSILON);
        assert!((normalized[2] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_l2_inplace() {
        let mut v = [3.0, 4.0, 0.0];
        let original_norm = normalize_l2_inplace(&mut v);

        assert!((original_norm - 5.0).abs() < EPSILON);
        assert!((l2_norm(&v) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_is_normalized() {
        let normalized = [0.6, 0.8, 0.0];
        assert!(is_normalized(&normalized, 0.001));

        let not_normalized = [1.0, 1.0, 1.0];
        assert!(!is_normalized(&not_normalized, 0.001));
    }

    #[test]
    fn test_cosine_with_normalized() {
        let a = normalize_l2(&[1.0, 2.0, 3.0]);
        let b = normalize_l2(&[4.0, 5.0, 6.0]);

        let cosine = dot_product(&a, &b);

        // Cosine similarity should be between -1 and 1
        assert!(cosine >= -1.0 - EPSILON && cosine <= 1.0 + EPSILON);

        // Manual calculation for verification
        let a_raw = [1.0, 2.0, 3.0];
        let b_raw = [4.0, 5.0, 6.0];
        let dot_raw: f32 = a_raw.iter().zip(b_raw.iter()).map(|(x, y)| x * y).sum();
        let norm_a = l2_norm(&a_raw);
        let norm_b = l2_norm(&b_raw);
        let expected = dot_raw / (norm_a * norm_b);

        assert!((cosine - expected).abs() < EPSILON);
    }

    #[test]
    fn test_metric_trait() {
        let a = normalize_l2(&[1.0, 0.0, 0.0]);
        let b = normalize_l2(&[0.0, 1.0, 0.0]);

        // Cosine of orthogonal vectors = 0
        let cosine = Metric::Cosine.compute(&a, &b);
        assert!((cosine - 0.0).abs() < EPSILON);
        assert!(Metric::Cosine.higher_is_better());

        // L2 of orthogonal unit vectors = 2 (squared)
        let l2 = Metric::L2.compute(&a, &b);
        assert!((l2 - 2.0).abs() < EPSILON);
        assert!(!Metric::L2.higher_is_better());
    }

    #[test]
    fn test_dot_product_odd_length() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        // 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5 + 8 + 9 + 8 + 5 = 35
        let result = dot_product(&a, &b);
        assert!((result - 35.0).abs() < EPSILON);
    }

    #[test]
    fn test_l2_squared_odd_length() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0];
        // (1)^2 * 5 = 5
        let result = l2_squared(&a, &b);
        assert!((result - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_large_vectors_384() {
        // Common embedding size
        let a: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..384).map(|i| (i as f32 * 0.1).cos()).collect();

        let dot_result = dot_product(&a, &b);
        let l2_result = l2_squared(&a, &b);

        // Just verify they complete without panic and produce reasonable values
        assert!(dot_result.is_finite());
        assert!(l2_result.is_finite());
        assert!(l2_result >= 0.0);
    }

    #[test]
    fn test_large_vectors_1536() {
        // OpenAI ada-002 embedding size
        let a: Vec<f32> = (0..1536).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i as f32 * 0.01).cos()).collect();

        let dot_result = dot_product(&a, &b);
        let l2_result = l2_squared(&a, &b);

        assert!(dot_result.is_finite());
        assert!(l2_result.is_finite());
        assert!(l2_result >= 0.0);
    }
}
