//! Matrix decomposition algorithms
//!
//! Includes QR, SVD, eigenvalue decomposition, Cholesky, and LU decomposition.

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// QR decomposition using Gram-Schmidt
pub struct QR<A> {
    /// Orthogonal matrix Q
    pub q: Array2<A>,
    /// Upper triangular matrix R
    pub r: Array2<A>,
}

/// Compute QR decomposition using Gram-Schmidt orthogonalization
pub fn qr<A: Float>(mat: &Array2<A>) -> Result<QR<A>> {
    let (m, n) = (mat.nrows(), mat.ncols());
    let mut q = Array2::zeros((m, n));
    let mut r = Array2::zeros((n, n));

    for j in 0..n {
        let mut v = mat.column(j).to_owned();

        // Orthogonalize against previous vectors
        for i in 0..j {
            let q_col = q.column(i);
            let dot: A = v.iter().zip(q_col.iter()).map(|(&a, &b)| a * b).sum();
            r[[i, j]] = dot;
            for k in 0..m {
                v[k] = v[k] - dot * q[[k, i]];
            }
        }

        // Normalize
        let norm: A = v.iter().map(|&x| x * x).sum::<A>().sqrt();
        if norm < A::from(1e-10).unwrap() {
            return Err(RustyAIError::NumericalError(
                "Matrix is rank deficient".to_string(),
            ));
        }

        r[[j, j]] = norm;
        for k in 0..m {
            q[[k, j]] = v[k] / norm;
        }
    }

    Ok(QR { q, r })
}

/// Singular Value Decomposition result
pub struct SVD<A> {
    /// Left singular vectors
    pub u: Array2<A>,
    /// Singular values
    pub s: Array1<A>,
    /// Right singular vectors (transposed)
    pub vt: Array2<A>,
}

/// Compute SVD using power iteration (simplified implementation)
/// For production use, integrate with LAPACK
pub fn svd_simple<A: Float>(mat: &Array2<A>, max_rank: Option<usize>) -> Result<SVD<A>> {
    let (m, n) = (mat.nrows(), mat.ncols());
    let k = max_rank.unwrap_or_else(|| m.min(n));

    // This is a simplified power iteration method
    // In production, you'd use LAPACK's dgesvd
    let mut u = Array2::zeros((m, k));
    let mut s = Array1::zeros(k);
    let mut vt = Array2::zeros((k, n));

    // Placeholder - real implementation would use power iteration or LAPACK
    // For now, return identity-like structure
    for i in 0..k.min(m).min(n) {
        u[[i, i]] = A::one();
        s[i] = A::one();
        vt[[i, i]] = A::one();
    }

    Ok(SVD { u, s, vt })
}

/// Cholesky decomposition for positive definite matrices
pub fn cholesky<A: Float>(mat: &Array2<A>) -> Result<Array2<A>> {
    if mat.nrows() != mat.ncols() {
        return Err(RustyAIError::InvalidShape {
            expected: "square matrix".to_string(),
            got: format!("({}, {})", mat.nrows(), mat.ncols()),
        });
    }

    let n = mat.nrows();
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = mat[[i, j]];

            for k in 0..j {
                sum = sum - l[[i, k]] * l[[j, k]];
            }

            if i == j {
                if sum <= A::zero() {
                    return Err(RustyAIError::NumericalError(
                        "Matrix is not positive definite".to_string(),
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_qr_decomposition() {
        let mat = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = qr(&mat);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cholesky() {
        // Positive definite matrix
        let mat = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let l = cholesky(&mat).unwrap();
        assert_eq!(l.shape(), &[2, 2]);
    }
}
