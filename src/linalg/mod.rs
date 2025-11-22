//! Linear algebra operations
//!
//! Comprehensive linear algebra functionality including matrix decompositions,
//! solvers, and numerical operations.

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

pub mod decomposition;
pub mod solve;

/// Compute the dot product of two vectors
pub fn dot<A: Float + ScalarOperand + Sum>(a: &Array1<A>, b: &Array1<A>) -> Result<A> {
    if a.len() != b.len() {
        return Err(RustyAIError::DimensionMismatch(format!(
            "Vector dimensions must match: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
}

/// Matrix-vector multiplication
pub fn matvec<A: Float + ScalarOperand + Sum>(mat: &Array2<A>, vec: &Array1<A>) -> Result<Array1<A>> {
    if mat.ncols() != vec.len() {
        return Err(RustyAIError::DimensionMismatch(format!(
            "Matrix columns ({}) must match vector length ({})",
            mat.ncols(),
            vec.len()
        )));
    }

    let mut result = Array1::zeros(mat.nrows());
    for (i, row) in mat.rows().into_iter().enumerate() {
        result[i] = row.iter().zip(vec.iter()).map(|(&a, &b)| a * b).sum();
    }
    Ok(result)
}

/// Matrix multiplication
pub fn matmul<A: Float + ScalarOperand + Sum>(a: &Array2<A>, b: &Array2<A>) -> Result<Array2<A>> {
    if a.ncols() != b.nrows() {
        return Err(RustyAIError::DimensionMismatch(format!(
            "Matrix dimensions incompatible: ({}, {}) x ({}, {})",
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }

    let mut result = Array2::zeros((a.nrows(), b.ncols()));
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            for k in 0..a.ncols() {
                result[[i, j]] = result[[i, j]] + a[[i, k]] * b[[k, j]];
            }
        }
    }
    Ok(result)
}

/// Compute the Frobenius norm of a matrix
pub fn norm_frobenius<A: Float + ScalarOperand + Sum>(mat: &Array2<A>) -> A {
    mat.iter().map(|&x| x * x).sum::<A>().sqrt()
}

/// Compute the L2 norm (Euclidean norm) of a vector
pub fn norm_l2<A: Float + ScalarOperand + Sum>(vec: &Array1<A>) -> A {
    vec.iter().map(|&x| x * x).sum::<A>().sqrt()
}

/// Compute the L1 norm (Manhattan norm) of a vector
pub fn norm_l1<A: Float + ScalarOperand + Sum>(vec: &Array1<A>) -> A {
    vec.iter().map(|&x| x.abs()).sum()
}

/// Transpose a matrix
pub fn transpose<A: Clone>(mat: &Array2<A>) -> Array2<A> {
    mat.t().to_owned()
}

/// Compute the trace of a square matrix
pub fn trace<A: Float + ScalarOperand + Sum>(mat: &Array2<A>) -> Result<A> {
    if mat.nrows() != mat.ncols() {
        return Err(RustyAIError::InvalidShape {
            expected: "square matrix".to_string(),
            got: format!("({}, {})", mat.nrows(), mat.ncols()),
        });
    }
    Ok((0..mat.nrows()).map(|i| mat[[i, i]]).sum())
}

/// Create an identity matrix
pub fn eye<A: Float + ScalarOperand + Sum>(n: usize) -> Array2<A> {
    let mut mat = Array2::zeros((n, n));
    for i in 0..n {
        mat[[i, i]] = A::one();
    }
    mat
}

/// Compute matrix determinant using LU decomposition (simple implementation)
pub fn det<A: Float + ScalarOperand + Sum>(mat: &Array2<A>) -> Result<A> {
    if mat.nrows() != mat.ncols() {
        return Err(RustyAIError::InvalidShape {
            expected: "square matrix".to_string(),
            got: format!("({}, {})", mat.nrows(), mat.ncols()),
        });
    }

    let n = mat.nrows();
    if n == 1 {
        return Ok(mat[[0, 0]]);
    }
    if n == 2 {
        return Ok(mat[[0, 0]] * mat[[1, 1]] - mat[[0, 1]] * mat[[1, 0]]);
    }

    // For larger matrices, use LU decomposition
    // This is a simplified implementation
    let mut a = mat.clone();
    let mut det = A::one();
    let mut sign = A::one();

    for k in 0..n {
        // Find pivot
        let mut pivot_row = k;
        let mut pivot_val = a[[k, k]].abs();
        for i in k + 1..n {
            if a[[i, k]].abs() > pivot_val {
                pivot_row = i;
                pivot_val = a[[i, k]].abs();
            }
        }

        if pivot_val < A::from(1e-10).unwrap() {
            return Ok(A::zero()); // Singular matrix
        }

        // Swap rows if needed
        if pivot_row != k {
            for j in 0..n {
                let temp = a[[k, j]];
                a[[k, j]] = a[[pivot_row, j]];
                a[[pivot_row, j]] = temp;
            }
            sign = -sign;
        }

        // Eliminate column
        for i in k + 1..n {
            let factor = a[[i, k]] / a[[k, k]];
            for j in k..n {
                a[[i, j]] = a[[i, j]] - factor * a[[k, j]];
            }
        }

        det = det * a[[k, k]];
    }

    Ok(det * sign)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dot_product() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let result = dot(&a, &b).unwrap();
        assert_abs_diff_eq!(result, 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_determinant() {
        let mat = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let d = det(&mat).unwrap();
        assert_abs_diff_eq!(d, -2.0, epsilon = 1e-10);
    }
}
