//! Linear system solvers

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Solve a linear system Ax = b using Gaussian elimination
pub fn solve<A: Float>(a: &Array2<A>, b: &Array1<A>) -> Result<Array1<A>> {
    if a.nrows() != a.ncols() {
        return Err(RustyAIError::InvalidShape {
            expected: "square matrix".to_string(),
            got: format!("({}, {})", a.nrows(), a.ncols()),
        });
    }

    if a.nrows() != b.len() {
        return Err(RustyAIError::DimensionMismatch(format!(
            "Matrix rows ({}) must match vector length ({})",
            a.nrows(),
            b.len()
        )));
    }

    let n = a.nrows();
    let mut aug = Array2::zeros((n, n + 1));

    // Create augmented matrix [A|b]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for k in 0..n {
        // Find pivot
        let mut pivot_row = k;
        let mut pivot_val = aug[[k, k]].abs();
        for i in k + 1..n {
            if aug[[i, k]].abs() > pivot_val {
                pivot_row = i;
                pivot_val = aug[[i, k]].abs();
            }
        }

        if pivot_val < A::from(1e-10).unwrap() {
            return Err(RustyAIError::SingularMatrix);
        }

        // Swap rows
        if pivot_row != k {
            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = temp;
            }
        }

        // Eliminate column
        for i in k + 1..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                aug[[i, j]] = aug[[i, j]] - factor * aug[[k, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in i + 1..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Solve least squares problem: minimize ||Ax - b||^2
pub fn lstsq<A: Float>(a: &Array2<A>, b: &Array1<A>) -> Result<Array1<A>> {
    // Use normal equations: A^T A x = A^T b
    let at = a.t();
    let ata = at.dot(a);
    let atb = at.dot(b);

    solve(&ata, &atb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_solve_linear_system() {
        let a = Array2::from_shape_vec((2, 2), vec![3.0, 1.0, 1.0, 2.0]).unwrap();
        let b = Array1::from_vec(vec![9.0, 8.0]);
        let x = solve(&a, &b).unwrap();
        assert_abs_diff_eq!(x[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 3.0, epsilon = 1e-10);
    }
}
