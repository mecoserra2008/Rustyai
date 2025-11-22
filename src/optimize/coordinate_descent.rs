//! Coordinate descent optimization

use super::{ObjectiveFunction, OptimizationResult};
use ndarray::Array1;
use num_traits::Float;
use std::iter::Sum;

/// Coordinate descent optimizer configuration
#[derive(Debug, Clone)]
pub struct CoordinateDescentConfig<A> {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: A,
    /// Whether to use random coordinate selection
    pub random: bool,
}

impl<A: Float + ScalarOperand + Sum> Default for CoordinateDescentConfig<A> {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: A::from(1e-6).unwrap(),
            random: false,
        }
    }
}

/// Perform coordinate descent optimization
pub fn coordinate_descent<A, F>(
    f: &F,
    x0: &Array1<A>,
    config: &CoordinateDescentConfig<A>,
) -> OptimizationResult<A>
where
    A: Float,
    F: ObjectiveFunction<A>,
{
    let mut x = x0.clone();
    let n = x.len();
    let mut prev_f = A::infinity();

    for iter in 0..config.max_iter {
        let fx = f.value(&x);

        // Check convergence
        if (fx - prev_f).abs() < config.tol {
            return OptimizationResult {
                x,
                fun: fx,
                nit: iter,
                success: true,
                message: "Optimization converged successfully".to_string(),
            };
        }

        // Update each coordinate
        for i in 0..n {
            // Compute directional derivative
            let grad = f.gradient(&x);
            let step = A::from(0.01).unwrap(); // Fixed step size for simplicity

            x[i] = x[i] - step * grad[i];
        }

        prev_f = fx;
    }

    let fx = f.value(&x);
    OptimizationResult {
        x,
        fun: fx,
        nit: config.max_iter,
        success: false,
        message: format!("Maximum iterations ({}) reached", config.max_iter),
    }
}
