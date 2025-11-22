//! Optimization algorithms
//!
//! Includes gradient descent variants, L-BFGS, coordinate descent,
//! and other optimization methods for machine learning.

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2};
use num_traits::Float;

pub mod gradient_descent;
pub mod lbfgs;
pub mod coordinate_descent;

/// Trait for objective functions
pub trait ObjectiveFunction<A: Float> {
    /// Evaluate the objective function
    fn value(&self, x: &Array1<A>) -> A;

    /// Compute the gradient
    fn gradient(&self, x: &Array1<A>) -> Array1<A>;

    /// Compute both value and gradient
    fn value_gradient(&self, x: &Array1<A>) -> (A, Array1<A>) {
        (self.value(x), self.gradient(x))
    }
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult<A> {
    /// Optimal parameters
    pub x: Array1<A>,
    /// Final objective value
    pub fun: A,
    /// Number of iterations
    pub nit: usize,
    /// Whether optimization converged
    pub success: bool,
    /// Status message
    pub message: String,
}

/// Line search using backtracking
pub fn backtracking_line_search<A, F>(
    f: &F,
    x: &Array1<A>,
    direction: &Array1<A>,
    gradient: &Array1<A>,
    alpha: A,
    rho: A,
    c: A,
) -> A
where
    A: Float,
    F: ObjectiveFunction<A>,
{
    let mut alpha = alpha;
    let fx = f.value(x);
    let slope: A = gradient.iter().zip(direction.iter()).map(|(&g, &d)| g * d).sum();

    for _ in 0..50 {
        let x_new: Array1<A> = x.iter().zip(direction.iter()).map(|(&xi, &di)| xi + alpha * di).collect();
        let fx_new = f.value(&x_new);

        if fx_new <= fx + c * alpha * slope {
            return alpha;
        }

        alpha = alpha * rho;
    }

    alpha
}

/// Compute numerical gradient using finite differences
pub fn numerical_gradient<A, F>(f: &F, x: &Array1<A>, epsilon: A) -> Array1<A>
where
    A: Float,
    F: Fn(&Array1<A>) -> A,
{
    let mut grad = Array1::zeros(x.len());

    for i in 0..x.len() {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();

        x_plus[i] = x_plus[i] + epsilon;
        x_minus[i] = x_minus[i] - epsilon;

        grad[i] = (f(&x_plus) - f(&x_minus)) / (A::from(2.0).unwrap() * epsilon);
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    struct QuadraticFunction;

    impl ObjectiveFunction<f64> for QuadraticFunction {
        fn value(&self, x: &Array1<f64>) -> f64 {
            x.iter().map(|&xi| xi * xi).sum()
        }

        fn gradient(&self, x: &Array1<f64>) -> Array1<f64> {
            x.mapv(|xi| 2.0 * xi)
        }
    }

    #[test]
    fn test_objective_function() {
        let f = QuadraticFunction;
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let val = f.value(&x);
        assert_eq!(val, 14.0);
    }
}
