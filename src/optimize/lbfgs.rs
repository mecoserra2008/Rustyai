//! L-BFGS optimization algorithm

use super::{ObjectiveFunction, OptimizationResult};
use ndarray::{Array1, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

/// L-BFGS optimizer configuration
#[derive(Debug, Clone)]
pub struct LBFGSConfig<A> {
    /// Number of corrections to approximate the Hessian
    pub m: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: A,
}

impl<A: Float + ScalarOperand + Sum> Default for LBFGSConfig<A> {
    fn default() -> Self {
        Self {
            m: 10,
            max_iter: 1000,
            tol: A::from(1e-6).unwrap(),
        }
    }
}

/// Perform L-BFGS optimization (simplified implementation)
pub fn lbfgs<A, F>(
    f: &F,
    x0: &Array1<A>,
    config: &LBFGSConfig<A>,
) -> OptimizationResult<A>
where
    A: Float,
    F: ObjectiveFunction<A>,
{
    let mut x = x0.clone();
    let mut s_history: Vec<Array1<A>> = Vec::new();
    let mut y_history: Vec<Array1<A>> = Vec::new();
    let mut prev_grad = f.gradient(&x);

    for iter in 0..config.max_iter {
        let (fx, grad) = f.value_gradient(&x);

        // Check convergence
        let grad_norm: A = grad.iter().map(|&g| g * g).sum::<A>().sqrt();
        if grad_norm < config.tol {
            return OptimizationResult {
                x,
                fun: fx,
                nit: iter,
                success: true,
                message: "Optimization converged successfully".to_string(),
            };
        }

        // Compute search direction using L-BFGS two-loop recursion
        let direction = lbfgs_direction(&grad, &s_history, &y_history);

        // Line search (simple backtracking)
        let mut alpha = A::one();
        let mut x_new = &x - &direction.mapv(|d| d * alpha);
        let mut fx_new = f.value(&x_new);

        while fx_new > fx && alpha > A::from(1e-10).unwrap() {
            alpha = alpha * A::from(0.5).unwrap();
            x_new = &x - &direction.mapv(|d| d * alpha);
            fx_new = f.value(&x_new);
        }

        // Update history
        let s = &x_new - &x;
        let y = &f.gradient(&x_new) - &grad;

        s_history.push(s);
        y_history.push(y);

        // Keep only m most recent updates
        if s_history.len() > config.m {
            s_history.remove(0);
            y_history.remove(0);
        }

        x = x_new;
        prev_grad = grad;
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

fn lbfgs_direction<A: Float + ScalarOperand + Sum>(
    grad: &Array1<A>,
    s_history: &[Array1<A>],
    y_history: &[Array1<A>],
) -> Array1<A> {
    if s_history.is_empty() {
        return grad.clone();
    }

    let m = s_history.len();
    let mut q = grad.clone();
    let mut alpha_vec = vec![A::zero(); m];

    // First loop
    for i in (0..m).rev() {
        let rho = A::one()
            / s_history[i]
                .iter()
                .zip(y_history[i].iter())
                .map(|(&si, &yi)| si * yi)
                .sum::<A>();

        let alpha = rho
            * s_history[i]
                .iter()
                .zip(q.iter())
                .map(|(&si, &qi)| si * qi)
                .sum::<A>();

        alpha_vec[i] = alpha;
        q = &q - &y_history[i].mapv(|yi| yi * alpha);
    }

    // Scale by H0 (using identity for simplicity)
    let mut r = q;

    // Second loop
    for i in 0..m {
        let rho = A::one()
            / s_history[i]
                .iter()
                .zip(y_history[i].iter())
                .map(|(&si, &yi)| si * yi)
                .sum::<A>();

        let beta = rho
            * y_history[i]
                .iter()
                .zip(r.iter())
                .map(|(&yi, &ri)| yi * ri)
                .sum::<A>();

        r = &r + &s_history[i].mapv(|si| si * (alpha_vec[i] - beta));
    }

    r
}
