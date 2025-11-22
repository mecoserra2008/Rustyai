//! Gradient descent optimization

use super::{ObjectiveFunction, OptimizationResult};
use ndarray::{Array1, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

/// Gradient descent optimizer configuration
#[derive(Debug, Clone)]
pub struct GradientDescentConfig<A> {
    /// Learning rate
    pub learning_rate: A,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: A,
    /// Whether to use momentum
    pub momentum: Option<A>,
    /// Whether to use Nesterov momentum
    pub nesterov: bool,
}

impl<A: Float + ScalarOperand + Sum> Default for GradientDescentConfig<A> {
    fn default() -> Self {
        Self {
            learning_rate: A::from(0.01).unwrap(),
            max_iter: 1000,
            tol: A::from(1e-6).unwrap(),
            momentum: None,
            nesterov: false,
        }
    }
}

/// Perform gradient descent optimization
pub fn gradient_descent<A, F>(
    f: &F,
    x0: &Array1<A>,
    config: &GradientDescentConfig<A>,
) -> OptimizationResult<A>
where
    A: Float,
    F: ObjectiveFunction<A>,
{
    let mut x = x0.clone();
    let mut velocity = Array1::zeros(x.len());
    let mut prev_f = A::infinity();

    for iter in 0..config.max_iter {
        let (fx, grad) = f.value_gradient(&x);

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

        // Update with momentum if specified
        if let Some(momentum) = config.momentum {
            velocity = velocity.mapv(|v| v * momentum) - grad.mapv(|g| g * config.learning_rate);
            x = &x + &velocity;
        } else {
            x = &x - &grad.mapv(|g| g * config.learning_rate);
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

/// Adam optimizer configuration
#[derive(Debug, Clone)]
pub struct AdamConfig<A> {
    /// Learning rate
    pub learning_rate: A,
    /// Beta1 (exponential decay rate for first moment)
    pub beta1: A,
    /// Beta2 (exponential decay rate for second moment)
    pub beta2: A,
    /// Epsilon for numerical stability
    pub epsilon: A,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: A,
}

impl<A: Float + ScalarOperand + Sum> Default for AdamConfig<A> {
    fn default() -> Self {
        Self {
            learning_rate: A::from(0.001).unwrap(),
            beta1: A::from(0.9).unwrap(),
            beta2: A::from(0.999).unwrap(),
            epsilon: A::from(1e-8).unwrap(),
            max_iter: 1000,
            tol: A::from(1e-6).unwrap(),
        }
    }
}

/// Perform Adam optimization
pub fn adam<A, F>(
    f: &F,
    x0: &Array1<A>,
    config: &AdamConfig<A>,
) -> OptimizationResult<A>
where
    A: Float,
    F: ObjectiveFunction<A>,
{
    let mut x = x0.clone();
    let mut m = Array1::zeros(x.len()); // First moment
    let mut v = Array1::zeros(x.len()); // Second moment
    let mut prev_f = A::infinity();

    for t in 1..=config.max_iter {
        let (fx, grad) = f.value_gradient(&x);

        // Check convergence
        if (fx - prev_f).abs() < config.tol {
            return OptimizationResult {
                x,
                fun: fx,
                nit: t,
                success: true,
                message: "Optimization converged successfully".to_string(),
            };
        }

        // Update biased first moment estimate
        m = m.mapv(|mi| mi * config.beta1)
            + grad.mapv(|gi| gi * (A::one() - config.beta1));

        // Update biased second raw moment estimate
        v = v.mapv(|vi| vi * config.beta2)
            + grad.mapv(|gi| gi * gi * (A::one() - config.beta2));

        // Compute bias-corrected first moment estimate
        let t_float = A::from(t).unwrap();
        let m_hat = m.mapv(|mi| mi / (A::one() - config.beta1.powf(t_float)));

        // Compute bias-corrected second raw moment estimate
        let v_hat = v.mapv(|vi| vi / (A::one() - config.beta2.powf(t_float)));

        // Update parameters
        let update: Array1<A> = m_hat.iter()
            .zip(v_hat.iter())
            .map(|(mi, vi): (&A, &A)| config.learning_rate * (*mi) / (vi.sqrt() + config.epsilon))
            .collect();
        x = x - update;

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
    fn test_gradient_descent() {
        let f = QuadraticFunction;
        let x0 = Array1::from_vec(vec![5.0, 5.0]);
        let config = GradientDescentConfig::default();
        let result = gradient_descent(&f, &x0, &config);
        assert!(result.success);
        assert!(result.fun < 0.001);
    }
}
