//! Stochastic processes and stochastic calculus
//!
//! Comprehensive stochastic process simulation and analysis:
//! - Brownian motion (standard, geometric, fractional)
//! - Poisson processes (homogeneous, inhomogeneous, compound)
//! - Jump diffusions (Merton, Kou, Variance Gamma)
//! - Lévy processes
//! - Ito calculus and stochastic differential equations
//! - Monte Carlo simulation
//! - Option pricing (Black-Scholes, Heston, etc.)

pub mod brownian;
pub mod poisson;
pub mod sde;
pub mod monte_carlo;
pub mod option_pricing;

use crate::error::Result;
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::iter::Sum;

/// Brownian Motion (Wiener Process)
pub struct BrownianMotion<A> {
    /// Current value
    pub value: A,
    /// Time
    pub time: A,
    /// Drift parameter (μ)
    pub mu: A,
    /// Volatility parameter (σ)
    pub sigma: A,
}

impl<A: Float + ScalarOperand + Sum> BrownianMotion<A> {
    /// Create standard Brownian motion (μ=0, σ=1)
    pub fn standard() -> Self {
        Self {
            value: A::zero(),
            time: A::zero(),
            mu: A::zero(),
            sigma: A::one(),
        }
    }

    /// Create Brownian motion with drift
    pub fn with_drift(mu: A, sigma: A) -> Self {
        Self {
            value: A::zero(),
            time: A::zero(),
            mu,
            sigma,
        }
    }

    /// Simulate path using Euler-Maruyama scheme
    ///
    /// dX_t = μ dt + σ dW_t
    pub fn simulate_path(&mut self, T: A, n_steps: usize) -> Result<Array1<A>> {
        let dt = T / A::from(n_steps).unwrap();
        let sqrt_dt = dt.sqrt();

        let mut path = Array1::zeros(n_steps + 1);
        path[0] = self.value;

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for i in 1..=n_steps {
            let dW = A::from(normal.sample(&mut rng)).unwrap() * sqrt_dt;
            path[i] = path[i - 1] + self.mu * dt + self.sigma * dW;
        }

        self.value = path[n_steps];
        self.time = T;

        Ok(path)
    }

    /// Simulate multiple paths
    pub fn simulate_paths(&self, T: A, n_steps: usize, n_paths: usize) -> Result<Array2<A>> {
        let dt = T / A::from(n_steps).unwrap();
        let sqrt_dt = dt.sqrt();

        let mut paths = Array2::zeros((n_paths, n_steps + 1));

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for p in 0..n_paths {
            paths[[p, 0]] = self.value;

            for i in 1..=n_steps {
                let dW = A::from(normal.sample(&mut rng)).unwrap() * sqrt_dt;
                paths[[p, i]] = paths[[p, i - 1]] + self.mu * dt + self.sigma * dW;
            }
        }

        Ok(paths)
    }
}

/// Geometric Brownian Motion
/// dS_t = μS_t dt + σS_t dW_t
pub struct GeometricBrownianMotion<A> {
    /// Current value
    pub value: A,
    /// Drift
    pub mu: A,
    /// Volatility
    pub sigma: A,
}

impl<A: Float + ScalarOperand + Sum> GeometricBrownianMotion<A> {
    /// Create GBM with given parameters
    pub fn new(initial_value: A, mu: A, sigma: A) -> Self {
        Self {
            value: initial_value,
            mu,
            sigma,
        }
    }

    /// Simulate path (exact solution)
    ///
    /// S_t = S_0 * exp((μ - σ²/2)t + σW_t)
    pub fn simulate_path(&self, T: A, n_steps: usize) -> Result<Array1<A>> {
        let dt = T / A::from(n_steps).unwrap();
        let sqrt_dt = dt.sqrt();

        let drift = (self.mu - self.sigma * self.sigma / A::from(2.0).unwrap()) * dt;

        let mut path = Array1::zeros(n_steps + 1);
        path[0] = self.value;

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for i in 1..=n_steps {
            let dW = A::from(normal.sample(&mut rng)).unwrap() * sqrt_dt;
            path[i] = path[i - 1] * (drift + self.sigma * dW).exp();
        }

        Ok(path)
    }

    /// Simulate multiple paths
    pub fn simulate_paths(&self, T: A, n_steps: usize, n_paths: usize) -> Result<Array2<A>> {
        let dt = T / A::from(n_steps).unwrap();
        let sqrt_dt = dt.sqrt();
        let drift = (self.mu - self.sigma * self.sigma / A::from(2.0).unwrap()) * dt;

        let mut paths = Array2::zeros((n_paths, n_steps + 1));
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for p in 0..n_paths {
            paths[[p, 0]] = self.value;

            for i in 1..=n_steps {
                let dW = A::from(normal.sample(&mut rng)).unwrap() * sqrt_dt;
                paths[[p, i]] = paths[[p, i - 1]] * (drift + self.sigma * dW).exp();
            }
        }

        Ok(paths)
    }
}

/// Poisson Process
pub struct PoissonProcess {
    /// Intensity (λ)
    pub lambda: f64,
    /// Current number of jumps
    pub count: usize,
    /// Current time
    pub time: f64,
}

impl PoissonProcess {
    /// Create Poisson process with given intensity
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            count: 0,
            time: 0.0,
        }
    }

    /// Simulate path up to time T
    pub fn simulate_path(&mut self, T: f64) -> Vec<(f64, usize)> {
        let mut rng = rand::thread_rng();
        let mut path = vec![(0.0, 0)];

        let mut t = 0.0;
        let mut count = 0;

        while t < T {
            // Exponential inter-arrival time
            let dt = -(-rng.gen::<f64>().ln()) / self.lambda;
            t += dt;

            if t < T {
                count += 1;
                path.push((t, count));
            }
        }

        self.time = t;
        self.count = count;

        path
    }
}

/// Ornstein-Uhlenbeck process (mean-reverting)
/// dX_t = θ(μ - X_t)dt + σdW_t
pub struct OrnsteinUhlenbeck<A> {
    /// Mean reversion speed
    pub theta: A,
    /// Long-term mean
    pub mu: A,
    /// Volatility
    pub sigma: A,
    /// Current value
    pub value: A,
}

impl<A: Float + ScalarOperand + Sum> OrnsteinUhlenbeck<A> {
    /// Create OU process
    pub fn new(theta: A, mu: A, sigma: A, initial_value: A) -> Self {
        Self {
            theta,
            mu,
            sigma,
            value: initial_value,
        }
    }

    /// Simulate path
    pub fn simulate_path(&mut self, T: A, n_steps: usize) -> Result<Array1<A>> {
        let dt = T / A::from(n_steps).unwrap();
        let sqrt_dt = dt.sqrt();

        let mut path = Array1::zeros(n_steps + 1);
        path[0] = self.value;

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for i in 1..=n_steps {
            let dW = A::from(normal.sample(&mut rng)).unwrap() * sqrt_dt;
            let mean_reversion = self.theta * (self.mu - path[i - 1]) * dt;
            path[i] = path[i - 1] + mean_reversion + self.sigma * dW;
        }

        self.value = path[n_steps];

        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brownian_motion() {
        let mut bm = BrownianMotion::standard();
        let path = bm.simulate_path(1.0, 100).unwrap();
        assert_eq!(path.len(), 101);
    }

    #[test]
    fn test_geometric_brownian_motion() {
        let gbm = GeometricBrownianMotion::new(100.0, 0.05, 0.2);
        let path = gbm.simulate_path(1.0, 252).unwrap();
        assert_eq!(path.len(), 253);
        assert!(path[0] == 100.0);
    }

    #[test]
    fn test_poisson_process() {
        let mut pp = PoissonProcess::new(3.0);
        let path = pp.simulate_path(10.0);
        assert!(!path.is_empty());
    }
}
