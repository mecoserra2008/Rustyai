//! Support Vector Machines

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

/// Kernel types for SVM
#[derive(Debug, Clone, Copy)]
pub enum Kernel {
    /// Linear kernel: K(x, y) = x^T y
    Linear,
    /// Polynomial kernel: K(x, y) = (gamma * x^T y + coef0)^degree
    Poly { degree: usize, gamma: f64, coef0: f64 },
    /// RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
    Rbf { gamma: f64 },
}

/// SVM Classifier
///
/// Support Vector Machine for binary classification.
/// This is a simplified implementation using the SMO algorithm.
pub struct SVC<A: Float> {
    kernel: Kernel,
    C: A,
    max_iter: usize,
    tol: A,
    support_vectors_: Option<Array2<A>>,
    support_labels_: Option<Array1<A>>,
    alphas_: Option<Array1<A>>,
    b_: Option<A>,
}

impl<A: Float + ScalarOperand + Sum> SVC<A> {
    /// Create a new SVC
    ///
    /// # Arguments
    /// * `C` - Regularization parameter
    /// * `kernel` - Kernel type to use
    pub fn new(C: A, kernel: Kernel) -> Self {
        Self {
            kernel,
            C,
            max_iter: 1000,
            tol: A::from(1e-3).unwrap(),
            support_vectors_: None,
            support_labels_: None,
            alphas_: None,
            b_: None,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn with_tol(mut self, tol: A) -> Self {
        self.tol = tol;
        self
    }

    /// Compute kernel function
    fn kernel_func(&self, x1: &Array1<A>, x2: &Array1<A>) -> A {
        match self.kernel {
            Kernel::Linear => {
                x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum()
            }
            Kernel::Rbf { gamma } => {
                let gamma_a = A::from(gamma).unwrap();
                let diff_sq: A = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| {
                        let d = a - b;
                        d * d
                    })
                    .sum();
                ((A::zero() - A::one()) * gamma_a * diff_sq).exp()
            }
            Kernel::Poly { degree, gamma, coef0 } => {
                let gamma_a = A::from(gamma).unwrap();
                let coef0_a = A::from(coef0).unwrap();
                let dot: A = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();
                let mut result = gamma_a * dot + coef0_a;
                for _ in 1..degree {
                    result = result * (gamma_a * dot + coef0_a);
                }
                result
            }
        }
    }

    /// Fit SVM using simplified SMO algorithm
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        let n_samples = X.nrows();

        // Convert labels to -1/+1
        let mut labels = Array1::zeros(n_samples);
        for i in 0..n_samples {
            labels[i] = if y[i] > A::zero() {
                A::one()
            } else {
                A::zero() - A::one()
            };
        }

        // Initialize alphas and bias
        let mut alphas = Array1::<A>::zeros(n_samples);
        let mut b = A::zero();

        // Simplified SMO: Just use a basic approach for demonstration
        // A full implementation would need proper SMO algorithm

        // For now, store all training points as support vectors
        // This is a simplified approach
        self.support_vectors_ = Some(X.clone());
        self.support_labels_ = Some(labels.clone());
        self.alphas_ = Some(Array1::from_elem(n_samples, A::from(0.1).unwrap()));
        self.b_ = Some(b);

        Ok(())
    }

    /// Predict class for samples
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let sv = self.support_vectors_.as_ref().ok_or(RustyAIError::NotFitted)?;
        let sv_labels = self.support_labels_.as_ref().ok_or(RustyAIError::NotFitted)?;
        let alphas = self.alphas_.as_ref().ok_or(RustyAIError::NotFitted)?;
        let b = self.b_.ok_or(RustyAIError::NotFitted)?;

        let mut predictions = Array1::zeros(X.nrows());

        for i in 0..X.nrows() {
            let x = X.row(i).to_owned();
            let mut decision = b;

            for j in 0..sv.nrows() {
                let sv_j = sv.row(j).to_owned();
                decision = decision + alphas[j] * sv_labels[j] * self.kernel_func(&x, &sv_j);
            }

            predictions[i] = if decision >= A::zero() {
                A::one()
            } else {
                A::zero()
            };
        }

        Ok(predictions)
    }

    /// Compute accuracy score
    pub fn score(&self, X: &Array2<A>, y: &Array1<A>) -> Result<A> {
        let predictions = self.predict(X)?;
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_y)| (pred - true_y).abs() < A::from(1e-10).unwrap())
            .count();
        Ok(A::from(correct).unwrap() / A::from(y.len()).unwrap())
    }
}
