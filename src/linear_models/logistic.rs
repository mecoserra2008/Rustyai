//! Logistic regression for binary and multiclass classification

use crate::error::{Result, RustyAIError};
use crate::optimize::{gradient_descent::*, ObjectiveFunction};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::iter::Sum;
use serde::{Deserialize, Serialize};

/// Logistic Regression classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression<A> {
    /// Model coefficients
    pub coef: Option<Array1<A>>,
    /// Intercept term
    pub intercept: Option<A>,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Learning rate
    pub learning_rate: A,
    /// Regularization strength
    pub C: A,
}

impl<A: Float + ScalarOperand + Sum> LogisticRegression<A> {
    /// Create a new LogisticRegression model
    pub fn new() -> Self {
        Self {
            coef: None,
            intercept: None,
            max_iter: 1000,
            learning_rate: A::from(0.01).unwrap(),
            C: A::one(),
        }
    }

    /// Sigmoid function
    fn sigmoid(z: A) -> A {
        A::one() / (A::one() + (-z).exp())
    }

    /// Fit the model
    pub fn fit(mut self, X: &Array2<A>, y: &Array1<A>) -> Result<Self> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        let n_features = X.ncols();
        let mut coef = Array1::zeros(n_features + 1);

        // Simple gradient descent
        for _ in 0..self.max_iter {
            let mut grad = Array1::zeros(n_features + 1);

            for i in 0..X.nrows() {
                let mut z = coef[0]; // intercept
                for j in 0..n_features {
                    z = z + coef[j + 1] * X[[i, j]];
                }

                let pred = Self::sigmoid(z);
                let error = pred - y[i];

                grad[0] = grad[0] + error;
                for j in 0..n_features {
                    grad[j + 1] = grad[j + 1] + error * X[[i, j]];
                }
            }

            let n = A::from(X.nrows()).unwrap();
            grad = grad.mapv(|g| g / n);

            coef = &coef - &grad.mapv(|g| g * self.learning_rate);
        }

        self.intercept = Some(coef[0]);
        self.coef = Some(coef.slice(ndarray::s![1..]).to_owned());

        Ok(self)
    }

    /// Predict probabilities
    pub fn predict_proba(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let coef = self.coef.as_ref().ok_or(RustyAIError::NotFitted)?;
        let intercept = self.intercept.ok_or(RustyAIError::NotFitted)?;

        let mut proba = Array1::zeros(X.nrows());
        for i in 0..X.nrows() {
            let mut z = intercept;
            for j in 0..X.ncols() {
                z = z + coef[j] * X[[i, j]];
            }
            proba[i] = Self::sigmoid(z);
        }

        Ok(proba)
    }

    /// Predict classes
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let proba = self.predict_proba(X)?;
        Ok(proba.mapv(|p| if p >= A::from(0.5).unwrap() { A::one() } else { A::zero() }))
    }
}

impl<A: Float + ScalarOperand + Sum> Default for LogisticRegression<A> {
    fn default() -> Self {
        Self::new()
    }
}
