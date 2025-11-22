//! Linear models for regression and classification
//!
//! Includes linear regression, logistic regression, Ridge, Lasso, ElasticNet,
//! and generalized linear models.

use crate::error::{Result, RustyAIError};
use crate::linalg::solve::lstsq;
use crate::traits::{Predictor, Regressor};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;
use serde::{Deserialize, Serialize};

pub mod logistic;
pub mod regularized;

// Re-export sub-module types
pub use logistic::LogisticRegression;
pub use regularized::{Ridge, Lasso};

/// Linear Regression model using ordinary least squares
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression<A> {
    /// Whether to fit an intercept
    fit_intercept: bool,
    /// Model coefficients
    coef: Option<Array1<A>>,
    /// Intercept term
    intercept: Option<A>,
}

impl<A: Float + ScalarOperand + Sum> LinearRegression<A> {
    /// Create a new LinearRegression model
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            coef: None,
            intercept: None,
        }
    }

    /// Create model without fitting intercept
    pub fn without_intercept() -> Self {
        Self {
            fit_intercept: false,
            coef: None,
            intercept: None,
        }
    }

    /// Fit the linear regression model using Ordinary Least Squares
    ///
    /// Solves: β = (X^T X)^(-1) X^T y
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        let (n_samples, n_features) = X.dim();

        if n_samples != y.len() {
            return Err(RustyAIError::InvalidParameter {
                name: "X, y".to_string(),
                reason: format!("n_samples mismatch: X has {}, y has {}", n_samples, y.len()),
            });
        }

        if self.fit_intercept {
            // Add intercept column (column of ones)
            let mut X_with_intercept = Array2::ones((n_samples, n_features + 1));
            for i in 0..n_samples {
                for j in 0..n_features {
                    X_with_intercept[[i, j + 1]] = X[[i, j]];
                }
            }

            // Solve using least squares
            let params = lstsq(&X_with_intercept, y)?;

            self.intercept = Some(params[0]);
            self.coef = Some(params.slice(ndarray::s![1..]).to_owned());
        } else {
            // No intercept
            let params = lstsq(X, y)?;
            self.coef = Some(params);
            self.intercept = Some(A::zero());
        }

        Ok(())
    }

    /// Predict target values for samples in X
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let coef = self.coef.as_ref()
            .ok_or_else(|| RustyAIError::NotFitted)?;
        let intercept = self.intercept
            .ok_or_else(|| RustyAIError::NotFitted)?;

        let n_samples = X.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut pred = intercept;
            for j in 0..coef.len() {
                pred = pred + X[[i, j]] * coef[j];
            }
            predictions[i] = pred;
        }

        Ok(predictions)
    }

    /// Compute R² score (coefficient of determination)
    ///
    /// R² = 1 - SS_res / SS_tot
    /// where SS_res = Σ(y - ŷ)² and SS_tot = Σ(y - ȳ)²
    pub fn score(&self, X: &Array2<A>, y: &Array1<A>) -> Result<A> {
        let predictions = self.predict(X)?;

        // Compute mean of y
        let y_mean = y.sum() / A::from(y.len()).unwrap();

        // SS_res: residual sum of squares
        let ss_res: A = y.iter().zip(predictions.iter())
            .map(|(&true_y, &pred_y)| {
                let diff = true_y - pred_y;
                diff * diff
            })
            .sum();

        // SS_tot: total sum of squares
        let ss_tot: A = y.iter()
            .map(|&yi| {
                let diff = yi - y_mean;
                diff * diff
            })
            .sum();

        if ss_tot == A::zero() {
            return Ok(A::one()); // Perfect fit if no variance
        }

        Ok(A::one() - ss_res / ss_tot)
    }

    /// Get the model coefficients
    pub fn coefficients(&self) -> Option<&Array1<A>> {
        self.coef.as_ref()
    }

    /// Get the intercept term
    pub fn intercept(&self) -> Option<A> {
        self.intercept
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.coef.is_some()
    }
}

impl<A: Float + ScalarOperand + Sum> Default for LinearRegression<A> {
    fn default() -> Self {
        Self::new()
    }
}
