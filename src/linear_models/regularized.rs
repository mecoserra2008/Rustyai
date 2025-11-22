//! Regularized linear models (Ridge, Lasso, ElasticNet)

use crate::error::{Result, RustyAIError};
use crate::linalg::solve::solve;
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;
use serde::{Deserialize, Serialize};

/// Ridge regression (L2 regularization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ridge<A> {
    /// Regularization strength
    pub alpha: A,
    /// Model coefficients
    pub coef: Option<Array1<A>>,
    /// Intercept term
    pub intercept: Option<A>,
}

impl<A: Float + ScalarOperand + Sum> Ridge<A> {
    /// Create a new Ridge model
    pub fn new(alpha: A) -> Self {
        Self {
            alpha,
            coef: None,
            intercept: None,
        }
    }

    /// Fit the Ridge regression model
    ///
    /// Solves: (X^T X + alpha * I) beta = X^T y
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        // Ridge regression: (X^T X + alpha * I) beta = X^T y
        let xt = X.t();
        let xtx = xt.dot(X);
        let xty = xt.dot(y);

        // Add regularization term
        let mut xtx_reg = xtx.clone();
        for i in 0..xtx_reg.nrows() {
            xtx_reg[[i, i]] = xtx_reg[[i, i]] + self.alpha;
        }

        let coef = solve(&xtx_reg, &xty)?;

        self.coef = Some(coef);
        self.intercept = Some(A::zero());

        Ok(())
    }

    /// Compute R² score
    pub fn score(&self, X: &Array2<A>, y: &Array1<A>) -> Result<A> {
        let predictions = self.predict(X)?;
        let y_mean = y.sum() / A::from(y.len()).unwrap();

        let ss_res: A = y.iter().zip(predictions.iter())
            .map(|(&true_y, &pred_y)| {
                let diff = true_y - pred_y;
                diff * diff
            })
            .sum();

        let ss_tot: A = y.iter()
            .map(|&yi| {
                let diff = yi - y_mean;
                diff * diff
            })
            .sum();

        if ss_tot == A::zero() {
            return Ok(A::one());
        }

        Ok(A::one() - ss_res / ss_tot)
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.coef.is_some()
    }

    /// Predict
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let coef = self.coef.as_ref().ok_or(RustyAIError::NotFitted)?;

        let mut predictions = Array1::zeros(X.nrows());
        for (i, row) in X.rows().into_iter().enumerate() {
            predictions[i] = row.iter().zip(coef.iter()).map(|(&x, &c)| x * c).sum();
        }

        Ok(predictions)
    }
}

/// Lasso regression (L1 regularization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lasso<A> {
    /// Regularization strength
    pub alpha: A,
    /// Model coefficients
    pub coef: Option<Array1<A>>,
    /// Intercept term
    pub intercept: Option<A>,
    /// Maximum iterations
    pub max_iter: usize,
}

impl<A: Float + ScalarOperand + Sum> Lasso<A> {
    /// Create a new Lasso model
    pub fn new(alpha: A) -> Self {
        Self {
            alpha,
            coef: None,
            intercept: None,
            max_iter: 1000,
        }
    }

    /// Soft thresholding operator
    fn soft_threshold(x: A, lambda: A) -> A {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            A::zero()
        }
    }

    /// Fit Lasso model using coordinate descent
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        let mut coef = Array1::zeros(X.ncols());

        // Coordinate descent
        for _ in 0..self.max_iter {
            for j in 0..X.ncols() {
                let mut residual = y.clone();
                for k in 0..X.ncols() {
                    if k != j {
                        for i in 0..X.nrows() {
                            residual[i] = residual[i] - coef[k] * X[[i, k]];
                        }
                    }
                }

                let mut rho = A::zero();
                let mut norm_sq = A::zero();
                for i in 0..X.nrows() {
                    rho = rho + X[[i, j]] * residual[i];
                    norm_sq = norm_sq + X[[i, j]] * X[[i, j]];
                }

                if norm_sq > A::from(1e-10).unwrap() {
                    coef[j] = Self::soft_threshold(rho / norm_sq, self.alpha / norm_sq);
                }
            }
        }

        self.coef = Some(coef);
        self.intercept = Some(A::zero());

        Ok(())
    }

    /// Compute R² score
    pub fn score(&self, X: &Array2<A>, y: &Array1<A>) -> Result<A> {
        let predictions = self.predict(X)?;
        let y_mean = y.sum() / A::from(y.len()).unwrap();

        let ss_res: A = y.iter().zip(predictions.iter())
            .map(|(&true_y, &pred_y)| {
                let diff = true_y - pred_y;
                diff * diff
            })
            .sum();

        let ss_tot: A = y.iter()
            .map(|&yi| {
                let diff = yi - y_mean;
                diff * diff
            })
            .sum();

        if ss_tot == A::zero() {
            return Ok(A::one());
        }

        Ok(A::one() - ss_res / ss_tot)
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.coef.is_some()
    }

    /// Predict
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let coef = self.coef.as_ref().ok_or(RustyAIError::NotFitted)?;

        let mut predictions = Array1::zeros(X.nrows());
        for (i, row) in X.rows().into_iter().enumerate() {
            predictions[i] = row.iter().zip(coef.iter()).map(|(&x, &c)| x * c).sum();
        }

        Ok(predictions)
    }
}
