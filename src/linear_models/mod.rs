//! Linear models for regression and classification
//!
//! Includes linear regression, logistic regression, Ridge, Lasso, ElasticNet,
//! and generalized linear models.

use crate::error::{Result, RustyAIError};
use crate::linalg::solve::lstsq;
use crate::traits::{Predictor, Regressor};
use ndarray::{Array1, Array2};
use num_traits::Float;
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

impl<A: Float> LinearRegression<A> {
    /// Create a new LinearRegression model
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            coef: None,
            intercept: None,
        }
    }
}

impl<A: Float> Default for LinearRegression<A> {
    fn default() -> Self {
        Self::new()
    }
}
