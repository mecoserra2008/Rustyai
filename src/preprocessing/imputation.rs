//! Missing value imputation strategies

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

/// Strategy for imputing missing values
#[derive(Debug, Clone, Copy)]
pub enum ImputationStrategy {
    Mean,
    Median,
    Mode,
    Constant,
}

/// Simple imputer for missing values
#[derive(Debug, Clone)]
pub struct SimpleImputer<A> {
    /// Imputation strategy
    strategy: ImputationStrategy,
    /// Fill value for constant strategy
    fill_value: Option<A>,
    /// Statistics for each feature
    statistics: Option<Array1<A>>,
}

impl<A: Float + ScalarOperand + Sum> SimpleImputer<A> {
    /// Create a new SimpleImputer with mean strategy
    pub fn new() -> Self {
        Self {
            strategy: ImputationStrategy::Mean,
            fill_value: None,
            statistics: None,
        }
    }

    /// Set the imputation strategy
    pub fn with_strategy(mut self, strategy: ImputationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the fill value for constant strategy
    pub fn with_fill_value(mut self, fill_value: A) -> Self {
        self.fill_value = Some(fill_value);
        self
    }

    /// Fit the imputer to the data (NaN values are considered missing)
    pub fn fit(mut self, X: &Array2<A>) -> Result<Self> {
        let mut statistics = Array1::zeros(X.ncols());

        for (col_idx, mut col) in X.columns().into_iter().enumerate() {
            let valid_values: Vec<A> = col.iter().filter(|&&x| !x.is_nan()).copied().collect();

            if valid_values.is_empty() {
                statistics[col_idx] = A::zero();
                continue;
            }

            statistics[col_idx] = match self.strategy {
                ImputationStrategy::Mean => {
                    valid_values.iter().sum::<A>() / A::from(valid_values.len()).unwrap()
                }
                ImputationStrategy::Median => {
                    let mut sorted = valid_values.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = sorted.len() / 2;
                    if sorted.len() % 2 == 0 {
                        (sorted[mid - 1] + sorted[mid]) / A::from(2.0).unwrap()
                    } else {
                        sorted[mid]
                    }
                }
                ImputationStrategy::Mode => {
                    // For simplicity, use mean for numeric data
                    valid_values.iter().sum::<A>() / A::from(valid_values.len()).unwrap()
                }
                ImputationStrategy::Constant => self.fill_value.unwrap_or(A::zero()),
            };
        }

        self.statistics = Some(statistics);
        Ok(self)
    }

    /// Transform the data by imputing missing values
    pub fn transform(&self, X: &Array2<A>) -> Result<Array2<A>> {
        let statistics = self
            .statistics
            .as_ref()
            .ok_or(RustyAIError::NotFitted)?;

        let mut X_imputed = X.clone();

        for (col_idx, mut col) in X_imputed.columns_mut().into_iter().enumerate() {
            for val in col.iter_mut() {
                if val.is_nan() {
                    *val = statistics[col_idx];
                }
            }
        }

        Ok(X_imputed)
    }
}

impl<A: Float + ScalarOperand + Sum> Default for SimpleImputer<A> {
    fn default() -> Self {
        Self::new()
    }
}
