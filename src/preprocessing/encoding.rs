//! Categorical encoding methods

use crate::error::{Result, RustyAIError};
use ndarray::{ScalarOperand, Array1, Array2, ScalarOperand};
use std::collections::HashMap;

/// One-hot encoder for categorical variables
#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    /// Categories for each feature
    categories: Option<Vec<Vec<usize>>>,
}

impl OneHotEncoder {
    /// Create a new OneHotEncoder
    pub fn new() -> Self {
        Self { categories: None }
    }

    /// Fit the encoder to the data
    pub fn fit(mut self, X: &Array2<usize>) -> Result<Self> {
        let mut categories = Vec::new();

        for col_idx in 0..X.ncols() {
            let mut unique = Vec::new();
            for row in X.rows() {
                let val = row[col_idx];
                if !unique.contains(&val) {
                    unique.push(val);
                }
            }
            unique.sort();
            categories.push(unique);
        }

        self.categories = Some(categories);
        Ok(self)
    }

    /// Transform the data
    pub fn transform(&self, X: &Array2<usize>) -> Result<Array2<f64>> {
        let categories = self
            .categories
            .as_ref()
            .ok_or(RustyAIError::NotFitted)?;

        let n_samples = X.nrows();
        let n_features: usize = categories.iter().map(|c| c.len()).sum();

        let mut X_encoded = Array2::zeros((n_samples, n_features));
        let mut col_offset = 0;

        for (feat_idx, cats) in categories.iter().enumerate() {
            for row_idx in 0..n_samples {
                let val = X[[row_idx, feat_idx]];
                if let Some(cat_idx) = cats.iter().position(|&c| c == val) {
                    X_encoded[[row_idx, col_offset + cat_idx]] = 1.0;
                }
            }
            col_offset += cats.len();
        }

        Ok(X_encoded)
    }

    /// Fit and transform in one step
    pub fn fit_transform(self, X: &Array2<usize>) -> Result<(Self, Array2<f64>)> {
        let fitted = self.fit(X)?;
        let transformed = fitted.transform(X)?;
        Ok((fitted, transformed))
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}
