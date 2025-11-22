//! Data preprocessing and feature engineering
//!
//! Comprehensive preprocessing tools including scaling, normalization,
//! encoding, and imputation.

use crate::error::{Result, RustyAIError};
use crate::traits::Transformer;
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::iter::Sum;

pub mod encoding;
pub mod imputation;

/// Standard scaler - standardizes features by removing mean and scaling to unit variance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler<A> {
    /// Mean of each feature
    mean: Option<Array1<A>>,
    /// Standard deviation of each feature
    std: Option<Array1<A>>,
    /// Whether to center the data
    with_mean: bool,
    /// Whether to scale the data
    with_std: bool,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> StandardScaler<A> {
    /// Create a new StandardScaler
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
            with_mean: true,
            with_std: true,
        }
    }

    /// Set whether to center the data
    pub fn with_mean(mut self, with_mean: bool) -> Self {
        self.with_mean = with_mean;
        self
    }

    /// Set whether to scale the data
    pub fn with_std(mut self, with_std: bool) -> Self {
        self.with_std = with_std;
        self
    }

    /// Fit the scaler to the data
    pub fn fit(&mut self, X: &Array2<A>) -> Result<()> {
        if X.is_empty() {
            return Err(RustyAIError::EmptyData);
        }

        let n_samples = A::from(X.nrows()).unwrap();

        if self.with_mean {
            let mean = X.mean_axis(Axis(0)).unwrap();
            self.mean = Some(mean);
        }

        if self.with_std {
            let mean = self.mean.as_ref().map(|m| m.clone()).unwrap_or_else(|| {
                X.mean_axis(Axis(0)).unwrap()
            });

            let mut std = Array1::zeros(X.ncols());
            for (i, col) in X.columns().into_iter().enumerate() {
                let variance: A = col
                    .iter()
                    .map(|&x| (x - mean[i]) * (x - mean[i]))
                    .sum::<A>()
                    / n_samples;
                std[i] = variance.sqrt();

                // Avoid division by zero
                if std[i] < A::from(1e-10).unwrap() {
                    std[i] = A::one();
                }
            }
            self.std = Some(std);
        }

        Ok(())
    }

    /// Transform the data
    pub fn transform(&self, X: &Array2<A>) -> Array2<A> {
        let mut X_transformed = X.clone();

        if self.with_mean {
            if let Some(ref mean) = self.mean {
                for (mut row, _) in X_transformed.rows_mut().into_iter().zip(0..) {
                    for (i, val) in row.iter_mut().enumerate() {
                        *val = *val - mean[i];
                    }
                }
            }
        }

        if self.with_std {
            if let Some(ref std) = self.std {
                for (mut row, _) in X_transformed.rows_mut().into_iter().zip(0..) {
                    for (i, val) in row.iter_mut().enumerate() {
                        *val = *val / std[i];
                    }
                }
            }
        }

        X_transformed
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, X: &Array2<A>) -> Result<Array2<A>> {
        self.fit(X)?;
        Ok(self.transform(X))
    }

    /// Inverse transform the data
    pub fn inverse_transform(&self, X: &Array2<A>) -> Array2<A> {
        let mut X_inv = X.clone();

        if self.with_std {
            if let Some(ref std) = self.std {
                for (mut row, _) in X_inv.rows_mut().into_iter().zip(0..) {
                    for (i, val) in row.iter_mut().enumerate() {
                        *val = *val * std[i];
                    }
                }
            }
        }

        if self.with_mean {
            if let Some(ref mean) = self.mean {
                for (mut row, _) in X_inv.rows_mut().into_iter().zip(0..) {
                    for (i, val) in row.iter_mut().enumerate() {
                        *val = *val + mean[i];
                    }
                }
            }
        }

        X_inv
    }
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> Default for StandardScaler<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Min-Max scaler - scales features to a given range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScaler<A> {
    /// Minimum value of each feature
    min: Option<Array1<A>>,
    /// Maximum value of each feature
    max: Option<Array1<A>>,
    /// Target minimum value
    feature_min: A,
    /// Target maximum value
    feature_max: A,
}

impl<A: Float + ScalarOperand + Sum> MinMaxScaler<A> {
    /// Create a new MinMaxScaler with range [0, 1]
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
            feature_min: A::zero(),
            feature_max: A::one(),
        }
    }

    /// Set the target range
    pub fn with_range(mut self, min: A, max: A) -> Self {
        self.feature_min = min;
        self.feature_max = max;
        self
    }

    /// Fit the scaler to the data
    pub fn fit(&mut self, X: &Array2<A>) -> Result<()> {
        if X.is_empty() {
            return Err(RustyAIError::EmptyData);
        }

        let mut min = Array1::from_elem(X.ncols(), A::infinity());
        let mut max = Array1::from_elem(X.ncols(), A::neg_infinity());

        for col_idx in 0..X.ncols() {
            for row in X.rows() {
                let val = row[col_idx];
                if val < min[col_idx] {
                    min[col_idx] = val;
                }
                if val > max[col_idx] {
                    max[col_idx] = val;
                }
            }
        }

        self.min = Some(min);
        self.max = Some(max);
        Ok(())
    }

    /// Transform the data
    pub fn transform(&self, X: &Array2<A>) -> Array2<A> {
        let mut X_transformed = X.clone();

        if let (Some(ref min), Some(ref max)) = (&self.min, &self.max) {
            let range = self.feature_max - self.feature_min;

            for (mut row, _) in X_transformed.rows_mut().into_iter().zip(0..) {
                for (i, val) in row.iter_mut().enumerate() {
                    let data_range = max[i] - min[i];
                    if data_range > A::from(1e-10).unwrap() {
                        *val = (*val - min[i]) / data_range * range + self.feature_min;
                    } else {
                        *val = self.feature_min;
                    }
                }
            }
        }

        X_transformed
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, X: &Array2<A>) -> Result<Array2<A>> {
        self.fit(X)?;
        Ok(self.transform(X))
    }
}

impl<A: Float + ScalarOperand + Sum> Default for MinMaxScaler<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalizer - normalizes samples individually to unit norm
#[derive(Debug, Clone)]
pub struct Normalizer<A> {
    /// Norm to use: 'l1', 'l2', or 'max'
    norm: NormType,
    _phantom: std::marker::PhantomData<A>,
}

#[derive(Debug, Clone, Copy)]
pub enum NormType {
    L1,
    L2,
    Max,
}

impl<A: Float + ScalarOperand + Sum> Normalizer<A> {
    /// Create a new Normalizer with L2 norm
    pub fn new() -> Self {
        Self {
            norm: NormType::L2,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the norm type
    pub fn with_norm(mut self, norm: NormType) -> Self {
        self.norm = norm;
        self
    }

    /// Transform the data
    pub fn transform(&self, X: &Array2<A>) -> Array2<A> {
        let mut X_normalized = X.clone();

        for (mut row, _) in X_normalized.rows_mut().into_iter().zip(0..) {
            let norm = match self.norm {
                NormType::L1 => row.iter().map(|&x| x.abs()).sum(),
                NormType::L2 => row.iter().map(|&x| x * x).sum::<A>().sqrt(),
                NormType::Max => row
                    .iter()
                    .map(|&x| x.abs())
                    .fold(A::zero(), |a, b| if a > b { a } else { b }),
            };

            if norm > A::from(1e-10).unwrap() {
                for val in row.iter_mut() {
                    *val = *val / norm;
                }
            }
        }

        X_normalized
    }
}

impl<A: Float + ScalarOperand + Sum> Default for Normalizer<A> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_standard_scaler() {
        let X = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut scaler = StandardScaler::new();
        scaler.fit(&X).unwrap();
        let X_scaled = scaler.transform(&X);

        // Check that mean is approximately 0
        let mean = X_scaled.mean_axis(Axis(0)).unwrap();
        assert_abs_diff_eq!(mean[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mean[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_minmax_scaler() {
        let X = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut scaler = MinMaxScaler::new();
        scaler.fit(&X).unwrap();
        let X_scaled = scaler.transform(&X);

        // Check that values are in [0, 1]
        for val in X_scaled.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }
}
