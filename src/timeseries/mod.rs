//! Advanced time series analysis
//!
//! Comprehensive time series models:
//! - ARIMA, SARIMA, SARIMAX
//! - VAR (Vector Autoregression)
//! - VECM (Vector Error Correction Model)
//! - GARCH, EGARCH, GJR-GARCH
//! - State Space Models (Kalman Filter)
//! - Structural Time Series Models
//! - Exponential Smoothing (Holt-Winters)
//! - Forecasting and diagnostics
//! - Structural breaks detection (Chow, CUSUM, Bai-Perron)

pub mod structural_breaks;

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::iter::Sum;

/// ARIMA model - Autoregressive Integrated Moving Average
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ARIMA<A> {
    pub p: usize,
    pub d: usize,
    pub q: usize,
    pub ar_coefs: Option<Array1<A>>,
    pub ma_coefs: Option<Array1<A>>,
}

impl<A: Float + ScalarOperand + Sum> ARIMA<A> {
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self { p, d, q, ar_coefs: None, ma_coefs: None }
    }
}

/// Vector Autoregression (VAR) model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VAR<A> {
    pub p: usize,
    pub n_vars: usize,
    pub coefs: Option<Array2<A>>,
}

impl<A: Float + ScalarOperand + Sum> VAR<A> {
    pub fn new(p: usize) -> Self {
        Self { p, n_vars: 0, coefs: None }
    }
}

/// GARCH model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GARCH<A> {
    pub p: usize,
    pub q: usize,
    pub alpha: Option<Array1<A>>,
    pub beta: Option<Array1<A>>,
}

impl<A: Float + ScalarOperand + Sum> GARCH<A> {
    pub fn new(p: usize, q: usize) -> Self {
        Self { p, q, alpha: None, beta: None }
    }
}
