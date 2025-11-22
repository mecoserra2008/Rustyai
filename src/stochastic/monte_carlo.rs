//! Monte Carlo simulation methods

use ndarray::Array1;
use ndarray::ScalarOperand;
use num_traits::Float;
use std::iter::Sum;

/// Variance reduction techniques
pub enum VarianceReduction {
    AntitheticVariates,
    ControlVariates,
    ImportanceSampling,
}
