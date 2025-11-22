//! Monte Carlo simulation methods

use ndarray::Array1;
use num_traits::Float;

/// Variance reduction techniques
pub enum VarianceReduction {
    AntitheticVariates,
    ControlVariates,
    ImportanceSampling,
}
