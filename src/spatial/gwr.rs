//! Geographically Weighted Regression

use crate::error::Result;
use ndarray::{Array1, Array2};
use num_traits::Float;

/// Geographically Weighted Regression
/// Allows parameters to vary spatially
pub struct GWR<A> {
    /// Local coefficients for each location
    pub local_coefs: Option<Array2<A>>,
    /// Bandwidth
    pub bandwidth: Option<A>,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float> GWR<A> {
    pub fn new() -> Self {
        Self {
            local_coefs: None,
            bandwidth: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<A: Float> Default for GWR<A> {
    fn default() -> Self {
        Self::new()
    }
}
