//! Stochastic Differential Equations solvers

use crate::error::Result;
use ndarray::Array1;
use num_traits::Float;

/// SDE Solver using Euler-Maruyama method
pub struct EulerMaruyama<A> {
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float> EulerMaruyama<A> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}
