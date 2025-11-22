//! Stochastic Differential Equations solvers

use crate::error::Result;
use ndarray::{Array1, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

/// SDE Solver using Euler-Maruyama method
pub struct EulerMaruyama<A> {
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float + ScalarOperand + Sum> EulerMaruyama<A> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}
