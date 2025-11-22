//! Optimizers for neural network training

use ndarray::{Array2, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

/// SGD optimizer
pub struct SGD<A: Float + ScalarOperand> {
    learning_rate: A,
}

impl<A: Float + ScalarOperand> SGD<A> {
    pub fn new(learning_rate: A) -> Self {
        Self { learning_rate }
    }

    pub fn step(&self, params: &mut Vec<&mut Array2<A>>, grads: &[&Array2<A>]) {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            **param = &**param - &(*grad * self.learning_rate);
        }
    }
}

/// Adam optimizer
pub struct Adam<A: Float> {
    learning_rate: A,
    beta1: A,
    beta2: A,
    epsilon: A,
    t: usize,
    m: Vec<Array2<A>>,
    v: Vec<Array2<A>>,
}

impl<A: Float + ScalarOperand + Sum> Adam<A> {
    pub fn new(learning_rate: A, n_params: usize) -> Self {
        Self {
            learning_rate,
            beta1: A::from(0.9).unwrap(),
            beta2: A::from(0.999).unwrap(),
            epsilon: A::from(1e-8).unwrap(),
            t: 0,
            m: vec![],
            v: vec![],
        }
    }
}
