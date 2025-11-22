//! Loss functions for neural networks

use ndarray::{Array2, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

/// Mean Squared Error Loss
pub fn mse<A: Float + ScalarOperand + Sum>(predictions: &Array2<A>, targets: &Array2<A>) -> A {
    let diff = predictions - targets;
    (&diff * &diff).sum() / A::from(predictions.len()).unwrap()
}

/// Mean Absolute Error Loss
pub fn mae<A: Float + ScalarOperand + Sum>(predictions: &Array2<A>, targets: &Array2<A>) -> A {
    let diff = predictions - targets;
    diff.mapv(|x| x.abs()).sum() / A::from(predictions.len()).unwrap()
}

/// Cross Entropy Loss
pub fn cross_entropy<A: Float + ScalarOperand + Sum>(predictions: &Array2<A>, targets: &Array2<A>) -> A {
    let eps = A::from(1e-10).unwrap();
    -(targets * &predictions.mapv(|x| (x + eps).ln())).sum() / A::from(predictions.nrows()).unwrap()
}

/// Binary Cross Entropy Loss
pub fn binary_cross_entropy<A: Float + ScalarOperand + Sum>(predictions: &Array2<A>, targets: &Array2<A>) -> A {
    let eps = A::from(1e-10).unwrap();
    let ones = Array2::ones(predictions.dim());
    -(targets * &predictions.mapv(|x: A| (x + eps).ln()) +
      &((&ones - targets) * &(&ones - predictions).mapv(|x: A| (x + eps).ln())))
        .sum() / A::from(predictions.nrows()).unwrap()
}
