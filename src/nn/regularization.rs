//! Regularization techniques

use ndarray::{Array2, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

/// L1 regularization (Lasso)
pub fn l1_penalty<A: Float + ScalarOperand + Sum>(weights: &Array2<A>, lambda: A) -> A {
    lambda * weights.mapv(|w| w.abs()).sum()
}

/// L2 regularization (Ridge)
pub fn l2_penalty<A: Float + ScalarOperand + Sum>(weights: &Array2<A>, lambda: A) -> A {
    lambda * weights.mapv(|w| w * w).sum() / A::from(2.0).unwrap()
}

/// Elastic Net regularization (L1 + L2)
pub fn elastic_net_penalty<A: Float + ScalarOperand + Sum>(
    weights: &Array2<A>,
    l1_ratio: A,
    lambda: A,
) -> A {
    l1_ratio * l1_penalty(weights, lambda) + (A::one() - l1_ratio) * l2_penalty(weights, lambda)
}
