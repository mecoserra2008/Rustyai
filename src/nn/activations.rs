//! Activation functions for neural networks

use ndarray::{Array2, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

/// Apply ReLU activation
pub fn relu<A: Float + ScalarOperand + Sum>(x: &Array2<A>) -> Array2<A> {
    x.mapv(|v| if v > A::zero() { v } else { A::zero() })
}

/// Apply Leaky ReLU activation
pub fn leaky_relu<A: Float + ScalarOperand + Sum>(x: &Array2<A>, alpha: A) -> Array2<A> {
    x.mapv(|v| if v > A::zero() { v } else { alpha * v })
}

/// Apply ELU activation
pub fn elu<A: Float + ScalarOperand + Sum>(x: &Array2<A>, alpha: A) -> Array2<A> {
    x.mapv(|v| if v > A::zero() { v } else { alpha * (v.exp() - A::one()) })
}

/// Apply GELU activation (Gaussian Error Linear Unit)
pub fn gelu<A: Float + ScalarOperand + Sum>(x: &Array2<A>) -> Array2<A> {
    let sqrt_2_pi = A::from(0.7978845608).unwrap(); // sqrt(2/pi)
    x.mapv(|v| {
        A::from(0.5).unwrap() * v * (A::one() + (sqrt_2_pi * (v + A::from(0.044715).unwrap() * v.powi(3))).tanh())
    })
}

/// Apply Swish activation
pub fn swish<A: Float + ScalarOperand + Sum>(x: &Array2<A>) -> Array2<A> {
    x.mapv(|v| v / (A::one() + (-v).exp()))
}

/// Apply sigmoid activation
pub fn sigmoid<A: Float + ScalarOperand + Sum>(x: &Array2<A>) -> Array2<A> {
    x.mapv(|v| A::one() / (A::one() + (-v).exp()))
}

/// Apply tanh activation
pub fn tanh<A: Float + ScalarOperand + Sum>(x: &Array2<A>) -> Array2<A> {
    x.mapv(|v| v.tanh())
}

/// Apply softmax activation
pub fn softmax<A: Float + ScalarOperand + Sum>(x: &Array2<A>) -> Array2<A> {
    let mut result = Array2::zeros(x.dim());

    for (i, row) in x.rows().into_iter().enumerate() {
        let max_val = row.iter().fold(A::neg_infinity(), |a, &b| if b > a { b } else { a });
        let exp_row: Vec<A> = row.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: A = exp_row.iter().copied().sum();

        for (j, &val) in exp_row.iter().enumerate() {
            result[[i, j]] = val / sum;
        }
    }

    result
}

/// Derivative of ReLU
pub fn relu_derivative<A: Float + ScalarOperand + Sum>(x: &Array2<A>) -> Array2<A> {
    x.mapv(|v| if v > A::zero() { A::one() } else { A::zero() })
}

/// Derivative of sigmoid
pub fn sigmoid_derivative<A: Float + ScalarOperand + Sum>(x: &Array2<A>) -> Array2<A> {
    let sig = sigmoid(x);
    &sig * &(Array2::ones(x.dim()) - &sig)
}

/// Derivative of tanh
pub fn tanh_derivative<A: Float + ScalarOperand + Sum>(x: &Array2<A>) -> Array2<A> {
    x.mapv(|v| {
        let t = v.tanh();
        A::one() - t * t
    })
}
