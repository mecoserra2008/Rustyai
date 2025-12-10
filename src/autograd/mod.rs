//! Automatic differentiation engine
//!
//! Provides automatic differentiation for building neural networks.
//! This is a simplified implementation for demonstration purposes.

use crate::error::Result;
use ndarray::{Array, ArrayD, IxDyn};
use std::sync::Arc;

/// Operation type for backpropagation
#[derive(Debug, Clone)]
pub enum Op {
    /// Leaf node (no operation)
    Leaf,
    /// Addition
    Add,
    /// Multiplication
    Mul,
    /// Matrix multiplication
    MatMul,
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Sum reduction
    Sum,
}

/// Tensor with gradient tracking
///
/// A simplified automatic differentiation tensor that tracks
/// computational graphs for backpropagation.
pub struct Tensor {
    /// The data stored in this tensor
    data: ArrayD<f64>,
    /// The gradient with respect to this tensor
    grad: Option<ArrayD<f64>>,
    /// Whether this tensor requires gradient computation
    requires_grad: bool,
    /// Operation that created this tensor
    op: Op,
    /// Parent tensors in computation graph
    parents: Vec<Arc<Tensor>>,
}

impl Tensor {
    /// Create a new tensor from data
    pub fn new(data: ArrayD<f64>, requires_grad: bool) -> Self {
        Self {
            data,
            grad: None,
            requires_grad,
            op: Op::Leaf,
            parents: Vec::new(),
        }
    }

    /// Create a tensor from a 1D array
    pub fn from_vec(data: Vec<f64>, requires_grad: bool) -> Self {
        let shape = IxDyn(&[data.len()]);
        let arr = Array::from_shape_vec(shape, data).unwrap();
        Self::new(arr, requires_grad)
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        let arr = ArrayD::zeros(IxDyn(shape));
        Self::new(arr, requires_grad)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        let arr = ArrayD::ones(IxDyn(shape));
        Self::new(arr, requires_grad)
    }

    /// Get the data
    pub fn data(&self) -> &ArrayD<f64> {
        &self.data
    }

    /// Get the gradient
    pub fn grad(&self) -> Option<&ArrayD<f64>> {
        self.grad.as_ref()
    }

    /// Get mutable gradient
    pub fn grad_mut(&mut self) -> &mut Option<ArrayD<f64>> {
        &mut self.grad
    }

    /// Zero the gradient
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            self.grad = Some(ArrayD::zeros(self.data.raw_dim()));
        }
    }

    /// Add two tensors
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let result_data = &self.data + &other.data;
        let mut result = Tensor::new(result_data, self.requires_grad || other.requires_grad);
        result.op = Op::Add;
        // Note: In a full implementation, we'd store Arc<Tensor> parents
        Ok(result)
    }

    /// Multiply two tensors element-wise
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let result_data = &self.data * &other.data;
        let mut result = Tensor::new(result_data, self.requires_grad || other.requires_grad);
        result.op = Op::Mul;
        Ok(result)
    }

    /// Apply ReLU activation
    pub fn relu(&self) -> Result<Tensor> {
        let result_data = self.data.mapv(|x| x.max(0.0));
        let mut result = Tensor::new(result_data, self.requires_grad);
        result.op = Op::ReLU;
        Ok(result)
    }

    /// Apply sigmoid activation
    pub fn sigmoid(&self) -> Result<Tensor> {
        let result_data = self.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let mut result = Tensor::new(result_data, self.requires_grad);
        result.op = Op::Sigmoid;
        Ok(result)
    }

    /// Sum all elements
    pub fn sum(&self) -> Result<Tensor> {
        let sum_val = self.data.iter().sum::<f64>();
        let result_data = ArrayD::from_elem(IxDyn(&[]), sum_val);
        let mut result = Tensor::new(result_data, self.requires_grad);
        result.op = Op::Sum;
        Ok(result)
    }

    /// Backward pass (simplified - does not fully implement backprop)
    ///
    /// This is a placeholder for a full autograd implementation.
    /// A complete implementation would:
    /// 1. Build a computation graph
    /// 2. Traverse in reverse topological order
    /// 3. Apply chain rule to compute gradients
    pub fn backward(&mut self) {
        if !self.requires_grad {
            return;
        }

        // Initialize gradient to 1 for scalar output
        if self.grad.is_none() {
            self.grad = Some(ArrayD::ones(self.data.raw_dim()));
        }

        // In a full implementation, we would:
        // - Traverse the computation graph in reverse
        // - Accumulate gradients using the chain rule
        // - Update gradients of all parent tensors
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Check if this tensor requires gradient
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}
