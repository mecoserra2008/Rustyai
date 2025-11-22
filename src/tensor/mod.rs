//! Unified tensor abstraction over ndarray and nalgebra
//!
//! This module provides a flexible tensor type that can work with both ndarray
//! and nalgebra backends, allowing for optimal performance in different scenarios.

use ndarray::{Array, Array1, Array2, ArrayD, Dimension, IxDyn};
use num_traits::{Float, NumAssign};
use serde::{Deserialize, Serialize};

pub mod ops;

/// A unified tensor type that wraps ndarray
///
/// This provides a consistent API while allowing backend flexibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<A, D>
where
    D: Dimension,
{
    data: Array<A, D>,
}

impl<A, D> Tensor<A, D>
where
    A: Clone,
    D: Dimension,
{
    /// Create a new tensor from an ndarray Array
    pub fn new(data: Array<A, D>) -> Self {
        Self { data }
    }

    /// Get a reference to the underlying array
    pub fn as_array(&self) -> &Array<A, D> {
        &self.data
    }

    /// Get a mutable reference to the underlying array
    pub fn as_array_mut(&mut self) -> &mut Array<A, D> {
        &mut self.data
    }

    /// Convert into the underlying array
    pub fn into_array(self) -> Array<A, D> {
        self.data
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<A> Tensor<A, IxDyn>
where
    A: Clone,
{
    /// Create a tensor from a flat vector with the given shape
    pub fn from_vec(vec: Vec<A>, shape: Vec<usize>) -> Result<Self, crate::error::RustyAIError> {
        let array = ArrayD::from_shape_vec(IxDyn(&shape), vec)
            .map_err(|e| crate::error::RustyAIError::InvalidShape {
                expected: format!("{:?}", shape),
                got: e.to_string(),
            })?;
        Ok(Self::new(array))
    }

    /// Reshape the tensor
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, crate::error::RustyAIError> {
        let reshaped = self
            .data
            .clone()
            .into_shape(IxDyn(&shape))
            .map_err(|e| crate::error::RustyAIError::InvalidShape {
                expected: format!("{:?}", shape),
                got: e.to_string(),
            })?;
        Ok(Self::new(reshaped))
    }
}

impl<A> Tensor<A, ndarray::Ix1>
where
    A: Clone + Float,
{
    /// Create a 1D tensor filled with zeros
    pub fn zeros(len: usize) -> Self {
        Self::new(Array1::from_elem(len, A::zero()))
    }

    /// Create a 1D tensor filled with ones
    pub fn ones(len: usize) -> Self {
        Self::new(Array1::from_elem(len, A::one()))
    }

    /// Dot product with another 1D tensor
    pub fn dot(&self, other: &Self) -> A
    where
        A: NumAssign + 'static,
    {
        self.data.dot(&other.data)
    }
}

impl<A> Tensor<A, ndarray::Ix2>
where
    A: Clone + Float,
{
    /// Create a 2D tensor filled with zeros
    pub fn zeros_2d(rows: usize, cols: usize) -> Self {
        Self::new(Array2::from_elem((rows, cols), A::zero()))
    }

    /// Create a 2D tensor filled with ones
    pub fn ones_2d(rows: usize, cols: usize) -> Self {
        Self::new(Array2::from_elem((rows, cols), A::one()))
    }

    /// Create an identity matrix
    pub fn eye(n: usize) -> Self {
        Self::new(Array2::eye(n))
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Self) -> Self
    where
        A: NumAssign + 'static,
    {
        Self::new(self.data.dot(&other.data))
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Self {
        Self::new(self.data.t().to_owned())
    }

    /// Get number of rows
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Get number of columns
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }
}

impl<A, D> From<Array<A, D>> for Tensor<A, D>
where
    A: Clone,
    D: Dimension,
{
    fn from(array: Array<A, D>) -> Self {
        Self::new(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(5);
        assert_eq!(t.len(), 5);
        assert_eq!(t.shape(), &[5]);
    }

    #[test]
    fn test_matrix_operations() {
        let a = Tensor::eye(3);
        let b = Tensor::ones_2d(3, 3);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[3, 3]);
    }
}
