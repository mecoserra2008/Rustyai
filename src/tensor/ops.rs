//! Tensor operations

use super::Tensor;
use ndarray::{Dimension, ScalarOperand};
use num_traits::{Float, NumAssign};
use std::ops::{Add, Div, Mul, Sub};

// Implement arithmetic operations for Tensor

impl<A, D> Add for Tensor<A, D>
where
    A: Clone + NumAssign,
    D: Dimension,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(&self.data + &rhs.data)
    }
}

impl<A, D> Sub for Tensor<A, D>
where
    A: Clone + NumAssign,
    D: Dimension,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(&self.data - &rhs.data)
    }
}

impl<A, D> Mul for Tensor<A, D>
where
    A: Clone + NumAssign,
    D: Dimension,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(&self.data * &rhs.data)
    }
}

impl<A, D> Div for Tensor<A, D>
where
    A: Clone + NumAssign,
    D: Dimension,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::new(&self.data / &rhs.data)
    }
}

// Scalar operations
impl<A, D> Mul<A> for Tensor<A, D>
where
    A: Clone + NumAssign + ScalarOperand,
    D: Dimension,
{
    type Output = Self;

    fn mul(self, scalar: A) -> Self::Output {
        Self::new(&self.data * scalar)
    }
}

impl<A, D> Div<A> for Tensor<A, D>
where
    A: Clone + NumAssign + ScalarOperand,
    D: Dimension,
{
    type Output = Self;

    fn div(self, scalar: A) -> Self::Output {
        Self::new(&self.data / scalar)
    }
}
