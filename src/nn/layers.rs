//! Neural network layers

use super::Layer;
use ndarray::{Array1, Array2, ScalarOperand, Axis};
use num_traits::{Float, FromPrimitive};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::iter::Sum;

/// Dense (fully connected) layer with automatic backpropagation
pub struct Dense<A: Float> {
    /// Weight matrix (input_dim x output_dim)
    pub weights: Array2<A>,
    /// Bias vector
    pub bias: Array1<A>,
    /// Cached input for backprop
    cached_input: Option<Array2<A>>,
    /// Weight gradients
    weight_grad: Array2<A>,
    /// Bias gradients
    bias_grad: Array1<A>,
}

impl<A: Float + ScalarOperand + Sum> Dense<A> {
    /// Create a new dense layer with Xavier initialization
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = thread_rng();
        let std = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights: Array2<A> = Array2::from_shape_fn((input_dim, output_dim), |_| {
            A::from(normal.sample(&mut rng)).unwrap()
        });

        let bias = Array1::zeros(output_dim);

        Self {
            weights,
            bias,
            cached_input: None,
            weight_grad: Array2::zeros((input_dim, output_dim)),
            bias_grad: Array1::zeros(output_dim),
        }
    }

    /// Create with He initialization (better for ReLU)
    pub fn new_he(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = thread_rng();
        let std = (2.0 / input_dim as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights: Array2<A> = Array2::from_shape_fn((input_dim, output_dim), |_| {
            A::from(normal.sample(&mut rng)).unwrap()
        });

        let bias = Array1::zeros(output_dim);

        Self {
            weights,
            bias,
            cached_input: None,
            weight_grad: Array2::zeros((input_dim, output_dim)),
            bias_grad: Array1::zeros(output_dim),
        }
    }
}

impl<A: Float + ScalarOperand + Sum + 'static> Layer<A> for Dense<A> {
    fn forward(&mut self, input: &Array2<A>) -> Array2<A> {
        // Cache input for backward pass
        self.cached_input = Some(input.clone());

        // output = input @ weights + bias
        let mut output = input.dot(&self.weights);
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                output[[i, j]] = output[[i, j]] + self.bias[j];
            }
        }

        output
    }

    fn backward(&mut self, grad_output: &Array2<A>) -> Array2<A> {
        let input = self.cached_input.as_ref().unwrap();

        // Compute weight gradient: input.T @ grad_output
        self.weight_grad = input.t().dot(grad_output);

        // Compute bias gradient: sum(grad_output, axis=0)
        self.bias_grad = grad_output.sum_axis(Axis(0));

        // Compute input gradient: grad_output @ weights.T
        grad_output.dot(&self.weights.t())
    }

    fn parameters(&self) -> Vec<&Array2<A>> {
        vec![&self.weights]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<A>> {
        vec![&mut self.weights]
    }

    fn gradients(&self) -> Vec<&Array2<A>> {
        vec![&self.weight_grad]
    }
}

/// Batch Normalization layer
pub struct BatchNorm<A: Float> {
    /// Number of features
    num_features: usize,
    /// Learnable scale parameter
    gamma: Array1<A>,
    /// Learnable shift parameter
    beta: Array1<A>,
    /// Running mean
    running_mean: Array1<A>,
    /// Running variance
    running_var: Array1<A>,
    /// Momentum for running stats
    momentum: A,
    /// Small constant for numerical stability
    eps: A,
    /// Cached values for backprop
    cached_input: Option<Array2<A>>,
    cached_normalized: Option<Array2<A>>,
    cached_std: Option<Array1<A>>,
    /// Gradients
    gamma_grad: Array1<A>,
    beta_grad: Array1<A>,
    /// Training mode
    training: bool,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> BatchNorm<A> {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            momentum: A::from(0.9).unwrap(),
            eps: A::from(1e-5).unwrap(),
            cached_input: None,
            cached_normalized: None,
            cached_std: None,
            gamma_grad: Array1::zeros(num_features),
            beta_grad: Array1::zeros(num_features),
            training: true,
        }
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn train(&mut self) {
        self.training = true;
    }
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive + 'static> Layer<A> for BatchNorm<A> {
    fn forward(&mut self, input: &Array2<A>) -> Array2<A> {
        let batch_size = A::from(input.nrows()).unwrap();

        if self.training {
            // Compute batch statistics
            let mean = input.mean_axis(Axis(0)).unwrap();
            let var = input.var_axis(Axis(0), A::zero());

            // Update running statistics
            self.running_mean = &self.running_mean * self.momentum + &(&mean * (A::one() - self.momentum));
            self.running_var = &self.running_var * self.momentum + &(&var * (A::one() - self.momentum));

            // Normalize
            let std = var.mapv(|v| (v + self.eps).sqrt());
            let mut normalized = Array2::zeros(input.dim());

            for (i, mut row) in normalized.rows_mut().into_iter().enumerate() {
                let input_row = input.row(i);
                row.assign(&((&input_row - &mean) / &std));
            }

            // Cache for backward
            self.cached_input = Some(input.clone());
            self.cached_normalized = Some(normalized.clone());
            self.cached_std = Some(std);

            // Scale and shift
            let mut output = normalized;
            for mut row in output.rows_mut() {
                row.assign(&(&row * &self.gamma + &self.beta));
            }

            output
        } else {
            // Use running statistics
            let std = self.running_var.mapv(|v| (v + self.eps).sqrt());
            let mut normalized = Array2::zeros(input.dim());

            for (i, mut row) in normalized.rows_mut().into_iter().enumerate() {
                let input_row = input.row(i);
                row.assign(&((&input_row - &self.running_mean) / &std));
            }

            // Scale and shift
            for mut row in normalized.rows_mut() {
                row.assign(&(&row * &self.gamma + &self.beta));
            }

            normalized
        }
    }

    fn backward(&mut self, grad_output: &Array2<A>) -> Array2<A> {
        let input = self.cached_input.as_ref().unwrap();
        let normalized = self.cached_normalized.as_ref().unwrap();
        let std = self.cached_std.as_ref().unwrap();

        let m = A::from(input.nrows()).unwrap();

        // Compute gamma gradient
        self.gamma_grad = (grad_output * normalized).sum_axis(Axis(0));

        // Compute beta gradient
        self.beta_grad = grad_output.sum_axis(Axis(0));

        // Compute input gradient (simplified)
        let mut grad_input = Array2::zeros(input.dim());
        for (i, mut row) in grad_input.rows_mut().into_iter().enumerate() {
            let grad_row = grad_output.row(i);
            row.assign(&(&grad_row * &self.gamma / std));
        }

        grad_input
    }

    fn parameters(&self) -> Vec<&Array2<A>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<A>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&Array2<A>> {
        vec![]
    }
}

/// Dropout layer for regularization
pub struct Dropout<A: Float> {
    /// Dropout rate (probability of dropping)
    rate: A,
    /// Dropout mask
    mask: Option<Array2<A>>,
    /// Training mode
    training: bool,
}

impl<A: Float + ScalarOperand + Sum> Dropout<A> {
    pub fn new(rate: f64) -> Self {
        Self {
            rate: A::from(rate).unwrap(),
            mask: None,
            training: true,
        }
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn train(&mut self) {
        self.training = true;
    }
}

impl<A: Float + ScalarOperand + Sum + 'static> Layer<A> for Dropout<A> {
    fn forward(&mut self, input: &Array2<A>) -> Array2<A> {
        if self.training {
            // Generate dropout mask
            let mut rng = thread_rng();
            let mask: Array2<A> = Array2::from_shape_fn(input.dim(), |_| {
                if rng.gen::<f64>() > self.rate.to_f64().unwrap() {
                    A::one() / (A::one() - self.rate)
                } else {
                    A::zero()
                }
            });

            self.mask = Some(mask.clone());
            input * &mask
        } else {
            input.clone()
        }
    }

    fn backward(&mut self, grad_output: &Array2<A>) -> Array2<A> {
        if let Some(ref mask) = self.mask {
            grad_output * mask
        } else {
            grad_output.clone()
        }
    }

    fn parameters(&self) -> Vec<&Array2<A>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<A>> {
        vec![]
    }

    fn gradients(&self) -> Vec<&Array2<A>> {
        vec![]
    }
}

use rand::Rng;
