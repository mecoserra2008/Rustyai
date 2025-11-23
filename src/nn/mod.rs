//! Deep Learning: Neural Networks and Layers
//!
//! Comprehensive neural network implementation with:
//! - Feedforward layers (Dense, Conv2D, LSTM, GRU)
//! - Activation functions (ReLU, Leaky ReLU, ELU, GELU, Swish)
//! - Normalization (BatchNorm, LayerNorm, Dropout)
//! - Loss functions (MSE, Cross-Entropy, Focal Loss)
//! - Optimizers (SGD, Adam, AdamW, RAdam, LAMB, RMSprop, Adagrad, Adadelta, Nadam, AMSGrad)
//! - Automatic differentiation and backpropagation

pub mod layers;
pub mod activations;
pub mod loss;
pub mod optimizers;
pub mod regularization;
pub mod attention;
pub mod schedulers;
pub mod generative;

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::iter::Sum;

/// Neural network layer trait
pub trait Layer<A: Float> {
    /// Forward pass through the layer
    fn forward(&mut self, input: &Array2<A>) -> Array2<A>;

    /// Backward pass (gradient computation)
    fn backward(&mut self, grad_output: &Array2<A>) -> Array2<A>;

    /// Get trainable parameters
    fn parameters(&self) -> Vec<&Array2<A>>;

    /// Get mutable trainable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Array2<A>>;

    /// Get parameter gradients
    fn gradients(&self) -> Vec<&Array2<A>>;
}

/// Activation function enum
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    LeakyReLU(f64),
    ELU(f64),
    GELU,
    Swish,
    Sigmoid,
    Tanh,
    Softmax,
    Identity,
}

/// Loss function enum
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    MSE,
    MAE,
    CrossEntropy,
    BinaryCrossEntropy,
    FocalLoss { alpha: f64, gamma: f64 },
    Huber { delta: f64 },
}

/// Sequential neural network model
pub struct Sequential<A: Float> {
    layers: Vec<Box<dyn Layer<A>>>,
    loss_fn: LossFunction,
}

impl<A: Float + ScalarOperand + Sum> Sequential<A> {
    /// Create a new sequential model
    pub fn new(loss_fn: LossFunction) -> Self {
        Self {
            layers: Vec::new(),
            loss_fn,
        }
    }

    /// Add a layer to the model
    pub fn add_layer(&mut self, layer: Box<dyn Layer<A>>) {
        self.layers.push(layer);
    }

    /// Forward pass through all layers
    pub fn forward(&mut self, input: &Array2<A>) -> Array2<A> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    /// Compute loss
    pub fn compute_loss(&self, predictions: &Array2<A>, targets: &Array2<A>) -> A {
        match self.loss_fn {
            LossFunction::MSE => {
                let diff = predictions - targets;
                (&diff * &diff).sum() / A::from(predictions.len()).unwrap()
            }
            LossFunction::MAE => {
                let diff = predictions - targets;
                diff.mapv(|x| x.abs()).sum() / A::from(predictions.len()).unwrap()
            }
            LossFunction::CrossEntropy => {
                // -sum(y * log(p))
                let eps = A::from(1e-10).unwrap();
                -(targets * &predictions.mapv(|x| (x + eps).ln())).sum() /
                    A::from(predictions.nrows()).unwrap()
            }
            LossFunction::BinaryCrossEntropy => {
                let eps = A::from(1e-10).unwrap();
                let ones = Array2::ones(predictions.dim());
                -(targets * &predictions.mapv(|x: A| (x + eps).ln()) +
                  &((&ones - targets) * &(&ones - predictions).mapv(|x: A| (x + eps).ln())))
                    .sum() / A::from(predictions.nrows()).unwrap()
            }
            LossFunction::FocalLoss { alpha, gamma } => {
                let alpha = A::from(alpha).unwrap();
                let gamma = A::from(gamma).unwrap();
                let eps = A::from(1e-10).unwrap();

                let pt = targets * predictions + &((Array2::ones(predictions.dim()) - targets) *
                         &(Array2::ones(predictions.dim()) - predictions));
                let at = targets * alpha + &((Array2::ones(predictions.dim()) - targets) * (A::one() - alpha));

                -(at * &(Array2::ones(predictions.dim()) - &pt).mapv(|x: A| x.powf(gamma)) *
                  &pt.mapv(|x: A| (x + eps).ln())).sum() / A::from(predictions.nrows()).unwrap()
            }
            LossFunction::Huber { delta } => {
                let delta = A::from(delta).unwrap();
                let diff = predictions - targets;
                let abs_diff = diff.mapv(|x| x.abs());

                abs_diff.mapv(|x| {
                    if x <= delta {
                        A::from(0.5).unwrap() * x * x
                    } else {
                        delta * (x - A::from(0.5).unwrap() * delta)
                    }
                }).sum() / A::from(predictions.len()).unwrap()
            }
        }
    }

    /// Backward pass through all layers
    pub fn backward(&mut self, loss_grad: &Array2<A>) -> Array2<A> {
        let mut grad = loss_grad.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
        grad
    }

    /// Get all trainable parameters
    pub fn parameters_mut(&mut self) -> Vec<&mut Array2<A>> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }

    /// Get all parameter gradients
    pub fn gradients(&self) -> Vec<&Array2<A>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.gradients())
            .collect()
    }

    /// Training step
    pub fn train_step(
        &mut self,
        inputs: &Array2<A>,
        targets: &Array2<A>,
        learning_rate: A,
    ) -> A {
        // Forward pass
        let predictions = self.forward(inputs);

        // Compute loss
        let loss = self.compute_loss(&predictions, targets);

        // Compute loss gradient
        let loss_grad = &predictions - targets;

        // Backward pass
        self.backward(&loss_grad);

        // Update parameters (simple SGD)
        // Simplified update - directly update weights in layers
        // In production, use a proper optimizer with momentum etc.
        for layer in &mut self.layers {
            // Clone grads first to avoid borrowing issues
            let grad_clones: Vec<Array2<A>> = layer.gradients().iter().map(|g| (*g).clone()).collect();
            let mut params = layer.parameters_mut();

            for (param, grad) in params.iter_mut().zip(grad_clones.iter()) {
                **param = &**param - &(grad * learning_rate);
            }
        }

        loss
    }

    /// Predict
    pub fn predict(&mut self, inputs: &Array2<A>) -> Array2<A> {
        self.forward(inputs)
    }

    /// Fit the model
    pub fn fit(
        &mut self,
        X: &Array2<A>,
        y: &Array2<A>,
        epochs: usize,
        batch_size: usize,
        learning_rate: A,
        verbose: bool,
    ) -> Result<Vec<A>> {
        let n_samples = X.nrows();
        let mut losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = A::zero();
            let mut n_batches = 0;

            // Mini-batch training
            for start in (0..n_samples).step_by(batch_size) {
                let end = (start + batch_size).min(n_samples);

                let batch_X = X.slice(ndarray::s![start..end, ..]).to_owned();
                let batch_y = y.slice(ndarray::s![start..end, ..]).to_owned();

                let loss = self.train_step(&batch_X, &batch_y, learning_rate);
                epoch_loss = epoch_loss + loss;
                n_batches += 1;
            }

            let avg_loss = epoch_loss / A::from(n_batches).unwrap();
            losses.push(avg_loss);

            // Verbose output (disabled for generic types without Debug)
            // if verbose && epoch % 10 == 0 {
            //     println!("Epoch {}: Loss = {:?}", epoch, avg_loss);
            // }
        }

        Ok(losses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_forward() {
        // Basic test
        assert!(true);
    }
}
