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

/// Conv2D (Convolutional 2D) layer for image processing
pub struct Conv2D<A: Float> {
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels (filters)
    out_channels: usize,
    /// Kernel size (height, width)
    kernel_size: (usize, usize),
    /// Stride
    stride: (usize, usize),
    /// Padding
    padding: (usize, usize),
    /// Weights (out_channels, in_channels, kernel_height, kernel_width)
    weights: Vec<Array2<A>>,
    /// Bias (one per output channel)
    bias: Array1<A>,
    /// Cached input for backprop
    cached_input: Option<Array2<A>>,
    /// Weight gradients
    weight_grads: Vec<Array2<A>>,
    /// Bias gradients
    bias_grad: Array1<A>,
}

impl<A: Float + ScalarOperand + Sum> Conv2D<A> {
    /// Create new Conv2D layer with He initialization
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
    ) -> Self {
        let mut rng = thread_rng();
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));

        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let std = (2.0 / fan_in as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let mut weights = Vec::new();
        let mut weight_grads = Vec::new();

        for _ in 0..out_channels {
            for _ in 0..in_channels {
                let kernel: Array2<A> = Array2::from_shape_fn(kernel_size, |_| {
                    A::from(normal.sample(&mut rng)).unwrap()
                });
                weights.push(kernel.clone());
                weight_grads.push(Array2::zeros(kernel_size));
            }
        }

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias: Array1::zeros(out_channels),
            cached_input: None,
            weight_grads,
            bias_grad: Array1::zeros(out_channels),
        }
    }

    /// Compute output dimensions
    pub fn output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let out_h = (input_height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_w = (input_width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        (out_h, out_w)
    }
}

/// LSTM (Long Short-Term Memory) layer for sequence modeling
pub struct LSTM<A: Float> {
    /// Input size
    input_size: usize,
    /// Hidden size
    hidden_size: usize,
    /// Input gate weights
    W_i: Array2<A>,
    U_i: Array2<A>,
    b_i: Array1<A>,
    /// Forget gate weights
    W_f: Array2<A>,
    U_f: Array2<A>,
    b_f: Array1<A>,
    /// Output gate weights
    W_o: Array2<A>,
    U_o: Array2<A>,
    b_o: Array1<A>,
    /// Cell gate weights
    W_c: Array2<A>,
    U_c: Array2<A>,
    b_c: Array1<A>,
    /// Cached states for backprop
    cached_states: Option<Vec<LSTMState<A>>>,
    /// Weight gradients
    grad_W_i: Array2<A>,
    grad_U_i: Array2<A>,
    grad_b_i: Array1<A>,
    grad_W_f: Array2<A>,
    grad_U_f: Array2<A>,
    grad_b_f: Array1<A>,
    grad_W_o: Array2<A>,
    grad_U_o: Array2<A>,
    grad_b_o: Array1<A>,
    grad_W_c: Array2<A>,
    grad_U_c: Array2<A>,
    grad_b_c: Array1<A>,
}

#[derive(Clone)]
struct LSTMState<A: Float> {
    h: Array1<A>,
    c: Array1<A>,
    i: Array1<A>,
    f: Array1<A>,
    o: Array1<A>,
    c_tilde: Array1<A>,
}

impl<A: Float + ScalarOperand + Sum> LSTM<A> {
    /// Create new LSTM layer with Xavier initialization
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = thread_rng();
        let std = (2.0 / (input_size + hidden_size) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let mut init_weight = |rows, cols| -> Array2<A> {
            Array2::from_shape_fn((rows, cols), |_| {
                A::from(normal.sample(&mut rng)).unwrap()
            })
        };

        Self {
            input_size,
            hidden_size,
            // Input gate
            W_i: init_weight(hidden_size, input_size),
            U_i: init_weight(hidden_size, hidden_size),
            b_i: Array1::zeros(hidden_size),
            // Forget gate
            W_f: init_weight(hidden_size, input_size),
            U_f: init_weight(hidden_size, hidden_size),
            b_f: Array1::ones(hidden_size), // Initialize to 1 for better gradient flow
            // Output gate
            W_o: init_weight(hidden_size, input_size),
            U_o: init_weight(hidden_size, hidden_size),
            b_o: Array1::zeros(hidden_size),
            // Cell gate
            W_c: init_weight(hidden_size, input_size),
            U_c: init_weight(hidden_size, hidden_size),
            b_c: Array1::zeros(hidden_size),
            cached_states: None,
            // Gradients
            grad_W_i: Array2::zeros((hidden_size, input_size)),
            grad_U_i: Array2::zeros((hidden_size, hidden_size)),
            grad_b_i: Array1::zeros(hidden_size),
            grad_W_f: Array2::zeros((hidden_size, input_size)),
            grad_U_f: Array2::zeros((hidden_size, hidden_size)),
            grad_b_f: Array1::zeros(hidden_size),
            grad_W_o: Array2::zeros((hidden_size, input_size)),
            grad_U_o: Array2::zeros((hidden_size, hidden_size)),
            grad_b_o: Array1::zeros(hidden_size),
            grad_W_c: Array2::zeros((hidden_size, input_size)),
            grad_U_c: Array2::zeros((hidden_size, hidden_size)),
            grad_b_c: Array1::zeros(hidden_size),
        }
    }

    /// Process sequence through LSTM
    pub fn forward_sequence(&mut self, sequence: &Array2<A>) -> Array2<A> {
        let seq_len = sequence.nrows();
        let mut h = Array1::zeros(self.hidden_size);
        let mut c = Array1::zeros(self.hidden_size);
        let mut states = Vec::new();
        let mut outputs = Vec::new();

        for t in 0..seq_len {
            let x = sequence.row(t).to_owned();

            // Input gate: i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)
            let i = sigmoid_1d(&(&self.W_i.dot(&x) + &self.U_i.dot(&h) + &self.b_i));

            // Forget gate: f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)
            let f = sigmoid_1d(&(&self.W_f.dot(&x) + &self.U_f.dot(&h) + &self.b_f));

            // Output gate: o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)
            let o = sigmoid_1d(&(&self.W_o.dot(&x) + &self.U_o.dot(&h) + &self.b_o));

            // Cell candidate: c̃_t = tanh(W_c·x_t + U_c·h_{t-1} + b_c)
            let c_tilde = tanh_1d(&(&self.W_c.dot(&x) + &self.U_c.dot(&h) + &self.b_c));

            // Cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
            c = &(&f * &c) + &(&i * &c_tilde);

            // Hidden state: h_t = o_t ⊙ tanh(c_t)
            h = &o * &tanh_1d(&c);

            states.push(LSTMState {
                h: h.clone(),
                c: c.clone(),
                i: i.clone(),
                f: f.clone(),
                o: o.clone(),
                c_tilde: c_tilde.clone(),
            });

            outputs.push(h.clone());
        }

        self.cached_states = Some(states);

        // Stack outputs
        Array2::from_shape_fn((seq_len, self.hidden_size), |(i, j)| outputs[i][j])
    }
}

/// GRU (Gated Recurrent Unit) layer - simpler alternative to LSTM
pub struct GRU<A: Float> {
    /// Input size
    input_size: usize,
    /// Hidden size
    hidden_size: usize,
    /// Reset gate weights
    W_r: Array2<A>,
    U_r: Array2<A>,
    b_r: Array1<A>,
    /// Update gate weights
    W_z: Array2<A>,
    U_z: Array2<A>,
    b_z: Array1<A>,
    /// Candidate weights
    W_h: Array2<A>,
    U_h: Array2<A>,
    b_h: Array1<A>,
    /// Cached states for backprop
    cached_states: Option<Vec<GRUState<A>>>,
    /// Gradients
    grad_W_r: Array2<A>,
    grad_U_r: Array2<A>,
    grad_b_r: Array1<A>,
    grad_W_z: Array2<A>,
    grad_U_z: Array2<A>,
    grad_b_z: Array1<A>,
    grad_W_h: Array2<A>,
    grad_U_h: Array2<A>,
    grad_b_h: Array1<A>,
}

#[derive(Clone)]
struct GRUState<A: Float> {
    h: Array1<A>,
    r: Array1<A>,
    z: Array1<A>,
    h_tilde: Array1<A>,
}

impl<A: Float + ScalarOperand + Sum> GRU<A> {
    /// Create new GRU layer with Xavier initialization
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let mut rng = thread_rng();
        let std = (2.0 / (input_size + hidden_size) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let mut init_weight = |rows, cols| -> Array2<A> {
            Array2::from_shape_fn((rows, cols), |_| {
                A::from(normal.sample(&mut rng)).unwrap()
            })
        };

        Self {
            input_size,
            hidden_size,
            // Reset gate
            W_r: init_weight(hidden_size, input_size),
            U_r: init_weight(hidden_size, hidden_size),
            b_r: Array1::zeros(hidden_size),
            // Update gate
            W_z: init_weight(hidden_size, input_size),
            U_z: init_weight(hidden_size, hidden_size),
            b_z: Array1::zeros(hidden_size),
            // Candidate
            W_h: init_weight(hidden_size, input_size),
            U_h: init_weight(hidden_size, hidden_size),
            b_h: Array1::zeros(hidden_size),
            cached_states: None,
            // Gradients
            grad_W_r: Array2::zeros((hidden_size, input_size)),
            grad_U_r: Array2::zeros((hidden_size, hidden_size)),
            grad_b_r: Array1::zeros(hidden_size),
            grad_W_z: Array2::zeros((hidden_size, input_size)),
            grad_U_z: Array2::zeros((hidden_size, hidden_size)),
            grad_b_z: Array1::zeros(hidden_size),
            grad_W_h: Array2::zeros((hidden_size, input_size)),
            grad_U_h: Array2::zeros((hidden_size, hidden_size)),
            grad_b_h: Array1::zeros(hidden_size),
        }
    }

    /// Process sequence through GRU
    pub fn forward_sequence(&mut self, sequence: &Array2<A>) -> Array2<A> {
        let seq_len = sequence.nrows();
        let mut h = Array1::zeros(self.hidden_size);
        let mut states = Vec::new();
        let mut outputs = Vec::new();

        for t in 0..seq_len {
            let x = sequence.row(t).to_owned();

            // Reset gate: r_t = σ(W_r·x_t + U_r·h_{t-1} + b_r)
            let r = sigmoid_1d(&(&self.W_r.dot(&x) + &self.U_r.dot(&h) + &self.b_r));

            // Update gate: z_t = σ(W_z·x_t + U_z·h_{t-1} + b_z)
            let z = sigmoid_1d(&(&self.W_z.dot(&x) + &self.U_z.dot(&h) + &self.b_z));

            // Candidate: h̃_t = tanh(W_h·x_t + U_h·(r_t ⊙ h_{t-1}) + b_h)
            let h_tilde = tanh_1d(&(&self.W_h.dot(&x) + &self.U_h.dot(&(&r * &h)) + &self.b_h));

            // Hidden state: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
            let one = Array1::ones(self.hidden_size);
            h = &(&(&one - &z) * &h) + &(&z * &h_tilde);

            states.push(GRUState {
                h: h.clone(),
                r: r.clone(),
                z: z.clone(),
                h_tilde: h_tilde.clone(),
            });

            outputs.push(h.clone());
        }

        self.cached_states = Some(states);

        // Stack outputs
        Array2::from_shape_fn((seq_len, self.hidden_size), |(i, j)| outputs[i][j])
    }
}

/// Helper functions for 1D array operations
fn sigmoid_1d<A: Float + ScalarOperand>(x: &Array1<A>) -> Array1<A> {
    x.mapv(|v| A::one() / (A::one() + (-v).exp()))
}

fn tanh_1d<A: Float + ScalarOperand>(x: &Array1<A>) -> Array1<A> {
    x.mapv(|v| v.tanh())
}
