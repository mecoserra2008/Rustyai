//! Attention Mechanisms and Transformers
//!
//! State-of-the-art attention mechanisms including:
//! - Scaled Dot-Product Attention
//! - Multi-Head Attention
//! - Transformer Encoder/Decoder
//! - Self-Attention
//! - Cross-Attention

use super::Layer;
use ndarray::{Array1, Array2, Array3, ScalarOperand, Axis};
use num_traits::{Float, FromPrimitive};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::iter::Sum;

/// Scaled Dot-Product Attention
///
/// Attention(Q, K, V) = softmax(QK^T / √d_k) V
pub struct ScaledDotProductAttention<A: Float> {
    /// Scale factor (√d_k)
    scale: A,
    /// Cached attention weights for visualization
    attention_weights: Option<Array2<A>>,
}

impl<A: Float + ScalarOperand + Sum> ScaledDotProductAttention<A> {
    pub fn new(d_k: usize) -> Self {
        let scale = A::from(d_k).unwrap().sqrt();
        Self {
            scale,
            attention_weights: None,
        }
    }

    /// Compute attention
    ///
    /// # Arguments
    /// * `Q` - Query matrix (batch_size, seq_len_q, d_k)
    /// * `K` - Key matrix (batch_size, seq_len_k, d_k)
    /// * `V` - Value matrix (batch_size, seq_len_v, d_v)
    /// * `mask` - Optional attention mask
    pub fn forward(
        &mut self,
        Q: &Array2<A>,
        K: &Array2<A>,
        V: &Array2<A>,
        mask: Option<&Array2<A>>,
    ) -> Array2<A> {
        // Compute scores: QK^T / √d_k
        let scores = Q.dot(&K.t()) / self.scale;

        // Apply mask if provided (set masked positions to large negative value)
        let scores = if let Some(m) = mask {
            let mut masked_scores = scores.clone();
            let neg_inf = A::from(-1e9).unwrap();
            for i in 0..masked_scores.nrows() {
                for j in 0..masked_scores.ncols() {
                    if m[[i, j]] == A::zero() {
                        masked_scores[[i, j]] = neg_inf;
                    }
                }
            }
            masked_scores
        } else {
            scores
        };

        // Apply softmax
        let attention_weights = softmax_2d(&scores);

        // Store for visualization
        self.attention_weights = Some(attention_weights.clone());

        // Compute weighted sum: Attention · V
        attention_weights.dot(V)
    }

    /// Get cached attention weights
    pub fn get_attention_weights(&self) -> Option<&Array2<A>> {
        self.attention_weights.as_ref()
    }
}

/// Multi-Head Attention layer
///
/// Allows the model to jointly attend to information from different representation subspaces.
pub struct MultiHeadAttention<A: Float> {
    /// Number of attention heads
    num_heads: usize,
    /// Dimension of model
    d_model: usize,
    /// Dimension per head (d_model / num_heads)
    d_k: usize,
    /// Query projection weights for all heads
    W_q: Array2<A>,
    /// Key projection weights for all heads
    W_k: Array2<A>,
    /// Value projection weights for all heads
    W_v: Array2<A>,
    /// Output projection weights
    W_o: Array2<A>,
    /// Attention mechanism
    attention: ScaledDotProductAttention<A>,
    /// Cached for backprop
    cached_input: Option<(Array2<A>, Array2<A>, Array2<A>)>,
}

impl<A: Float + ScalarOperand + Sum> MultiHeadAttention<A> {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");

        let d_k = d_model / num_heads;
        let mut rng = thread_rng();
        let std = (2.0 / d_model as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let mut init_weight = |rows, cols| -> Array2<A> {
            Array2::from_shape_fn((rows, cols), |_| {
                A::from(normal.sample(&mut rng)).unwrap()
            })
        };

        Self {
            num_heads,
            d_model,
            d_k,
            W_q: init_weight(d_model, d_model),
            W_k: init_weight(d_model, d_model),
            W_v: init_weight(d_model, d_model),
            W_o: init_weight(d_model, d_model),
            attention: ScaledDotProductAttention::new(d_k),
            cached_input: None,
        }
    }

    /// Forward pass through multi-head attention
    ///
    /// # Arguments
    /// * `Q` - Query (seq_len, d_model)
    /// * `K` - Key (seq_len, d_model)
    /// * `V` - Value (seq_len, d_model)
    /// * `mask` - Optional attention mask
    pub fn forward(
        &mut self,
        Q: &Array2<A>,
        K: &Array2<A>,
        V: &Array2<A>,
        mask: Option<&Array2<A>>,
    ) -> Array2<A> {
        self.cached_input = Some((Q.clone(), K.clone(), V.clone()));

        let seq_len = Q.nrows();

        // Linear projections
        let Q_proj = Q.dot(&self.W_q);
        let K_proj = K.dot(&self.W_k);
        let V_proj = V.dot(&self.W_v);

        // Split into heads and apply attention
        let mut head_outputs = Vec::new();

        for h in 0..self.num_heads {
            let start = h * self.d_k;
            let end = start + self.d_k;

            // Extract head-specific projections
            let Q_h = Q_proj.slice(ndarray::s![.., start..end]).to_owned();
            let K_h = K_proj.slice(ndarray::s![.., start..end]).to_owned();
            let V_h = V_proj.slice(ndarray::s![.., start..end]).to_owned();

            // Apply attention for this head
            let head_output = self.attention.forward(&Q_h, &K_h, &V_h, mask);
            head_outputs.push(head_output);
        }

        // Concatenate heads
        let concat = Array2::from_shape_fn((seq_len, self.d_model), |(i, j)| {
            let head_idx = j / self.d_k;
            let in_head_idx = j % self.d_k;
            head_outputs[head_idx][[i, in_head_idx]]
        });

        // Final linear projection
        concat.dot(&self.W_o)
    }
}

/// Transformer Encoder Layer
///
/// Combines multi-head attention with feed-forward network and residual connections.
pub struct TransformerEncoderLayer<A: Float> {
    /// Multi-head self-attention
    self_attention: MultiHeadAttention<A>,
    /// Feed-forward network layer 1
    ff_W1: Array2<A>,
    ff_b1: Array1<A>,
    /// Feed-forward network layer 2
    ff_W2: Array2<A>,
    ff_b2: Array1<A>,
    /// Layer norm parameters for attention
    ln1_gamma: Array1<A>,
    ln1_beta: Array1<A>,
    /// Layer norm parameters for feed-forward
    ln2_gamma: Array1<A>,
    ln2_beta: Array1<A>,
    /// Dimension
    d_model: usize,
    /// Feed-forward dimension
    d_ff: usize,
    /// Dropout rate
    dropout_rate: A,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> TransformerEncoderLayer<A> {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout: f64) -> Self {
        let mut rng = thread_rng();
        let std = (2.0 / d_model as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let mut init_weight = |rows, cols| -> Array2<A> {
            Array2::from_shape_fn((rows, cols), |_| {
                A::from(normal.sample(&mut rng)).unwrap()
            })
        };

        Self {
            self_attention: MultiHeadAttention::new(d_model, num_heads),
            ff_W1: init_weight(d_model, d_ff),
            ff_b1: Array1::zeros(d_ff),
            ff_W2: init_weight(d_ff, d_model),
            ff_b2: Array1::zeros(d_model),
            ln1_gamma: Array1::ones(d_model),
            ln1_beta: Array1::zeros(d_model),
            ln2_gamma: Array1::ones(d_model),
            ln2_beta: Array1::zeros(d_model),
            d_model,
            d_ff,
            dropout_rate: A::from(dropout).unwrap(),
        }
    }

    /// Forward pass through encoder layer
    pub fn forward(&mut self, x: &Array2<A>, mask: Option<&Array2<A>>) -> Array2<A> {
        // Multi-head self-attention with residual connection
        let attn_output = self.self_attention.forward(x, x, x, mask);
        let x = x + &attn_output;

        // Layer normalization 1
        let x = layer_norm_2d(&x, &self.ln1_gamma, &self.ln1_beta);

        // Feed-forward network with residual connection
        let ff_output = {
            // First layer with ReLU
            let mut hidden = x.dot(&self.ff_W1);
            for i in 0..hidden.nrows() {
                for j in 0..hidden.ncols() {
                    hidden[[i, j]] = hidden[[i, j]] + self.ff_b1[j];
                    hidden[[i, j]] = hidden[[i, j]].max(A::zero()); // ReLU
                }
            }

            // Second layer
            let mut output = hidden.dot(&self.ff_W2);
            for i in 0..output.nrows() {
                for j in 0..output.ncols() {
                    output[[i, j]] = output[[i, j]] + self.ff_b2[j];
                }
            }
            output
        };

        let x = &x + &ff_output;

        // Layer normalization 2
        layer_norm_2d(&x, &self.ln2_gamma, &self.ln2_beta)
    }
}

/// Transformer Decoder Layer
pub struct TransformerDecoderLayer<A: Float> {
    /// Masked multi-head self-attention
    self_attention: MultiHeadAttention<A>,
    /// Multi-head cross-attention (decoder-encoder)
    cross_attention: MultiHeadAttention<A>,
    /// Feed-forward network layer 1
    ff_W1: Array2<A>,
    ff_b1: Array1<A>,
    /// Feed-forward network layer 2
    ff_W2: Array2<A>,
    ff_b2: Array1<A>,
    /// Layer norm parameters
    ln1_gamma: Array1<A>,
    ln1_beta: Array1<A>,
    ln2_gamma: Array1<A>,
    ln2_beta: Array1<A>,
    ln3_gamma: Array1<A>,
    ln3_beta: Array1<A>,
    d_model: usize,
    d_ff: usize,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> TransformerDecoderLayer<A> {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        let mut rng = thread_rng();
        let std = (2.0 / d_model as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let mut init_weight = |rows, cols| -> Array2<A> {
            Array2::from_shape_fn((rows, cols), |_| {
                A::from(normal.sample(&mut rng)).unwrap()
            })
        };

        Self {
            self_attention: MultiHeadAttention::new(d_model, num_heads),
            cross_attention: MultiHeadAttention::new(d_model, num_heads),
            ff_W1: init_weight(d_model, d_ff),
            ff_b1: Array1::zeros(d_ff),
            ff_W2: init_weight(d_ff, d_model),
            ff_b2: Array1::zeros(d_model),
            ln1_gamma: Array1::ones(d_model),
            ln1_beta: Array1::zeros(d_model),
            ln2_gamma: Array1::ones(d_model),
            ln2_beta: Array1::zeros(d_model),
            ln3_gamma: Array1::ones(d_model),
            ln3_beta: Array1::zeros(d_model),
            d_model,
            d_ff,
        }
    }

    /// Forward pass through decoder layer
    ///
    /// # Arguments
    /// * `x` - Decoder input
    /// * `encoder_output` - Output from encoder
    /// * `tgt_mask` - Causal mask for decoder self-attention
    /// * `src_mask` - Mask for encoder-decoder attention
    pub fn forward(
        &mut self,
        x: &Array2<A>,
        encoder_output: &Array2<A>,
        tgt_mask: Option<&Array2<A>>,
        src_mask: Option<&Array2<A>>,
    ) -> Array2<A> {
        // Masked self-attention with residual
        let self_attn_output = self.self_attention.forward(x, x, x, tgt_mask);
        let x = x + &self_attn_output;
        let x = layer_norm_2d(&x, &self.ln1_gamma, &self.ln1_beta);

        // Cross-attention with encoder output
        let cross_attn_output = self.cross_attention.forward(
            &x,
            encoder_output,
            encoder_output,
            src_mask,
        );
        let x = &x + &cross_attn_output;
        let x = layer_norm_2d(&x, &self.ln2_gamma, &self.ln2_beta);

        // Feed-forward network
        let ff_output = {
            let mut hidden = x.dot(&self.ff_W1);
            for i in 0..hidden.nrows() {
                for j in 0..hidden.ncols() {
                    hidden[[i, j]] = hidden[[i, j]] + self.ff_b1[j];
                    hidden[[i, j]] = hidden[[i, j]].max(A::zero());
                }
            }

            let mut output = hidden.dot(&self.ff_W2);
            for i in 0..output.nrows() {
                for j in 0..output.ncols() {
                    output[[i, j]] = output[[i, j]] + self.ff_b2[j];
                }
            }
            output
        };

        let x = &x + &ff_output;
        layer_norm_2d(&x, &self.ln3_gamma, &self.ln3_beta)
    }
}

/// Create causal mask for decoder (prevents attending to future tokens)
pub fn create_causal_mask<A: Float>(seq_len: usize) -> Array2<A> {
    Array2::from_shape_fn((seq_len, seq_len), |(i, j)| {
        if j <= i {
            A::one()
        } else {
            A::zero()
        }
    })
}

/// Positional encoding using sinusoidal functions
pub fn positional_encoding<A: Float + ScalarOperand>(
    seq_len: usize,
    d_model: usize,
) -> Array2<A> {
    Array2::from_shape_fn((seq_len, d_model), |(pos, i)| {
        let pos_f = A::from(pos).unwrap();
        let i_f = A::from(i).unwrap();
        let d_model_f = A::from(d_model).unwrap();

        let angle = pos_f / A::from(10000.0)
            .unwrap()
            .powf(A::from(2.0).unwrap() * i_f / d_model_f);

        if i % 2 == 0 {
            angle.sin()
        } else {
            angle.cos()
        }
    })
}

/// Helper: Softmax over rows of 2D array
fn softmax_2d<A: Float + ScalarOperand + Sum>(x: &Array2<A>) -> Array2<A> {
    let mut result = x.clone();

    for mut row in result.rows_mut() {
        // Subtract max for numerical stability
        let max_val = row.fold(A::neg_infinity(), |acc, &val| acc.max(val));
        row.mapv_inplace(|v| (v - max_val).exp());

        // Normalize
        let sum: A = row.iter().copied().sum();
        row.mapv_inplace(|v| v / sum);
    }

    result
}

/// Helper: Layer normalization for 2D arrays
fn layer_norm_2d<A: Float + ScalarOperand + Sum + FromPrimitive>(
    x: &Array2<A>,
    gamma: &Array1<A>,
    beta: &Array1<A>,
) -> Array2<A> {
    let eps = A::from(1e-5).unwrap();
    let mut result = x.clone();

    for mut row in result.rows_mut() {
        // Compute mean and variance
        let mean = row.mean().unwrap();
        let variance = row.mapv(|v| (v - mean) * (v - mean)).sum() / A::from(row.len()).unwrap();
        let std = (variance + eps).sqrt();

        // Normalize and scale
        row.mapv_inplace(|v| (v - mean) / std);
        row.assign(&(&row * gamma + beta));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_dot_product_attention() {
        let mut attn = ScaledDotProductAttention::<f64>::new(4);
        let Q = Array2::from_shape_fn((2, 4), |(i, j)| (i * 4 + j) as f64);
        let K = Array2::from_shape_fn((2, 4), |(i, j)| (i * 4 + j) as f64);
        let V = Array2::from_shape_fn((2, 4), |(i, j)| (i * 4 + j) as f64);

        let output = attn.forward(&Q, &K, &V, None);
        assert_eq!(output.shape(), &[2, 4]);
    }

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask::<f64>(4);
        assert_eq!(mask[[0, 0]], 1.0);
        assert_eq!(mask[[0, 1]], 0.0);
        assert_eq!(mask[[3, 3]], 1.0);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = positional_encoding::<f64>(10, 512);
        assert_eq!(pe.shape(), &[10, 512]);
    }
}
