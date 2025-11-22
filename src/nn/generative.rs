//! Generative Models
//!
//! Foundational architectures for generative modeling:
//! - VAE (Variational Autoencoder)
//! - GAN (Generative Adversarial Network) components

use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::iter::Sum;

/// Variational Autoencoder (VAE)
///
/// Learns a probabilistic latent representation of data.
pub struct VAE<A: Float> {
    /// Latent dimension
    pub latent_dim: usize,
    /// Encoder mean weights
    pub encoder_mean_weights: Array2<A>,
    /// Encoder logvar weights
    pub encoder_logvar_weights: Array2<A>,
    /// Decoder weights
    pub decoder_weights: Array2<A>,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> VAE<A> {
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.01).unwrap();

        let mut init_weight = |rows, cols| -> Array2<A> {
            Array2::from_shape_fn((rows, cols), |_| {
                A::from(normal.sample(&mut rng)).unwrap()
            })
        };

        Self {
            latent_dim,
            encoder_mean_weights: init_weight(hidden_dim, latent_dim),
            encoder_logvar_weights: init_weight(hidden_dim, latent_dim),
            decoder_weights: init_weight(latent_dim, input_dim),
        }
    }

    /// Encode input to latent distribution parameters
    pub fn encode(&self, x: &Array2<A>) -> (Array2<A>, Array2<A>) {
        // Simplified: just linear transformations
        let mean = x.dot(&self.encoder_mean_weights);
        let logvar = x.dot(&self.encoder_logvar_weights);
        (mean, logvar)
    }

    /// Reparameterization trick
    pub fn reparameterize(&self, mean: &Array2<A>, logvar: &Array2<A>) -> Array2<A> {
        // z = mean + std * epsilon
        // std = exp(0.5 * logvar)
        let std = logvar.mapv(|x| (x * A::from(0.5).unwrap()).exp());
        let epsilon = Array2::from_shape_fn(mean.dim(), |_| {
            A::from(rand::random::<f64>()).unwrap()
        });
        mean + &(std * epsilon)
    }

    /// Decode latent to reconstruction
    pub fn decode(&self, z: &Array2<A>) -> Array2<A> {
        z.dot(&self.decoder_weights)
    }

    /// Compute VAE loss (reconstruction + KL divergence)
    pub fn loss(&self, x: &Array2<A>, x_recon: &Array2<A>, mean: &Array2<A>, logvar: &Array2<A>) -> A {
        // Reconstruction loss (MSE)
        let recon_loss = (x - x_recon).mapv(|v| v * v).sum() / A::from(x.len()).unwrap();

        // KL divergence: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        let mean_sq = mean.mapv(|m| m * m);
        let logvar_exp = logvar.mapv(|lv| lv.exp());
        let kl_div = -A::from(0.5).unwrap() * (logvar - &mean_sq - &logvar_exp).sum() / A::from(mean.nrows()).unwrap()
            + A::from(0.5).unwrap();

        recon_loss + kl_div
    }
}

/// GAN Generator
pub struct Generator<A: Float> {
    /// Latent dimension
    pub latent_dim: usize,
    /// Hidden layers
    pub layers: Vec<Array2<A>>,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> Generator<A> {
    pub fn new(latent_dim: usize, hidden_dims: Vec<usize>, output_dim: usize) -> Self {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let mut init_weight = |rows, cols| -> Array2<A> {
            Array2::from_shape_fn((rows, cols), |_| {
                A::from(normal.sample(&mut rng)).unwrap()
            })
        };

        let mut layers = Vec::new();
        let mut prev_dim = latent_dim;

        for &hidden_dim in &hidden_dims {
            layers.push(init_weight(prev_dim, hidden_dim));
            prev_dim = hidden_dim;
        }

        layers.push(init_weight(prev_dim, output_dim));

        Self { latent_dim, layers }
    }

    /// Generate samples from noise
    pub fn generate(&self, z: &Array2<A>) -> Array2<A> {
        let mut activation = z.clone();

        for layer in &self.layers {
            activation = activation.dot(layer);
            // ReLU activation (except last layer)
            activation.mapv_inplace(|x| x.max(A::zero()));
        }

        // Tanh for final layer (output in [-1, 1])
        activation.mapv(|x| x.tanh())
    }
}

/// GAN Discriminator
pub struct Discriminator<A: Float> {
    /// Hidden layers
    pub layers: Vec<Array2<A>>,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> Discriminator<A> {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>) -> Self {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let mut init_weight = |rows, cols| -> Array2<A> {
            Array2::from_shape_fn((rows, cols), |_| {
                A::from(normal.sample(&mut rng)).unwrap()
            })
        };

        let mut layers = Vec::new();
        let mut prev_dim = input_dim;

        for &hidden_dim in &hidden_dims {
            layers.push(init_weight(prev_dim, hidden_dim));
            prev_dim = hidden_dim;
        }

        // Output: single neuron for binary classification
        layers.push(init_weight(prev_dim, 1));

        Self { layers }
    }

    /// Discriminate real vs fake
    pub fn discriminate(&self, x: &Array2<A>) -> Array2<A> {
        let mut activation = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            activation = activation.dot(layer);

            if i < self.layers.len() - 1 {
                // Leaky ReLU for hidden layers
                activation.mapv_inplace(|x| {
                    if x > A::zero() {
                        x
                    } else {
                        x * A::from(0.2).unwrap()
                    }
                });
            } else {
                // Sigmoid for output
                activation.mapv_inplace(|x| A::one() / (A::one() + (-x).exp()));
            }
        }

        activation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vae_creation() {
        let vae = VAE::<f64>::new(784, 400, 20);
        assert_eq!(vae.latent_dim, 20);
    }

    #[test]
    fn test_gan_creation() {
        let gen = Generator::<f64>::new(100, vec![256, 512], 784);
        let disc = Discriminator::<f64>::new(784, vec![512, 256]);
        assert_eq!(gen.latent_dim, 100);
        assert!(disc.layers.len() > 0);
    }
}
