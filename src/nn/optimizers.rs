//! Advanced optimizers for neural network training
//!
//! State-of-the-art optimization algorithms including:
//! - SGD with momentum and Nesterov acceleration
//! - Adam (Adaptive Moment Estimation)
//! - AdamW (Adam with decoupled weight decay)
//! - RAdam (Rectified Adam)
//! - LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
//! - RMSprop

use ndarray::{Array2, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

/// SGD optimizer with momentum and Nesterov acceleration
pub struct SGD<A: Float + ScalarOperand> {
    learning_rate: A,
    momentum: A,
    nesterov: bool,
    velocities: Vec<Array2<A>>,
}

impl<A: Float + ScalarOperand + Sum> SGD<A> {
    pub fn new(learning_rate: A, momentum: Option<A>, nesterov: Option<bool>) -> Self {
        Self {
            learning_rate,
            momentum: momentum.unwrap_or(A::zero()),
            nesterov: nesterov.unwrap_or(false),
            velocities: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &mut Vec<&mut Array2<A>>, grads: &[&Array2<A>]) {
        // Initialize velocities on first call
        if self.velocities.is_empty() {
            self.velocities = grads.iter().map(|g| Array2::zeros(g.dim())).collect();
        }

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            if self.momentum == A::zero() {
                // Standard SGD
                **param = &**param - &(*grad * self.learning_rate);
            } else {
                // SGD with momentum
                self.velocities[i] = &(&self.velocities[i] * self.momentum) + &(*grad * self.learning_rate);

                if self.nesterov {
                    // Nesterov momentum
                    **param = &**param - &(&(*grad * self.learning_rate) + &(&self.velocities[i] * self.momentum));
                } else {
                    // Classical momentum
                    **param = &**param - &self.velocities[i];
                }
            }
        }
    }
}

/// Adam optimizer (Adaptive Moment Estimation)
///
/// Computes adaptive learning rates for each parameter by maintaining
/// running averages of gradients and squared gradients.
pub struct Adam<A: Float> {
    learning_rate: A,
    beta1: A,
    beta2: A,
    epsilon: A,
    t: usize,
    m: Vec<Array2<A>>,  // First moment estimate
    v: Vec<Array2<A>>,  // Second moment estimate
}

impl<A: Float + ScalarOperand + Sum> Adam<A> {
    pub fn new(learning_rate: A, beta1: Option<A>, beta2: Option<A>) -> Self {
        Self {
            learning_rate,
            beta1: beta1.unwrap_or(A::from(0.9).unwrap()),
            beta2: beta2.unwrap_or(A::from(0.999).unwrap()),
            epsilon: A::from(1e-8).unwrap(),
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &mut Vec<&mut Array2<A>>, grads: &[&Array2<A>]) {
        // Initialize moments on first call
        if self.m.is_empty() {
            self.m = grads.iter().map(|g| Array2::zeros(g.dim())).collect();
            self.v = grads.iter().map(|g| Array2::zeros(g.dim())).collect();
        }

        self.t += 1;
        let t = A::from(self.t).unwrap();

        // Bias correction factors
        let bias_correction1 = A::one() - self.beta1.powf(t);
        let bias_correction2 = A::one() - self.beta2.powf(t);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Update biased first moment estimate
            self.m[i] = &(&self.m[i] * self.beta1) + &(*grad * (A::one() - self.beta1));

            // Update biased second raw moment estimate
            self.v[i] = &(&self.v[i] * self.beta2) + &(grad.mapv(|g| g * g) * (A::one() - self.beta2));

            // Compute bias-corrected moments
            let m_hat = &self.m[i] / bias_correction1;
            let v_hat = &self.v[i] / bias_correction2;

            // Update parameters
            **param = &**param - &(&m_hat * self.learning_rate / &(v_hat.mapv(|v| v.sqrt()) + self.epsilon));
        }
    }
}

/// AdamW optimizer (Adam with decoupled Weight decay)
///
/// Improves upon Adam by decoupling weight decay from gradient-based update.
pub struct AdamW<A: Float> {
    learning_rate: A,
    beta1: A,
    beta2: A,
    epsilon: A,
    weight_decay: A,
    t: usize,
    m: Vec<Array2<A>>,
    v: Vec<Array2<A>>,
}

impl<A: Float + ScalarOperand + Sum> AdamW<A> {
    pub fn new(
        learning_rate: A,
        beta1: Option<A>,
        beta2: Option<A>,
        weight_decay: Option<A>,
    ) -> Self {
        Self {
            learning_rate,
            beta1: beta1.unwrap_or(A::from(0.9).unwrap()),
            beta2: beta2.unwrap_or(A::from(0.999).unwrap()),
            epsilon: A::from(1e-8).unwrap(),
            weight_decay: weight_decay.unwrap_or(A::from(0.01).unwrap()),
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &mut Vec<&mut Array2<A>>, grads: &[&Array2<A>]) {
        if self.m.is_empty() {
            self.m = grads.iter().map(|g| Array2::zeros(g.dim())).collect();
            self.v = grads.iter().map(|g| Array2::zeros(g.dim())).collect();
        }

        self.t += 1;
        let t = A::from(self.t).unwrap();

        let bias_correction1 = A::one() - self.beta1.powf(t);
        let bias_correction2 = A::one() - self.beta2.powf(t);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Update moments (same as Adam)
            self.m[i] = &(&self.m[i] * self.beta1) + &(*grad * (A::one() - self.beta1));
            self.v[i] = &(&self.v[i] * self.beta2) + &(grad.mapv(|g| g * g) * (A::one() - self.beta2));

            let m_hat = &self.m[i] / bias_correction1;
            let v_hat = &self.v[i] / bias_correction2;

            // AdamW: Decoupled weight decay
            // θ = θ - lr * (m_hat / (√v_hat + ε) + λ * θ)
            let adam_update = &m_hat / &(v_hat.mapv(|v| v.sqrt()) + self.epsilon);
            let weight_decay_update = &**param * self.weight_decay;

            **param = &**param - &(&(&adam_update + &weight_decay_update) * self.learning_rate);
        }
    }
}

/// RAdam optimizer (Rectified Adam)
///
/// Addresses the large variance issue in Adam during early training
/// by rectifying the variance of adaptive learning rate.
pub struct RAdam<A: Float> {
    learning_rate: A,
    beta1: A,
    beta2: A,
    epsilon: A,
    t: usize,
    m: Vec<Array2<A>>,
    v: Vec<Array2<A>>,
}

impl<A: Float + ScalarOperand + Sum> RAdam<A> {
    pub fn new(learning_rate: A, beta1: Option<A>, beta2: Option<A>) -> Self {
        Self {
            learning_rate,
            beta1: beta1.unwrap_or(A::from(0.9).unwrap()),
            beta2: beta2.unwrap_or(A::from(0.999).unwrap()),
            epsilon: A::from(1e-8).unwrap(),
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &mut Vec<&mut Array2<A>>, grads: &[&Array2<A>]) {
        if self.m.is_empty() {
            self.m = grads.iter().map(|g| Array2::zeros(g.dim())).collect();
            self.v = grads.iter().map(|g| Array2::zeros(g.dim())).collect();
        }

        self.t += 1;
        let t_f = self.t as f64;
        let beta1_f = self.beta1.to_f64().unwrap();
        let beta2_f = self.beta2.to_f64().unwrap();

        // Compute maximum length of approximated SMA
        let rho_inf = 2.0 / (1.0 - beta2_f) - 1.0;

        // Compute bias-corrected second moment
        let rho_t = rho_inf - 2.0 * t_f * beta2_f.powf(t_f) / (1.0 - beta2_f.powf(t_f));

        let bias_correction1 = A::one() - self.beta1.powf(A::from(t_f).unwrap());

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Update moments
            self.m[i] = &(&self.m[i] * self.beta1) + &(*grad * (A::one() - self.beta1));
            self.v[i] = &(&self.v[i] * self.beta2) + &(grad.mapv(|g| g * g) * (A::one() - self.beta2));

            let m_hat = &self.m[i] / bias_correction1;

            if rho_t > 4.0 {
                // Variance is tractable, use adaptive learning rate
                let bias_correction2 = A::one() - self.beta2.powf(A::from(t_f).unwrap());
                let v_hat = &self.v[i] / bias_correction2;

                // Compute rectification term
                let r_t = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                          ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)).sqrt();
                let r_t = A::from(r_t).unwrap();

                let update = &m_hat * r_t / &(v_hat.mapv(|v| v.sqrt()) + self.epsilon);
                **param = &**param - &(&update * self.learning_rate);
            } else {
                // Variance is not tractable, use SGD-like update
                **param = &**param - &(&m_hat * self.learning_rate);
            }
        }
    }
}

/// LAMB optimizer (Layer-wise Adaptive Moments optimizer for Batch training)
///
/// Designed for large batch training, uses layer-wise adaptation
/// to improve convergence and generalization.
pub struct LAMB<A: Float> {
    learning_rate: A,
    beta1: A,
    beta2: A,
    epsilon: A,
    weight_decay: A,
    t: usize,
    m: Vec<Array2<A>>,
    v: Vec<Array2<A>>,
}

impl<A: Float + ScalarOperand + Sum> LAMB<A> {
    pub fn new(
        learning_rate: A,
        beta1: Option<A>,
        beta2: Option<A>,
        weight_decay: Option<A>,
    ) -> Self {
        Self {
            learning_rate,
            beta1: beta1.unwrap_or(A::from(0.9).unwrap()),
            beta2: beta2.unwrap_or(A::from(0.999).unwrap()),
            epsilon: A::from(1e-6).unwrap(),
            weight_decay: weight_decay.unwrap_or(A::from(0.01).unwrap()),
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &mut Vec<&mut Array2<A>>, grads: &[&Array2<A>]) {
        if self.m.is_empty() {
            self.m = grads.iter().map(|g| Array2::zeros(g.dim())).collect();
            self.v = grads.iter().map(|g| Array2::zeros(g.dim())).collect();
        }

        self.t += 1;
        let t = A::from(self.t).unwrap();

        let bias_correction1 = A::one() - self.beta1.powf(t);
        let bias_correction2 = A::one() - self.beta2.powf(t);

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Update moments
            self.m[i] = &(&self.m[i] * self.beta1) + &(*grad * (A::one() - self.beta1));
            self.v[i] = &(&self.v[i] * self.beta2) + &(grad.mapv(|g| g * g) * (A::one() - self.beta2));

            let m_hat = &self.m[i] / bias_correction1;
            let v_hat = &self.v[i] / bias_correction2;

            // Compute adaptive update
            let update = &m_hat / &(v_hat.mapv(|v| v.sqrt()) + self.epsilon) +
                        &(&**param * self.weight_decay);

            // Compute trust ratio (layer-wise adaptation)
            let param_norm = param.mapv(|p| p * p).sum().sqrt();
            let update_norm = update.mapv(|u| u * u).sum().sqrt();

            let trust_ratio = if param_norm > A::zero() && update_norm > A::zero() {
                param_norm / update_norm
            } else {
                A::one()
            };

            // Apply update with trust ratio
            **param = &**param - &(&update * self.learning_rate * trust_ratio);
        }
    }
}

/// RMSprop optimizer
///
/// Uses moving average of squared gradients to normalize the gradient.
pub struct RMSprop<A: Float> {
    learning_rate: A,
    alpha: A,
    epsilon: A,
    v: Vec<Array2<A>>,
}

impl<A: Float + ScalarOperand + Sum> RMSprop<A> {
    pub fn new(learning_rate: A, alpha: Option<A>) -> Self {
        Self {
            learning_rate,
            alpha: alpha.unwrap_or(A::from(0.99).unwrap()),
            epsilon: A::from(1e-8).unwrap(),
            v: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &mut Vec<&mut Array2<A>>, grads: &[&Array2<A>]) {
        if self.v.is_empty() {
            self.v = grads.iter().map(|g| Array2::zeros(g.dim())).collect();
        }

        for (i, (param, grad)) in params.iter_mut().zip(grads.iter()).enumerate() {
            // Update moving average of squared gradients
            self.v[i] = &(&self.v[i] * self.alpha) + &(grad.mapv(|g| g * g) * (A::one() - self.alpha));

            // Update parameters
            **param = &**param - &(*grad * self.learning_rate / &(self.v[i].mapv(|v| v.sqrt()) + self.epsilon));
        }
    }
}
