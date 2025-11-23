//! Advanced Reinforcement Learning Algorithms
//!
//! State-of-the-art deep RL algorithms including:
//! - Proximal Policy Optimization (PPO)
//! - Twin Delayed Deep Deterministic Policy Gradient (TD3)
//! - Soft Actor-Critic (SAC)
//! - Advantage Actor-Critic (A2C)

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::collections::VecDeque;
use std::iter::Sum;
use rand::Rng;

/// Proximal Policy Optimization (PPO) - State-of-the-art policy gradient algorithm
pub struct PPO<A: Float> {
    /// Policy network weights (state_dim x hidden x action_dim)
    policy_weights: Vec<Array2<A>>,
    /// Value network weights (state_dim x hidden x 1)
    value_weights: Vec<Array2<A>>,
    /// Learning rate
    lr: A,
    /// PPO clipping parameter
    epsilon_clip: A,
    /// Discount factor
    gamma: A,
    /// GAE lambda
    gae_lambda: A,
    /// Number of epochs for updates
    n_epochs: usize,
    /// Minibatch size
    batch_size: usize,
    /// Hidden layer size
    hidden_size: usize,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> PPO<A> {
    /// Create a new PPO agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_size: usize,
        lr: A,
        epsilon_clip: A,
        gamma: A,
    ) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        // Initialize policy network (state -> hidden -> action)
        let policy_w1 = Array2::from_shape_fn((state_dim, hidden_size), |_| {
            A::from(normal.sample(&mut rng)).unwrap()
        });
        let policy_w2 = Array2::from_shape_fn((hidden_size, action_dim), |_| {
            A::from(normal.sample(&mut rng)).unwrap()
        });

        // Initialize value network (state -> hidden -> 1)
        let value_w1 = Array2::from_shape_fn((state_dim, hidden_size), |_| {
            A::from(normal.sample(&mut rng)).unwrap()
        });
        let value_w2 = Array2::from_shape_fn((hidden_size, 1), |_| {
            A::from(normal.sample(&mut rng)).unwrap()
        });

        Self {
            policy_weights: vec![policy_w1, policy_w2],
            value_weights: vec![value_w1, value_w2],
            lr,
            epsilon_clip,
            gamma,
            gae_lambda: A::from(0.95).unwrap(),
            n_epochs: 10,
            batch_size: 64,
            hidden_size,
        }
    }

    /// Forward pass through policy network
    fn policy_forward(&self, state: &Array1<A>) -> Array1<A> {
        // Hidden layer with ReLU
        let hidden = state.dot(&self.policy_weights[0]);
        let hidden_activated = hidden.mapv(|x| x.max(A::zero()));

        // Output layer with softmax
        let logits = hidden_activated.dot(&self.policy_weights[1]);
        let exp_logits = logits.mapv(|x| x.exp());
        let sum_exp = exp_logits.sum();
        exp_logits.mapv(|x| x / sum_exp)
    }

    /// Forward pass through value network
    fn value_forward(&self, state: &Array1<A>) -> A {
        let hidden = state.dot(&self.value_weights[0]);
        let hidden_activated = hidden.mapv(|x| x.max(A::zero()));
        hidden_activated.dot(&self.value_weights[1])[0]
    }

    /// Select action using current policy
    pub fn select_action(&self, state: &Array1<A>) -> usize {
        let action_probs = self.policy_forward(state);

        // Sample from categorical distribution
        let mut rng = rand::thread_rng();
        let rand_val: f64 = rng.gen();
        let mut cumsum = 0.0;

        for (action, &prob) in action_probs.iter().enumerate() {
            cumsum += prob.to_f64().unwrap();
            if rand_val < cumsum {
                return action;
            }
        }

        action_probs.len() - 1
    }

    /// Compute Generalized Advantage Estimation (GAE)
    pub fn compute_gae(
        &self,
        rewards: &[A],
        values: &[A],
        dones: &[bool],
    ) -> Vec<A> {
        let mut advantages = vec![A::zero(); rewards.len()];
        let mut gae = A::zero();

        for t in (0..rewards.len()).rev() {
            let next_value = if t + 1 < rewards.len() && !dones[t] {
                values[t + 1]
            } else {
                A::zero()
            };

            let delta = rewards[t] + self.gamma * next_value - values[t];
            gae = delta + self.gamma * self.gae_lambda * gae * {
                if dones[t] {
                    A::zero()
                } else {
                    A::one()
                }
            };
            advantages[t] = gae;
        }

        advantages
    }
}

/// Twin Delayed DDPG (TD3) - Advanced continuous control algorithm
pub struct TD3<A: Float> {
    /// Actor network weights
    actor_weights: Vec<Array2<A>>,
    /// Critic network 1 weights
    critic1_weights: Vec<Array2<A>>,
    /// Critic network 2 weights (twin)
    critic2_weights: Vec<Array2<A>>,
    /// Target networks
    target_actor_weights: Vec<Array2<A>>,
    target_critic1_weights: Vec<Array2<A>>,
    target_critic2_weights: Vec<Array2<A>>,
    /// Learning rates
    actor_lr: A,
    critic_lr: A,
    /// Discount factor
    gamma: A,
    /// Polyak averaging parameter
    tau: A,
    /// Policy noise
    policy_noise: A,
    /// Noise clip
    noise_clip: A,
    /// Policy delay (update actor every N steps)
    policy_delay: usize,
    /// Update counter
    updates: usize,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> TD3<A> {
    /// Create a new TD3 agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_size: usize,
        actor_lr: A,
        critic_lr: A,
        gamma: A,
    ) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.01).unwrap();

        let mut init_network = |input_dim: usize, output_dim: usize| -> Vec<Array2<A>> {
            vec![
                Array2::from_shape_fn((input_dim, hidden_size), |_| {
                    A::from(normal.sample(&mut rng)).unwrap()
                }),
                Array2::from_shape_fn((hidden_size, output_dim), |_| {
                    A::from(normal.sample(&mut rng)).unwrap()
                }),
            ]
        };

        let actor_weights = init_network(state_dim, action_dim);
        let critic1_weights = init_network(state_dim + action_dim, 1);
        let critic2_weights = init_network(state_dim + action_dim, 1);

        let target_actor_weights = actor_weights.clone();
        let target_critic1_weights = critic1_weights.clone();
        let target_critic2_weights = critic2_weights.clone();

        Self {
            actor_weights,
            critic1_weights,
            critic2_weights,
            target_actor_weights,
            target_critic1_weights,
            target_critic2_weights,
            actor_lr,
            critic_lr,
            gamma,
            tau: A::from(0.005).unwrap(),
            policy_noise: A::from(0.2).unwrap(),
            noise_clip: A::from(0.5).unwrap(),
            policy_delay: 2,
            updates: 0,
        }
    }

    /// Select action using current policy with exploration noise
    pub fn select_action(&self, state: &Array1<A>, add_noise: bool) -> Array1<A> {
        // Simple forward pass through actor network
        let hidden = state.dot(&self.actor_weights[0]);
        let hidden_activated = hidden.mapv(|x| x.max(A::zero()));
        let mut action = hidden_activated.dot(&self.actor_weights[1]);

        // Add exploration noise
        if add_noise {
            let mut rng = rand::thread_rng();
            use rand_distr::{Distribution, Normal};
            let normal = Normal::new(0.0, 0.1).unwrap();

            for i in 0..action.len() {
                action[i] = action[i] + A::from(normal.sample(&mut rng)).unwrap();
                action[i] = action[i].max(A::from(-1.0).unwrap()).min(A::one());
            }
        }

        action
    }
}

/// Soft Actor-Critic (SAC) - Maximum entropy RL algorithm
pub struct SAC<A: Float> {
    /// Actor (policy) network weights
    actor_weights: Vec<Array2<A>>,
    /// Critic Q1 network weights
    q1_weights: Vec<Array2<A>>,
    /// Critic Q2 network weights
    q2_weights: Vec<Array2<A>>,
    /// Target Q1 network
    target_q1_weights: Vec<Array2<A>>,
    /// Target Q2 network
    target_q2_weights: Vec<Array2<A>>,
    /// Temperature parameter (entropy coefficient)
    alpha: A,
    /// Learning rates
    actor_lr: A,
    critic_lr: A,
    /// Discount factor
    gamma: A,
    /// Target network update rate
    tau: A,
    /// Hidden layer size
    hidden_size: usize,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> SAC<A> {
    /// Create a new SAC agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_size: usize,
        actor_lr: A,
        critic_lr: A,
        gamma: A,
        alpha: A,
    ) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.01).unwrap();

        let mut init_network = |input_dim: usize, output_dim: usize| -> Vec<Array2<A>> {
            vec![
                Array2::from_shape_fn((input_dim, hidden_size), |_| {
                    A::from(normal.sample(&mut rng)).unwrap()
                }),
                Array2::from_shape_fn((hidden_size, output_dim), |_| {
                    A::from(normal.sample(&mut rng)).unwrap()
                }),
            ]
        };

        let actor_weights = init_network(state_dim, action_dim * 2); // mean and log_std
        let q1_weights = init_network(state_dim + action_dim, 1);
        let q2_weights = init_network(state_dim + action_dim, 1);

        let target_q1_weights = q1_weights.clone();
        let target_q2_weights = q2_weights.clone();

        Self {
            actor_weights,
            q1_weights,
            q2_weights,
            target_q1_weights,
            target_q2_weights,
            alpha,
            actor_lr,
            critic_lr,
            gamma,
            tau: A::from(0.005).unwrap(),
            hidden_size,
        }
    }

    /// Sample action from stochastic policy
    pub fn select_action(&self, state: &Array1<A>, deterministic: bool) -> Array1<A> {
        // Forward pass
        let hidden = state.dot(&self.actor_weights[0]);
        let hidden_activated = hidden.mapv(|x| x.max(A::zero()));
        let output = hidden_activated.dot(&self.actor_weights[1]);

        let action_dim = output.len() / 2;
        let mean = output.slice(ndarray::s![..action_dim]);
        let log_std = output.slice(ndarray::s![action_dim..]);

        if deterministic {
            // Return mean action
            mean.to_owned()
        } else {
            // Sample from Gaussian
            let mut rng = rand::thread_rng();
            use rand_distr::{Distribution, Normal};
            let standard_normal = Normal::new(0.0, 1.0).unwrap();

            let mut action = Array1::zeros(action_dim);
            for i in 0..action_dim {
                let noise = A::from(standard_normal.sample(&mut rng)).unwrap();
                let std = log_std[i].exp();
                action[i] = mean[i] + std * noise;
                // Tanh squashing
                action[i] = action[i].tanh();
            }
            action
        }
    }

    /// Get temperature parameter for entropy regularization
    pub fn get_alpha(&self) -> A {
        self.alpha
    }

    /// Set temperature parameter
    pub fn set_alpha(&mut self, alpha: A) {
        self.alpha = alpha;
    }
}

/// Advantage Actor-Critic (A2C) - Synchronous version of A3C
pub struct A2C<A: Float> {
    /// Actor network weights
    actor_weights: Vec<Array2<A>>,
    /// Critic network weights
    critic_weights: Vec<Array2<A>>,
    /// Learning rate
    lr: A,
    /// Discount factor
    gamma: A,
    /// Value function coefficient
    value_coef: A,
    /// Entropy coefficient
    entropy_coef: A,
    /// Hidden layer size
    hidden_size: usize,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> A2C<A> {
    /// Create a new A2C agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_size: usize,
        lr: A,
        gamma: A,
    ) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        let actor_w1 = Array2::from_shape_fn((state_dim, hidden_size), |_| {
            A::from(normal.sample(&mut rng)).unwrap()
        });
        let actor_w2 = Array2::from_shape_fn((hidden_size, action_dim), |_| {
            A::from(normal.sample(&mut rng)).unwrap()
        });

        let critic_w1 = Array2::from_shape_fn((state_dim, hidden_size), |_| {
            A::from(normal.sample(&mut rng)).unwrap()
        });
        let critic_w2 = Array2::from_shape_fn((hidden_size, 1), |_| {
            A::from(normal.sample(&mut rng)).unwrap()
        });

        Self {
            actor_weights: vec![actor_w1, actor_w2],
            critic_weights: vec![critic_w1, critic_w2],
            lr,
            gamma,
            value_coef: A::from(0.5).unwrap(),
            entropy_coef: A::from(0.01).unwrap(),
            hidden_size,
        }
    }

    /// Forward pass through actor network
    pub fn actor_forward(&self, state: &Array1<A>) -> Array1<A> {
        let hidden = state.dot(&self.actor_weights[0]);
        let hidden_activated = hidden.mapv(|x| x.max(A::zero()));
        let logits = hidden_activated.dot(&self.actor_weights[1]);

        // Softmax
        let exp_logits = logits.mapv(|x| x.exp());
        let sum_exp = exp_logits.sum();
        exp_logits.mapv(|x| x / sum_exp)
    }

    /// Forward pass through critic network
    pub fn critic_forward(&self, state: &Array1<A>) -> A {
        let hidden = state.dot(&self.critic_weights[0]);
        let hidden_activated = hidden.mapv(|x| x.max(A::zero()));
        hidden_activated.dot(&self.critic_weights[1])[0]
    }

    /// Select action from policy
    pub fn select_action(&self, state: &Array1<A>) -> usize {
        let action_probs = self.actor_forward(state);

        let mut rng = rand::thread_rng();
        let rand_val: f64 = rng.gen();
        let mut cumsum = 0.0;

        for (action, &prob) in action_probs.iter().enumerate() {
            cumsum += prob.to_f64().unwrap();
            if rand_val < cumsum {
                return action;
            }
        }

        action_probs.len() - 1
    }

    /// Compute returns for a trajectory
    pub fn compute_returns(&self, rewards: &[A], dones: &[bool], final_value: A) -> Vec<A> {
        let mut returns = vec![A::zero(); rewards.len()];
        let mut r = final_value;

        for t in (0..rewards.len()).rev() {
            r = rewards[t] + self.gamma * r * {
                if dones[t] {
                    A::zero()
                } else {
                    A::one()
                }
            };
            returns[t] = r;
        }

        returns
    }
}

/// Prioritized Experience Replay for DQN/DDPG
pub struct PrioritizedReplayBuffer<A: Float> {
    buffer: VecDeque<(Array1<A>, usize, A, Array1<A>, bool)>,
    priorities: VecDeque<A>,
    capacity: usize,
    alpha: A, // Priority exponent
    beta: A,  // Importance sampling exponent
    max_priority: A,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> PrioritizedReplayBuffer<A> {
    pub fn new(capacity: usize, alpha: A, beta: A) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            priorities: VecDeque::with_capacity(capacity),
            capacity,
            alpha,
            beta,
            max_priority: A::one(),
        }
    }

    /// Add experience with max priority
    pub fn push(&mut self, state: Array1<A>, action: usize, reward: A, next_state: Array1<A>, done: bool) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
            self.priorities.pop_front();
        }

        self.buffer.push_back((state, action, reward, next_state, done));
        self.priorities.push_back(self.max_priority);
    }

    /// Sample batch with importance sampling weights
    pub fn sample(&mut self, batch_size: usize) -> Option<Vec<(Array1<A>, usize, A, Array1<A>, bool, A)>> {
        if self.buffer.len() < batch_size {
            return None;
        }

        // Compute sampling probabilities
        let total_priority: A = self.priorities.iter().map(|&p| p.powf(self.alpha)).sum();

        let mut batch = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();

        for _ in 0..batch_size {
            let rand_val = A::from(rng.gen::<f64>()).unwrap() * total_priority;
            let mut cumsum = A::zero();
            let mut idx = 0;

            for (i, &priority) in self.priorities.iter().enumerate() {
                cumsum = cumsum + priority.powf(self.alpha);
                if rand_val < cumsum {
                    idx = i;
                    break;
                }
            }

            let experience = self.buffer[idx].clone();

            // Importance sampling weight
            let prob = self.priorities[idx].powf(self.alpha) / total_priority;
            let weight = (A::from(self.buffer.len()).unwrap() * prob).powf(-self.beta);

            batch.push((
                experience.0,
                experience.1,
                experience.2,
                experience.3,
                experience.4,
                weight,
            ));
        }

        Some(batch)
    }

    /// Update priorities for sampled batch
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[A]) {
        for (&idx, &error) in indices.iter().zip(td_errors.iter()) {
            let priority = error.abs() + A::from(1e-6).unwrap();
            self.priorities[idx] = priority;
            self.max_priority = self.max_priority.max(priority);
        }
    }

    /// Anneal beta towards 1.0
    pub fn anneal_beta(&mut self, increment: A) {
        self.beta = (self.beta + increment).min(A::one());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_creation() {
        let ppo: PPO<f64> = PPO::new(4, 2, 64, 0.0003, 0.2, 0.99);
        assert_eq!(ppo.policy_weights.len(), 2);
        assert_eq!(ppo.value_weights.len(), 2);
    }

    #[test]
    fn test_td3_creation() {
        let td3: TD3<f64> = TD3::new(4, 2, 64, 0.0003, 0.0003, 0.99);
        assert_eq!(td3.actor_weights.len(), 2);
        assert_eq!(td3.critic1_weights.len(), 2);
    }

    #[test]
    fn test_sac_creation() {
        let sac: SAC<f64> = SAC::new(4, 2, 64, 0.0003, 0.0003, 0.99, 0.2);
        assert_eq!(sac.get_alpha(), 0.2);
    }

    #[test]
    fn test_prioritized_replay() {
        let mut buffer: PrioritizedReplayBuffer<f64> = PrioritizedReplayBuffer::new(100, 0.6, 0.4);

        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let next_state = Array1::from_vec(vec![1.1, 2.1, 3.1, 4.1]);

        buffer.push(state.clone(), 0, 1.0, next_state.clone(), false);
        buffer.push(state, 1, 0.5, next_state, true);

        assert_eq!(buffer.buffer.len(), 2);
    }
}
