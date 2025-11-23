//! Reinforcement Learning
//!
//! Foundational and Advanced RL algorithms:
//! - Q-Learning (tabular)
//! - Deep Q-Network (DQN)
//! - Experience Replay
//! - Epsilon-Greedy exploration
//! - PPO, TD3, SAC, A2C (advanced algorithms)
//! - Prioritized Experience Replay

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::collections::VecDeque;
use std::iter::Sum;

pub mod advanced;

pub use advanced::{PPO, TD3, SAC, A2C, PrioritizedReplayBuffer};

/// Q-Learning agent (tabular)
pub struct QLearning<A: Float> {
    /// Q-table (states x actions)
    pub q_table: Array2<A>,
    /// Learning rate
    pub alpha: A,
    /// Discount factor
    pub gamma: A,
    /// Exploration rate
    pub epsilon: A,
    /// Number of states
    pub n_states: usize,
    /// Number of actions
    pub n_actions: usize,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive> QLearning<A> {
    pub fn new(n_states: usize, n_actions: usize, alpha: A, gamma: A, epsilon: A) -> Self {
        Self {
            q_table: Array2::zeros((n_states, n_actions)),
            alpha,
            gamma,
            epsilon,
            n_states,
            n_actions,
        }
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: usize) -> usize {
        if rand::random::<f64>() < self.epsilon.to_f64().unwrap() {
            // Explore: random action
            (rand::random::<f64>() * self.n_actions as f64) as usize % self.n_actions
        } else {
            // Exploit: best action
            self.greedy_action(state)
        }
    }

    /// Get greedy action (argmax Q(s,a))
    pub fn greedy_action(&self, state: usize) -> usize {
        let q_values = self.q_table.row(state);
        let mut best_action = 0;
        let mut best_value = q_values[0];

        for (action, &value) in q_values.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        best_action
    }

    /// Update Q-value using Q-learning rule
    pub fn update(
        &mut self,
        state: usize,
        action: usize,
        reward: A,
        next_state: usize,
        done: bool,
    ) {
        let current_q = self.q_table[[state, action]];

        let next_max_q = if done {
            A::zero()
        } else {
            let next_q_values = self.q_table.row(next_state);
            next_q_values.fold(A::neg_infinity(), |acc, &val| acc.max(val))
        };

        // Q(s,a) <- Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        let target = reward + self.gamma * next_max_q;
        self.q_table[[state, action]] = current_q + self.alpha * (target - current_q);
    }

    /// Decay epsilon (for exploration-exploitation tradeoff)
    pub fn decay_epsilon(&mut self, decay_rate: f64) {
        self.epsilon = self.epsilon * A::from(decay_rate).unwrap();
    }
}

/// Experience for replay buffer
#[derive(Clone)]
pub struct Experience<A: Float> {
    pub state: Array1<A>,
    pub action: usize,
    pub reward: A,
    pub next_state: Array1<A>,
    pub done: bool,
}

/// Experience Replay Buffer
pub struct ReplayBuffer<A: Float> {
    buffer: VecDeque<Experience<A>>,
    capacity: usize,
}

impl<A: Float + Clone> ReplayBuffer<A> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add experience to buffer
    pub fn push(&mut self, experience: Experience<A>) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    /// Sample random batch
    pub fn sample(&self, batch_size: usize) -> Vec<Experience<A>> {
        let mut samples = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            if self.buffer.is_empty() {
                break;
            }
            let idx = (rand::random::<f64>() * self.buffer.len() as f64) as usize % self.buffer.len();
            samples.push(self.buffer[idx].clone());
        }

        samples
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Deep Q-Network (DQN) - simplified version
pub struct DQN<A: Float> {
    /// Neural network weights (simplified as single layer)
    pub weights: Array2<A>,
    /// Replay buffer
    pub replay_buffer: ReplayBuffer<A>,
    /// Target network weights (for stable learning)
    pub target_weights: Array2<A>,
    /// Learning rate
    pub alpha: A,
    /// Discount factor
    pub gamma: A,
    /// Epsilon for exploration
    pub epsilon: A,
}

impl<A: Float + ScalarOperand + Sum + FromPrimitive + Clone> DQN<A> {
    pub fn new(
        state_dim: usize,
        n_actions: usize,
        buffer_capacity: usize,
        alpha: A,
        gamma: A,
        epsilon: A,
    ) -> Self {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.01).unwrap();

        let mut init_weight = |rows, cols| -> Array2<A> {
            Array2::from_shape_fn((rows, cols), |_| {
                A::from(normal.sample(&mut rng)).unwrap()
            })
        };

        let weights = init_weight(state_dim, n_actions);
        let target_weights = weights.clone();

        Self {
            weights,
            replay_buffer: ReplayBuffer::new(buffer_capacity),
            target_weights,
            alpha,
            gamma,
            epsilon,
        }
    }

    /// Forward pass to get Q-values
    pub fn q_values(&self, state: &Array1<A>) -> Array1<A> {
        // Simplified: state.dot(weights)
        let mut q_vals = Array1::zeros(self.weights.ncols());
        for j in 0..self.weights.ncols() {
            for i in 0..state.len() {
                q_vals[j] = q_vals[j] + state[i] * self.weights[[i, j]];
            }
        }
        q_vals
    }

    /// Select action using epsilon-greedy
    pub fn select_action(&self, state: &Array1<A>) -> usize {
        if rand::random::<f64>() < self.epsilon.to_f64().unwrap() {
            (rand::random::<f64>() * self.weights.ncols() as f64) as usize % self.weights.ncols()
        } else {
            self.greedy_action(state)
        }
    }

    /// Greedy action selection
    pub fn greedy_action(&self, state: &Array1<A>) -> usize {
        let q_vals = self.q_values(state);
        let mut best_action = 0;
        let mut best_value = q_vals[0];

        for (action, &value) in q_vals.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }

        best_action
    }

    /// Add experience to replay buffer
    pub fn remember(&mut self, experience: Experience<A>) {
        self.replay_buffer.push(experience);
    }

    /// Train on a batch from replay buffer
    pub fn train(&mut self, batch_size: usize) {
        if self.replay_buffer.len() < batch_size {
            return;
        }

        let batch = self.replay_buffer.sample(batch_size);

        for exp in batch {
            let q_values = self.q_values(&exp.state);
            let next_q_values = self.q_values(&exp.next_state);

            let target = if exp.done {
                exp.reward
            } else {
                let max_next_q = next_q_values.fold(A::neg_infinity(), |acc, &val| acc.max(val));
                exp.reward + self.gamma * max_next_q
            };

            // Simplified gradient descent update
            let td_error = target - q_values[exp.action];

            for i in 0..exp.state.len() {
                self.weights[[i, exp.action]] =
                    self.weights[[i, exp.action]] + self.alpha * td_error * exp.state[i];
            }
        }
    }

    /// Update target network
    pub fn update_target_network(&mut self) {
        self.target_weights = self.weights.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q_learning() {
        let mut agent = QLearning::new(10, 4, 0.1, 0.99, 0.1);
        let action = agent.select_action(0);
        assert!(action < 4);

        agent.update(0, action, 1.0, 1, false);
        assert!(agent.q_table[[0, action]] > 0.0);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        let exp = Experience {
            state: Array1::zeros(4),
            action: 0,
            reward: 1.0,
            next_state: Array1::zeros(4),
            done: false,
        };

        buffer.push(exp);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_dqn() {
        let dqn = DQN::new(4, 2, 1000, 0.001, 0.99, 0.1);
        let state = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let action = dqn.select_action(&state);
        assert!(action < 2);
    }
}
