//! Learning Rate Schedulers
//!
//! Advanced learning rate scheduling strategies for better training:
//! - Cosine Annealing (with warm restarts)
//! - OneCycle Policy (super-convergence)
//! - Reduce on Plateau (adaptive reduction)
//! - Step Decay
//! - Exponential Decay
//! - Polynomial Decay
//! - Warmup Schedulers

use num_traits::Float;
use std::f64::consts::PI;

/// Learning rate scheduler trait
pub trait LRScheduler<A: Float> {
    /// Get current learning rate
    fn get_lr(&self) -> A;

    /// Step the scheduler (call after each epoch or batch)
    fn step(&mut self);

    /// Step with metric (for ReduceOnPlateau)
    fn step_with_metric(&mut self, metric: A) {
        let _ = metric;
        self.step();
    }
}

/// Cosine Annealing scheduler
///
/// Decreases learning rate following a cosine curve.
/// Can optionally use warm restarts (SGDR).
pub struct CosineAnnealingLR<A: Float> {
    /// Initial learning rate
    base_lr: A,
    /// Minimum learning rate
    eta_min: A,
    /// Period of cosine cycle
    T_max: usize,
    /// Current step
    current_step: usize,
    /// Whether to use warm restarts
    warm_restarts: bool,
    /// Multiplication factor for period after restart
    T_mult: usize,
    /// Current period
    current_T_max: usize,
    /// Current learning rate
    current_lr: A,
}

impl<A: Float> CosineAnnealingLR<A> {
    pub fn new(base_lr: A, T_max: usize, eta_min: Option<A>, warm_restarts: Option<bool>) -> Self {
        let eta_min = eta_min.unwrap_or(A::zero());
        let warm_restarts = warm_restarts.unwrap_or(false);

        Self {
            base_lr,
            eta_min,
            T_max,
            current_step: 0,
            warm_restarts,
            T_mult: 2,
            current_T_max: T_max,
            current_lr: base_lr,
        }
    }
}

impl<A: Float> LRScheduler<A> for CosineAnnealingLR<A> {
    fn get_lr(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_step += 1;

        if self.warm_restarts && self.current_step >= self.current_T_max {
            // Restart
            self.current_step = 0;
            self.current_T_max *= self.T_mult;
        }

        let t = A::from(self.current_step).unwrap();
        let t_max = A::from(self.current_T_max).unwrap();
        let pi = A::from(PI).unwrap();

        // Cosine annealing formula
        self.current_lr = self.eta_min +
            (self.base_lr - self.eta_min) *
            (A::one() + (pi * t / t_max).cos()) /
            A::from(2.0).unwrap();
    }
}

/// OneCycle learning rate policy
///
/// Implements the 1cycle policy from "Super-Convergence" paper.
/// Cycles learning rate from low to high and back to low.
pub struct OneCycleLR<A: Float> {
    /// Maximum learning rate
    max_lr: A,
    /// Total number of steps
    total_steps: usize,
    /// Current step
    current_step: usize,
    /// Percentage of cycle spent increasing LR
    pct_start: f64,
    /// Division factor for initial LR
    div_factor: f64,
    /// Division factor for final LR
    final_div_factor: f64,
    /// Current learning rate
    current_lr: A,
}

impl<A: Float> OneCycleLR<A> {
    pub fn new(
        max_lr: A,
        total_steps: usize,
        pct_start: Option<f64>,
        div_factor: Option<f64>,
        final_div_factor: Option<f64>,
    ) -> Self {
        let pct_start = pct_start.unwrap_or(0.3);
        let div_factor = div_factor.unwrap_or(25.0);
        let final_div_factor = final_div_factor.unwrap_or(10000.0);

        let initial_lr = max_lr / A::from(div_factor).unwrap();

        Self {
            max_lr,
            total_steps,
            current_step: 0,
            pct_start,
            div_factor,
            final_div_factor,
            current_lr: initial_lr,
        }
    }
}

impl<A: Float> LRScheduler<A> for OneCycleLR<A> {
    fn get_lr(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) {
        let step_num = self.current_step as f64;
        let total = self.total_steps as f64;
        let step_size_up = self.pct_start * total;
        let step_size_down = total - step_size_up;

        if step_num < step_size_up {
            // Increasing phase
            let pct = step_num / step_size_up;
            let initial_lr = self.max_lr / A::from(self.div_factor).unwrap();
            self.current_lr = initial_lr + (self.max_lr - initial_lr) * A::from(pct).unwrap();
        } else {
            // Decreasing phase
            let pct = (step_num - step_size_up) / step_size_down;
            let final_lr = self.max_lr / A::from(self.final_div_factor).unwrap();
            self.current_lr = self.max_lr - (self.max_lr - final_lr) * A::from(pct).unwrap();
        }

        self.current_step += 1;
    }
}

/// Reduce learning rate on plateau
///
/// Reduces LR when a metric has stopped improving.
pub struct ReduceLROnPlateau<A: Float> {
    /// Current learning rate
    current_lr: A,
    /// Factor to reduce LR by
    factor: f64,
    /// Number of epochs with no improvement before reducing
    patience: usize,
    /// Minimum LR
    min_lr: A,
    /// Mode: "min" or "max"
    mode: PlateauMode,
    /// Threshold for measuring improvement
    threshold: f64,
    /// Cooldown period after LR reduction
    cooldown: usize,

    // Internal state
    best_metric: A,
    num_bad_epochs: usize,
    cooldown_counter: usize,
}

#[derive(Clone, Copy)]
pub enum PlateauMode {
    Min,
    Max,
}

impl<A: Float> ReduceLROnPlateau<A> {
    pub fn new(
        initial_lr: A,
        mode: PlateauMode,
        factor: Option<f64>,
        patience: Option<usize>,
        threshold: Option<f64>,
        min_lr: Option<A>,
        cooldown: Option<usize>,
    ) -> Self {
        let best_metric = match mode {
            PlateauMode::Min => A::infinity(),
            PlateauMode::Max => A::neg_infinity(),
        };

        Self {
            current_lr: initial_lr,
            factor: factor.unwrap_or(0.1),
            patience: patience.unwrap_or(10),
            min_lr: min_lr.unwrap_or(A::zero()),
            mode,
            threshold: threshold.unwrap_or(1e-4),
            cooldown: cooldown.unwrap_or(0),
            best_metric,
            num_bad_epochs: 0,
            cooldown_counter: 0,
        }
    }
}

impl<A: Float> LRScheduler<A> for ReduceLROnPlateau<A> {
    fn get_lr(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) {
        // This scheduler needs metric, so do nothing here
    }

    fn step_with_metric(&mut self, metric: A) {
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            return;
        }

        let improved = match self.mode {
            PlateauMode::Min => {
                metric < self.best_metric - A::from(self.threshold).unwrap()
            }
            PlateauMode::Max => {
                metric > self.best_metric + A::from(self.threshold).unwrap()
            }
        };

        if improved {
            self.best_metric = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;

            if self.num_bad_epochs >= self.patience {
                // Reduce learning rate
                let new_lr = self.current_lr * A::from(self.factor).unwrap();
                self.current_lr = new_lr.max(self.min_lr);
                self.num_bad_epochs = 0;
                self.cooldown_counter = self.cooldown;
            }
        }
    }
}

/// Step decay scheduler
///
/// Decreases LR by a factor every N steps.
pub struct StepLR<A: Float> {
    /// Base learning rate
    base_lr: A,
    /// Step size (epochs)
    step_size: usize,
    /// Multiplicative factor
    gamma: f64,
    /// Current epoch
    current_epoch: usize,
    /// Current LR
    current_lr: A,
}

impl<A: Float> StepLR<A> {
    pub fn new(base_lr: A, step_size: usize, gamma: Option<f64>) -> Self {
        Self {
            base_lr,
            step_size,
            gamma: gamma.unwrap_or(0.1),
            current_epoch: 0,
            current_lr: base_lr,
        }
    }
}

impl<A: Float> LRScheduler<A> for StepLR<A> {
    fn get_lr(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_epoch += 1;
        let num_reductions = self.current_epoch / self.step_size;
        self.current_lr = self.base_lr * A::from(self.gamma.powi(num_reductions as i32)).unwrap();
    }
}

/// Exponential decay scheduler
pub struct ExponentialLR<A: Float> {
    /// Base learning rate
    base_lr: A,
    /// Decay factor
    gamma: f64,
    /// Current epoch
    current_epoch: usize,
    /// Current LR
    current_lr: A,
}

impl<A: Float> ExponentialLR<A> {
    pub fn new(base_lr: A, gamma: f64) -> Self {
        Self {
            base_lr,
            gamma,
            current_epoch: 0,
            current_lr: base_lr,
        }
    }
}

impl<A: Float> LRScheduler<A> for ExponentialLR<A> {
    fn get_lr(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_epoch += 1;
        self.current_lr = self.base_lr * A::from(self.gamma.powi(self.current_epoch as i32)).unwrap();
    }
}

/// Polynomial learning rate decay
pub struct PolynomialLR<A: Float> {
    /// Initial LR
    base_lr: A,
    /// Final LR
    end_lr: A,
    /// Total steps
    total_steps: usize,
    /// Power
    power: f64,
    /// Current step
    current_step: usize,
    /// Current LR
    current_lr: A,
}

impl<A: Float> PolynomialLR<A> {
    pub fn new(base_lr: A, end_lr: A, total_steps: usize, power: Option<f64>) -> Self {
        Self {
            base_lr,
            end_lr,
            total_steps,
            power: power.unwrap_or(1.0),
            current_step: 0,
            current_lr: base_lr,
        }
    }
}

impl<A: Float> LRScheduler<A> for PolynomialLR<A> {
    fn get_lr(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) {
        self.current_step += 1;
        let decay_steps = self.total_steps.min(self.current_step);
        let pct_remaining = 1.0 - (decay_steps as f64 / self.total_steps as f64);
        let decay = pct_remaining.powf(self.power);

        self.current_lr = (self.base_lr - self.end_lr) * A::from(decay).unwrap() + self.end_lr;
    }
}

/// Linear warmup scheduler wrapper
///
/// Wraps another scheduler and adds linear warmup.
pub struct WarmupScheduler<A: Float, S: LRScheduler<A>> {
    /// Base scheduler
    base_scheduler: S,
    /// Warmup steps
    warmup_steps: usize,
    /// Current step
    current_step: usize,
    /// Initial LR (start of warmup)
    initial_lr: A,
    /// Target LR (end of warmup)
    target_lr: A,
}

impl<A: Float, S: LRScheduler<A>> WarmupScheduler<A, S> {
    pub fn new(base_scheduler: S, warmup_steps: usize, initial_lr: A, target_lr: A) -> Self {
        Self {
            base_scheduler,
            warmup_steps,
            current_step: 0,
            initial_lr,
            target_lr,
        }
    }
}

impl<A: Float, S: LRScheduler<A>> LRScheduler<A> for WarmupScheduler<A, S> {
    fn get_lr(&self) -> A {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            let pct = A::from(self.current_step as f64 / self.warmup_steps as f64).unwrap();
            self.initial_lr + (self.target_lr - self.initial_lr) * pct
        } else {
            self.base_scheduler.get_lr()
        }
    }

    fn step(&mut self) {
        self.current_step += 1;
        if self.current_step > self.warmup_steps {
            self.base_scheduler.step();
        }
    }

    fn step_with_metric(&mut self, metric: A) {
        self.current_step += 1;
        if self.current_step > self.warmup_steps {
            self.base_scheduler.step_with_metric(metric);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_annealing() {
        let mut scheduler = CosineAnnealingLR::new(0.1f64, 100, Some(0.0), Some(false));
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);

        for _ in 0..50 {
            scheduler.step();
        }
        // At halfway point, should be near minimum
        assert!(scheduler.get_lr() < 0.05);
    }

    #[test]
    fn test_onecycle() {
        let mut scheduler = OneCycleLR::new(0.1f64, 100, None, None, None);
        let initial_lr = scheduler.get_lr();

        for _ in 0..30 {
            scheduler.step();
        }
        // Should have increased
        assert!(scheduler.get_lr() > initial_lr);
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut scheduler = ReduceLROnPlateau::new(
            0.1f64,
            PlateauMode::Min,
            Some(0.5),
            Some(2),
            None,
            None,
            None,
        );

        let initial_lr = scheduler.get_lr();

        // Simulate no improvement
        for _ in 0..3 {
            scheduler.step_with_metric(1.0);
        }

        // LR should have been reduced
        assert!(scheduler.get_lr() < initial_lr);
    }
}
