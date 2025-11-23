//! Bootstrap and Resampling Methods
//!
//! Advanced resampling techniques for model evaluation and uncertainty estimation:
//! - Bootstrap aggregating (bagging)
//! - Out-of-bag (OOB) estimation
//! - Stratified bootstrap
//! - Balanced bootstrap
//! - Block bootstrap for time series
//! - Jackknife resampling

use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Bootstrap sampler for generating resampled datasets
pub struct Bootstrap {
    n_iterations: usize,
    sample_size: Option<usize>,
    replace: bool,
}

impl Bootstrap {
    /// Create a new Bootstrap sampler
    pub fn new(n_iterations: usize) -> Self {
        Self {
            n_iterations,
            sample_size: None,
            replace: true,
        }
    }

    /// Set sample size (defaults to original size)
    pub fn with_sample_size(mut self, size: usize) -> Self {
        self.sample_size = Some(size);
        self
    }

    /// Set whether to sample with replacement
    pub fn with_replacement(mut self, replace: bool) -> Self {
        self.replace = replace;
        self
    }

    /// Generate bootstrap samples (indices)
    pub fn generate_indices(&self, n_samples: usize) -> Vec<Vec<usize>> {
        let mut rng = rand::thread_rng();
        let sample_size = self.sample_size.unwrap_or(n_samples);
        let mut samples = Vec::with_capacity(self.n_iterations);

        for _ in 0..self.n_iterations {
            let mut indices = Vec::with_capacity(sample_size);

            if self.replace {
                // Sample with replacement (standard bootstrap)
                for _ in 0..sample_size {
                    indices.push(rng.gen_range(0..n_samples));
                }
            } else {
                // Sample without replacement (subsample)
                let mut all_indices: Vec<usize> = (0..n_samples).collect();
                all_indices.shuffle(&mut rng);
                indices = all_indices[..sample_size].to_vec();
            }

            samples.push(indices);
        }

        samples
    }

    /// Get out-of-bag (OOB) samples for each bootstrap iteration
    pub fn get_oob_indices(&self, n_samples: usize) -> Vec<Vec<usize>> {
        let bootstrap_samples = self.generate_indices(n_samples);
        let mut oob_samples = Vec::with_capacity(self.n_iterations);

        for bootstrap_idx in bootstrap_samples {
            let bootstrap_set: std::collections::HashSet<usize> = bootstrap_idx.into_iter().collect();
            let oob: Vec<usize> = (0..n_samples)
                .filter(|i| !bootstrap_set.contains(i))
                .collect();
            oob_samples.push(oob);
        }

        oob_samples
    }
}

/// Stratified Bootstrap - maintains class distribution
pub struct StratifiedBootstrap {
    n_iterations: usize,
}

impl StratifiedBootstrap {
    pub fn new(n_iterations: usize) -> Self {
        Self { n_iterations }
    }

    /// Generate stratified bootstrap samples preserving class distribution
    pub fn generate_indices(&self, labels: &Array1<usize>) -> Vec<Vec<usize>> {
        let mut rng = rand::thread_rng();
        let n_samples = labels.len();

        // Group indices by class
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &label) in labels.iter().enumerate() {
            class_indices.entry(label).or_insert_with(Vec::new).push(idx);
        }

        let mut samples = Vec::with_capacity(self.n_iterations);

        for _ in 0..self.n_iterations {
            let mut bootstrap_sample = Vec::with_capacity(n_samples);

            // Sample from each class proportionally
            for (_, indices) in &class_indices {
                let n_class_samples = indices.len();
                for _ in 0..n_class_samples {
                    let random_idx = rng.gen_range(0..n_class_samples);
                    bootstrap_sample.push(indices[random_idx]);
                }
            }

            bootstrap_sample.shuffle(&mut rng);
            samples.push(bootstrap_sample);
        }

        samples
    }
}

/// Block Bootstrap for time series data
pub struct BlockBootstrap {
    n_iterations: usize,
    block_length: usize,
}

impl BlockBootstrap {
    /// Create a new block bootstrap sampler
    pub fn new(n_iterations: usize, block_length: usize) -> Self {
        Self {
            n_iterations,
            block_length,
        }
    }

    /// Generate block bootstrap samples for time series
    pub fn generate_indices(&self, n_samples: usize) -> Vec<Vec<usize>> {
        let mut rng = rand::thread_rng();
        let n_blocks = (n_samples as f64 / self.block_length as f64).ceil() as usize;
        let mut samples = Vec::with_capacity(self.n_iterations);

        for _ in 0..self.n_iterations {
            let mut bootstrap_sample = Vec::with_capacity(n_samples);

            for _ in 0..n_blocks {
                // Random block start
                let start = rng.gen_range(0..(n_samples - self.block_length + 1));

                // Add block
                for offset in 0..self.block_length {
                    if bootstrap_sample.len() < n_samples {
                        bootstrap_sample.push(start + offset);
                    }
                }
            }

            bootstrap_sample.truncate(n_samples);
            samples.push(bootstrap_sample);
        }

        samples
    }
}

/// Jackknife resampling - leave-one-out cross-validation
pub struct Jackknife {
    // Jackknife uses all n samples, leaving one out each time
}

impl Jackknife {
    pub fn new() -> Self {
        Self {}
    }

    /// Generate jackknife samples (leave-one-out)
    pub fn generate_indices(&self, n_samples: usize) -> Vec<Vec<usize>> {
        let mut samples = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let sample: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
            samples.push(sample);
        }

        samples
    }

    /// Get the left-out indices for each jackknife sample
    pub fn get_test_indices(&self, n_samples: usize) -> Vec<usize> {
        (0..n_samples).collect()
    }
}

impl Default for Jackknife {
    fn default() -> Self {
        Self::new()
    }
}

/// Balanced Bootstrap - ensures each sample appears exactly once per iteration
pub struct BalancedBootstrap {
    n_iterations: usize,
}

impl BalancedBootstrap {
    pub fn new(n_iterations: usize) -> Self {
        Self { n_iterations }
    }

    /// Generate balanced bootstrap samples
    pub fn generate_indices(&self, n_samples: usize) -> Vec<Vec<usize>> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(self.n_iterations);

        for _ in 0..self.n_iterations {
            // Create n_samples copies of each index
            let mut pool: Vec<usize> = (0..n_samples)
                .flat_map(|i| std::iter::repeat(i).take(n_samples))
                .collect();

            pool.shuffle(&mut rng);

            // Take first n_samples elements (each index appears exactly once)
            let sample = pool[..n_samples].to_vec();
            samples.push(sample);
        }

        samples
    }
}

/// Bootstrap confidence interval estimation
pub struct BootstrapCI<F, A: Float> {
    n_iterations: usize,
    confidence_level: f64,
    statistic_fn: F,
    _phantom: std::marker::PhantomData<A>,
}

impl<F, A> BootstrapCI<F, A>
where
    F: Fn(&Array1<A>) -> A,
    A: Float + PartialOrd,
{
    /// Create a new bootstrap confidence interval estimator
    pub fn new(n_iterations: usize, confidence_level: f64, statistic_fn: F) -> Self {
        Self {
            n_iterations,
            confidence_level,
            statistic_fn,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute bootstrap confidence interval
    pub fn estimate(&self, data: &Array1<A>) -> (A, A, A) {
        let mut rng = rand::thread_rng();
        let n_samples = data.len();
        let mut statistics = Vec::with_capacity(self.n_iterations);

        // Original statistic
        let original_stat = (self.statistic_fn)(data);

        // Bootstrap iterations
        for _ in 0..self.n_iterations {
            let mut bootstrap_sample = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let idx = rng.gen_range(0..n_samples);
                bootstrap_sample[i] = data[idx];
            }

            let stat = (self.statistic_fn)(&bootstrap_sample);
            statistics.push(stat);
        }

        // Sort statistics
        statistics.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute percentile confidence intervals
        let alpha = (1.0 - self.confidence_level) / 2.0;
        let lower_idx = (self.n_iterations as f64 * alpha) as usize;
        let upper_idx = (self.n_iterations as f64 * (1.0 - alpha)) as usize;

        let lower_bound = statistics[lower_idx];
        let upper_bound = statistics[upper_idx.min(statistics.len() - 1)];

        (original_stat, lower_bound, upper_bound)
    }
}

/// Permutation test for hypothesis testing
pub struct PermutationTest {
    n_permutations: usize,
}

impl PermutationTest {
    pub fn new(n_permutations: usize) -> Self {
        Self { n_permutations }
    }

    /// Perform permutation test to assess statistical significance
    pub fn test<A, F>(&self, group1: &Array1<A>, group2: &Array1<A>, statistic_fn: F) -> f64
    where
        A: Float + Copy,
        F: Fn(&Array1<A>, &Array1<A>) -> A,
    {
        let mut rng = rand::thread_rng();

        // Observed test statistic
        let observed = statistic_fn(group1, group2);

        // Combine groups
        let mut combined = Vec::with_capacity(group1.len() + group2.len());
        combined.extend(group1.iter());
        combined.extend(group2.iter());

        let n1 = group1.len();
        let n_total = combined.len();

        // Permutation test
        let mut count_extreme = 0;

        for _ in 0..self.n_permutations {
            // Shuffle combined data
            combined.shuffle(&mut rng);

            // Split into two groups
            let perm_group1 = Array1::from_vec(combined[..n1].to_vec());
            let perm_group2 = Array1::from_vec(combined[n1..].to_vec());

            let perm_stat = statistic_fn(&perm_group1, &perm_group2);

            // Count how many permutations are as extreme as observed
            if perm_stat.to_f64().unwrap().abs() >= observed.to_f64().unwrap().abs() {
                count_extreme += 1;
            }
        }

        // P-value
        count_extreme as f64 / self.n_permutations as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap() {
        let bootstrap = Bootstrap::new(10);
        let samples = bootstrap.generate_indices(100);
        assert_eq!(samples.len(), 10);
        assert_eq!(samples[0].len(), 100);
    }

    #[test]
    fn test_stratified_bootstrap() {
        let labels = Array1::from_vec(vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 2]);
        let bootstrap = StratifiedBootstrap::new(5);
        let samples = bootstrap.generate_indices(&labels);

        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0].len(), 10);
    }

    #[test]
    fn test_block_bootstrap() {
        let bootstrap = BlockBootstrap::new(5, 10);
        let samples = bootstrap.generate_indices(100);
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0].len(), 100);
    }

    #[test]
    fn test_jackknife() {
        let jackknife = Jackknife::new();
        let samples = jackknife.generate_indices(10);
        assert_eq!(samples.len(), 10);
        assert_eq!(samples[0].len(), 9);
    }

    #[test]
    fn test_bootstrap_ci() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let mean_fn = |x: &Array1<f64>| x.sum() / x.len() as f64;

        let ci = BootstrapCI::new(100, 0.95, mean_fn);
        let (original, lower, upper) = ci.estimate(&data);

        assert!(lower <= original);
        assert!(original <= upper);
    }
}
