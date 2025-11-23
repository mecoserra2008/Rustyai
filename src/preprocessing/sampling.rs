//! Advanced Sampling Techniques
//!
//! Includes SMOTE, ADASYN, importance sampling, reservoir sampling,
//! and other advanced resampling methods for imbalanced datasets.

use ndarray::{Array1, Array2, Axis, s};
use num_traits::Float;
use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use crate::error::{Result, RustyAIError};

/// SMOTE (Synthetic Minority Over-sampling Technique)
///
/// Generates synthetic samples for minority classes using k-nearest neighbors
pub struct SMOTE {
    k_neighbors: usize,
    sampling_strategy: SamplingStrategy,
}

#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Oversample all minority classes to match majority class
    Auto,
    /// Oversample all classes to specific size
    AllToSize(usize),
    /// Custom ratios for each class
    Custom(HashMap<usize, usize>),
}

impl SMOTE {
    /// Create a new SMOTE sampler
    pub fn new(k_neighbors: usize) -> Self {
        Self {
            k_neighbors,
            sampling_strategy: SamplingStrategy::Auto,
        }
    }

    /// Set sampling strategy
    pub fn with_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Fit and resample the dataset
    pub fn fit_resample<A: Float + Copy>(
        &self,
        X: &Array2<A>,
        y: &Array1<usize>,
    ) -> Result<(Array2<A>, Array1<usize>)> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Count samples per class
        let mut class_counts: HashMap<usize, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        // Determine target counts based on strategy
        let target_counts = match &self.sampling_strategy {
            SamplingStrategy::Auto => {
                let max_count = *class_counts.values().max().unwrap();
                class_counts.keys().map(|&k| (k, max_count)).collect()
            }
            SamplingStrategy::AllToSize(size) => {
                class_counts.keys().map(|&k| (k, *size)).collect()
            }
            SamplingStrategy::Custom(custom) => custom.clone(),
        };

        // Separate samples by class
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_insert_with(Vec::new).push(idx);
        }

        // Generate synthetic samples
        let mut X_resampled = X.to_owned();
        let mut y_resampled = y.to_owned();

        for (&class_label, &target_count) in &target_counts {
            let current_count = class_counts.get(&class_label).copied().unwrap_or(0);
            if target_count <= current_count {
                continue; // No oversampling needed
            }

            let n_synthetic = target_count - current_count;
            let class_idx = class_indices.get(&class_label).unwrap();

            // Extract minority class samples
            let mut minority_samples = Array2::zeros((class_idx.len(), n_features));
            for (i, &idx) in class_idx.iter().enumerate() {
                for j in 0..n_features {
                    minority_samples[[i, j]] = X[[idx, j]];
                }
            }

            // Generate synthetic samples
            let synthetic = self.generate_synthetic_samples(
                &minority_samples,
                n_synthetic,
            )?;

            // Append to resampled data
            X_resampled = concatenate_rows(&X_resampled, &synthetic);
            let synthetic_labels = Array1::from_elem(n_synthetic, class_label);
            y_resampled = concatenate_1d_usize(&y_resampled, &synthetic_labels);
        }

        Ok((X_resampled, y_resampled))
    }

    fn generate_synthetic_samples<A: Float + Copy>(
        &self,
        X: &Array2<A>,
        n_samples: usize,
    ) -> Result<Array2<A>> {
        let mut rng = rand::thread_rng();
        let n_minority = X.nrows();
        let n_features = X.ncols();
        let mut synthetic = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            // Randomly select a minority sample
            let idx = rng.gen_range(0..n_minority);
            let sample = X.row(idx);

            // Find k nearest neighbors
            let neighbors = self.find_k_neighbors(X, idx, self.k_neighbors);

            // Randomly select one neighbor
            let neighbor_idx = neighbors[rng.gen_range(0..neighbors.len())];
            let neighbor = X.row(neighbor_idx);

            // Generate synthetic sample between sample and neighbor
            let alpha = A::from(rng.gen::<f64>()).unwrap();
            for j in 0..n_features {
                synthetic[[i, j]] = sample[j] + alpha * (neighbor[j] - sample[j]);
            }
        }

        Ok(synthetic)
    }

    fn find_k_neighbors<A: Float + Copy>(
        &self,
        X: &Array2<A>,
        idx: usize,
        k: usize,
    ) -> Vec<usize> {
        let sample = X.row(idx);
        let n_samples = X.nrows();

        // Compute distances to all other samples
        let mut distances: Vec<(usize, A)> = Vec::new();
        for i in 0..n_samples {
            if i == idx {
                continue;
            }

            let dist = euclidean_distance(&sample, &X.row(i));
            distances.push((i, dist));
        }

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.iter().take(k.min(distances.len())).map(|&(i, _)| i).collect()
    }
}

/// ADASYN (Adaptive Synthetic Sampling)
///
/// Adaptively generates more synthetic samples for harder-to-learn minority samples
pub struct ADASYN {
    k_neighbors: usize,
    beta: f64,
}

impl ADASYN {
    pub fn new(k_neighbors: usize) -> Self {
        Self {
            k_neighbors,
            beta: 1.0,
        }
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    pub fn fit_resample<A: Float + Copy>(
        &self,
        X: &Array2<A>,
        y: &Array1<usize>,
    ) -> Result<(Array2<A>, Array1<usize>)> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Find minority and majority classes
        let mut class_counts: HashMap<usize, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let minority_class = *class_counts.iter().min_by_key(|(_, &count)| count).unwrap().0;
        let majority_class = *class_counts.iter().max_by_key(|(_, &count)| count).unwrap().0;

        let n_minority = class_counts[&minority_class];
        let n_majority = class_counts[&majority_class];

        // Calculate number of synthetic samples to generate
        let n_synthetic = ((n_majority - n_minority) as f64 * self.beta) as usize;

        // Find minority class indices
        let minority_indices: Vec<usize> = y.iter()
            .enumerate()
            .filter(|(_, &label)| label == minority_class)
            .map(|(idx, _)| idx)
            .collect();

        // Calculate density distribution (ratio of majority neighbors)
        let mut densities = Vec::new();
        for &idx in &minority_indices {
            let neighbors = self.find_k_neighbors(X, y, idx, self.k_neighbors);
            let majority_neighbors = neighbors.iter()
                .filter(|&&n_idx| y[n_idx] == majority_class)
                .count();
            densities.push(majority_neighbors as f64 / self.k_neighbors as f64);
        }

        // Normalize densities
        let density_sum: f64 = densities.iter().sum();
        let normalized_densities: Vec<f64> = if density_sum > 0.0 {
            densities.iter().map(|&d| d / density_sum).collect()
        } else {
            vec![1.0 / densities.len() as f64; densities.len()]
        };

        // Generate synthetic samples based on density distribution
        let mut rng = rand::thread_rng();
        let mut X_resampled = X.to_owned();
        let mut y_resampled = y.to_owned();

        for _ in 0..n_synthetic {
            // Sample minority instance based on density
            let idx = sample_weighted(&normalized_densities, &mut rng);
            let sample_idx = minority_indices[idx];
            let sample = X.row(sample_idx);

            // Find minority neighbors
            let neighbors = self.find_k_neighbors(X, y, sample_idx, self.k_neighbors);
            let minority_neighbors: Vec<usize> = neighbors.iter()
                .filter(|&&n_idx| y[n_idx] == minority_class)
                .copied()
                .collect();

            if minority_neighbors.is_empty() {
                continue;
            }

            // Generate synthetic sample
            let neighbor_idx = minority_neighbors[rng.gen_range(0..minority_neighbors.len())];
            let neighbor = X.row(neighbor_idx);

            let alpha = A::from(rng.gen::<f64>()).unwrap();
            let mut synthetic_sample = Array1::zeros(n_features);
            for j in 0..n_features {
                synthetic_sample[j] = sample[j] + alpha * (neighbor[j] - sample[j]);
            }

            X_resampled = concatenate_rows(&X_resampled, &synthetic_sample.insert_axis(Axis(0)));
            y_resampled = concatenate_1d_usize(&y_resampled, &Array1::from_elem(1, minority_class));
        }

        Ok((X_resampled, y_resampled))
    }

    fn find_k_neighbors<A: Float + Copy>(
        &self,
        X: &Array2<A>,
        y: &Array1<usize>,
        idx: usize,
        k: usize,
    ) -> Vec<usize> {
        let sample = X.row(idx);
        let n_samples = X.nrows();

        let mut distances: Vec<(usize, A)> = Vec::new();
        for i in 0..n_samples {
            if i == idx {
                continue;
            }
            let dist = euclidean_distance(&sample, &X.row(i));
            distances.push((i, dist));
        }

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.iter().take(k.min(distances.len())).map(|&(i, _)| i).collect()
    }
}

/// Importance Sampling
///
/// Samples from a distribution with importance weights
pub struct ImportanceSampling {
    n_samples: usize,
}

impl ImportanceSampling {
    pub fn new(n_samples: usize) -> Self {
        Self { n_samples }
    }

    /// Sample indices with replacement based on weights
    pub fn sample<A: Float + Copy>(
        &self,
        weights: &Array1<A>,
    ) -> Vec<usize> {
        let mut rng = rand::thread_rng();
        let n = weights.len();

        // Normalize weights
        let weight_sum: A = weights.sum();
        let normalized: Vec<f64> = weights.iter()
            .map(|&w| (w / weight_sum).to_f64().unwrap())
            .collect();

        let mut samples = Vec::with_capacity(self.n_samples);
        for _ in 0..self.n_samples {
            let idx = sample_weighted(&normalized, &mut rng);
            samples.push(idx);
        }

        samples
    }

    /// Compute importance sampling weights for reweighting
    pub fn compute_weights<A: Float + Copy>(
        &self,
        target_probs: &Array1<A>,
        proposal_probs: &Array1<A>,
    ) -> Array1<A> {
        let mut weights = Array1::zeros(target_probs.len());
        for i in 0..target_probs.len() {
            weights[i] = target_probs[i] / proposal_probs[i];
        }
        weights
    }
}

/// Reservoir Sampling
///
/// Efficiently samples k items from a stream of unknown size
pub struct ReservoirSampling {
    k: usize,
}

impl ReservoirSampling {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Sample k items from iterator
    pub fn sample<A: Clone, I: Iterator<Item = A>>(&self, stream: I) -> Vec<A> {
        let mut rng = rand::thread_rng();
        let mut reservoir: Vec<A> = Vec::with_capacity(self.k);

        for (i, item) in stream.enumerate() {
            if i < self.k {
                reservoir.push(item);
            } else {
                let j = rng.gen_range(0..=i);
                if j < self.k {
                    reservoir[j] = item;
                }
            }
        }

        reservoir
    }

    /// Sample from array
    pub fn sample_array<A: Float + Copy>(&self, X: &Array2<A>) -> Array2<A> {
        let mut rng = rand::thread_rng();
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if self.k >= n_samples {
            return X.to_owned();
        }

        let mut reservoir = Array2::zeros((self.k, n_features));

        // Fill reservoir with first k samples
        for i in 0..self.k {
            for j in 0..n_features {
                reservoir[[i, j]] = X[[i, j]];
            }
        }

        // Replace elements with decreasing probability
        for i in self.k..n_samples {
            let j = rng.gen_range(0..=i);
            if j < self.k {
                for feat in 0..n_features {
                    reservoir[[j, feat]] = X[[i, feat]];
                }
            }
        }

        reservoir
    }
}

/// Stratified Sampling
///
/// Ensures proportional representation of different strata
pub struct StratifiedSampling {
    n_samples: usize,
}

impl StratifiedSampling {
    pub fn new(n_samples: usize) -> Self {
        Self { n_samples }
    }

    /// Stratified sample maintaining class proportions
    pub fn sample<A: Float + Copy>(
        &self,
        X: &Array2<A>,
        y: &Array1<usize>,
    ) -> Result<(Array2<A>, Array1<usize>)> {
        let mut rng = rand::thread_rng();

        // Count samples per class
        let mut class_counts: HashMap<usize, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        // Calculate samples per class
        let total_samples = y.len();
        let mut samples_per_class: HashMap<usize, usize> = HashMap::new();
        for (&class_label, &count) in &class_counts {
            let n = (self.n_samples as f64 * count as f64 / total_samples as f64).round() as usize;
            samples_per_class.insert(class_label, n.max(1));
        }

        // Separate indices by class
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_insert_with(Vec::new).push(idx);
        }

        // Sample from each class
        let mut selected_indices = Vec::new();
        for (&class_label, indices) in &class_indices {
            let n_to_sample = *samples_per_class.get(&class_label).unwrap_or(&0);
            let mut class_idx = indices.clone();
            class_idx.shuffle(&mut rng);
            selected_indices.extend(&class_idx[..n_to_sample.min(indices.len())]);
        }

        // Build sampled arrays
        let n_features = X.ncols();
        let actual_samples = selected_indices.len();
        let mut X_sampled = Array2::zeros((actual_samples, n_features));
        let mut y_sampled = Array1::zeros(actual_samples);

        for (i, &idx) in selected_indices.iter().enumerate() {
            for j in 0..n_features {
                X_sampled[[i, j]] = X[[idx, j]];
            }
            y_sampled[i] = y[idx];
        }

        Ok((X_sampled, y_sampled))
    }
}

// Helper functions

fn euclidean_distance<A: Float + Copy>(a: &ndarray::ArrayView1<A>, b: &ndarray::ArrayView1<A>) -> A {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .fold(A::zero(), |acc, x| acc + x)
        .sqrt()
}

fn concatenate_rows<A: Float + Copy>(a: &Array2<A>, b: &Array2<A>) -> Array2<A> {
    let n_rows = a.nrows() + b.nrows();
    let n_cols = a.ncols();
    let mut result = Array2::zeros((n_rows, n_cols));

    for i in 0..a.nrows() {
        for j in 0..n_cols {
            result[[i, j]] = a[[i, j]];
        }
    }

    for i in 0..b.nrows() {
        for j in 0..n_cols {
            result[[a.nrows() + i, j]] = b[[i, j]];
        }
    }

    result
}

fn concatenate_1d_usize(a: &Array1<usize>, b: &Array1<usize>) -> Array1<usize> {
    let n = a.len() + b.len();
    let mut result = Vec::with_capacity(n);
    result.extend(a.iter().copied());
    result.extend(b.iter().copied());
    Array1::from_vec(result)
}

fn concatenate_1d<A: Float + Copy>(a: &Array1<A>, b: &Array1<A>) -> Array1<A> {
    let n = a.len() + b.len();
    let mut result = Array1::zeros(n);

    for i in 0..a.len() {
        result[i] = a[i];
    }

    for i in 0..b.len() {
        result[a.len() + i] = b[i];
    }

    result
}

fn sample_weighted<R: Rng>(weights: &[f64], rng: &mut R) -> usize {
    let total: f64 = weights.iter().sum();
    let mut threshold = rng.gen::<f64>() * total;

    for (i, &weight) in weights.iter().enumerate() {
        threshold -= weight;
        if threshold <= 0.0 {
            return i;
        }
    }

    weights.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smote() {
        let X = Array2::from_shape_vec((6, 2), vec![
            1.0, 1.0,
            1.1, 1.0,
            1.0, 1.1,
            5.0, 5.0,
            5.1, 5.0,
            5.0, 5.1,
        ]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let smote = SMOTE::new(2);
        let (X_res, y_res) = smote.fit_resample(&X, &y).unwrap();

        assert_eq!(X_res.nrows(), y_res.len());
        assert!(X_res.nrows() >= X.nrows());
    }

    #[test]
    fn test_reservoir_sampling() {
        let X = Array2::from_shape_vec((100, 5), (0..500).map(|x| x as f64).collect()).unwrap();
        let sampler = ReservoirSampling::new(20);
        let sampled = sampler.sample_array(&X);

        assert_eq!(sampled.nrows(), 20);
        assert_eq!(sampled.ncols(), 5);
    }
}
