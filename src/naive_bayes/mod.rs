//! Naive Bayes classifiers
//!
//! Probabilistic classifiers based on Bayes' theorem with strong independence assumptions.

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use std::collections::HashMap;
use std::iter::Sum;

/// Gaussian Naive Bayes
///
/// Assumes features follow a Gaussian distribution.
pub struct GaussianNB<A: Float> {
    /// Class priors
    class_prior_: Option<HashMap<i32, A>>,
    /// Mean of each feature per class
    theta_: Option<HashMap<i32, Array1<A>>>,
    /// Variance of each feature per class
    sigma_: Option<HashMap<i32, Array1<A>>>,
    /// Classes seen during fit
    classes_: Option<Vec<i32>>,
}

impl<A: Float + ScalarOperand + Sum> GaussianNB<A> {
    /// Create a new Gaussian Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            class_prior_: None,
            theta_: None,
            sigma_: None,
            classes_: None,
        }
    }

    /// Fit Gaussian Naive Bayes classifier
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Get unique classes
        let mut classes_set: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for &label in y {
            classes_set.insert(label.to_i32().unwrap_or(0));
        }
        let mut classes: Vec<i32> = classes_set.into_iter().collect();
        classes.sort();

        let mut class_prior = HashMap::new();
        let mut theta = HashMap::new();
        let mut sigma = HashMap::new();

        // For each class, compute mean and variance
        for &class in &classes {
            let class_a = A::from(class).unwrap();

            // Filter samples of this class
            let mut class_samples = Vec::new();
            for i in 0..n_samples {
                if (y[i] - class_a).abs() < A::from(1e-10).unwrap() {
                    class_samples.push(i);
                }
            }

            let n_class = class_samples.len();
            class_prior.insert(class, A::from(n_class).unwrap() / A::from(n_samples).unwrap());

            // Compute mean for each feature
            let mut mean = Array1::zeros(n_features);
            for &idx in &class_samples {
                for j in 0..n_features {
                    mean[j] = mean[j] + X[[idx, j]];
                }
            }
            mean.mapv_inplace(|x| x / A::from(n_class).unwrap());

            // Compute variance for each feature
            let mut var = Array1::zeros(n_features);
            for &idx in &class_samples {
                for j in 0..n_features {
                    let diff = X[[idx, j]] - mean[j];
                    var[j] = var[j] + diff * diff;
                }
            }
            var.mapv_inplace(|x| x / A::from(n_class).unwrap() + A::from(1e-9).unwrap()); // Add small value for numerical stability

            theta.insert(class, mean);
            sigma.insert(class, var);
        }

        self.class_prior_ = Some(class_prior);
        self.theta_ = Some(theta);
        self.sigma_ = Some(sigma);
        self.classes_ = Some(classes);

        Ok(())
    }

    /// Predict class labels
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let log_proba = self.predict_log_proba(X)?;
        let mut predictions = Array1::zeros(X.nrows());

        let classes = self.classes_.as_ref().ok_or(RustyAIError::NotFitted)?;

        for i in 0..X.nrows() {
            let mut max_log_prob = A::neg_infinity();
            let mut best_class = classes[0];

            for (j, &class) in classes.iter().enumerate() {
                if log_proba[[i, j]] > max_log_prob {
                    max_log_prob = log_proba[[i, j]];
                    best_class = class;
                }
            }

            predictions[i] = A::from(best_class).unwrap();
        }

        Ok(predictions)
    }

    /// Predict log probability
    pub fn predict_log_proba(&self, X: &Array2<A>) -> Result<Array2<A>> {
        let class_prior = self.class_prior_.as_ref().ok_or(RustyAIError::NotFitted)?;
        let theta = self.theta_.as_ref().ok_or(RustyAIError::NotFitted)?;
        let sigma = self.sigma_.as_ref().ok_or(RustyAIError::NotFitted)?;
        let classes = self.classes_.as_ref().ok_or(RustyAIError::NotFitted)?;

        let n_samples = X.nrows();
        let n_classes = classes.len();
        let mut log_proba = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for (j, &class) in classes.iter().enumerate() {
                let prior = class_prior[&class];
                let mean = &theta[&class];
                let var = &sigma[&class];

                // Log probability = log(prior) + sum of log(Gaussian PDF)
                let mut log_prob = prior.ln();

                for k in 0..X.ncols() {
                    let x = X[[i, k]];
                    let mu = mean[k];
                    let sigma2 = var[k];

                    // log(Gaussian PDF) = -0.5 * log(2*pi*sigma^2) - (x-mu)^2 / (2*sigma^2)
                    let diff = x - mu;
                    log_prob = log_prob - A::from(0.5).unwrap() * (A::from(2.0 * std::f64::consts::PI).unwrap() * sigma2).ln()
                        - (diff * diff) / (A::from(2.0).unwrap() * sigma2);
                }

                log_proba[[i, j]] = log_prob;
            }
        }

        Ok(log_proba)
    }

    /// Predict probability
    pub fn predict_proba(&self, X: &Array2<A>) -> Result<Array2<A>> {
        let log_proba = self.predict_log_proba(X)?;

        // Convert log probabilities to probabilities using log-sum-exp trick
        let mut proba = Array2::zeros(log_proba.dim());

        for i in 0..log_proba.nrows() {
            // Find max log probability for numerical stability
            let mut max_log_prob = A::neg_infinity();
            for j in 0..log_proba.ncols() {
                if log_proba[[i, j]] > max_log_prob {
                    max_log_prob = log_proba[[i, j]];
                }
            }

            // Compute exp(log_prob - max) and sum
            let mut sum = A::zero();
            for j in 0..log_proba.ncols() {
                let exp_val = (log_proba[[i, j]] - max_log_prob).exp();
                proba[[i, j]] = exp_val;
                sum = sum + exp_val;
            }

            // Normalize
            for j in 0..log_proba.ncols() {
                proba[[i, j]] = proba[[i, j]] / sum;
            }
        }

        Ok(proba)
    }

    /// Compute accuracy score
    pub fn score(&self, X: &Array2<A>, y: &Array1<A>) -> Result<A> {
        let predictions = self.predict(X)?;
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_y)| (pred - true_y).abs() < A::from(1e-10).unwrap())
            .count();
        Ok(A::from(correct).unwrap() / A::from(y.len()).unwrap())
    }
}

impl<A: Float + ScalarOperand + Sum> Default for GaussianNB<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Multinomial Naive Bayes
///
/// Suitable for discrete features (e.g., word counts).
pub struct MultinomialNB<A: Float> {
    /// Class priors
    class_prior_: Option<HashMap<i32, A>>,
    /// Feature log probabilities per class
    feature_log_prob_: Option<HashMap<i32, Array1<A>>>,
    /// Classes seen during fit
    classes_: Option<Vec<i32>>,
    /// Smoothing parameter
    alpha: A,
}

impl<A: Float + ScalarOperand + Sum> MultinomialNB<A> {
    /// Create a new Multinomial Naive Bayes classifier
    ///
    /// # Arguments
    /// * `alpha` - Additive (Laplace/Lidstone) smoothing parameter (default 1.0)
    pub fn new(alpha: A) -> Self {
        Self {
            class_prior_: None,
            feature_log_prob_: None,
            classes_: None,
            alpha,
        }
    }

    /// Fit Multinomial Naive Bayes classifier
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Get unique classes
        let mut classes_set: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for &label in y {
            classes_set.insert(label.to_i32().unwrap_or(0));
        }
        let mut classes: Vec<i32> = classes_set.into_iter().collect();
        classes.sort();

        let mut class_prior = HashMap::new();
        let mut feature_log_prob = HashMap::new();

        for &class in &classes {
            let class_a = A::from(class).unwrap();

            // Count samples in this class
            let mut class_count = 0;
            let mut feature_counts = Array1::zeros(n_features);

            for i in 0..n_samples {
                if (y[i] - class_a).abs() < A::from(1e-10).unwrap() {
                    class_count += 1;
                    for j in 0..n_features {
                        feature_counts[j] = feature_counts[j] + X[[i, j]];
                    }
                }
            }

            class_prior.insert(class, A::from(class_count).unwrap() / A::from(n_samples).unwrap());

            // Compute log probabilities with smoothing
            let total_count: A = feature_counts.sum() + self.alpha * A::from(n_features).unwrap();
            let log_prob = feature_counts.mapv(|count: A| {
                ((count + self.alpha) / total_count).ln()
            });

            feature_log_prob.insert(class, log_prob);
        }

        self.class_prior_ = Some(class_prior);
        self.feature_log_prob_ = Some(feature_log_prob);
        self.classes_ = Some(classes);

        Ok(())
    }

    /// Predict class labels
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let log_proba = self.predict_log_proba(X)?;
        let mut predictions = Array1::zeros(X.nrows());

        let classes = self.classes_.as_ref().ok_or(RustyAIError::NotFitted)?;

        for i in 0..X.nrows() {
            let mut max_log_prob = A::neg_infinity();
            let mut best_class = classes[0];

            for (j, &class) in classes.iter().enumerate() {
                if log_proba[[i, j]] > max_log_prob {
                    max_log_prob = log_proba[[i, j]];
                    best_class = class;
                }
            }

            predictions[i] = A::from(best_class).unwrap();
        }

        Ok(predictions)
    }

    /// Predict log probability
    pub fn predict_log_proba(&self, X: &Array2<A>) -> Result<Array2<A>> {
        let class_prior = self.class_prior_.as_ref().ok_or(RustyAIError::NotFitted)?;
        let feature_log_prob = self.feature_log_prob_.as_ref().ok_or(RustyAIError::NotFitted)?;
        let classes = self.classes_.as_ref().ok_or(RustyAIError::NotFitted)?;

        let n_samples = X.nrows();
        let n_classes = classes.len();
        let mut log_proba = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for (j, &class) in classes.iter().enumerate() {
                let prior = class_prior[&class];
                let feature_lp = &feature_log_prob[&class];

                // Log probability = log(prior) + sum of (count * log_prob)
                let mut log_prob = prior.ln();

                for k in 0..X.ncols() {
                    log_prob = log_prob + X[[i, k]] * feature_lp[k];
                }

                log_proba[[i, j]] = log_prob;
            }
        }

        Ok(log_proba)
    }

    /// Compute accuracy score
    pub fn score(&self, X: &Array2<A>, y: &Array1<A>) -> Result<A> {
        let predictions = self.predict(X)?;
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_y)| (pred - true_y).abs() < A::from(1e-10).unwrap())
            .count();
        Ok(A::from(correct).unwrap() / A::from(y.len()).unwrap())
    }
}

impl<A: Float + ScalarOperand + Sum> Default for MultinomialNB<A> {
    fn default() -> Self {
        Self::new(A::one())
    }
}
