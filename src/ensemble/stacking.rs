//! Stacking and Voting Ensemble Methods
//!
//! Advanced ensemble techniques that combine multiple models:
//! - Stacking (meta-learning with multiple base estimators)
//! - Voting (hard and soft voting for classification)
//! - Blending (holdout-based ensemble)

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::Float;
use std::collections::HashMap;

/// Voting Classifier
///
/// Combines predictions from multiple classifiers using majority voting (hard)
/// or averaged probabilities (soft).
pub struct VotingClassifier<A: Float> {
    estimators: Vec<Box<dyn Predictor<A>>>,
    voting: VotingType,
    weights: Option<Vec<A>>,
}

#[derive(Debug, Clone, Copy)]
pub enum VotingType {
    /// Majority voting on predicted class labels
    Hard,
    /// Average predicted probabilities (requires predict_proba)
    Soft,
}

/// Trait for predictors that can be used in ensembles
pub trait Predictor<A: Float> {
    fn predict(&self, X: &Array2<A>) -> Result<Array1<A>>;
    fn predict_proba(&self, X: &Array2<A>) -> Result<Array2<A>> {
        Err(RustyAIError::Other(
            "predict_proba not implemented for this estimator".to_string(),
        ))
    }
}

impl<A: Float + Copy + ScalarOperand> VotingClassifier<A> {
    /// Create a new voting classifier
    pub fn new(voting: VotingType) -> Self {
        Self {
            estimators: Vec::new(),
            voting,
            weights: None,
        }
    }

    /// Add an estimator to the ensemble
    pub fn add_estimator(&mut self, estimator: Box<dyn Predictor<A>>) {
        self.estimators.push(estimator);
    }

    /// Set voting weights for each estimator
    pub fn with_weights(mut self, weights: Vec<A>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Make predictions using voting
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.estimators.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let n_samples = X.nrows();
        let mut predictions = Array1::zeros(n_samples);

        match self.voting {
            VotingType::Hard => {
                // Collect all predictions
                let all_preds: Vec<Array1<A>> = self
                    .estimators
                    .iter()
                    .map(|est| est.predict(X))
                    .collect::<Result<Vec<_>>>()?;

                // Majority vote for each sample
                for i in 0..n_samples {
                    let mut votes: HashMap<i32, A> = HashMap::new();

                    for (j, pred) in all_preds.iter().enumerate() {
                        let class = pred[i].to_i32().unwrap_or(0);
                        let weight = self.weights.as_ref().map(|w| w[j]).unwrap_or(A::one());
                        let current = *votes.entry(class).or_insert(A::zero());
                        votes.insert(class, current + weight);
                    }

                    let majority_class = votes
                        .iter()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(&class, _)| A::from(class).unwrap())
                        .unwrap_or(A::zero());

                    predictions[i] = majority_class;
                }
            }
            VotingType::Soft => {
                // Average predicted probabilities
                let all_probas: Vec<Array2<A>> = self
                    .estimators
                    .iter()
                    .map(|est| est.predict_proba(X))
                    .collect::<Result<Vec<_>>>()?;

                if all_probas.is_empty() {
                    return Err(RustyAIError::Other(
                        "No estimators support predict_proba for soft voting".to_string(),
                    ));
                }

                let n_classes = all_probas[0].ncols();
                let mut avg_probas = Array2::zeros((n_samples, n_classes));

                for (j, proba) in all_probas.iter().enumerate() {
                    let weight = self.weights.as_ref().map(|w| w[j]).unwrap_or(A::one());
                    avg_probas = avg_probas + &(proba * weight);
                }

                let total_weight: A = self
                    .weights
                    .as_ref()
                    .map(|w| w.iter().copied().fold(A::zero(), |acc, x| acc + x))
                    .unwrap_or_else(|| A::from(self.estimators.len()).unwrap());

                avg_probas = avg_probas / total_weight;

                // Argmax to get predicted class
                for i in 0..n_samples {
                    let row = avg_probas.row(i);
                    let max_idx = row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a): &(usize, &A), (_, b): &(usize, &A)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    predictions[i] = A::from(max_idx).unwrap();
                }
            }
        }

        Ok(predictions)
    }
}

/// Stacking Classifier/Regressor
///
/// Uses a meta-learner to combine predictions from multiple base estimators.
/// Base estimators are trained on the full training set, and the meta-learner
/// is trained on their predictions using cross-validation.
pub struct StackingEnsemble<A: Float> {
    base_estimators: Vec<Box<dyn Predictor<A>>>,
    meta_learner: Option<Box<dyn Predictor<A>>>,
    use_features: bool,
    cv_folds: usize,
}

impl<A: Float + Copy + ScalarOperand> StackingEnsemble<A> {
    /// Create a new stacking ensemble
    pub fn new(cv_folds: usize) -> Self {
        Self {
            base_estimators: Vec::new(),
            meta_learner: None,
            use_features: true,
            cv_folds,
        }
    }

    /// Add a base estimator
    pub fn add_base_estimator(&mut self, estimator: Box<dyn Predictor<A>>) {
        self.base_estimators.push(estimator);
    }

    /// Set the meta-learner
    pub fn set_meta_learner(&mut self, meta_learner: Box<dyn Predictor<A>>) {
        self.meta_learner = Some(meta_learner);
    }

    /// Set whether to include original features in meta-learner input
    pub fn with_features(mut self, use_features: bool) -> Self {
        self.use_features = use_features;
        self
    }

    /// Make predictions
    ///
    /// Note: This is a simplified implementation. In practice, you'd need to
    /// implement fit() with proper cross-validation training.
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.base_estimators.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let meta_learner = self
            .meta_learner
            .as_ref()
            .ok_or(RustyAIError::NotFitted)?;

        // Get predictions from all base estimators
        let base_predictions: Vec<Array1<A>> = self
            .base_estimators
            .iter()
            .map(|est| est.predict(X))
            .collect::<Result<Vec<_>>>()?;

        // Stack predictions into meta-features
        let n_samples = X.nrows();
        let n_base = base_predictions.len();

        let meta_features = if self.use_features {
            // Include original features
            let n_features = X.ncols();
            let mut meta_X = Array2::zeros((n_samples, n_base + n_features));

            for i in 0..n_samples {
                // Base predictions
                for j in 0..n_base {
                    meta_X[[i, j]] = base_predictions[j][i];
                }
                // Original features
                for j in 0..n_features {
                    meta_X[[i, n_base + j]] = X[[i, j]];
                }
            }
            meta_X
        } else {
            // Only use base predictions
            let mut meta_X = Array2::zeros((n_samples, n_base));
            for i in 0..n_samples {
                for j in 0..n_base {
                    meta_X[[i, j]] = base_predictions[j][i];
                }
            }
            meta_X
        };

        // Meta-learner prediction
        meta_learner.predict(&meta_features)
    }
}

/// Blending Ensemble
///
/// Similar to stacking but simpler: uses a holdout set to train the meta-learner
/// instead of cross-validation.
pub struct BlendingEnsemble<A: Float> {
    base_estimators: Vec<Box<dyn Predictor<A>>>,
    meta_learner: Option<Box<dyn Predictor<A>>>,
    holdout_ratio: f64,
}

impl<A: Float + Copy + ScalarOperand> BlendingEnsemble<A> {
    /// Create a new blending ensemble
    pub fn new(holdout_ratio: f64) -> Self {
        Self {
            base_estimators: Vec::new(),
            meta_learner: None,
            holdout_ratio,
        }
    }

    /// Add a base estimator
    pub fn add_base_estimator(&mut self, estimator: Box<dyn Predictor<A>>) {
        self.base_estimators.push(estimator);
    }

    /// Set the meta-learner
    pub fn set_meta_learner(&mut self, meta_learner: Box<dyn Predictor<A>>) {
        self.meta_learner = Some(meta_learner);
    }

    /// Make predictions
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.base_estimators.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let meta_learner = self
            .meta_learner
            .as_ref()
            .ok_or(RustyAIError::NotFitted)?;

        // Get predictions from all base estimators
        let base_predictions: Vec<Array1<A>> = self
            .base_estimators
            .iter()
            .map(|est| est.predict(X))
            .collect::<Result<Vec<_>>>()?;

        // Stack predictions into meta-features
        let n_samples = X.nrows();
        let n_base = base_predictions.len();
        let mut meta_X = Array2::zeros((n_samples, n_base));

        for i in 0..n_samples {
            for j in 0..n_base {
                meta_X[[i, j]] = base_predictions[j][i];
            }
        }

        // Meta-learner prediction
        meta_learner.predict(&meta_X)
    }
}

/// Bagging Ensemble (Bootstrap Aggregating)
///
/// Generic bagging implementation that can work with any base estimator.
pub struct BaggingEnsemble<A: Float> {
    estimators: Vec<Box<dyn Predictor<A>>>,
    n_estimators: usize,
    max_samples: f64,
    max_features: f64,
    bootstrap: bool,
}

impl<A: Float + Copy + ScalarOperand> BaggingEnsemble<A> {
    /// Create a new bagging ensemble
    pub fn new(n_estimators: usize) -> Self {
        Self {
            estimators: Vec::new(),
            n_estimators,
            max_samples: 1.0,
            max_features: 1.0,
            bootstrap: true,
        }
    }

    /// Set the fraction of samples to draw
    pub fn with_max_samples(mut self, max_samples: f64) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Set the fraction of features to draw
    pub fn with_max_features(mut self, max_features: f64) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set whether to use bootstrap sampling
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Add an estimator
    pub fn add_estimator(&mut self, estimator: Box<dyn Predictor<A>>) {
        self.estimators.push(estimator);
    }

    /// Make predictions (for classification: majority vote, for regression: average)
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.estimators.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let n_samples = X.nrows();
        let mut predictions = Array1::zeros(n_samples);

        // Collect all predictions
        let all_preds: Vec<Array1<A>> = self
            .estimators
            .iter()
            .map(|est| est.predict(X))
            .collect::<Result<Vec<_>>>()?;

        // Average predictions (suitable for regression)
        // For classification, would need to use voting instead
        for i in 0..n_samples {
            let sum: A = all_preds.iter().map(|pred| pred[i]).fold(A::zero(), |acc, x| acc + x);
            predictions[i] = sum / A::from(all_preds.len()).unwrap();
        }

        Ok(predictions)
    }

    /// Make predictions using majority voting (for classification)
    pub fn predict_vote(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.estimators.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let n_samples = X.nrows();
        let mut predictions = Array1::zeros(n_samples);

        // Collect all predictions
        let all_preds: Vec<Array1<A>> = self
            .estimators
            .iter()
            .map(|est| est.predict(X))
            .collect::<Result<Vec<_>>>()?;

        // Majority vote for each sample
        for i in 0..n_samples {
            let mut votes: HashMap<i32, usize> = HashMap::new();

            for pred in &all_preds {
                let class = pred[i].to_i32().unwrap_or(0);
                *votes.entry(class).or_insert(0) += 1;
            }

            let majority_class = votes
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&class, _)| A::from(class).unwrap())
                .unwrap_or(A::zero());

            predictions[i] = majority_class;
        }

        Ok(predictions)
    }
}

/// Pasting Ensemble
///
/// Similar to bagging but without replacement (deterministic sampling).
pub type PastingEnsemble<A> = BaggingEnsemble<A>;

/// Random Subspaces Ensemble
///
/// Trains estimators on random subsets of features.
pub struct RandomSubspacesEnsemble<A: Float> {
    estimators: Vec<Box<dyn Predictor<A>>>,
    n_estimators: usize,
    max_features: f64,
}

impl<A: Float + Copy + ScalarOperand> RandomSubspacesEnsemble<A> {
    pub fn new(n_estimators: usize, max_features: f64) -> Self {
        Self {
            estimators: Vec::new(),
            n_estimators,
            max_features,
        }
    }

    pub fn add_estimator(&mut self, estimator: Box<dyn Predictor<A>>) {
        self.estimators.push(estimator);
    }

    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.estimators.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let n_samples = X.nrows();
        let mut predictions = Array1::zeros(n_samples);

        // Collect all predictions (simplified - would need feature selection logic)
        let all_preds: Vec<Array1<A>> = self
            .estimators
            .iter()
            .map(|est| est.predict(X))
            .collect::<Result<Vec<_>>>()?;

        // Average predictions
        for i in 0..n_samples {
            let sum: A = all_preds.iter().map(|pred| pred[i]).fold(A::zero(), |acc, x| acc + x);
            predictions[i] = sum / A::from(all_preds.len()).unwrap();
        }

        Ok(predictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock predictor for testing
    struct MockPredictor {
        predictions: Vec<f64>,
    }

    impl Predictor<f64> for MockPredictor {
        fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>> {
            Ok(Array1::from_vec(self.predictions.clone()))
        }
    }

    #[test]
    fn test_voting_classifier_hard() {
        let mut voting = VotingClassifier::new(VotingType::Hard);
        voting.add_estimator(Box::new(MockPredictor {
            predictions: vec![0.0, 1.0, 0.0],
        }));
        voting.add_estimator(Box::new(MockPredictor {
            predictions: vec![0.0, 1.0, 1.0],
        }));
        voting.add_estimator(Box::new(MockPredictor {
            predictions: vec![0.0, 0.0, 1.0],
        }));

        let X = Array2::zeros((3, 2));
        let preds = voting.predict(&X).unwrap();

        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn test_bagging_ensemble() {
        let mut bagging = BaggingEnsemble::new(3);
        bagging.add_estimator(Box::new(MockPredictor {
            predictions: vec![1.0, 2.0, 3.0],
        }));
        bagging.add_estimator(Box::new(MockPredictor {
            predictions: vec![1.5, 2.5, 3.5],
        }));

        let X = Array2::zeros((3, 2));
        let preds = bagging.predict(&X).unwrap();

        assert!((preds[0] - 1.25).abs() < 1e-10);
        assert!((preds[1] - 2.25).abs() < 1e-10);
    }
}
