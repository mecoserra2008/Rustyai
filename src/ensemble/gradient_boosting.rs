//! Gradient Boosting implementation (XGBoost-like)

use super::DecisionTree;
use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};

/// Gradient Boosting for regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoosting<A> {
    trees: Vec<DecisionTree<A>>,
    n_estimators: usize,
    learning_rate: A,
    max_depth: Option<usize>,
    min_samples_split: usize,
}

impl<A: Float> GradientBoosting<A> {
    /// Create a new GradientBoosting model
    pub fn new() -> Self {
        Self {
            trees: Vec::new(),
            n_estimators: 100,
            learning_rate: A::from(0.1).unwrap(),
            max_depth: Some(3),
            min_samples_split: 2,
        }
    }

    /// Fit the model
    pub fn fit(mut self, X: &Array2<A>, y: &Array1<A>) -> Result<Self> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        self.trees.clear();

        // Initialize predictions with mean
        let mut predictions = Array1::from_elem(y.len(), y.sum() / A::from(y.len()).unwrap());

        for _ in 0..self.n_estimators {
            // Compute residuals
            let residuals = y - &predictions;

            // Fit tree to residuals
            let tree = DecisionTree::new(self.max_depth, self.min_samples_split)
                .fit(X, &residuals)?;

            // Update predictions
            let tree_pred = tree.predict(X)?;
            predictions = predictions + tree_pred.mapv(|p| p * self.learning_rate);

            self.trees.push(tree);
        }

        Ok(self)
    }

    /// Make predictions
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.trees.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let mut predictions = Array1::zeros(X.nrows());

        for tree in &self.trees {
            let tree_pred = tree.predict(X)?;
            predictions = predictions + tree_pred.mapv(|p| p * self.learning_rate);
        }

        Ok(predictions)
    }
}

impl<A: Float> Default for GradientBoosting<A> {
    fn default() -> Self {
        Self::new()
    }
}
