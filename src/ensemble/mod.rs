//! Ensemble methods
//!
//! Includes Random Forests, Gradient Boosting, AdaBoost, and stacking ensembles.

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::iter::Sum;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub mod random_forest;
pub mod gradient_boosting;

pub use random_forest::RandomForest;
pub use gradient_boosting::GradientBoosting;

/// Decision tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TreeNode<A> {
    feature_idx: Option<usize>,
    threshold: Option<A>,
    left: Option<Box<TreeNode<A>>>,
    right: Option<Box<TreeNode<A>>>,
    value: Option<A>,
}

impl<A: Float + ScalarOperand + Sum> TreeNode<A> {
    fn leaf(value: A) -> Self {
        Self {
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            value: Some(value),
        }
    }

    fn split(feature_idx: usize, threshold: A, left: TreeNode<A>, right: TreeNode<A>) -> Self {
        Self {
            feature_idx: Some(feature_idx),
            threshold: Some(threshold),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            value: None,
        }
    }

    fn predict(&self, x: &Array1<A>) -> A {
        if let Some(value) = self.value {
            return value;
        }

        let feature_idx = self.feature_idx.unwrap();
        let threshold = self.threshold.unwrap();

        if x[feature_idx] <= threshold {
            self.left.as_ref().unwrap().predict(x)
        } else {
            self.right.as_ref().unwrap().predict(x)
        }
    }
}

/// Decision tree for regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree<A> {
    root: Option<TreeNode<A>>,
    max_depth: Option<usize>,
    min_samples_split: usize,
}

impl<A: Float + ScalarOperand + Sum> DecisionTree<A> {
    pub fn new(max_depth: Option<usize>, min_samples_split: usize) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split,
        }
    }

    pub fn fit(mut self, X: &Array2<A>, y: &Array1<A>) -> Result<Self> {
        self.root = Some(self.build_tree(X, y, 0));
        Ok(self)
    }

    fn build_tree(&self, X: &Array2<A>, y: &Array1<A>, depth: usize) -> TreeNode<A> {
        let n_samples = X.nrows();

        // Stopping criteria
        if n_samples < self.min_samples_split
            || self.max_depth.map_or(false, |max_d| depth >= max_d)
        {
            let mean = y.sum() / A::from(n_samples).unwrap();
            return TreeNode::leaf(mean);
        }

        // Find best split
        let (best_feature, best_threshold, best_gain) = self.find_best_split(X, y);

        if best_gain < A::from(1e-7).unwrap() {
            let mean = y.sum() / A::from(n_samples).unwrap();
            return TreeNode::leaf(mean);
        }

        // Split data
        let (left_idx, right_idx) = self.split_data(X, best_feature, best_threshold);

        if left_idx.is_empty() || right_idx.is_empty() {
            let mean = y.sum() / A::from(n_samples).unwrap();
            return TreeNode::leaf(mean);
        }

        let X_left = X.select(ndarray::Axis(0), &left_idx);
        let y_left = y.select(ndarray::Axis(0), &left_idx);
        let X_right = X.select(ndarray::Axis(0), &right_idx);
        let y_right = y.select(ndarray::Axis(0), &right_idx);

        let left = self.build_tree(&X_left, &y_left, depth + 1);
        let right = self.build_tree(&X_right, &y_right, depth + 1);

        TreeNode::split(best_feature, best_threshold, left, right)
    }

    fn find_best_split(&self, X: &Array2<A>, y: &Array1<A>) -> (usize, A, A) {
        let mut best_gain = A::zero();
        let mut best_feature = 0;
        let mut best_threshold = A::zero();

        for feature in 0..X.ncols() {
            let col = X.column(feature);
            let unique_vals: Vec<A> = col.iter().copied().collect();

            for &threshold in &unique_vals {
                let gain = self.information_gain(X, y, feature, threshold);
                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    fn information_gain(&self, X: &Array2<A>, y: &Array1<A>, feature: usize, threshold: A) -> A {
        let (left_idx, right_idx) = self.split_data(X, feature, threshold);

        if left_idx.is_empty() || right_idx.is_empty() {
            return A::zero();
        }

        let n = A::from(y.len()).unwrap();
        let n_left = A::from(left_idx.len()).unwrap();
        let n_right = A::from(right_idx.len()).unwrap();

        let parent_var = self.variance(y);
        let y_left = y.select(ndarray::Axis(0), &left_idx);
        let y_right = y.select(ndarray::Axis(0), &right_idx);

        let left_var = self.variance(&y_left);
        let right_var = self.variance(&y_right);

        parent_var - (n_left / n * left_var + n_right / n * right_var)
    }

    fn variance(&self, y: &Array1<A>) -> A {
        if y.is_empty() {
            return A::zero();
        }
        let mean = y.sum() / A::from(y.len()).unwrap();
        y.iter().map(|&yi| (yi - mean) * (yi - mean)).sum::<A>()
            / A::from(y.len()).unwrap()
    }

    fn split_data(&self, X: &Array2<A>, feature: usize, threshold: A) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::new();
        let mut right = Vec::new();

        for i in 0..X.nrows() {
            if X[[i, feature]] <= threshold {
                left.push(i);
            } else {
                right.push(i);
            }
        }

        (left, right)
    }

    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        let root = self.root.as_ref().ok_or(RustyAIError::NotFitted)?;

        let mut predictions = Array1::zeros(X.nrows());
        for (i, row) in X.rows().into_iter().enumerate() {
            predictions[i] = root.predict(&row.to_owned());
        }

        Ok(predictions)
    }
}
