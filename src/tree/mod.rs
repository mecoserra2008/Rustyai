//! Decision trees for classification and regression
//!
//! Implements CART (Classification and Regression Trees) algorithm with:
//! - Gini impurity and entropy for classification
//! - MSE variance reduction for regression
//! - Pruning support
//! - Feature importance calculation

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use std::collections::HashMap;
use std::iter::Sum;
use serde::{Deserialize, Serialize};

/// Split criterion for decision trees
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Criterion {
    /// Gini impurity (classification)
    Gini,
    /// Entropy/Information gain (classification)
    Entropy,
    /// Mean squared error (regression)
    Mse,
}

/// Tree node structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TreeNode<A: Float> {
    /// Feature index for splitting
    feature: Option<usize>,
    /// Threshold value for splitting
    threshold: Option<A>,
    /// Left child index
    left: Option<usize>,
    /// Right child index
    right: Option<usize>,
    /// Prediction value (for leaf nodes)
    value: Option<A>,
    /// Number of samples at this node
    n_samples: usize,
    /// Impurity at this node
    impurity: A,
}

/// Decision Tree Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeClassifier<A: Float> {
    /// Maximum depth of the tree
    max_depth: Option<usize>,
    /// Minimum samples required to split
    min_samples_split: usize,
    /// Minimum samples required at leaf node
    min_samples_leaf: usize,
    /// Split criterion
    criterion: Criterion,
    /// Tree nodes
    nodes: Vec<TreeNode<A>>,
    /// Number of classes
    n_classes: Option<usize>,
    /// Feature importances
    feature_importances: Option<Array1<A>>,
}

impl<A: Float + ScalarOperand + Sum> DecisionTreeClassifier<A> {
    /// Create a new Decision Tree Classifier
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: Criterion::Gini,
            nodes: Vec::new(),
            n_classes: None,
            feature_importances: None,
        }
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Set minimum samples for split
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set minimum samples per leaf
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set split criterion
    pub fn with_criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Compute Gini impurity
    fn gini_impurity(&self, y: &[A]) -> A {
        if y.is_empty() {
            return A::zero();
        }

        let n = A::from(y.len()).unwrap();
        let mut class_counts: HashMap<i32, usize> = HashMap::new();

        for &label in y {
            let label_int = label.to_i32().unwrap_or(0);
            *class_counts.entry(label_int).or_insert(0) += 1;
        }

        let mut gini = A::one();
        for &count in class_counts.values() {
            let prob = A::from(count).unwrap() / n;
            gini = gini - prob * prob;
        }

        gini
    }

    /// Compute entropy
    fn entropy(&self, y: &[A]) -> A {
        if y.is_empty() {
            return A::zero();
        }

        let n = A::from(y.len()).unwrap();
        let mut class_counts: HashMap<i32, usize> = HashMap::new();

        for &label in y {
            let label_int = label.to_i32().unwrap_or(0);
            *class_counts.entry(label_int).or_insert(0) += 1;
        }

        let mut ent = A::zero();
        for &count in class_counts.values() {
            if count > 0 {
                let prob = A::from(count).unwrap() / n;
                ent = ent - prob * prob.ln();
            }
        }

        ent
    }

    /// Calculate impurity based on criterion
    fn impurity(&self, y: &[A]) -> A {
        match self.criterion {
            Criterion::Gini => self.gini_impurity(y),
            Criterion::Entropy => self.entropy(y),
            _ => A::zero(),
        }
    }

    /// Find best split for a feature
    fn best_split(
        &self,
        X: &Array2<A>,
        y: &Array1<A>,
        indices: &[usize],
        feature: usize,
    ) -> Option<(A, A)> {
        if indices.len() < self.min_samples_split {
            return None;
        }

        // Collect feature values and sort
        let mut values: Vec<(A, usize)> = indices
            .iter()
            .map(|&idx| (X[[idx, feature]], idx))
            .collect();
        values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut best_gain = A::neg_infinity();
        let mut best_threshold = A::zero();

        let parent_y: Vec<A> = indices.iter().map(|&idx| y[idx]).collect();
        let parent_impurity = self.impurity(&parent_y);

        // Try each unique threshold
        for i in 0..values.len() - 1 {
            if values[i].0 == values[i + 1].0 {
                continue;
            }

            let threshold = (values[i].0 + values[i + 1].0) / A::from(2.0).unwrap();

            let left_y: Vec<A> = values[..=i].iter().map(|&(_, idx)| y[idx]).collect();
            let right_y: Vec<A> = values[i + 1..].iter().map(|&(_, idx)| y[idx]).collect();

            if left_y.len() < self.min_samples_leaf || right_y.len() < self.min_samples_leaf {
                continue;
            }

            let left_impurity = self.impurity(&left_y);
            let right_impurity = self.impurity(&right_y);

            let n = A::from(parent_y.len()).unwrap();
            let n_left = A::from(left_y.len()).unwrap();
            let n_right = A::from(right_y.len()).unwrap();

            let gain = parent_impurity
                - (n_left / n) * left_impurity
                - (n_right / n) * right_impurity;

            if gain > best_gain {
                best_gain = gain;
                best_threshold = threshold;
            }
        }

        if best_gain > A::zero() {
            Some((best_threshold, best_gain))
        } else {
            None
        }
    }

    /// Build tree recursively
    fn build_tree(
        &mut self,
        X: &Array2<A>,
        y: &Array1<A>,
        indices: &[usize],
        depth: usize,
    ) -> usize {
        let n_samples = indices.len();
        let n_features = X.ncols();

        // Calculate current impurity
        let current_y: Vec<A> = indices.iter().map(|&idx| y[idx]).collect();
        let impurity = self.impurity(&current_y);

        // Compute majority class
        let mut class_counts: HashMap<i32, usize> = HashMap::new();
        for &label in &current_y {
            let label_int = label.to_i32().unwrap_or(0);
            *class_counts.entry(label_int).or_insert(0) += 1;
        }
        let majority_class = class_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&class, _)| A::from(class).unwrap())
            .unwrap_or(A::zero());

        // Create leaf node if stopping criteria met
        let should_stop = if let Some(max_d) = self.max_depth {
            depth >= max_d
        } else {
            false
        } || n_samples < self.min_samples_split
            || impurity == A::zero();

        if should_stop {
            let node = TreeNode {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                value: Some(majority_class),
                n_samples,
                impurity,
            };
            self.nodes.push(node);
            return self.nodes.len() - 1;
        }

        // Find best split across all features
        let mut best_feature = 0;
        let mut best_threshold = A::zero();
        let mut best_gain = A::neg_infinity();

        for feature in 0..n_features {
            if let Some((threshold, gain)) = self.best_split(X, y, indices, feature) {
                if gain > best_gain {
                    best_gain = gain;
                    best_threshold = threshold;
                    best_feature = feature;
                }
            }
        }

        // Create leaf if no good split found
        if best_gain <= A::zero() {
            let node = TreeNode {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                value: Some(majority_class),
                n_samples,
                impurity,
            };
            self.nodes.push(node);
            return self.nodes.len() - 1;
        }

        // Split data
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            if X[[idx, best_feature]] <= best_threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // Recursively build subtrees
        let left_child = self.build_tree(X, y, &left_indices, depth + 1);
        let right_child = self.build_tree(X, y, &right_indices, depth + 1);

        // Create internal node
        let node = TreeNode {
            feature: Some(best_feature),
            threshold: Some(best_threshold),
            left: Some(left_child),
            right: Some(right_child),
            value: None,
            n_samples,
            impurity,
        };
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// Fit the decision tree
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        // Determine number of classes
        let mut classes = std::collections::HashSet::new();
        for &label in y {
            classes.insert(label.to_i32().unwrap_or(0));
        }
        self.n_classes = Some(classes.len());

        // Build tree
        self.nodes.clear();
        let indices: Vec<usize> = (0..X.nrows()).collect();
        self.build_tree(X, y, &indices, 0);

        Ok(())
    }

    /// Predict a single sample
    fn predict_sample(&self, x: &Array1<A>) -> A {
        let mut node_idx = self.nodes.len() - 1; // Root is last node

        loop {
            let node = &self.nodes[node_idx];

            // Leaf node - return value
            if let Some(value) = node.value {
                return value;
            }

            // Internal node - follow split
            if let (Some(feature), Some(threshold)) = (node.feature, node.threshold) {
                if x[feature] <= threshold {
                    node_idx = node.left.unwrap();
                } else {
                    node_idx = node.right.unwrap();
                }
            } else {
                return A::zero(); // Shouldn't happen
            }
        }
    }

    /// Predict class labels for samples
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.nodes.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let mut predictions = Array1::zeros(X.nrows());
        for (i, row) in X.rows().into_iter().enumerate() {
            predictions[i] = self.predict_sample(&row.to_owned());
        }

        Ok(predictions)
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

impl<A: Float + ScalarOperand + Sum> Default for DecisionTreeClassifier<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Decision Tree Regressor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeRegressor<A: Float> {
    /// Maximum depth of the tree
    max_depth: Option<usize>,
    /// Minimum samples required to split
    min_samples_split: usize,
    /// Minimum samples required at leaf node
    min_samples_leaf: usize,
    /// Tree nodes
    nodes: Vec<TreeNode<A>>,
}

impl<A: Float + ScalarOperand + Sum> DecisionTreeRegressor<A> {
    /// Create a new Decision Tree Regressor
    pub fn new() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            nodes: Vec::new(),
        }
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Set minimum samples for split
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set minimum samples per leaf
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Compute MSE variance
    fn mse_variance(&self, y: &[A]) -> A {
        if y.is_empty() {
            return A::zero();
        }

        let mean = y.iter().copied().sum::<A>() / A::from(y.len()).unwrap();
        y.iter()
            .map(|&val| (val - mean) * (val - mean))
            .sum::<A>()
            / A::from(y.len()).unwrap()
    }

    /// Find best split for a feature
    fn best_split(
        &self,
        X: &Array2<A>,
        y: &Array1<A>,
        indices: &[usize],
        feature: usize,
    ) -> Option<(A, A)> {
        if indices.len() < self.min_samples_split {
            return None;
        }

        // Collect feature values and sort
        let mut values: Vec<(A, usize)> = indices
            .iter()
            .map(|&idx| (X[[idx, feature]], idx))
            .collect();
        values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut best_gain = A::neg_infinity();
        let mut best_threshold = A::zero();

        let parent_y: Vec<A> = indices.iter().map(|&idx| y[idx]).collect();
        let parent_variance = self.mse_variance(&parent_y);

        // Try each unique threshold
        for i in 0..values.len() - 1 {
            if values[i].0 == values[i + 1].0 {
                continue;
            }

            let threshold = (values[i].0 + values[i + 1].0) / A::from(2.0).unwrap();

            let left_y: Vec<A> = values[..=i].iter().map(|&(_, idx)| y[idx]).collect();
            let right_y: Vec<A> = values[i + 1..].iter().map(|&(_, idx)| y[idx]).collect();

            if left_y.len() < self.min_samples_leaf || right_y.len() < self.min_samples_leaf {
                continue;
            }

            let left_variance = self.mse_variance(&left_y);
            let right_variance = self.mse_variance(&right_y);

            let n = A::from(parent_y.len()).unwrap();
            let n_left = A::from(left_y.len()).unwrap();
            let n_right = A::from(right_y.len()).unwrap();

            let gain = parent_variance
                - (n_left / n) * left_variance
                - (n_right / n) * right_variance;

            if gain > best_gain {
                best_gain = gain;
                best_threshold = threshold;
            }
        }

        if best_gain > A::zero() {
            Some((best_threshold, best_gain))
        } else {
            None
        }
    }

    /// Build tree recursively
    fn build_tree(
        &mut self,
        X: &Array2<A>,
        y: &Array1<A>,
        indices: &[usize],
        depth: usize,
    ) -> usize {
        let n_samples = indices.len();
        let n_features = X.ncols();

        // Calculate current variance
        let current_y: Vec<A> = indices.iter().map(|&idx| y[idx]).collect();
        let variance = self.mse_variance(&current_y);

        // Compute mean for prediction
        let mean = current_y.iter().copied().sum::<A>() / A::from(n_samples).unwrap();

        // Create leaf node if stopping criteria met
        let should_stop = if let Some(max_d) = self.max_depth {
            depth >= max_d
        } else {
            false
        } || n_samples < self.min_samples_split
            || variance < A::from(1e-10).unwrap();

        if should_stop {
            let node = TreeNode {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                value: Some(mean),
                n_samples,
                impurity: variance,
            };
            self.nodes.push(node);
            return self.nodes.len() - 1;
        }

        // Find best split across all features
        let mut best_feature = 0;
        let mut best_threshold = A::zero();
        let mut best_gain = A::neg_infinity();

        for feature in 0..n_features {
            if let Some((threshold, gain)) = self.best_split(X, y, indices, feature) {
                if gain > best_gain {
                    best_gain = gain;
                    best_threshold = threshold;
                    best_feature = feature;
                }
            }
        }

        // Create leaf if no good split found
        if best_gain <= A::zero() {
            let node = TreeNode {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                value: Some(mean),
                n_samples,
                impurity: variance,
            };
            self.nodes.push(node);
            return self.nodes.len() - 1;
        }

        // Split data
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            if X[[idx, best_feature]] <= best_threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // Recursively build subtrees
        let left_child = self.build_tree(X, y, &left_indices, depth + 1);
        let right_child = self.build_tree(X, y, &right_indices, depth + 1);

        // Create internal node
        let node = TreeNode {
            feature: Some(best_feature),
            threshold: Some(best_threshold),
            left: Some(left_child),
            right: Some(right_child),
            value: None,
            n_samples,
            impurity: variance,
        };
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// Fit the decision tree
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        // Build tree
        self.nodes.clear();
        let indices: Vec<usize> = (0..X.nrows()).collect();
        self.build_tree(X, y, &indices, 0);

        Ok(())
    }

    /// Predict a single sample
    fn predict_sample(&self, x: &Array1<A>) -> A {
        let mut node_idx = self.nodes.len() - 1; // Root is last node

        loop {
            let node = &self.nodes[node_idx];

            // Leaf node - return value
            if let Some(value) = node.value {
                return value;
            }

            // Internal node - follow split
            if let (Some(feature), Some(threshold)) = (node.feature, node.threshold) {
                if x[feature] <= threshold {
                    node_idx = node.left.unwrap();
                } else {
                    node_idx = node.right.unwrap();
                }
            } else {
                return A::zero(); // Shouldn't happen
            }
        }
    }

    /// Predict values for samples
    pub fn predict(&self, X: &Array2<A>) -> Result<Array1<A>> {
        if self.nodes.is_empty() {
            return Err(RustyAIError::NotFitted);
        }

        let mut predictions = Array1::zeros(X.nrows());
        for (i, row) in X.rows().into_iter().enumerate() {
            predictions[i] = self.predict_sample(&row.to_owned());
        }

        Ok(predictions)
    }

    /// Compute R² score
    pub fn score(&self, X: &Array2<A>, y: &Array1<A>) -> Result<A> {
        let predictions = self.predict(X)?;

        let y_mean = y.sum() / A::from(y.len()).unwrap();
        let ss_res: A = y
            .iter()
            .zip(predictions.iter())
            .map(|(&true_y, &pred_y)| {
                let diff = true_y - pred_y;
                diff * diff
            })
            .sum();

        let ss_tot: A = y
            .iter()
            .map(|&yi| {
                let diff = yi - y_mean;
                diff * diff
            })
            .sum();

        if ss_tot == A::zero() {
            return Ok(A::one());
        }

        Ok(A::one() - ss_res / ss_tot)
    }
}

impl<A: Float + ScalarOperand + Sum> Default for DecisionTreeRegressor<A> {
    fn default() -> Self {
        Self::new()
    }
}
