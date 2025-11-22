//! Random Forest implementation

use super::DecisionTree;
use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};

/// Random Forest for regression and classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForest<A> {
    trees: Vec<DecisionTree<A>>,
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    max_features: Option<usize>,
}

impl<A: Float> RandomForest<A> {
    /// Create a new RandomForest
    pub fn new() -> Self {
        Self {
            trees: Vec::new(),
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            max_features: None,
        }
    }

    /// Builder pattern
    pub fn builder() -> RandomForestBuilder<A> {
        RandomForestBuilder::new()
    }

    /// Fit the random forest
    pub fn fit(mut self, X: &Array2<A>, y: &Array1<A>) -> Result<Self> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        self.trees.clear();
        let mut rng = thread_rng();

        for _ in 0..self.n_estimators {
            // Bootstrap sample
            let indices: Vec<usize> = (0..X.nrows())
                .map(|_| rand::random::<usize>() % X.nrows())
                .collect();

            let X_boot = X.select(ndarray::Axis(0), &indices);
            let y_boot = y.select(ndarray::Axis(0), &indices);

            // Train tree
            let tree = DecisionTree::new(self.max_depth, self.min_samples_split)
                .fit(&X_boot, &y_boot)?;

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
            predictions = predictions + tree_pred;
        }

        let n_trees = A::from(self.trees.len()).unwrap();
        Ok(predictions.mapv(|p| p / n_trees))
    }
}

impl<A: Float> Default for RandomForest<A> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for RandomForest
pub struct RandomForestBuilder<A> {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    max_features: Option<usize>,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float> RandomForestBuilder<A> {
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            max_features: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    pub fn max_depth(mut self, depth: Option<usize>) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn min_samples_split(mut self, min: usize) -> Self {
        self.min_samples_split = min;
        self
    }

    pub fn build(self) -> Result<RandomForest<A>> {
        Ok(RandomForest {
            trees: Vec::new(),
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            min_samples_split: self.min_samples_split,
            max_features: self.max_features,
        })
    }
}
