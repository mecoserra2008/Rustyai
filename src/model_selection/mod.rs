//! Model selection and hyperparameter tuning
//!
//! Tools for cross-validation, train-test splitting, bootstrap resampling,
//! and hyperparameter tuning.

use crate::error::Result;
use ndarray::{Array1, Array2, Axis};
use num_traits::Float;
use std::iter::Sum;

pub mod bootstrap;

pub use bootstrap::{
    Bootstrap, StratifiedBootstrap, BlockBootstrap, Jackknife,
    BalancedBootstrap, BootstrapCI, PermutationTest,
};

/// K-Fold cross-validation
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
}

impl KFold {
    /// Create a new K-Fold splitter
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: false,
        }
    }

    /// Enable shuffling before splitting
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Generate train/test indices for each fold
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        let fold_size = n_samples / self.n_splits;
        let mut splits = Vec::new();

        for i in 0..self.n_splits {
            let test_start = i * fold_size;
            let test_end = if i == self.n_splits - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let train: Vec<usize> = indices[0..test_start]
                .iter()
                .chain(&indices[test_end..n_samples])
                .copied()
                .collect();
            let test: Vec<usize> = indices[test_start..test_end].to_vec();

            splits.push((train, test));
        }

        splits
    }

    /// Split data matrices into train/test folds
    pub fn split_data<A: Float + Copy>(
        &self,
        X: &Array2<A>,
        y: &Array1<A>,
    ) -> Vec<(Array2<A>, Array1<A>, Array2<A>, Array1<A>)> {
        let splits = self.split(X.nrows());
        let mut result = Vec::new();

        for (train_idx, test_idx) in splits {
            let X_train = select_rows(X, &train_idx);
            let y_train = select_elements(y, &train_idx);
            let X_test = select_rows(X, &test_idx);
            let y_test = select_elements(y, &test_idx);

            result.push((X_train, y_train, X_test, y_test));
        }

        result
    }
}

/// Train-test split
///
/// Split arrays or matrices into random train and test subsets.
/// Returns (X_train, X_test, y_train, y_test)
pub fn train_test_split<A: Float + Copy>(
    X: &Array2<A>,
    y: &Array1<A>,
    test_size: f64,
    shuffle: bool,
) -> (Array2<A>, Array2<A>, Array1<A>, Array1<A>) {
    let n_samples = X.nrows();
    let n_test = (n_samples as f64 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    let mut indices: Vec<usize> = (0..n_samples).collect();

    if shuffle {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
    }

    let train_idx = &indices[0..n_train];
    let test_idx = &indices[n_train..];

    let X_train = select_rows(X, train_idx);
    let X_test = select_rows(X, test_idx);
    let y_train = select_elements(y, train_idx);
    let y_test = select_elements(y, test_idx);

    (X_train, X_test, y_train, y_test)
}

/// Helper function to select rows from a matrix by indices
fn select_rows<A: Float + Copy>(X: &Array2<A>, indices: &[usize]) -> Array2<A> {
    let n_cols = X.ncols();
    let mut result = Array2::zeros((indices.len(), n_cols));

    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..n_cols {
            result[[i, j]] = X[[idx, j]];
        }
    }

    result
}

/// Helper function to select elements from an array by indices
fn select_elements<A: Float + Copy>(arr: &Array1<A>, indices: &[usize]) -> Array1<A> {
    let mut result = Array1::zeros(indices.len());

    for (i, &idx) in indices.iter().enumerate() {
        result[i] = arr[idx];
    }

    result
}

/// Cross-validation score
///
/// Evaluate a metric across all folds and return the scores.
pub fn cross_val_score<A, F>(
    X: &Array2<A>,
    y: &Array1<A>,
    cv: &KFold,
    mut score_fn: F,
) -> Vec<A>
where
    A: Float + Copy + Sum,
    F: FnMut(&Array2<A>, &Array1<A>, &Array2<A>, &Array1<A>) -> A,
{
    let folds = cv.split_data(X, y);
    let mut scores = Vec::new();

    for (X_train, y_train, X_test, y_test) in folds {
        let score = score_fn(&X_train, &y_train, &X_test, &y_test);
        scores.push(score);
    }

    scores
}

/// Parameter grid for GridSearchCV
pub type ParameterGrid = std::collections::HashMap<String, Vec<f64>>;

/// Grid search for hyperparameter tuning
///
/// GridSearchCV exhaustively searches over specified parameter values
/// for an estimator and selects the best combination based on cross-validation.
pub struct GridSearchCV {
    cv: usize,
    best_params_: Option<std::collections::HashMap<String, f64>>,
    best_score_: Option<f64>,
    cv_results_: Option<Vec<(std::collections::HashMap<String, f64>, f64)>>,
}

impl GridSearchCV {
    /// Create a new GridSearchCV
    ///
    /// # Arguments
    /// * `cv` - Number of cross-validation folds
    pub fn new(cv: usize) -> Self {
        Self {
            cv,
            best_params_: None,
            best_score_: None,
            cv_results_: None,
        }
    }

    /// Generate all combinations from parameter grid
    fn generate_param_combinations(
        &self,
        param_grid: &ParameterGrid,
    ) -> Vec<std::collections::HashMap<String, f64>> {
        let keys: Vec<&String> = param_grid.keys().collect();
        let mut combinations = Vec::new();

        if keys.is_empty() {
            return combinations;
        }

        // Recursive helper to generate combinations
        fn generate_recursive(
            keys: &[&String],
            param_grid: &ParameterGrid,
            current: std::collections::HashMap<String, f64>,
            combinations: &mut Vec<std::collections::HashMap<String, f64>>,
        ) {
            if keys.is_empty() {
                combinations.push(current);
                return;
            }

            let key = keys[0];
            let values = &param_grid[key];

            for &value in values {
                let mut new_current = current.clone();
                new_current.insert(key.clone(), value);
                generate_recursive(&keys[1..], param_grid, new_current, combinations);
            }
        }

        generate_recursive(&keys, param_grid, std::collections::HashMap::new(), &mut combinations);
        combinations
    }

    /// Fit the grid search (simplified version for demonstration)
    ///
    /// Note: This is a basic implementation that demonstrates the structure.
    /// A full implementation would need to work with generic estimators.
    pub fn fit_simple<A: Float + Copy>(
        &mut self,
        X: &Array2<A>,
        y: &Array1<A>,
        param_grid: &ParameterGrid,
        mut score_fn: impl FnMut(&Array2<A>, &Array1<A>, &Array2<A>, &Array1<A>, &std::collections::HashMap<String, f64>) -> A,
    ) -> Result<()> {
        let kfold = KFold::new(self.cv);
        let folds = kfold.split_data(X, y);

        let param_combinations = self.generate_param_combinations(param_grid);
        let mut results = Vec::new();
        let mut best_score = A::neg_infinity();
        let mut best_params = std::collections::HashMap::new();

        // Try each parameter combination
        for params in param_combinations {
            let mut fold_scores = Vec::new();

            // Evaluate on each fold
            for (X_train, y_train, X_test, y_test) in &folds {
                let score = score_fn(X_train, y_train, X_test, y_test, &params);
                fold_scores.push(score);
            }

            // Compute mean score
            let sum: A = fold_scores.iter().fold(A::zero(), |acc, &x| acc + x);
            let mean_score: A = sum / A::from(fold_scores.len()).unwrap();

            results.push((params.clone(), mean_score.to_f64().unwrap_or(0.0)));

            if mean_score > best_score {
                best_score = mean_score;
                best_params = params;
            }
        }

        self.best_params_ = Some(best_params);
        self.best_score_ = Some(best_score.to_f64().unwrap_or(0.0));
        self.cv_results_ = Some(results);

        Ok(())
    }

    /// Get the best parameters
    pub fn best_params(&self) -> Option<&std::collections::HashMap<String, f64>> {
        self.best_params_.as_ref()
    }

    /// Get the best score
    pub fn best_score(&self) -> Option<f64> {
        self.best_score_
    }

    /// Get all cross-validation results
    pub fn cv_results(&self) -> Option<&Vec<(std::collections::HashMap<String, f64>, f64)>> {
        self.cv_results_.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kfold_split() {
        let kf = KFold::new(5);
        let splits = kf.split(100);

        assert_eq!(splits.len(), 5);

        // Check that all samples are used
        for (train, test) in &splits {
            assert_eq!(train.len() + test.len(), 100);
        }
    }

    #[test]
    fn test_train_test_split() {
        let X = Array2::from_shape_vec((100, 5), (0..500).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..100).map(|x| x as f64).collect());

        let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.2, false);

        assert_eq!(X_train.nrows(), 80);
        assert_eq!(X_test.nrows(), 20);
        assert_eq!(y_train.len(), 80);
        assert_eq!(y_test.len(), 20);
    }
}
