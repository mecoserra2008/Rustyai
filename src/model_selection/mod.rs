//! Model selection and hyperparameter tuning

use crate::error::Result;
use ndarray::{Array1, Array2};

/// K-Fold cross-validation
pub struct KFold {
    n_splits: usize,
}

impl KFold {
    pub fn new(n_splits: usize) -> Self {
        Self { n_splits }
    }

    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let fold_size = n_samples / self.n_splits;
        let mut splits = Vec::new();

        for i in 0..self.n_splits {
            let test_start = i * fold_size;
            let test_end = if i == self.n_splits - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let train: Vec<usize> = (0..test_start)
                .chain(test_end..n_samples)
                .collect();
            let test: Vec<usize> = (test_start..test_end).collect();

            splits.push((train, test));
        }

        splits
    }
}

/// Grid search for hyperparameter tuning
pub struct GridSearchCV {
    // To be implemented
}
