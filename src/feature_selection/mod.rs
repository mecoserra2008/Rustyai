//! Feature selection methods

use crate::error::{Result, RustyAIError};
use ndarray::{Array1, Array2, Axis, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

/// Scoring function for SelectKBest
#[derive(Debug, Clone, Copy)]
pub enum ScoreFunc {
    /// F-statistic for classification (ANOVA F-test)
    FClassif,
    /// F-statistic for regression
    FRegression,
    /// Mutual information for classification
    MutualInfoClassif,
}

/// Select K best features based on statistical tests
///
/// SelectKBest removes all but the k highest scoring features.
/// It uses univariate statistical tests to select features.
pub struct SelectKBest<A: Float> {
    k: usize,
    score_func: ScoreFunc,
    scores_: Option<Array1<A>>,
    selected_features_: Option<Vec<usize>>,
}

impl<A: Float + ScalarOperand + Sum> SelectKBest<A> {
    /// Create a new SelectKBest feature selector
    ///
    /// # Arguments
    /// * `k` - Number of top features to select
    /// * `score_func` - Scoring function to use
    pub fn new(k: usize, score_func: ScoreFunc) -> Self {
        Self {
            k,
            score_func,
            scores_: None,
            selected_features_: None,
        }
    }

    /// Compute F-statistic for classification (ANOVA F-test)
    fn f_classif(&self, X: &Array2<A>, y: &Array1<A>) -> Array1<A> {
        let n_features = X.ncols();
        let mut f_scores = Array1::zeros(n_features);

        // Get unique classes
        let mut classes_vec: Vec<i32> = y
            .iter()
            .map(|&val| val.to_i32().unwrap_or(0))
            .collect();
        classes_vec.sort_unstable();
        classes_vec.dedup();

        for feature_idx in 0..n_features {
            let feature = X.column(feature_idx);

            // Compute between-group and within-group variance
            let mut group_means = Vec::new();
            let mut group_sizes = Vec::new();

            let overall_mean: A = feature.sum() / A::from(feature.len()).unwrap();

            for &class in &classes_vec {
                let class_a = A::from(class).unwrap();
                let mask: Vec<bool> = y.iter().map(|&val| (val - class_a).abs() < A::from(1e-10).unwrap()).collect();
                let group_values: Vec<A> = feature
                    .iter()
                    .zip(mask.iter())
                    .filter_map(|(&val, &m)| if m { Some(val) } else { None })
                    .collect();

                if !group_values.is_empty() {
                    let group_mean = group_values.iter().copied().sum::<A>()
                        / A::from(group_values.len()).unwrap();
                    group_means.push(group_mean);
                    group_sizes.push(group_values.len());
                }
            }

            // Between-group variance
            let mut ss_between = A::zero();
            for (mean, size) in group_means.iter().zip(group_sizes.iter()) {
                let diff = *mean - overall_mean;
                ss_between = ss_between + A::from(*size).unwrap() * diff * diff;
            }

            // Within-group variance
            let mut ss_within = A::zero();
            for (class_idx, &class) in classes_vec.iter().enumerate() {
                let class_a = A::from(class).unwrap();
                let mask: Vec<bool> = y.iter().map(|&val| (val - class_a).abs() < A::from(1e-10).unwrap()).collect();
                let group_values: Vec<A> = feature
                    .iter()
                    .zip(mask.iter())
                    .filter_map(|(&val, &m)| if m { Some(val) } else { None })
                    .collect();

                if !group_values.is_empty() && class_idx < group_means.len() {
                    let group_mean = group_means[class_idx];
                    for &val in &group_values {
                        let diff = val - group_mean;
                        ss_within = ss_within + diff * diff;
                    }
                }
            }

            // F-statistic
            let df_between = A::from(classes_vec.len() - 1).unwrap();
            let df_within = A::from(feature.len() - classes_vec.len()).unwrap();

            let f_stat = if ss_within > A::zero() {
                (ss_between / df_between) / (ss_within / df_within)
            } else {
                A::zero()
            };

            f_scores[feature_idx] = f_stat;
        }

        f_scores
    }

    /// Compute F-statistic for regression
    fn f_regression(&self, X: &Array2<A>, y: &Array1<A>) -> Array1<A> {
        let n_features = X.ncols();
        let n_samples = X.nrows();
        let mut f_scores = Array1::zeros(n_features);

        let y_mean = y.sum() / A::from(n_samples).unwrap();
        let y_var: A = y
            .iter()
            .map(|&val| {
                let diff = val - y_mean;
                diff * diff
            })
            .sum::<A>()
            / A::from(n_samples).unwrap();

        for feature_idx in 0..n_features {
            let feature = X.column(feature_idx);
            let x_mean = feature.sum() / A::from(n_samples).unwrap();

            // Compute correlation
            let mut cov = A::zero();
            let mut x_var = A::zero();

            for i in 0..n_samples {
                let x_diff = feature[i] - x_mean;
                let y_diff = y[i] - y_mean;
                cov = cov + x_diff * y_diff;
                x_var = x_var + x_diff * x_diff;
            }

            cov = cov / A::from(n_samples).unwrap();
            x_var = x_var / A::from(n_samples).unwrap();

            // Correlation coefficient
            let r = if x_var > A::zero() && y_var > A::zero() {
                cov / (x_var * y_var).sqrt()
            } else {
                A::zero()
            };

            // F-statistic from correlation
            let r2 = r * r;
            let df = A::from(n_samples - 2).unwrap();
            let f_stat = if r2 < A::one() {
                r2 * df / (A::one() - r2)
            } else {
                A::zero()
            };

            f_scores[feature_idx] = f_stat;
        }

        f_scores
    }

    /// Fit the feature selector
    pub fn fit(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<&mut Self> {
        if X.nrows() != y.len() {
            return Err(RustyAIError::DimensionMismatch(format!(
                "X has {} samples but y has {} samples",
                X.nrows(),
                y.len()
            )));
        }

        // Compute scores based on the scoring function
        let scores = match self.score_func {
            ScoreFunc::FClassif => self.f_classif(X, y),
            ScoreFunc::FRegression => self.f_regression(X, y),
            ScoreFunc::MutualInfoClassif => {
                // Simplified: use F-classif for now
                self.f_classif(X, y)
            }
        };

        // Select top k features
        let mut score_indices: Vec<(usize, A)> = scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();
        score_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let k = self.k.min(X.ncols());
        let selected: Vec<usize> = score_indices.iter().take(k).map(|(idx, _)| *idx).collect();

        self.scores_ = Some(scores);
        self.selected_features_ = Some(selected);

        Ok(self)
    }

    /// Transform X to the selected features
    pub fn transform(&self, X: &Array2<A>) -> Result<Array2<A>> {
        let selected = self
            .selected_features_
            .as_ref()
            .ok_or(RustyAIError::NotFitted)?;

        let mut X_new = Array2::zeros((X.nrows(), selected.len()));
        for (new_idx, &old_idx) in selected.iter().enumerate() {
            for row in 0..X.nrows() {
                X_new[[row, new_idx]] = X[[row, old_idx]];
            }
        }

        Ok(X_new)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, X: &Array2<A>, y: &Array1<A>) -> Result<Array2<A>> {
        self.fit(X, y)?;
        self.transform(X)
    }

    /// Get the feature scores
    pub fn scores(&self) -> Option<&Array1<A>> {
        self.scores_.as_ref()
    }

    /// Get selected feature indices
    pub fn selected_features(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }
}

/// Recursive Feature Elimination
///
/// Feature ranking with recursive feature elimination.
/// Given an external estimator that assigns weights to features,
/// RFE recursively removes features and builds a model on the remaining features
/// until the desired number of features is reached.
pub struct RFE<A: Float> {
    n_features_to_select: usize,
    step: usize,
    selected_features_: Option<Vec<usize>>,
    ranking_: Option<Vec<usize>>,
    _phantom: std::marker::PhantomData<A>,
}

impl<A: Float> RFE<A> {
    /// Create a new RFE feature selector
    ///
    /// # Arguments
    /// * `n_features_to_select` - Number of features to select
    /// * `step` - Number of features to remove at each iteration
    pub fn new(n_features_to_select: usize, step: usize) -> Self {
        Self {
            n_features_to_select,
            step: step.max(1),
            selected_features_: None,
            ranking_: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get selected feature indices
    pub fn selected_features(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }

    /// Get feature ranking (1 = best)
    pub fn ranking(&self) -> Option<&Vec<usize>> {
        self.ranking_.as_ref()
    }
}
