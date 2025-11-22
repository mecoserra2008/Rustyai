//! Evaluation metrics for machine learning models
//!
//! Includes metrics for regression, classification, and clustering.

use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;
use std::collections::HashMap;

/// Mean Squared Error
pub fn mean_squared_error<A: Float + ScalarOperand + Sum>(y_true: &Array1<A>, y_pred: &Array1<A>) -> A {
    if y_true.len() != y_pred.len() {
        return A::nan();
    }

    let n = A::from(y_true.len()).unwrap();
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| (yt - yp) * (yt - yp))
        .sum::<A>()
        / n
}

/// Root Mean Squared Error
pub fn root_mean_squared_error<A: Float + ScalarOperand + Sum>(y_true: &Array1<A>, y_pred: &Array1<A>) -> A {
    mean_squared_error(y_true, y_pred).sqrt()
}

/// Mean Absolute Error
pub fn mean_absolute_error<A: Float + ScalarOperand + Sum>(y_true: &Array1<A>, y_pred: &Array1<A>) -> A {
    if y_true.len() != y_pred.len() {
        return A::nan();
    }

    let n = A::from(y_true.len()).unwrap();
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| (yt - yp).abs())
        .sum::<A>()
        / n
}

/// R² (coefficient of determination) score
pub fn r2_score<A: Float + ScalarOperand + Sum>(y_true: &Array1<A>, y_pred: &Array1<A>) -> A {
    if y_true.len() != y_pred.len() {
        return A::nan();
    }

    let mean = y_true.sum() / A::from(y_true.len()).unwrap();

    let ss_tot: A = y_true.iter().map(|&y| (y - mean) * (y - mean)).sum();
    let ss_res: A = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| (yt - yp) * (yt - yp))
        .sum();

    if ss_tot < A::from(1e-10).unwrap() {
        return A::zero();
    }

    A::one() - ss_res / ss_tot
}

/// Accuracy score for classification
pub fn accuracy<A: Float + ScalarOperand + Sum>(y_true: &Array1<A>, y_pred: &Array1<A>) -> A {
    if y_true.len() != y_pred.len() {
        return A::nan();
    }

    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&yt, &yp)| (yt - yp).abs() < A::from(1e-10).unwrap())
        .count();

    A::from(correct).unwrap() / A::from(y_true.len()).unwrap()
}

/// Precision score for binary classification
pub fn precision(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mut tp = 0;
    let mut fp = 0;

    for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
        if yp > 0.5 {
            if yt > 0.5 {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }

    if tp + fp == 0 {
        return 0.0;
    }

    tp as f64 / (tp + fp) as f64
}

/// Recall score for binary classification
pub fn recall(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mut tp = 0;
    let mut fn_count = 0;

    for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
        if yt > 0.5 {
            if yp > 0.5 {
                tp += 1;
            } else {
                fn_count += 1;
            }
        }
    }

    if tp + fn_count == 0 {
        return 0.0;
    }

    tp as f64 / (tp + fn_count) as f64
}

/// F1 score for binary classification
pub fn f1_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let p = precision(y_true, y_pred);
    let r = recall(y_true, y_pred);

    if p + r == 0.0 {
        return 0.0;
    }

    2.0 * p * r / (p + r)
}

/// Confusion matrix for classification
pub fn confusion_matrix(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> Array2<usize> {
    if y_true.len() != y_pred.len() {
        return Array2::zeros((0, 0));
    }

    let n_classes = y_true.iter().chain(y_pred.iter()).max().unwrap() + 1;
    let mut matrix = Array2::zeros((n_classes, n_classes));

    for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
        matrix[[yt, yp]] += 1;
    }

    matrix
}

/// Adjusted R² score (accounts for number of predictors)
pub fn adjusted_r2_score<A: Float + ScalarOperand + Sum>(
    y_true: &Array1<A>,
    y_pred: &Array1<A>,
    n_features: usize,
) -> A {
    let r2 = r2_score(y_true, y_pred);
    let n = A::from(y_true.len()).unwrap();
    let p = A::from(n_features).unwrap();

    if n <= p + A::one() {
        return A::nan();
    }

    A::one() - (A::one() - r2) * (n - A::one()) / (n - p - A::one())
}

/// Log loss (binary cross-entropy)
pub fn log_loss(y_true: &Array1<f64>, y_pred_proba: &Array1<f64>) -> f64 {
    if y_true.len() != y_pred_proba.len() {
        return f64::NAN;
    }

    let eps = 1e-15;
    let mut loss = 0.0;

    for (&yt, &yp) in y_true.iter().zip(y_pred_proba.iter()) {
        // Clip predictions to avoid log(0)
        let yp_clipped = yp.max(eps).min(1.0 - eps);
        loss += yt * yp_clipped.ln() + (1.0 - yt) * (1.0 - yp_clipped).ln();
    }

    -loss / y_true.len() as f64
}

/// Matthews Correlation Coefficient (MCC) for binary classification
pub fn matthews_corrcoef(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_count = 0;

    for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
        let yt_bin = yt > 0.5;
        let yp_bin = yp > 0.5;

        match (yt_bin, yp_bin) {
            (true, true) => tp += 1,
            (false, false) => tn += 1,
            (false, true) => fp += 1,
            (true, false) => fn_count += 1,
        }
    }

    let numerator = (tp * tn - fp * fn_count) as f64;
    let denominator = ((tp + fp) * (tp + fn_count) * (tn + fp) * (tn + fn_count)) as f64;

    if denominator == 0.0 {
        return 0.0;
    }

    numerator / denominator.sqrt()
}

/// Classification report - compute multiple metrics at once
pub struct ClassificationReport {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl ClassificationReport {
    pub fn new(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Self {
        Self {
            accuracy: accuracy(y_true, y_pred),
            precision: precision(y_true, y_pred),
            recall: recall(y_true, y_pred),
            f1: f1_score(y_true, y_pred),
        }
    }
}

/// Regression report - compute multiple metrics at once
pub struct RegressionReport<A: Float> {
    pub mse: A,
    pub rmse: A,
    pub mae: A,
    pub r2: A,
}

impl<A: Float + ScalarOperand + Sum> RegressionReport<A> {
    pub fn new(y_true: &Array1<A>, y_pred: &Array1<A>) -> Self {
        let mse = mean_squared_error(y_true, y_pred);
        Self {
            mse,
            rmse: mse.sqrt(),
            mae: mean_absolute_error(y_true, y_pred),
            r2: r2_score(y_true, y_pred),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mean_squared_error() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Array1::from_vec(vec![1.1, 2.1, 2.9, 4.1, 4.9]);
        let mse = mean_squared_error(&y_true, &y_pred);
        assert!(mse < 0.02);
    }

    #[test]
    fn test_accuracy() {
        let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        let acc = accuracy(&y_true, &y_pred);
        assert_abs_diff_eq!(acc, 0.75, epsilon = 1e-10);
    }
}
