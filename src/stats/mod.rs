//! Statistical functions and probability distributions
//!
//! Comprehensive statistical toolkit including distributions, hypothesis tests,
//! and statistical measures.

use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal, Uniform};
use std::f64::consts::PI;

pub mod distributions;
pub mod hypothesis;

/// Compute the mean of an array
pub fn mean<A: Float>(data: &Array1<A>) -> A {
    if data.is_empty() {
        return A::zero();
    }
    data.sum() / A::from(data.len()).unwrap()
}

/// Compute the variance of an array
pub fn variance<A: Float>(data: &Array1<A>, ddof: usize) -> A {
    if data.len() <= ddof {
        return A::zero();
    }

    let m = mean(data);
    let sum_sq_diff: A = data.iter().map(|&x| (x - m) * (x - m)).sum();
    sum_sq_diff / A::from(data.len() - ddof).unwrap()
}

/// Compute the standard deviation of an array
pub fn std<A: Float>(data: &Array1<A>, ddof: usize) -> A {
    variance(data, ddof).sqrt()
}

/// Compute the median of an array
pub fn median<A: Float>(data: &Array1<A>) -> Option<A> {
    if data.is_empty() {
        return None;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        Some((sorted[mid - 1] + sorted[mid]) / A::from(2.0).unwrap())
    } else {
        Some(sorted[mid])
    }
}

/// Compute quantiles of an array
pub fn quantile<A: Float>(data: &Array1<A>, q: A) -> Option<A> {
    if data.is_empty() || q < A::zero() || q > A::one() {
        return None;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pos = q * A::from(sorted.len() - 1).unwrap();
    let idx = pos.floor().to_usize().unwrap();

    if idx >= sorted.len() - 1 {
        return Some(sorted[sorted.len() - 1]);
    }

    let frac = pos - A::from(idx).unwrap();
    Some(sorted[idx] * (A::one() - frac) + sorted[idx + 1] * frac)
}

/// Compute covariance between two arrays
pub fn covariance<A: Float>(x: &Array1<A>, y: &Array1<A>, ddof: usize) -> Option<A> {
    if x.len() != y.len() || x.len() <= ddof {
        return None;
    }

    let mean_x = mean(x);
    let mean_y = mean(y);

    let cov: A = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    Some(cov / A::from(x.len() - ddof).unwrap())
}

/// Compute Pearson correlation coefficient
pub fn correlation<A: Float>(x: &Array1<A>, y: &Array1<A>) -> Option<A> {
    if x.len() != y.len() {
        return None;
    }

    let cov = covariance(x, y, 1)?;
    let std_x = std(x, 1);
    let std_y = std(y, 1);

    if std_x == A::zero() || std_y == A::zero() {
        return None;
    }

    Some(cov / (std_x * std_y))
}

/// Compute covariance matrix
pub fn cov_matrix<A: Float>(data: &Array2<A>, ddof: usize) -> Array2<A> {
    let n_features = data.ncols();
    let n_samples = data.nrows();
    let mut cov = Array2::zeros((n_features, n_features));

    // Compute means
    let means: Vec<A> = (0..n_features)
        .map(|i| {
            let col = data.column(i);
            col.sum() / A::from(n_samples).unwrap()
        })
        .collect();

    // Compute covariances
    for i in 0..n_features {
        for j in i..n_features {
            let mut sum = A::zero();
            for k in 0..n_samples {
                sum = sum + (data[[k, i]] - means[i]) * (data[[k, j]] - means[j]);
            }
            let val = sum / A::from(n_samples - ddof).unwrap();
            cov[[i, j]] = val;
            cov[[j, i]] = val;
        }
    }

    cov
}

/// Compute correlation matrix
pub fn corr_matrix<A: Float>(data: &Array2<A>) -> Array2<A> {
    let cov = cov_matrix(data, 1);
    let n = cov.nrows();
    let mut corr = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let std_i = cov[[i, i]].sqrt();
            let std_j = cov[[j, j]].sqrt();
            if std_i > A::zero() && std_j > A::zero() {
                corr[[i, j]] = cov[[i, j]] / (std_i * std_j);
            }
        }
    }

    corr
}

/// Compute skewness
pub fn skewness<A: Float>(data: &Array1<A>) -> A {
    let n = A::from(data.len()).unwrap();
    let m = mean(data);
    let s = std(data, 1);

    if s == A::zero() {
        return A::zero();
    }

    let skew: A = data.iter().map(|&x| ((x - m) / s).powi(3)).sum();
    skew * n / ((n - A::one()) * (n - A::from(2.0).unwrap()))
}

/// Compute kurtosis
pub fn kurtosis<A: Float>(data: &Array1<A>) -> A {
    let n = A::from(data.len()).unwrap();
    let m = mean(data);
    let s = std(data, 1);

    if s == A::zero() {
        return A::zero();
    }

    let kurt: A = data.iter().map(|&x| ((x - m) / s).powi(4)).sum();
    kurt / n - A::from(3.0).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mean() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_abs_diff_eq!(mean(&data), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_variance() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let var = variance(&data, 1);
        assert_abs_diff_eq!(var, 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_correlation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let corr = correlation(&x, &y).unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);
    }
}
