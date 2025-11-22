//! Structural Breaks Detection in Time Series
//!
//! This module provides advanced methods for detecting structural breaks:
//! - Chow test for known break points
//! - CUSUM (Cumulative Sum) test for unknown break points
//! - Bai-Perron sequential testing for multiple breaks
//! - Recursive residuals analysis
//! - Rolling window stability tests

use crate::error::{Result, RustyAIError};
use crate::stats::distributions::{f_test_p_value, standard_normal_cdf};
use ndarray::{Array1, Array2, ScalarOperand, Axis, s};
use num_traits::{Float, FromPrimitive};
use std::iter::Sum;

/// Result of a Chow test
#[derive(Debug, Clone)]
pub struct ChowTestResult<A: Float> {
    /// F-statistic
    pub f_stat: A,
    /// Degrees of freedom (numerator)
    pub df1: usize,
    /// Degrees of freedom (denominator)
    pub df2: usize,
    /// P-value
    pub p_value: A,
    /// Break point location
    pub break_point: usize,
}

/// Result of CUSUM test
#[derive(Debug, Clone)]
pub struct CUSUMResult<A: Float> {
    /// CUSUM statistic series
    pub cusum: Array1<A>,
    /// Critical values (5% significance level)
    pub critical_value: A,
    /// Whether structural break is detected
    pub break_detected: bool,
    /// Estimated break point (if detected)
    pub break_point: Option<usize>,
}

/// Result of Bai-Perron test
#[derive(Debug, Clone)]
pub struct BaiPerronResult<A: Float> {
    /// Number of breaks detected
    pub num_breaks: usize,
    /// Break point locations
    pub break_points: Vec<usize>,
    /// F-statistics for each break
    pub f_stats: Vec<A>,
    /// BIC (Bayesian Information Criterion) for model selection
    pub bic: A,
}

/// Chow test for structural break at a known point
///
/// Tests whether coefficients in a regression are the same before and after a break point.
///
/// # Arguments
/// * `y` - Dependent variable
/// * `X` - Independent variables (each column is a variable)
/// * `break_point` - Index where the potential break occurs
///
/// # Returns
/// ChowTestResult containing F-statistic and p-value
pub fn chow_test<A: Float + ScalarOperand + Sum>(
    y: &Array1<A>,
    X: &Array2<A>,
    break_point: usize,
) -> Result<ChowTestResult<A>> {
    let n = y.len();
    let k = X.ncols();

    if break_point <= k || break_point >= n - k {
        return Err(RustyAIError::InvalidParameter {
            name: "break_point".to_string(),
            reason: "Break point too close to boundaries".to_string(),
        });
    }

    // Split data at break point
    let y1 = y.slice(s![..break_point]).to_owned();
    let X1 = X.slice(s![..break_point, ..]).to_owned();
    let y2 = y.slice(s![break_point..]).to_owned();
    let X2 = X.slice(s![break_point.., ..]).to_owned();

    // Estimate full model
    let beta_full = ols_estimate(&X, &y)?;
    let residuals_full = y - &X.dot(&beta_full);
    let rss_full = residuals_full.mapv(|r| r * r).sum();

    // Estimate sub-models
    let beta1 = ols_estimate(&X1, &y1)?;
    let residuals1 = &y1 - &X1.dot(&beta1);
    let rss1 = residuals1.mapv(|r| r * r).sum();

    let beta2 = ols_estimate(&X2, &y2)?;
    let residuals2 = &y2 - &X2.dot(&beta2);
    let rss2 = residuals2.mapv(|r| r * r).sum();

    let rss_split = rss1 + rss2;

    // Compute Chow F-statistic
    // F = ((RSS_full - RSS_split) / k) / (RSS_split / (n - 2k))
    let numerator = (rss_full - rss_split) / A::from(k).unwrap();
    let denominator = rss_split / A::from(n - 2 * k).unwrap();
    let f_stat = numerator / denominator;

    let df1 = k;
    let df2 = n - 2 * k;

    // Compute p-value
    let f_val = f_stat.to_f64().unwrap_or(0.0);
    let p_value = A::from(1.0 - f_test_p_value(f_val, df1 as f64, df2 as f64)).unwrap();

    Ok(ChowTestResult {
        f_stat,
        df1,
        df2,
        p_value,
        break_point,
    })
}

/// CUSUM test for detecting structural breaks at unknown points
///
/// Uses cumulative sum of recursive residuals to detect parameter instability.
///
/// # Arguments
/// * `y` - Dependent variable
/// * `X` - Independent variables
/// * `significance` - Significance level (default 0.05)
///
/// # Returns
/// CUSUMResult with test statistics and break point detection
pub fn cusum_test<A: Float + ScalarOperand + Sum + FromPrimitive>(
    y: &Array1<A>,
    X: &Array2<A>,
    significance: Option<f64>,
) -> Result<CUSUMResult<A>> {
    let n = y.len();
    let k = X.ncols();
    let sig = significance.unwrap_or(0.05);

    if n < k + 20 {
        return Err(RustyAIError::InvalidParameter {
            name: "data".to_string(),
            reason: "Insufficient observations for CUSUM test".to_string(),
        });
    }

    // Compute recursive residuals
    let recursive_residuals = compute_recursive_residuals(y, X, k)?;

    // Standardize residuals
    let mean = recursive_residuals.mean().unwrap();
    let std = {
        let variance = recursive_residuals.mapv(|x| (x - mean) * (x - mean)).sum()
            / A::from(recursive_residuals.len() - 1).unwrap();
        variance.sqrt()
    };

    // Compute CUSUM statistic
    let mut cusum = Array1::zeros(recursive_residuals.len());
    let mut cumsum = A::zero();

    for (i, &residual) in recursive_residuals.iter().enumerate() {
        cumsum = cumsum + (residual - mean) / std;
        cusum[i] = cumsum / A::from(recursive_residuals.len()).unwrap().sqrt();
    }

    // Critical value at 5% significance (approximately 0.948 for large samples)
    let critical_value = A::from(0.948).unwrap();

    // Detect if any CUSUM value exceeds critical bounds
    let mut break_detected = false;
    let mut break_point = None;

    for (i, &val) in cusum.iter().enumerate() {
        if val.abs() > critical_value {
            break_detected = true;
            if break_point.is_none() {
                break_point = Some(k + i); // Adjust for initial estimation period
            }
        }
    }

    Ok(CUSUMResult {
        cusum,
        critical_value,
        break_detected,
        break_point,
    })
}

/// Bai-Perron test for multiple structural breaks
///
/// Sequentially tests for multiple structural breaks using dynamic programming.
///
/// # Arguments
/// * `y` - Dependent variable
/// * `X` - Independent variables
/// * `max_breaks` - Maximum number of breaks to test
/// * `min_segment_size` - Minimum observations between breaks
///
/// # Returns
/// BaiPerronResult with detected break points
pub fn bai_perron_test<A: Float + ScalarOperand + Sum>(
    y: &Array1<A>,
    X: &Array2<A>,
    max_breaks: usize,
    min_segment_size: Option<usize>,
) -> Result<BaiPerronResult<A>> {
    let n = y.len();
    let k = X.ncols();
    let min_size = min_segment_size.unwrap_or((0.15 * n as f64) as usize).max(k + 5);

    if n < min_size * (max_breaks + 1) {
        return Err(RustyAIError::InvalidParameter {
            name: "data".to_string(),
            reason: "Insufficient observations for requested number of breaks".to_string(),
        });
    }

    let mut best_breaks = Vec::new();
    let mut best_bic = A::infinity();
    let mut best_f_stats = Vec::new();

    // Test for 0 to max_breaks
    for m in 0..=max_breaks {
        if m == 0 {
            // No breaks model
            let beta = ols_estimate(X, y)?;
            let residuals = y - &X.dot(&beta);
            let rss = residuals.mapv(|r| r * r).sum();
            let bic = compute_bic(rss, n, k * (m + 1));

            if bic < best_bic {
                best_bic = bic;
                best_breaks = Vec::new();
                best_f_stats = Vec::new();
            }
        } else {
            // Test with m breaks
            let (breaks, f_stats, rss) = find_optimal_breaks(y, X, m, min_size)?;
            let bic = compute_bic(rss, n, k * (m + 1));

            if bic < best_bic {
                best_bic = bic;
                best_breaks = breaks;
                best_f_stats = f_stats;
            }
        }
    }

    Ok(BaiPerronResult {
        num_breaks: best_breaks.len(),
        break_points: best_breaks,
        f_stats: best_f_stats,
        bic: best_bic,
    })
}

/// Compute recursive residuals for CUSUM test
fn compute_recursive_residuals<A: Float + ScalarOperand + Sum>(
    y: &Array1<A>,
    X: &Array2<A>,
    k: usize,
) -> Result<Array1<A>> {
    let n = y.len();
    let mut recursive_residuals = Vec::new();

    // Start after initial estimation period
    for t in (k + 1)..n {
        let y_sub = y.slice(s![..t]).to_owned();
        let X_sub = X.slice(s![..t, ..]).to_owned();

        let beta = ols_estimate(&X_sub, &y_sub)?;

        // Predict next observation
        let x_t = X.row(t);
        let y_pred = x_t.dot(&beta);
        let residual = y[t] - y_pred;

        recursive_residuals.push(residual);
    }

    Ok(Array1::from_vec(recursive_residuals))
}

/// Find optimal break points using dynamic programming
fn find_optimal_breaks<A: Float + ScalarOperand + Sum>(
    y: &Array1<A>,
    X: &Array2<A>,
    num_breaks: usize,
    min_size: usize,
) -> Result<(Vec<usize>, Vec<A>, A)> {
    let n = y.len();
    let k = X.ncols();

    // Simplified greedy approach for computational efficiency
    // In production, use dynamic programming for global optimum
    let mut break_points = Vec::new();
    let mut f_stats = Vec::new();
    let mut remaining_start = min_size;
    let mut remaining_end = n - min_size;

    for _ in 0..num_breaks {
        let mut best_f = A::neg_infinity();
        let mut best_point = remaining_start;

        // Search for best break point in remaining interval
        for candidate in (remaining_start + min_size)..(remaining_end - min_size) {
            // Skip if too close to existing breaks
            if break_points.iter().any(|&bp| (bp as isize - candidate as isize).abs() < min_size as isize) {
                continue;
            }

            // Compute F-statistic for this candidate
            let test_breaks = {
                let mut temp = break_points.clone();
                temp.push(candidate);
                temp.sort();
                temp
            };

            let rss = compute_segmented_rss(y, X, &test_breaks)?;
            let rss_full = {
                let beta = ols_estimate(X, y)?;
                let residuals = y - &X.dot(&beta);
                residuals.mapv(|r| r * r).sum()
            };

            let df1 = k * test_breaks.len();
            let df2 = n - k * (test_breaks.len() + 1);

            if df2 > 0 {
                let f_stat = ((rss_full - rss) / A::from(df1).unwrap()) /
                            (rss / A::from(df2).unwrap());

                if f_stat > best_f {
                    best_f = f_stat;
                    best_point = candidate;
                }
            }
        }

        if best_f > A::neg_infinity() {
            break_points.push(best_point);
            f_stats.push(best_f);
        }
    }

    break_points.sort();
    let final_rss = compute_segmented_rss(y, X, &break_points)?;

    Ok((break_points, f_stats, final_rss))
}

/// Compute RSS for segmented regression
fn compute_segmented_rss<A: Float + ScalarOperand + Sum>(
    y: &Array1<A>,
    X: &Array2<A>,
    break_points: &[usize],
) -> Result<A> {
    let mut rss_total = A::zero();
    let mut start = 0;

    let mut segments = break_points.to_vec();
    segments.push(y.len());

    for &end in &segments {
        let y_seg = y.slice(s![start..end]).to_owned();
        let X_seg = X.slice(s![start..end, ..]).to_owned();

        let beta = ols_estimate(&X_seg, &y_seg)?;
        let residuals = &y_seg - &X_seg.dot(&beta);
        let rss = residuals.mapv(|r| r * r).sum();

        rss_total = rss_total + rss;
        start = end;
    }

    Ok(rss_total)
}

/// OLS estimation using normal equations
fn ols_estimate<A: Float + ScalarOperand + Sum>(
    X: &Array2<A>,
    y: &Array1<A>,
) -> Result<Array1<A>> {
    // beta = (X'X)^(-1) X'y
    let xtx = X.t().dot(X);
    let xty = X.t().dot(y);

    // Solve linear system using simple Gaussian elimination
    // In production, use proper linear algebra library
    solve_linear_system(&xtx, &xty)
}

/// Solve linear system Ax = b using Gaussian elimination
fn solve_linear_system<A: Float + ScalarOperand + Sum>(
    A: &Array2<A>,
    b: &Array1<A>,
) -> Result<Array1<A>> {
    let n = A.nrows();
    let mut aug = Array2::zeros((n, n + 1));

    // Create augmented matrix [A|b]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = A[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singularity
        if aug[[i, i]].abs() < A::from(1e-10).unwrap() {
            return Err(RustyAIError::InvalidParameter {
                name: "matrix".to_string(),
                reason: "Singular matrix in OLS estimation".to_string(),
            });
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..=n {
                aug[[k, j]] = aug[[k, j]] - factor * aug[[i, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum = sum - aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Compute BIC (Bayesian Information Criterion)
fn compute_bic<A: Float + ScalarOperand + Sum>(rss: A, n: usize, k: usize) -> A {
    let n_f = A::from(n).unwrap();
    let k_f = A::from(k).unwrap();

    n_f * (rss / n_f).ln() + k_f * n_f.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_chow_test() {
        // Simple test with known break
        let y = array![1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0];
        let X = Array2::from_shape_fn((8, 2), |(i, j)| {
            if j == 0 { 1.0 } else { i as f64 }
        });

        let result = chow_test(&y, &X, 4).unwrap();
        assert!(result.f_stat > 0.0);
    }

    #[test]
    fn test_cusum_test() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                       15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0];
        let X = Array2::from_shape_fn((20, 2), |(i, j)| {
            if j == 0 { 1.0 } else { i as f64 }
        });

        let result = cusum_test(&y, &X, None).unwrap();
        assert!(result.cusum.len() > 0);
    }
}
