//! Hypothesis testing functions

use ndarray::{Array1, ScalarOperand};
use num_traits::Float;
use std::iter::Sum;

use super::{mean, std, variance};

/// Result of a t-test
#[derive(Debug, Clone)]
pub struct TTestResult<A> {
    /// t-statistic
    pub statistic: A,
    /// degrees of freedom
    pub df: usize,
    /// p-value (two-tailed)
    pub pvalue: A,
}

/// Perform a one-sample t-test
pub fn ttest_1samp<A: Float + ScalarOperand + Sum>(data: &Array1<A>, popmean: A) -> TTestResult<A> {
    let n = data.len();
    let sample_mean = mean(data);
    let sample_std = std(data, 1);

    let statistic = (sample_mean - popmean) / (sample_std / A::from(n).unwrap().sqrt());
    let df = n - 1;

    // Simple p-value approximation (for demonstration)
    // In production, use proper t-distribution CDF
    let pvalue = A::from(2.0).unwrap() * (A::one() - A::from(0.95).unwrap()); // Placeholder

    TTestResult {
        statistic,
        df,
        pvalue,
    }
}

/// Perform a two-sample independent t-test
pub fn ttest_ind<A: Float + ScalarOperand + Sum>(a: &Array1<A>, b: &Array1<A>) -> TTestResult<A> {
    let n1 = a.len();
    let n2 = b.len();
    let mean1 = mean(a);
    let mean2 = mean(b);
    let var1 = variance(a, 1);
    let var2 = variance(b, 1);

    // Pooled standard deviation
    let pooled_std =
        ((var1 * A::from(n1 - 1).unwrap() + var2 * A::from(n2 - 1).unwrap())
            / A::from(n1 + n2 - 2).unwrap())
        .sqrt();

    let statistic = (mean1 - mean2)
        / (pooled_std
            * (A::one() / A::from(n1).unwrap() + A::one() / A::from(n2).unwrap()).sqrt());

    let df = n1 + n2 - 2;
    let pvalue = A::from(0.05).unwrap(); // Placeholder

    TTestResult {
        statistic,
        df,
        pvalue,
    }
}

/// Result of an F-test
#[derive(Debug, Clone)]
pub struct FTestResult<A> {
    /// F-statistic
    pub statistic: A,
    /// degrees of freedom (numerator, denominator)
    pub df: (usize, usize),
    /// p-value
    pub pvalue: A,
}

/// Perform an F-test for equality of variances
pub fn f_test<A: Float + ScalarOperand + Sum>(a: &Array1<A>, b: &Array1<A>) -> FTestResult<A> {
    let var1 = variance(a, 1);
    let var2 = variance(b, 1);

    let statistic = if var1 > var2 { var1 / var2 } else { var2 / var1 };
    let df = (a.len() - 1, b.len() - 1);
    let pvalue = A::from(0.05).unwrap(); // Placeholder

    FTestResult {
        statistic,
        df,
        pvalue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttest_1samp() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = ttest_1samp(&data, 3.0);
        assert!(result.statistic.abs() < 1e-10);
    }
}
