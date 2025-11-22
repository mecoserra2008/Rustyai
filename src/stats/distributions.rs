//! Probability distributions

use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::f64::consts::PI;

/// Probability density function of the normal distribution
pub fn normal_pdf<A: Float>(x: A, mu: A, sigma: A) -> A {
    let sqrt_2pi = A::from((2.0 * PI).sqrt()).unwrap();
    let coefficient = A::one() / (sigma * sqrt_2pi);
    let exponent = -((x - mu) * (x - mu)) / (A::from(2.0).unwrap() * sigma * sigma);
    coefficient * exponent.exp()
}

/// Cumulative distribution function of the standard normal distribution
pub fn standard_normal_cdf<A: Float>(x: A) -> A {
    // Using error function approximation
    let a1 = A::from(0.254829592).unwrap();
    let a2 = A::from(-0.284496736).unwrap();
    let a3 = A::from(1.421413741).unwrap();
    let a4 = A::from(-1.453152027).unwrap();
    let a5 = A::from(1.061405429).unwrap();
    let p = A::from(0.3275911).unwrap();

    let sign = if x < A::zero() {
        -A::one()
    } else {
        A::one()
    };
    let x_abs = x.abs();

    let t = A::one() / (A::one() + p * x_abs);
    let y = A::one()
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
            * (-(x_abs * x_abs) / A::from(2.0).unwrap()).exp();

    (A::one() + sign * y) / A::from(2.0).unwrap()
}

/// Normal distribution CDF
pub fn normal_cdf<A: Float>(x: A, mu: A, sigma: A) -> A {
    standard_normal_cdf((x - mu) / sigma)
}

/// Chi-squared distribution PDF
pub fn chi_squared_pdf(x: f64, k: usize) -> f64 {
    if x <= 0.0 || k == 0 {
        return 0.0;
    }

    let k_half = k as f64 / 2.0;
    let numerator = x.powf(k_half - 1.0) * (-x / 2.0).exp();
    let denominator = 2.0f64.powf(k_half) * gamma(k_half);

    numerator / denominator
}

/// Student's t-distribution PDF
pub fn t_pdf(x: f64, df: f64) -> f64 {
    let numerator = gamma((df + 1.0) / 2.0);
    let denominator = (df * PI).sqrt() * gamma(df / 2.0);
    let factor = (1.0 + x * x / df).powf(-(df + 1.0) / 2.0);

    numerator / denominator * factor
}

/// Gamma function approximation using Stirling's formula
fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        // Stirling's approximation
        let x = x - 1.0;
        let tmp = x + 5.5;
        let tmp = (x + 0.5) * tmp.ln() - tmp;
        let ser = 1.000000000190015
            + 76.18009172947146 / (x + 1.0)
            - 86.50532032941677 / (x + 2.0)
            + 24.01409824083091 / (x + 3.0)
            - 1.231739572450155 / (x + 4.0)
            + 0.1208650973866179e-2 / (x + 5.0)
            - 0.5395239384953e-5 / (x + 6.0);

        (tmp + ser.ln()).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_normal_pdf() {
        let pdf = normal_pdf(0.0, 0.0, 1.0);
        assert_abs_diff_eq!(pdf, 0.3989422804014327, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_cdf() {
        let cdf = normal_cdf(0.0, 0.0, 1.0);
        assert_abs_diff_eq!(cdf, 0.5, epsilon = 1e-6);
    }
}
