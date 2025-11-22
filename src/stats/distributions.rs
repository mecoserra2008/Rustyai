//! Probability distributions

use ndarray::ScalarOperand;
use num_traits::Float;
use std::iter::Sum;
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::f64::consts::PI;

/// Probability density function of the normal distribution
pub fn normal_pdf<A: Float + ScalarOperand + Sum>(x: A, mu: A, sigma: A) -> A {
    let sqrt_2pi = A::from((2.0 * PI).sqrt()).unwrap();
    let coefficient = A::one() / (sigma * sqrt_2pi);
    let exponent = -((x - mu) * (x - mu)) / (A::from(2.0).unwrap() * sigma * sigma);
    coefficient * exponent.exp()
}

/// Cumulative distribution function of the standard normal distribution
pub fn standard_normal_cdf<A: Float + ScalarOperand + Sum>(x: A) -> A {
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
pub fn normal_cdf<A: Float + ScalarOperand + Sum>(x: A, mu: A, sigma: A) -> A {
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

/// Student's t-distribution CDF (two-tailed p-value)
pub fn t_cdf(t: f64, df: f64) -> f64 {
    // For large df, use normal approximation
    if df > 30.0 {
        return standard_normal_cdf(t);
    }

    // Simpson's rule numerical integration
    let steps = 1000;
    let lower = -10.0;
    let upper = t;
    let h = (upper - lower) / steps as f64;

    let mut sum = t_pdf(lower, df) + t_pdf(upper, df);
    for i in 1..steps {
        let x = lower + i as f64 * h;
        let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * t_pdf(x, df);
    }

    sum * h / 3.0
}

/// Two-tailed t-test p-value
pub fn t_test_p_value(t: f64, df: f64) -> f64 {
    2.0 * (1.0 - t_cdf(t.abs(), df))
}

/// F-distribution PDF
pub fn f_pdf(x: f64, df1: f64, df2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    let d1_half = df1 / 2.0;
    let d2_half = df2 / 2.0;

    let numerator = (df1 / df2).powf(d1_half) * x.powf(d1_half - 1.0);
    let denominator = beta(d1_half, d2_half) * (1.0 + (df1 / df2) * x).powf(d1_half + d2_half);

    numerator / denominator
}

/// F-distribution CDF approximation
pub fn f_cdf(f: f64, df1: f64, df2: f64) -> f64 {
    // Simplified approximation - for production use beta distribution relationship
    if f <= 0.0 {
        return 0.0;
    }

    let x = df2 / (df2 + df1 * f);
    incomplete_beta(df2 / 2.0, df1 / 2.0, x)
}

/// F-test p-value
pub fn f_test_p_value(f: f64, df1: f64, df2: f64) -> f64 {
    1.0 - f_cdf(f, df1, df2)
}

/// Chi-squared CDF using gamma distribution relationship
pub fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // Chi-squared(k) = Gamma(k/2, 2)
    let k_half = df / 2.0;
    incomplete_gamma(k_half, x / 2.0)
}

/// Chi-squared p-value
pub fn chi_squared_p_value(chi2: f64, df: f64) -> f64 {
    1.0 - chi_squared_cdf(chi2, df)
}

/// Beta function
fn beta(a: f64, b: f64) -> f64 {
    gamma(a) * gamma(b) / gamma(a + b)
}

/// Incomplete beta function approximation
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Simplified approximation
    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (gamma(a + b) / (gamma(a) * gamma(b))) * x.powf(a) * (1.0 - x).powf(b)
    };

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for beta function
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-15;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;

    if d.abs() < EPS {
        d = EPS;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < EPS {
            d = EPS;
        }
        c = 1.0 + aa / c;
        if c.abs() < EPS {
            c = EPS;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < EPS {
            d = EPS;
        }
        c = 1.0 + aa / c;
        if c.abs() < EPS {
            c = EPS;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPS {
            break;
        }
    }

    h
}

/// Incomplete gamma function
fn incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_continued_fraction(a, x)
    }
}

/// Gamma function series expansion
fn gamma_series(a: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-15;

    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;

    for n in 1..=MAX_ITER {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < EPS * sum.abs() {
            break;
        }
    }

    sum * (-x).exp() * x.powf(a) / gamma(a)
}

/// Gamma continued fraction
fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-15;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / EPS;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=MAX_ITER {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < EPS {
            d = EPS;
        }
        c = b + an / c;
        if c.abs() < EPS {
            c = EPS;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }

    (-x).exp() * x.powf(a) * h / gamma(a)
}

/// Gamma function approximation using Lanczos approximation
fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        // Lanczos approximation
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
