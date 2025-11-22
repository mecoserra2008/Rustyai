//! Dataset loaders and generators
//!
//! Includes classic datasets (Iris, Boston, etc.) and synthetic data generators.

use crate::error::Result;
use ndarray::{Array1, Array2};

/// Load the Iris dataset
pub fn load_iris() -> Result<(Array2<f64>, Array1<f64>)> {
    // Simplified Iris dataset (first few samples of each class)
    let X = Array2::from_shape_vec(
        (15, 4),
        vec![
            // Setosa
            5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
            3.6, 1.4, 0.2, // Versicolor
            7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5,
            2.8, 4.6, 1.5, // Virginica
            6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5,
            3.0, 5.8, 2.2,
        ],
    )?;

    let y = Array1::from_vec(vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    ]);

    Ok((X, y))
}

/// Generate synthetic regression data
pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    noise: f64,
) -> Result<(Array2<f64>, Array1<f64>)> {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate random X
    let X: Vec<f64> = (0..n_samples * n_features)
        .map(|_| normal.sample(&mut rng))
        .collect();
    let X = Array2::from_shape_vec((n_samples, n_features), X)?;

    // Generate true coefficients
    let coef: Array1<f64> = Array1::from_vec(
        (0..n_features)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect(),
    );

    // Generate y = X * coef + noise
    let mut y = Array1::zeros(n_samples);
    for (i, row) in X.rows().into_iter().enumerate() {
        y[i] = row.iter().zip(coef.iter()).map(|(&x, &c)| x * c).sum();
        y[i] += normal.sample(&mut rng) * noise;
    }

    Ok((X, y))
}

/// Generate synthetic classification data
pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> Result<(Array2<f64>, Array1<f64>)> {
    use rand::Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let samples_per_class = n_samples / n_classes;
    let mut X = Vec::new();
    let mut y = Vec::new();

    for class in 0..n_classes {
        let center = class as f64 * 3.0;

        for _ in 0..samples_per_class {
            for _ in 0..n_features {
                X.push(center + normal.sample(&mut rng));
            }
            y.push(class as f64);
        }
    }

    Ok((
        Array2::from_shape_vec((n_samples, n_features), X)?,
        Array1::from_vec(y),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_iris() {
        let (X, y) = load_iris().unwrap();
        assert_eq!(X.nrows(), 15);
        assert_eq!(X.ncols(), 4);
        assert_eq!(y.len(), 15);
    }

    #[test]
    fn test_make_regression() {
        let (X, y) = make_regression(100, 5, 0.1).unwrap();
        assert_eq!(X.nrows(), 100);
        assert_eq!(X.ncols(), 5);
        assert_eq!(y.len(), 100);
    }
}
