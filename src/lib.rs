//! # RustyAI - Next-Generation Machine Learning and Data Science Library
//!
//! RustyAI is a comprehensive, production-ready machine learning and data science library
//! that combines the capabilities of scipy, statsmodels, scikit-learn, TensorFlow, and XGBoost
//! into a single, unified Rust framework.
//!
//! ## Key Features
//!
//! - **Unified API**: Consistent, ergonomic interfaces across all components
//! - **Production-Ready**: Type-safe, zero-cost abstractions with compile-time guarantees
//! - **Comprehensive**: Complete ML/DS toolkit from preprocessing to deployment
//! - **High Performance**: Leverages Rust's fearless concurrency and SIMD
//! - **Flexible Backend**: Support for both ndarray and nalgebra
//!
//! ## Architecture
//!
//! ```text
//! RustyAI
//! ├── Core Foundation
//! │   ├── tensor      - Unified tensor abstraction
//! │   ├── linalg      - Linear algebra operations
//! │   ├── stats       - Statistical functions & distributions
//! │   └── optimize    - Optimization algorithms
//! ├── Machine Learning
//! │   ├── preprocessing      - Data preprocessing & scaling
//! │   ├── feature_selection  - Feature selection methods
//! │   ├── model_selection    - Cross-validation & hyperparameter tuning
//! │   ├── linear_models      - Linear/logistic regression, GLM
//! │   ├── ensemble           - Random forests, gradient boosting, stacking
//! │   ├── svm                - Support vector machines
//! │   ├── cluster            - Clustering algorithms
//! │   ├── decomposition      - PCA, SVD, manifold learning
//! │   ├── neighbors          - KNN, ball tree
//! │   └── naive_bayes        - Naive Bayes classifiers
//! ├── Deep Learning
//! │   ├── nn         - Neural network layers & modules
//! │   ├── autograd   - Automatic differentiation
//! │   └── optim      - Neural network optimizers
//! ├── Time Series
//! │   └── timeseries - ARIMA, SARIMA, forecasting
//! └── Utilities
//!     ├── metrics    - Evaluation metrics
//!     ├── pipeline   - ML pipelines
//!     ├── datasets   - Dataset loaders
//!     └── io         - Model serialization
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use rustyai::prelude::*;
//!
//! // Load and preprocess data
//! let (X, y) = datasets::load_iris()?;
//! let scaler = StandardScaler::new().fit(&X)?;
//! let X_scaled = scaler.transform(&X);
//!
//! // Train a model
//! let model = RandomForest::builder()
//!     .n_estimators(100)
//!     .max_depth(Some(10))
//!     .build()?
//!     .fit(&X_scaled, &y)?;
//!
//! // Make predictions
//! let predictions = model.predict(&X_scaled);
//!
//! // Evaluate
//! let accuracy = metrics::accuracy(&y, &predictions);
//! println!("Accuracy: {:.2}%", accuracy * 100.0);
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

// Core foundation modules
pub mod tensor;
pub mod linalg;
pub mod stats;
pub mod optimize;

// Machine learning modules
pub mod preprocessing;
pub mod feature_selection;
pub mod model_selection;
pub mod linear_models;
pub mod ensemble;
pub mod svm;
pub mod cluster;
pub mod decomposition;
pub mod neighbors;
pub mod naive_bayes;

// Deep learning modules
pub mod nn;
pub mod autograd;

// Time series module
pub mod timeseries;

// Utility modules
pub mod metrics;
pub mod pipeline;
pub mod datasets;
pub mod io;

// Error handling
pub mod error;

/// Common traits and types used throughout the library
pub mod prelude {
    //! Prelude module containing commonly used imports

    // Core traits
    pub use crate::error::{Result, RustyAIError};

    // Estimator traits
    pub use crate::traits::{
        Estimator, Predictor, Transformer, Classifier, Regressor
    };

    // Common types
    pub use ndarray::{Array, Array1, Array2, ArrayD, ArrayView1, ArrayView2};
    pub use crate::tensor::Tensor;

    // Preprocessing
    pub use crate::preprocessing::{StandardScaler, MinMaxScaler, Normalizer};

    // Models
    pub use crate::linear_models::{LinearRegression, LogisticRegression, Ridge, Lasso};
    pub use crate::ensemble::{RandomForest, GradientBoosting};
    pub use crate::cluster::{KMeans, DBSCAN};

    // Metrics
    pub use crate::metrics;

    // Datasets
    pub use crate::datasets;
}

/// Core traits that define the ML API
pub mod traits {
    use crate::error::Result;

    /// Base estimator trait - all models implement this
    pub trait Estimator<X, Y> {
        /// The type of the fitted estimator
        type Fitted;

        /// Fit the estimator to the training data
        fn fit(self, X: &X, y: &Y) -> Result<Self::Fitted>;
    }

    /// Predictor trait for making predictions
    pub trait Predictor<X> {
        /// Output type of predictions
        type Output;

        /// Make predictions on new data
        fn predict(&self, X: &X) -> Self::Output;
    }

    /// Transformer trait for data transformations
    pub trait Transformer<X> {
        /// Output type of transformed data
        type Output;

        /// Transform the data
        fn transform(&self, X: &X) -> Self::Output;

        /// Fit and transform in one step
        fn fit_transform(self, X: &X) -> (Self, Self::Output)
        where
            Self: Sized;
    }

    /// Classifier trait for classification tasks
    pub trait Classifier<X>: Predictor<X> {
        /// Predict class probabilities
        fn predict_proba(&self, X: &X) -> Self::Output;

        /// Get the classes
        fn classes(&self) -> Vec<usize>;
    }

    /// Regressor trait for regression tasks
    pub trait Regressor<X>: Predictor<X> {
        /// Get feature importance if available
        fn feature_importances(&self) -> Option<Vec<f64>> {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Verify all modules are accessible
        // This is a smoke test to ensure compilation
    }
}
