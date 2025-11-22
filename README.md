# RustyAI 🦀🤖

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

**RustyAI** is a comprehensive, next-generation machine learning and data science library for Rust that combines the capabilities of scipy, statsmodels, scikit-learn, TensorFlow, and XGBoost into a single, unified framework.

## 🚀 Vision

RustyAI aims to revolutionize data science in Rust by:

- **Filling the gaps** in the Rust ML ecosystem identified through comprehensive analysis
- **Providing a unified API** across all ML/DS components
- **Leveraging Rust's strengths**: fearless concurrency, zero-cost abstractions, and memory safety
- **Production-ready from day one**: type-safe, well-tested, and performant

## ✨ Key Features

### **Comprehensive Coverage**
- ✅ **Linear Algebra**: Matrix operations, decompositions (QR, SVD, Cholesky), solvers
- ✅ **Statistics**: Distributions, hypothesis testing, descriptive statistics
- ✅ **Optimization**: Gradient descent, Adam, L-BFGS, coordinate descent
- ✅ **Preprocessing**: Standard scaling, min-max scaling, normalization, encoding, imputation
- ✅ **Linear Models**: Linear regression, logistic regression, Ridge, Lasso
- ✅ **Ensemble Methods**: Random forests, gradient boosting (XGBoost-like)
- ✅ **Clustering**: K-Means, DBSCAN
- ✅ **Metrics**: MSE, RMSE, R², accuracy, precision, recall, F1, confusion matrix
- ✅ **Datasets**: Iris, synthetic data generators
- 🚧 **Neural Networks**: Deep learning framework (in progress)
- 🚧 **Time Series**: ARIMA, SARIMA, forecasting (in progress)
- 🚧 **Feature Selection**: SelectKBest, RFE (in progress)
- 🚧 **Model Selection**: GridSearchCV, cross-validation (in progress)

### **Superior Developer Experience**
- **Consistent API**: Scikit-learn-inspired design adapted for Rust idioms
- **Type Safety**: Compile-time guarantees prevent common ML errors
- **Builder Patterns**: Ergonomic configuration for complex models
- **Comprehensive Docs**: Every module thoroughly documented
- **Extensive Examples**: Real-world usage patterns

### **Production-Ready**
- **No `unsafe` code**: Pure Rust implementation with safety guarantees
- **Serialization**: Save and load models with serde
- **Parallel by default**: Leverages rayon for multi-core performance
- **Flexible backends**: Optional BLAS/LAPACK for optimized linear algebra

## 📦 Installation

Add RustyAI to your `Cargo.toml`:

```toml
[dependencies]
rustyai = "0.1"

# With BLAS support for optimized linear algebra
rustyai = { version = "0.1", features = ["blas"] }

# Full feature set including Python interop
rustyai = { version = "0.1", features = ["full"] }
```

## 🎯 Quick Start

### Linear Regression

```rust
use rustyai::prelude::*;

fn main() -> Result<()> {
    // Load data
    let (X, y) = datasets::load_iris()?;

    // Preprocess
    let scaler = StandardScaler::new().fit(&X)?;
    let X_scaled = scaler.transform(&X);

    // Train model
    let model = LinearRegression::new()
        .fit(&X_scaled, &y)?;

    // Predict
    let predictions = model.predict(&X_scaled)?;

    // Evaluate
    let r2 = metrics::r2_score(&y, &predictions);
    println!("R² Score: {:.4}", r2);

    Ok(())
}
```

### Random Forest

```rust
use rustyai::prelude::*;

fn main() -> Result<()> {
    // Generate synthetic data
    let (X, y) = datasets::make_regression(1000, 10, 0.1)?;

    // Train random forest with builder pattern
    let model = RandomForest::builder()
        .n_estimators(100)
        .max_depth(Some(10))
        .build()?
        .fit(&X, &y)?;

    // Predict
    let predictions = model.predict(&X)?;

    // Evaluate
    let mse = metrics::mean_squared_error(&y, &predictions);
    println!("MSE: {:.4}", mse);

    Ok(())
}
```

### K-Means Clustering

```rust
use rustyai::prelude::*;

fn main() -> Result<()> {
    let (X, _) = datasets::load_iris()?;

    // Cluster into 3 groups
    let model = KMeans::new(3)
        .with_max_iter(300)
        .fit(&X)?;

    let labels = model.labels().unwrap();
    let centers = model.cluster_centers().unwrap();

    println!("Cluster centers:\n{:?}", centers);
    println!("Labels: {:?}", labels);

    Ok(())
}
```

### Gradient Boosting

```rust
use rustyai::ensemble::GradientBoosting;

fn main() -> Result<()> {
    let (X, y) = datasets::make_regression(500, 5, 0.1)?;

    // Train gradient boosting model (XGBoost-like)
    let model = GradientBoosting::new()
        .fit(&X, &y)?;

    let predictions = model.predict(&X)?;
    let rmse = metrics::root_mean_squared_error(&y, &predictions);

    println!("RMSE: {:.4}", rmse);

    Ok(())
}
```

## 🏗️ Architecture

RustyAI is organized into logical modules:

```
RustyAI
├── Core Foundation
│   ├── tensor      - Unified tensor abstraction
│   ├── linalg      - Linear algebra operations
│   ├── stats       - Statistical functions & distributions
│   └── optimize    - Optimization algorithms
├── Machine Learning
│   ├── preprocessing      - Data preprocessing & scaling
│   ├── feature_selection  - Feature selection methods
│   ├── model_selection    - Cross-validation & hyperparameter tuning
│   ├── linear_models      - Linear/logistic regression, GLM
│   ├── ensemble           - Random forests, gradient boosting
│   ├── svm                - Support vector machines
│   ├── cluster            - Clustering algorithms
│   ├── decomposition      - PCA, SVD, manifold learning
│   ├── neighbors          - KNN, ball tree
│   └── naive_bayes        - Naive Bayes classifiers
├── Deep Learning
│   ├── nn         - Neural network layers & modules
│   ├── autograd   - Automatic differentiation
│   └── optim      - Neural network optimizers (planned)
├── Time Series
│   └── timeseries - ARIMA, SARIMA, forecasting
└── Utilities
    ├── metrics    - Evaluation metrics
    ├── pipeline   - ML pipelines
    ├── datasets   - Dataset loaders
    └── io         - Model serialization
```

## 🎨 Design Principles

### **1. Unified API**

All estimators implement consistent traits:

```rust
pub trait Estimator<X, Y> {
    type Fitted;
    fn fit(self, X: &X, y: &Y) -> Result<Self::Fitted>;
}

pub trait Predictor<X> {
    type Output;
    fn predict(&self, X: &X) -> Self::Output;
}

pub trait Transformer<X> {
    type Output;
    fn transform(&self, X: &X) -> Self::Output;
}
```

### **2. Type Safety**

Leverage Rust's type system to prevent errors:

```rust
// Compile-time dimension checking
let X: Array2<f64> = Array2::zeros((100, 10));
let y: Array1<f64> = Array1::zeros(100);

let model = LinearRegression::new().fit(&X, &y)?; // ✅ Compiles
// let bad_y = Array1::zeros(50);
// let model = LinearRegression::new().fit(&X, &bad_y)?; // ❌ Runtime error caught
```

### **3. Builder Patterns**

Ergonomic configuration:

```rust
let model = RandomForest::builder()
    .n_estimators(100)
    .max_depth(Some(10))
    .min_samples_split(2)
    .build()?;
```

### **4. Zero-Cost Abstractions**

Generic programming with no runtime overhead:

```rust
// Works with f32 or f64 with no performance penalty
fn train<A: Float>(X: &Array2<A>, y: &Array1<A>) -> Result<LinearRegression<A>> {
    LinearRegression::new().fit(X, y)
}
```

## 🔬 Research-Backed Design

RustyAI was built after comprehensive research into the Rust ML ecosystem in 2025, identifying and addressing key gaps:

### **Gaps Identified and Addressed**

1. ✅ **Ecosystem Fragmentation**: Unified API across all components
2. ✅ **Missing Algorithms**: Comprehensive gradient boosting, advanced preprocessing
3. ✅ **Poor Ergonomics**: Builder patterns, intuitive API design
4. ✅ **Limited Statistics**: Full statistical toolkit matching statsmodels
5. ✅ **Incomplete Preprocessing**: One-hot encoding, imputation, scaling
6. 🚧 **No AutoML**: Hyperparameter optimization framework (in progress)
7. 🚧 **Limited Time Series**: SARIMA, forecasting models (in progress)

## 📊 Benchmarks

RustyAI leverages Rust's performance advantages:

- **Zero-cost abstractions**: Generic code with no runtime overhead
- **Fearless concurrency**: Parallel operations are safe by default
- **Memory efficiency**: No garbage collection pauses
- **SIMD**: Automatic vectorization for numeric operations

*Detailed benchmarks coming soon*

## 🛠️ Development

### Build

```bash
cargo build --release
```

### Test

```bash
cargo test
```

### Benchmarks

```bash
cargo bench
```

### Features

- `default`: Core functionality
- `blas`: BLAS/LAPACK backend for optimized linear algebra
- `nalgebra-backend`: Use nalgebra instead of ndarray
- `python`: PyO3 bindings for Python interop
- `full`: All features enabled

## 🗺️ Roadmap

### v0.1 (Current)
- [x] Core linear algebra
- [x] Statistics and distributions
- [x] Optimization algorithms
- [x] Preprocessing tools
- [x] Linear models
- [x] Ensemble methods
- [x] Clustering algorithms
- [x] Evaluation metrics

### v0.2 (Next)
- [ ] Neural network framework with autograd
- [ ] Time series analysis (ARIMA, SARIMA)
- [ ] SVM implementation
- [ ] PCA and dimensionality reduction
- [ ] Feature selection methods
- [ ] Cross-validation and grid search

### v0.3
- [ ] Advanced ensemble methods (stacking, blending)
- [ ] Bayesian methods
- [ ] Manifold learning (t-SNE, UMAP)
- [ ] Advanced clustering (HDBSCAN, spectral)
- [ ] Model explainability (SHAP-like)

### v1.0
- [ ] Production deployment tools
- [ ] Model serving framework
- [ ] Distributed training
- [ ] GPU acceleration
- [ ] AutoML framework
- [ ] Comprehensive documentation and tutorials

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas We Need Help With

- **Algorithms**: Implementing additional ML algorithms
- **Optimization**: Performance improvements and SIMD utilization
- **Documentation**: Examples, tutorials, and guides
- **Testing**: Property-based tests and benchmarks
- **Integration**: Python bindings, ONNX export

## 📚 Learning Resources

- [RustyAI Documentation](https://docs.rs/rustyai) (coming soon)
- [API Reference](https://docs.rs/rustyai/latest/rustyai)
- [Examples](./examples)
- [Tutorials](./tutorials) (coming soon)

## 🙏 Acknowledgments

RustyAI builds upon and is inspired by:

- **ndarray**: Fundamental array computing
- **linfa**: ML algorithms for Rust
- **smartcore**: Comprehensive ML library
- **burn**: Modern deep learning framework
- **polars**: High-performance DataFrames

And the Python ecosystem:
- **scikit-learn**: API design inspiration
- **scipy**: Statistical functions
- **XGBoost**: Gradient boosting algorithms
- **statsmodels**: Statistical models

## 📄 License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## 🌟 Citation

If you use RustyAI in your research, please cite:

```bibtex
@software{rustyai2025,
  title = {RustyAI: A Comprehensive Machine Learning and Data Science Library for Rust},
  author = {RustyAI Contributors},
  year = {2025},
  url = {https://github.com/mecoserra2008/Rustyai}
}
```

---

**Built with ❤️ in Rust** 🦀
