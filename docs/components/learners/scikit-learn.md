# Scikit-learn Learners

LearnM8 provides six scikit-learn based learners optimized for molecular property prediction. These models offer proven performance, computational efficiency, and robust handling of different dataset characteristics.

## RandomForestLearner

### Overview

Random Forest uses an ensemble of decision trees trained on bootstrapped samples of the data. Each tree votes on the prediction, with the average serving as the final output. This learner provides a fast, robust baseline with good performance across diverse molecular datasets.

**When to use:**
- Fast prototyping and baseline establishment
- Datasets of any size (particularly effective for 100-10,000 compounds)
- When interpretability through feature importance is valuable
- As a component in ensemble models

**Key characteristics:**
- No uncertainty quantification (base implementation)
- Parallel training across all CPU cores
- Out-of-bag (OOB) scoring for model validation
- Feature importance through tree-based analysis

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 100 | Number of trees in the forest. More trees generally improve performance but increase training time linearly. 100 provides good balance. |
| `max_depth` | int | None | Maximum tree depth. None allows unlimited depth. Limiting depth (e.g., 10-20) prevents overfitting on small datasets. |
| `min_samples_split` | int | 2 | Minimum samples required to split a node. Higher values (5-10) prevent overfitting. |
| `min_samples_leaf` | int | 1 | Minimum samples required at leaf nodes. Higher values (2-5) create smoother decision boundaries. |
| `max_features` | str | 'sqrt' | Features considered per split. 'sqrt' uses sqrt(n_features), 'log2' is more conservative, None uses all features. |
| `random_state` | int | 42 | Random seed for reproducibility. |
| `n_jobs` | int | -1 | Parallel jobs. -1 uses all CPU cores for 5-10x training speedup. |

**Performance notes:**
- Training time scales linearly with `n_estimators`
- Memory scales with `n_estimators` × tree size
- Prediction is fast (logarithmic in tree depth)
- OOB scoring adds negligible overhead

### Uncertainty Support

**No** - Base RandomForestLearner does not provide uncertainty estimates. Use AdvancedRandomForestLearner or ensemble variants for uncertainty quantification.

### Performance Characteristics

**Speed:** Fast (trains in seconds for typical molecular datasets)

**Scalability:**
- Excellent for 100-100,000 compounds
- Linear scaling with dataset size
- Parallel training utilizes all CPU cores

**Memory:** Moderate (trees stored in memory, ~1-10 MB per 100 trees)

**Best for:** Quick prototyping, baseline models, feature importance analysis

### Example (CLI)

```bash
learnm8 run compounds.csv --target Activity --learner rf --featurizer morgan --n-cycles 10
```

### Example (API)

```python
from learnm8 import run_active_learning
from learnm8.learners.sklearn import RandomForestLearner

learner = RandomForestLearner(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=learner,
    target_col='Activity',
    featurizer_type='morgan',
    n_cycles=10
)

oob_score = learner.get_oob_score()
feature_importance = learner.get_feature_importance()
```

---

## GaussianProcessLearner

### Overview

Gaussian Process (GP) regression provides the gold standard for uncertainty quantification in active learning. It models predictions as a Gaussian distribution, naturally providing both mean predictions and standard deviation estimates. GP excels on small to medium datasets where principled uncertainty is crucial.

**When to use:**
- Small to medium datasets (<5,000 compounds recommended)
- Uncertainty-based acquisition strategies (UCB, EI, Thompson sampling)
- When accurate uncertainty quantification is critical
- Benchmark studies requiring rigorous uncertainty estimates

**Key characteristics:**
- Native, principled uncertainty quantification
- Scales cubically with dataset size (O(n³))
- Hyperparameter optimization via marginal likelihood
- Smooth, continuous prediction surfaces

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kernel` | Kernel | None | GP kernel function. None uses RBF with automatic hyperparameter learning: C(1.0) * RBF(1.0). Custom kernels supported. |
| `alpha` | float | 1e-10 | Noise regularization parameter. Larger values (1e-6 to 1e-3) add stability for noisy data. Smaller values for low-noise targets. |
| `n_restarts_optimizer` | int | 5 | Optimizer restarts for hyperparameter optimization. More restarts (10-20) improve convergence but increase training time. |
| `normalize_y` | bool | True | Normalize target values. Highly recommended for GP stability and performance. |
| `random_state` | int | 42 | Random seed for reproducibility. |

**Performance notes:**
- Training time: O(n³) - becomes slow for >5,000 compounds
- Prediction time: O(n²) - scales quadratically with training set size
- Memory: O(n²) - stores full covariance matrix
- Hyperparameter optimization adds 5-10x overhead but critical for performance

### Uncertainty Support

**Yes** - GaussianProcessLearner provides principled uncertainty estimates through posterior standard deviation. This is the most theoretically sound uncertainty available in LearnM8.

**How it works:**
- GP models predictions as Gaussian distributions
- Returns mean (prediction) and standard deviation (uncertainty)
- Uncertainty increases in unexplored regions
- Uncertainty decreases near training data

### Performance Characteristics

**Speed:** Slow for large datasets (cubic scaling)

**Scalability:**
- Excellent for <1,000 compounds
- Acceptable for 1,000-5,000 compounds
- Not recommended for >5,000 compounds

**Memory:** High (stores n×n covariance matrix)

**Best for:** Small datasets requiring principled uncertainty, uncertainty-based acquisition strategies

### Example (CLI)

```bash
learnm8 run compounds.csv --target Activity --learner gp --featurizer descriptors --cycles "random:0.02 ucb:0.01*8"
```

### Example (API)

```python
from learnm8 import run_active_learning
from learnm8.learners.sklearn import GaussianProcessLearner
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

learner = GaussianProcessLearner(
    kernel=C(1.0) * Matern(length_scale=1.0, nu=2.5),
    alpha=1e-6,
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=42
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=learner,
    target_col='Activity',
    featurizer_type='descriptors',
    cycles=[('random', 0.02), ('ucb', 0.01)],
    acquisition_params={'beta': 2.0}
)

hyperparams = learner.get_learned_hyperparameters()
```

---

## XGBoostLearner

### Overview

XGBoost provides high-performance gradient boosting optimized for speed and accuracy. It builds an ensemble of decision trees sequentially, with each tree correcting errors from previous trees. XGBoost excels on medium to large molecular datasets where prediction quality and computational efficiency are both priorities.

**When to use:**
- Medium to large datasets (1,000-100,000+ compounds)
- When prediction accuracy is the top priority
- Production systems requiring fast predictions
- Datasets with complex non-linear relationships

**Key characteristics:**
- State-of-the-art tabular data performance
- Efficient parallel training
- Regularization prevents overfitting
- No native uncertainty quantification (use XGBEnsemble for uncertainty)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 100 | Number of boosting rounds. More rounds improve performance but risk overfitting. 100-300 typical. |
| `learning_rate` | float | 0.1 | Step size shrinkage. Lower values (0.01-0.05) require more estimators but often improve generalization. |
| `max_depth` | int | 6 | Maximum tree depth. Controls model complexity. 3-10 typical for molecular data. |
| `min_child_weight` | int | 1 | Minimum sum of instance weight in child. Higher values (5-10) prevent overfitting on small datasets. |
| `subsample` | float | 0.8 | Row subsampling fraction. 0.6-0.9 adds regularization through randomness. |
| `colsample_bytree` | float | 0.8 | Column subsampling per tree. 0.6-0.9 prevents feature dominance. |
| `reg_alpha` | float | 0.0 | L1 regularization. Increase (0.1-1.0) for sparse feature importance. |
| `reg_lambda` | float | 1.0 | L2 regularization. Increase (1.0-10.0) for smoother predictions. |
| `random_state` | int | 42 | Random seed for reproducibility. |
| `n_jobs` | int | -1 | Parallel jobs. -1 uses all CPU cores for fast training. |

**Performance notes:**
- Training time scales with `n_estimators` × `max_depth`
- Histogram-based algorithm (`tree_method='hist'`) for efficiency
- Memory efficient compared to Random Forest
- Feature importance through gain/weight/cover metrics

### Uncertainty Support

**No** - Base XGBoostLearner does not provide uncertainty estimates. Use XGBEnsemble for uncertainty quantification through model disagreement.

### Performance Characteristics

**Speed:** Very fast (optimized C++ implementation)

**Scalability:**
- Excellent for 1,000-100,000+ compounds
- Sub-linear scaling with efficient algorithms
- Parallel training utilizes all CPU cores

**Memory:** Low (gradient statistics stored efficiently)

**Best for:** Large datasets, production systems, maximum prediction accuracy

### Example (CLI)

```bash
learnm8 run compounds.csv --target Activity --learner xgb --featurizer morgan --n-cycles 15
```

### Example (API)

```python
from learnm8 import run_active_learning
from learnm8.learners.sklearn import XGBoostLearner

learner = XGBoostLearner(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=3,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=42
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=learner,
    target_col='Activity',
    featurizer_type='morgan',
    n_cycles=15
)

feature_importance = learner.get_feature_importance()
booster_stats = learner.get_booster_stats()
```

---

## DecisionTreeLearner

### Overview

Decision Tree provides a single interpretable tree structure for predictions. Unlike ensemble methods, a single tree offers complete transparency in decision-making, making it valuable for understanding model logic and identifying key features. However, single trees are prone to overfitting and generally provide lower accuracy than ensemble methods.

**When to use:**
- Model interpretability is paramount
- Understanding decision logic for domain insights
- Debugging feature extraction or data quality issues
- Teaching active learning concepts with simple models

**Key characteristics:**
- Fully interpretable decision paths
- Fast training and prediction
- Prone to overfitting without regularization
- No uncertainty quantification

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | int | 10 | Maximum tree depth. Limits model complexity. 5-15 typical for molecular data. Lower prevents overfitting. |
| `min_samples_split` | int | 10 | Minimum samples required to split. Higher values (10-20) create simpler trees. |
| `min_samples_leaf` | int | 5 | Minimum samples at leaf nodes. Higher values (5-10) prevent overfitting. |
| `max_features` | str | None | Features considered per split. None uses all features. 'sqrt' or 'log2' add regularization. |
| `random_state` | int | 42 | Random seed for reproducibility. |

**Performance notes:**
- Defaults designed to prevent overfitting (limited depth, higher min_samples)
- Training time logarithmic in dataset size
- Prediction extremely fast (single path through tree)
- Tree structure can be exported and visualized

### Uncertainty Support

**No** - DecisionTreeLearner does not provide uncertainty estimates. Each leaf node has a single prediction value.

### Performance Characteristics

**Speed:** Very fast (fastest learner in LearnM8)

**Scalability:**
- Excellent for any dataset size
- Logarithmic prediction time
- Minimal memory footprint

**Memory:** Very low (single tree structure)

**Best for:** Model interpretation, debugging, educational purposes

### Example (CLI)

```bash
learnm8 run compounds.csv --target Activity --learner dt --featurizer morgan --n-cycles 10
```

### Example (API)

```python
from learnm8 import run_active_learning
from learnm8.learners.sklearn import DecisionTreeLearner

learner = DecisionTreeLearner(
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=learner,
    target_col='Activity',
    featurizer_type='morgan',
    n_cycles=10
)

feature_importance = learner.get_feature_importance()
```

---

## LinearRegressionLearner

### Overview

Linear Regression models predictions as a linear combination of features. This learner provides the simplest baseline and works well when molecular properties have approximately linear relationships with features. Supports both standard linear regression and Ridge regression (L2 regularization) for handling feature collinearity.

**When to use:**
- Simple baseline establishment
- Linear relationships between features and targets
- Small datasets where complex models overfit
- Fast predictions required in production

**Key characteristics:**
- Analytical solution (no iterative optimization)
- Interpretable coefficients
- Very fast training and prediction
- Ridge variant handles collinear features

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | None | Regularization strength. None uses standard LinearRegression. Values like 0.1-10.0 enable Ridge regression with L2 regularization. |
| `fit_intercept` | bool | True | Whether to fit intercept term. Usually True unless features are centered. |
| `n_jobs` | int | -1 | Parallel jobs for LinearRegression only. -1 uses all CPU cores. Ridge doesn't parallelize. |
| `random_state` | int | 42 | Random seed for Ridge regression reproducibility. |

**Performance notes:**
- LinearRegression: Parallel computation via BLAS/LAPACK
- Ridge: Sequential but still very fast
- Training time: O(n² × d) for n samples, d features
- Analytical solution means no hyperparameter tuning needed

### Uncertainty Support

**No** - LinearRegressionLearner does not provide uncertainty estimates. Use LREnsemble for uncertainty through model disagreement.

### Performance Characteristics

**Speed:** Very fast (analytical solution)

**Scalability:**
- Excellent for any dataset size
- Quadratic scaling with features
- Parallel computation for LinearRegression

**Memory:** Very low (stores only coefficients)

**Best for:** Simple baselines, linear relationships, fast predictions

### Example (CLI)

```bash
learnm8 run compounds.csv --target Activity --learner lr --featurizer descriptors --n-cycles 10
```

### Example (API)

```python
from learnm8 import run_active_learning
from learnm8.learners.sklearn import LinearRegressionLearner

learner = LinearRegressionLearner(
    alpha=1.0,
    fit_intercept=True,
    random_state=42
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=learner,
    target_col='Activity',
    featurizer_type='descriptors',
    n_cycles=10
)

coefficients = learner.get_coefficients()
intercept = learner.get_intercept()
```

---

## AdvancedRandomForestLearner

### Overview

Advanced Random Forest extends the base RandomForestLearner with optimized hyperparameters, regularization techniques, and enhanced configuration for superior performance on molecular datasets. This learner includes 300 trees (vs 100), depth limits, cost complexity pruning, and bootstrap subsampling designed to balance accuracy and generalization.

**When to use:**
- When Random Forest is preferred but enhanced performance needed
- Medium to large molecular datasets (1,000-50,000 compounds)
- When out-of-bag validation is valuable
- As a strong single-model baseline before ensembles

**Key characteristics:**
- Optimized hyperparameters for molecular data
- Cost complexity pruning (ccp_alpha) prevents overfitting
- Bootstrap subsampling adds regularization
- Enhanced tree statistics and diagnostics

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | int | 300 | Number of trees (3x base RF). More trees improve stability and reduce variance. |
| `max_depth` | int | 15 | Maximum tree depth. Limited depth prevents overfitting compared to unlimited in base RF. |
| `min_samples_split` | int | 5 | Minimum samples to split. Higher than base RF (2) for regularization. |
| `min_samples_leaf` | int | 2 | Minimum samples at leaves. Prevents single-sample leaves. |
| `max_features` | str | 'sqrt' | Features per split. 'sqrt' provides good balance. |
| `max_samples` | float | 0.8 | Bootstrap sample fraction. <1.0 adds regularization through subsampling. |
| `min_impurity_decrease` | float | 0.0001 | Minimum improvement required to split. Prunes low-value splits. |
| `ccp_alpha` | float | 0.001 | Cost complexity pruning parameter. Non-zero enables pruning of weak branches. |
| `bootstrap` | bool | True | Enable bootstrap sampling. Recommended for OOB validation. |
| `oob_score` | bool | True | Calculate out-of-bag score. Free validation metric. |
| `random_state` | int | 42 | Random seed for reproducibility. |
| `n_jobs` | int | -1 | Parallel jobs. Auto-capped at 32 cores for efficiency. |

**Performance notes:**
- 3x training time vs base RF (300 vs 100 trees)
- Cost complexity pruning adds minimal overhead
- OOB scoring provides free validation
- Tree statistics available for analysis

### Uncertainty Support

**Yes** - AdvancedRandomForestLearner provides uncertainty estimates through tree prediction variance. Each tree in the forest makes a prediction, and the standard deviation across trees serves as uncertainty.

**How it works:**
- Collect predictions from all 300 trees
- Mean prediction = average across trees
- Uncertainty = standard deviation across trees
- Higher variance indicates model disagreement (uncertainty)

### Performance Characteristics

**Speed:** Fast (3x slower than base RF due to more trees)

**Scalability:**
- Excellent for 1,000-50,000 compounds
- Linear scaling with dataset size
- Parallel training utilizes up to 32 CPU cores

**Memory:** Moderate (300 trees stored, ~3x base RF memory)

**Best for:** Enhanced Random Forest performance, molecular datasets requiring regularization

### Example (CLI)

```bash
learnm8 run compounds.csv --target Activity --learner advanced_rf --featurizer morgan --cycles "random:0.02 greedy:0.01*9"
```

### Example (API)

```python
from learnm8 import run_active_learning
from learnm8.learners.sklearn import AdvancedRandomForestLearner

learner = AdvancedRandomForestLearner(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=3,
    max_samples=0.7,
    min_impurity_decrease=0.0005,
    ccp_alpha=0.005,
    random_state=42
)

results = run_active_learning(
    compound_pool='compounds.csv',
    oracle='oracle.csv',
    learner=learner,
    target_col='Activity',
    featurizer_type='morgan',
    cycles=[('random', 0.02), ('greedy', 0.01)],
    n_cycles=10
)

oob_score = learner.get_oob_score()
tree_stats = learner.get_tree_stats()
feature_importance = learner.get_feature_importance()
```

---

## Learner Comparison

### Performance Summary

| Learner | Speed | Scalability | Uncertainty | Memory | Best Use Case |
|---------|-------|-------------|-------------|--------|---------------|
| RandomForest | Fast | 100-100k | No | Moderate | Fast baseline |
| GaussianProcess | Slow | <5k | Yes (best) | High | Small datasets, principled uncertainty |
| XGBoost | Very fast | 1k-100k+ | No | Low | Large datasets, max accuracy |
| DecisionTree | Very fast | Any | No | Very low | Interpretability |
| LinearRegression | Very fast | Any | No | Very low | Simple baseline |
| AdvancedRandomForest | Fast | 1k-50k | Yes | Moderate | Enhanced RF performance |

### Featurizer Recommendations

**Best featurizer by learner type:**

| Learner | Recommended Featurizer | Rationale |
|---------|------------------------|-----------|
| RandomForest | `morgan` or `ecfp6` | Tree methods work well with binary fingerprints |
| GaussianProcess | `descriptors` | Continuous features better for kernel methods |
| XGBoost | `morgan` or `ecfp6` | Tree methods optimized for binary features |
| DecisionTree | `morgan` | Simpler fingerprints easier to interpret |
| LinearRegression | `descriptors` | Continuous features for linear relationships |
| AdvancedRandomForest | `morgan` or `ecfp6` | Tree methods with binary fingerprints |

### When to Use Each Learner

**Choose RandomForestLearner when:**
- Establishing baselines quickly
- Dataset size is 100-10,000 compounds
- Feature importance analysis needed
- Computational resources limited

**Choose GaussianProcessLearner when:**
- Dataset size <5,000 compounds
- Uncertainty-based acquisition critical (UCB, EI, Thompson)
- Principled uncertainty quantification required
- Computational time available for cubic scaling

**Choose XGBoostLearner when:**
- Dataset size >1,000 compounds
- Maximum prediction accuracy required
- Fast training and prediction needed
- Production deployment planned

**Choose DecisionTreeLearner when:**
- Model interpretability is paramount
- Understanding decision logic required
- Debugging data quality issues
- Educational/demonstration purposes

**Choose LinearRegressionLearner when:**
- Simple baseline needed
- Linear relationships suspected
- Extremely fast predictions required
- Interpretable coefficients valuable

**Choose AdvancedRandomForestLearner when:**
- Random Forest preferred but better performance needed
- Dataset size 1,000-50,000 compounds
- Uncertainty estimates desired without ensemble overhead
- Regularization important for generalization
