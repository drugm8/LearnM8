# LearnM8 Data Flow Documentation

## Overview

LearnM8 is an active learning framework designed for molecular screening applications. This document provides an extremely granular, step-by-step explanation of how data flows through the entire system, from initial input to final results.

## System Architecture

The LearnM8 system is organized into several modular components:

```
.
├── learnm8/                # Main package directory
│   ├── cli/                # Command-line interface
│   ├── core/               # Core active learning logic and interfaces
│   ├── evaluation/         # Metrics and monitoring
│   ├── experiments/        # Experiment orchestration
│   ├── learners/           # Machine learning models
│   ├── oracles/            # Data sources/ground truth providers
│   ├── strategies/         # Compound selection strategies
│   └── utils/              # Utility functions
├── data/                   # CSV data for various molecular targets
├── examples/               # Example scripts
├── benchmark_results/      # Contains results from benchmarking runs
├── setup.py                # Package setup and installation script
└── README.md               # Project README
```

## Complete Data Flow Walkthrough

### Phase 1: Initialization and Setup

#### 1.1. Command Line Entry Point (`learnm8/cli/main.py`)

**Input Data:**
- `compound_pool`: Path to a CSV file containing compound information, requiring `['ID', 'SMILES']` columns.
- `ground_truth`: Path to a CSV file with ground truth values, requiring `['ID', target_column]` columns.
- `target_column`: The name of the column in the ground truth file to be used as the learning target.
- Additional parameters controlling the experiment, such as the number of cycles, batch size, selection strategy, and more.

**Data Processing Steps:**

1.  **Argument Parsing**: The system parses command-line arguments to create a structured configuration for the experiment.
    ```python
    # Raw command line arguments are parsed into structured config
    args = parser.parse_args()
    ```

2.  **File Validation**: It verifies that the provided `compound_pool` and `ground_truth` files exist.
    ```python
    # File existence checks
    compound_pool_path = Path(args.compound_pool)
    ground_truth_path = Path(args.ground_truth)
    # Raises SystemExit if files don't exist
    ```

3.  **Score Direction Auto-Detection**: The system determines whether a higher or lower score is better by analyzing the `target_column` name and data patterns.
    ```python
    # Analyzes target column name and data patterns
    if score_direction == 'auto':
        score_direction = detect_score_direction(str(ground_truth_path), args.target_column)
    ```

4.  **Output Directory Creation**: A timestamped directory is created to store the results of the experiment.
    ```python
    # Creates timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"learnm8_results_{timestamp}")
    ```

#### 1.2. Experiment Configuration (`learnm8/experiments/config.py`)

**Data Transformation**: A configuration dictionary is created to hold all experiment parameters.
```python
config = create_experiment_config(
    compound_pool_path=str(compound_pool_path),
    ground_truth_path=str(ground_truth_path),
    target_column=args.target_column,
    # ... other parameters
)
```

**Configuration Dictionary Structure**:
```python
{
    'compound_pool_path': str,
    'ground_truth_path': str, 
    'target_column': str,
    'n_cycles': int,
    'batch_size_fraction': float,
    'selection_strategy': str,
    'initial_strategy': str,
    'score_direction': str,
    'learner_type': str,
    'random_state': int,
    'monitoring': {
        'top_k': int,
        'enrichment_percentile': float
    },
    'timestamp': str
}
```

### Phase 2: Data Loading and Component Initialization

#### 2.1 Compound Pool Loading (`experiments/runner.py`)

**Input Processing:**
```python
compound_pool = pd.read_csv(config['compound_pool_path'])
```

**Required Column Validation:**
```python
required_cols = ['ID', 'SMILES']
missing_cols = [col for col in required_cols if col not in compound_pool.columns]
if missing_cols:
    raise ValueError(f"Compound pool missing columns: {missing_cols}")
```

**Data Structure After Loading:**
```
compound_pool: DataFrame
├── ID: [str/int] - Unique compound identifiers
├── SMILES: [str] - SMILES molecular representations
└── [optional additional columns]

Shape: (n_compounds, n_columns)
Example: (50000, 2)
```

#### 2.2 Batch Size Calculation

**Dynamic Sizing:**
```python
batch_size = int(len(compound_pool) * config['batch_size_fraction'])
if batch_size < 1:
    raise ValueError(f"Batch size too small: {batch_size}")
```

**Data Flow:**
- `len(compound_pool)`: e.g., 50,000 compounds
- `batch_size_fraction`: e.g., 0.1 (10%)
- `batch_size`: 5,000 compounds per cycle

#### 2.3 Oracle Initialization (`oracles/csv_oracle.py`)

**Ground Truth Loading:**
```python
oracle = CSVOracle(config['ground_truth_path'])
# Inside CSVOracle.__init__:
self.ground_truth = pd.read_csv(csv_path)
```

**Validation:**
```python
if 'ID' not in self.ground_truth.columns:
    raise ValueError("CSV must contain an 'ID' column")
```

**Oracle Data Structure:**
```
oracle.ground_truth: DataFrame
├── ID: [str/int] - Compound identifiers
├── target_column: [float] - Target property values
├── Activity: [int] - Optional binary activity labels (0/1)
└── [other property columns]

Shape: (n_ground_truth_compounds, n_properties)
```

#### 2.4 Learner Initialization (`learners/random_forest.py` → `learners/base.py`)

**Component Creation:**
```python
learner = get_learner(config['learner_type'], config['random_state'])
# Returns RandomForestLearner instance
```

**RandomForestLearner Initialization:**
```python
model = RandomForestRegressor(
    n_estimators=100,
    n_jobs=min(os.cpu_count() or 1, 32),  # CPU optimization
    random_state=random_state
)
super().__init__(model)  # Calls SklearnLearner.__init__
```

**Learner State After Initialization:**
```python
learner.model: RandomForestRegressor (untrained)
learner.is_trained: False
learner.training_data: pd.DataFrame() (empty)
learner.target_column: None
```

#### 2.5 Strategy Function Loading

**Strategy Mapping:**
```python
strategies = {
    'greedy': select_greedy,
    'random': select_random, 
    'diverse': select_diverse,
    'diversity': select_diverse
}
selection_strategy = strategies[config['selection_strategy']]
initial_strategy = strategies[config['initial_strategy']]
```

### Phase 3: Active Learning Loop Execution (`core/active_learning.py`)

#### 3.1 Loop Initialization

**State Setup:**
```python
available_pool = compound_pool.copy()  # All compounds initially available
labeled_compounds = pd.DataFrame()     # No compounds labeled initially
monitoring_results = []                # Empty metrics history
```

**Ground Truth Preparation for Monitoring:**
```python
all_properties = [target_column]
if hasattr(oracle, 'ground_truth') and 'Activity' in oracle.ground_truth.columns:
    all_properties.append('Activity')
ground_truth_full = oracle.measure(compound_pool, all_properties)
```

**Data State at Loop Start:**
```
available_pool: DataFrame (50,000 compounds)
├── ID: compound identifiers
├── SMILES: molecular structures  
└── [prediction column added later]

labeled_compounds: DataFrame (empty initially)
monitoring_results: List (empty initially)
```

#### 3.2 Cycle Execution (Repeated n_cycles times)

##### 3.2.1 Cycle 0 (Initial Selection)

**Initial Strategy Execution:**
```python
if cycle == 0:
    selected = initial_selection_strategy(available_pool, batch_size, random_state)
```

**For Random Initial Strategy (`strategies/random.py`):**
```python
def select_random(compounds: pd.DataFrame, n_select: int, random_state: int = None):
    selected = compounds.sample(n=min(n_select, len(compounds)), random_state=random_state)
    return selected
```

**Data Transformation:**
- Input: `available_pool` (50,000 compounds)
- Output: `selected` (5,000 randomly chosen compounds)

##### 3.2.2 Cycles 1+ (Prediction-Based Selection)

**Prediction Generation:**
```python
predictions = learner.predict(available_pool)
available_pool['prediction'] = predictions
```

**Inside `learner.predict()` (`learners/base.py`):**

1. **SMILES to Fingerprint Conversion:**
   ```python
   X = smiles_to_fingerprints(available_pool['SMILES'].tolist())
   ```

2. **Fingerprint Generation Process (`utils/chemistry.py`):**
   ```python
   def smiles_to_fingerprints(smiles_list: list[str], n_jobs: int = -1):
       # Parallel processing with joblib
       fingerprints = Parallel(n_jobs=n_jobs)(
           delayed(smiles_to_morgan_fingerprint)(smiles) 
           for smiles in smiles_list
       )
       return np.array(fingerprints)
   ```

3. **Individual SMILES Processing:**
   ```python
   def smiles_to_morgan_fingerprint(smiles: str):
       mol = Chem.MolFromSmiles(smiles)  # RDKit molecule object
       morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
       fp = morgan_gen.GetFingerprint(mol)  # Morgan fingerprint
       return np.array(fp)  # 2048-dimensional binary vector
   ```

**Data Flow in Fingerprint Generation:**
```
SMILES: "CCO" → RDKit Mol → Morgan FP → [0,1,0,1,1,0,...] (2048 bits)
```

4. **Model Prediction:**
   ```python
   predictions = self.model.predict(X)  # RandomForestRegressor.predict()
   ```

**Prediction Data Structure:**
```
X: np.ndarray, shape (n_available_compounds, 2048)
predictions: np.ndarray, shape (n_available_compounds,)
available_pool after prediction:
├── ID: compound identifiers
├── SMILES: molecular structures
└── prediction: [float] - model predictions
```

**Strategy-Based Selection:**

**For Greedy Strategy (`strategies/greedy.py`):**
```python
def select_greedy(compounds: pd.DataFrame, n_select: int, score_direction: str = 'higher'):
    ascending = (score_direction == 'lower')
    sorted_compounds = compounds.sort_values('prediction', ascending=ascending)
    selected = sorted_compounds.head(n_select)
    return selected
```

**Data Transformation:**
- Input: `available_pool` with predictions (45,000 compounds after cycle 1)
- Processing: Sort by prediction values
- Output: `selected` (top 5,000 compounds by prediction)

##### 3.2.3 Oracle Measurement

**Property Measurement:**
```python
measured = oracle.measure(selected, [target_column])
```

**Inside `oracle.measure()` (`oracles/csv_oracle.py`):**
```python
def measure(self, compounds: pd.DataFrame, properties: list[str]):
    compound_ids = compounds[['ID']].copy()
    result = pd.merge(
        compound_ids,
        self.ground_truth[['ID'] + properties],
        on='ID',
        how='inner'
    )
    return result
```

**Data Flow:**
- Input: `selected` compounds with IDs
- Process: Lookup in ground truth CSV
- Output: `measured` DataFrame with true property values

**Measured Data Structure:**
```
measured: DataFrame
├── ID: compound identifiers
└── target_column: [float] - true property values

Shape: (batch_size, 2)
```

##### 3.2.4 Data Integration and Updates

**Label Integration:**
```python
selected_with_labels = pd.merge(selected[['ID', 'SMILES']], measured, on='ID')
```

**Training Data Accumulation:**
```python
if labeled_compounds.empty:
    labeled_compounds = selected_with_labels.copy()
else:
    labeled_compounds = pd.concat([labeled_compounds, selected_with_labels], ignore_index=True)
```

**Available Pool Update:**
```python
available_pool = available_pool[~available_pool['ID'].isin(selected['ID'])]
```

**Data State After Updates:**
```
labeled_compounds: DataFrame (cumulative)
├── ID: labeled compound identifiers
├── SMILES: molecular structures
└── target_column: true property values

Cycle 0: (5,000, 3)
Cycle 1: (10,000, 3)
...
Cycle n: (5,000 * (n+1), 3)

available_pool: DataFrame (remaining)
Cycle 0: (45,000, 2)
Cycle 1: (40,000, 3)  # includes prediction column
...
```

##### 3.2.5 Model Training

**Training Process (`learners/base.py`):**
```python
learner.train(selected_with_labels, target_column)
```

**Inside `train()` Method:**

1. **Training Data Accumulation:**
   ```python
   if self.training_data.empty:
       self.training_data = compounds.copy()
   else:
       self.training_data = pd.concat([self.training_data, compounds], ignore_index=True)
   ```

2. **Feature Generation:**
   ```python
   X = smiles_to_fingerprints(self.training_data['SMILES'].tolist())
   y = self.training_data[target_column].values
   ```

3. **Model Fitting:**
   ```python
   self.model.fit(X, y)  # RandomForestRegressor.fit()
   self.is_trained = True
   ```

**Training Data Evolution:**
```
Cycle 0: X.shape = (5,000, 2048), y.shape = (5,000,)
Cycle 1: X.shape = (10,000, 2048), y.shape = (10,000,)
...
Model performance improves with more training data
```

##### 3.2.6 Monitoring and Evaluation

**Full Pool Predictions for Monitoring:**
```python
all_predictions = learner.predict(compound_pool)
predictions_df = pd.DataFrame({
    'ID': compound_pool['ID'],
    'prediction': all_predictions
})
```

**Cycle Report Generation (`evaluation/monitor.py`):**
```python
report = create_cycle_report(
    cycle=cycle + 1,
    predictions_df=predictions_df,
    ground_truth_df=ground_truth_full,
    newly_selected_df=selected,
    target_column=target_column,
    score_direction=score_direction,
    top_k=monitoring_config['top_k'],
    enrichment_percentile=monitoring_config['enrichment_percentile']
)
```

**Inside `create_cycle_report()`:**

1. **Data Merging:**
   ```python
   merged = pd.merge(predictions_df, ground_truth_df[['ID', target_column]], on='ID')
   ```

2. **RMSE Calculation (`evaluation/metrics.py`):**
   ```python
   rmse = calculate_rmse(merged[target_column].values, merged['prediction'].values)
   # Uses sklearn.metrics.mean_squared_error internally
   ```

3. **Top-K Overlap Calculation:**
   ```python
   top_k_overlap = calculate_top_k_overlap(
       predictions_df, ground_truth_df, top_k, target_column, score_direction
   )
   ```

**Top-K Overlap Algorithm (`evaluation/metrics.py`):**
```python
def calculate_top_k_overlap(predictions_df, ground_truth_df, k, target_column, score_direction):
    merged = pd.merge(predictions_df, ground_truth_df[['ID', target_column]], on='ID')
    ascending = (score_direction == 'lower')
    
    # Top K by predictions
    top_k_predicted = set(merged.nlargest(k, 'prediction', keep='first')['ID'].values) \
                      if not ascending else \
                      set(merged.nsmallest(k, 'prediction', keep='first')['ID'].values)
    
    # Top K by ground truth
    top_k_true = set(merged.nlargest(k, target_column, keep='first')['ID'].values) \
                 if not ascending else \
                 set(merged.nsmallest(k, target_column, keep='first')['ID'].values)
    
    # Calculate overlap
    overlap_count = len(top_k_predicted & top_k_true)
    overlap_percentage = (overlap_count / k) * 100
    return overlap_percentage
```

4. **Average Score Calculation:**
   ```python
   newly_selected_with_truth = pd.merge(
       newly_selected_df, ground_truth_df[['ID', target_column]], on='ID'
   )
   avg_score_selected = calculate_average_score(newly_selected_with_truth[target_column].values)
   ```

5. **Enrichment Factor Calculation (if Activity column exists):**
   ```python
   if 'Activity' in ground_truth_df.columns:
       merged_activity = pd.merge(predictions_df, ground_truth_df[['ID', 'Activity']], on='ID')
       enrichment_factor = calculate_enrichment_factor(
           merged_activity['prediction'].values,
           merged_activity['Activity'].values,
           enrichment_percentile
       )
   ```

**Enrichment Factor Algorithm:**
```python
def calculate_enrichment_factor(scores, labels, percentile):
    # EF = (n_actives_selected / n_selected) / (n_actives_total / n_total)
    sorted_indices = np.argsort(scores)[::-1]  # Sort descending
    sorted_labels = labels[sorted_indices]
    
    n_total = len(labels)
    n_selected = max(1, int(n_total * percentile / 100))
    n_actives_total = np.sum(labels)
    n_actives_selected = np.sum(sorted_labels[:n_selected])
    
    ef = (n_total * n_actives_selected) / (n_actives_total * n_selected)
    return ef
```

**Report Data Structure:**
```python
report = {
    'cycle': int,
    'rmse': float,
    'top_k_overlap': float,  # Percentage 0-100
    'avg_score_selected': float,
    'enrichment_factor': float or None,
    'n_compounds_selected': int,
    'n_total_predictions': int
}
```

**Monitoring Results Accumulation:**
```python
monitoring_results.append(report)
```

### Phase 4: Selection Strategy Deep Dive

#### 4.1 Greedy Selection (`strategies/greedy.py`)

**Algorithm:**
1. **Input:** DataFrame with 'prediction' column
2. **Sorting:** Sort compounds by prediction values
3. **Direction:** Ascending if score_direction='lower', descending if 'higher'
4. **Selection:** Take top n_select compounds

**Data Transformation Example:**
```
Input (available_pool with predictions):
ID    SMILES    prediction
1     CCO       0.85
2     CC        0.92
3     CCC       0.78
...

After sorting (score_direction='higher'):
ID    SMILES    prediction
2     CC        0.92  ← selected
1     CCO       0.85  ← selected  
3     CCC       0.78
...

Output (selected):
First n_select compounds from sorted list
```

#### 4.2 Random Selection (`strategies/random.py`)

**Algorithm:**
```python
selected = compounds.sample(n=min(n_select, len(compounds)), random_state=random_state)
```

**Properties:**
- Uniform random sampling
- No bias towards any prediction values
- Reproducible with random_state

#### 4.3 Diversity Selection (`strategies/diversity.py`)

**Complex Multi-Step Algorithm:**

1. **Memory Management:**
   ```python
   if len(working_compounds) > max_compounds:  # Default: 35,000
       working_compounds = working_compounds.sample(n=max_compounds, random_state=random_state)
   ```

2. **Fingerprint Generation:**
   ```python
   rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
   fingerprints = []
   for smiles in working_compounds['SMILES']:
       mol = Chem.MolFromSmiles(smiles)
       fp = rdkit_gen.GetFingerprint(mol)  # RDKit fingerprint (different from Morgan)
       fingerprints.append(fp)
   ```

3. **Distance Matrix Calculation:**
   ```python
   def _calculate_tanimoto_distances(fingerprints):
       distances = []
       n_fps = len(fingerprints)
       for i in range(1, n_fps):
           similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
           distances.extend([1 - sim for sim in similarities])  # Convert similarity to distance
       return distances
   ```

4. **Butina Clustering:**
   ```python
   clusters = Butina.ClusterData(distances, len(fingerprints), 0.4, isDistData=True)
   # Threshold = 0.4 means compounds with Tanimoto distance > 0.4 are in different clusters
   ```

5. **Cluster-Based Selection:**
   ```python
   sorted_clusters = sorted(clusters, key=len, reverse=True)  # Largest clusters first
   
   # Step 1: Select cluster centers
   for cluster in sorted_clusters:
       if len(selected_indices) < n_select:
           selected_indices.append(cluster[0])  # Cluster center
   
   # Step 2: Select additional members from large clusters
   for cluster in sorted_clusters:
       if len(cluster) > 10:
           n_from_cluster = min(10, n_select - len(selected_indices))
       else:
           n_from_cluster = min(len(cluster) // 2 + 1, n_select - len(selected_indices))
       
       for i in range(1, min(n_from_cluster + 1, len(cluster))):
           if len(selected_indices) < n_select:
               selected_indices.append(cluster[i])
   ```

**Data Flow in Diversity Selection:**
```
SMILES → RDKit FPs → Tanimoto Distances → Butina Clusters → Representative Selection
Example:
50,000 compounds → 35,000 sampled → 12,000 clusters → 5,000 diverse representatives
```

### Phase 5: Final Results and Output

#### 5.1 Final Predictions Generation

**After All Cycles Complete:**
```python
final_predictions = pd.DataFrame({
    'ID': compound_pool['ID'],
    'SMILES': compound_pool['SMILES'], 
    'prediction': learner.predict(compound_pool)
})
```

**Data Structure:**
```
final_predictions: DataFrame
├── ID: all compound identifiers
├── SMILES: molecular structures
└── prediction: final model predictions for entire pool

Shape: (original_pool_size, 3)
Example: (50,000, 3)
```

#### 5.2 Results Persistence

**File Outputs:**
```python
if output_dir:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Monitoring results
    save_monitoring_results(monitoring_results, output_dir / "monitoring_results.csv")
    
    # Final predictions
    final_predictions.to_csv(output_dir / "final_predictions.csv", index=False)
    
    # Training data
    labeled_compounds.to_csv(output_dir / "labeled_compounds.csv", index=False)
    
    # Experiment configuration
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=2)
```

**Monitoring Results CSV Structure:**
```csv
cycle,rmse,top_k_overlap,avg_score_selected,enrichment_factor,n_compounds_selected,n_total_predictions
1,0.5234,12.50,0.7845,2.34,5000,50000
2,0.4987,18.75,0.8123,2.67,5000,50000
...
```

#### 5.3 Performance Summary

**Console Output Generation:**
```python
print("EXPERIMENT SUMMARY")
if monitoring_results:
    first_result = monitoring_results[0]
    last_result = monitoring_results[-1]
    
    # Top-K improvement
    improvement = last_result['top_k_overlap'] - first_result['top_k_overlap']
    print(f"Top-{config['monitoring']['top_k']} Overlap:")
    print(f"  First cycle: {first_result['top_k_overlap']:.2f}%")
    print(f"  Last cycle:  {last_result['top_k_overlap']:.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")
    
    # Score trend analysis
    direction_better = config['score_direction'] == 'higher'
    score_improved = (last_result['avg_score_selected'] > first_result['avg_score_selected']) == direction_better
    print(f"  Trend: {'Better' if score_improved else 'Worse'}")
```

## Data Validation and Error Handling

### Input Validation

1. **File Existence Checks:**
   ```python
   if not compound_pool_path.exists():
       print(f"Error: Compound pool file not found: {compound_pool_path}", file=sys.stderr)
       sys.exit(1)
   ```

2. **Required Column Validation:**
   ```python
   required_cols = ['ID', 'SMILES']
   missing_cols = [col for col in required_cols if col not in compound_pool.columns]
   if missing_cols:
       raise ValueError(f"Compound pool missing columns: {missing_cols}")
   ```

3. **Batch Size Validation:**
   ```python
   if batch_size < 1:
       raise ValueError(f"Batch size too small: {batch_size}. Increase batch_size_fraction.")
   ```

### Runtime Error Handling

1. **Invalid SMILES Handling:**
   ```python
   mol = Chem.MolFromSmiles(smiles)
   if mol is None:
       raise ValueError(f"Invalid SMILES: {smiles}")
   ```

2. **Missing Compounds in Ground Truth:**
   ```python
   if len(result) < len(compounds):
       missing_count = len(compounds) - len(result)
       print(f"Warning: {missing_count} compounds not found in ground truth")
   ```

3. **Model Training Validation:**
   ```python
   if not self.is_trained:
       raise RuntimeError("Model must be trained before prediction")
   ```

## Memory and Performance Optimizations

### Parallel Processing

1. **Fingerprint Generation:**
   ```python
   fingerprints = Parallel(n_jobs=n_jobs)(
       delayed(smiles_to_morgan_fingerprint)(smiles) 
       for smiles in smiles_list
   )
   ```

2. **CPU Core Management:**
   ```python
   if n_jobs == -1:
       n_jobs = min(os.cpu_count() or 1, 32)  # Cap at 32 cores
   ```

### Memory Management

1. **Diversity Selection Sampling:**
   ```python
   if len(working_compounds) > max_compounds:  # 35,000 limit
       working_compounds = working_compounds.sample(n=max_compounds, random_state=random_state)
   ```

2. **Efficient Data Structures:**
   - Use of pandas DataFrames for structured data
   - NumPy arrays for numerical computations
   - Sets for overlap calculations

## Data Quality Metrics

### Active Learning Performance Metrics

1. **Root Mean Squared Error (RMSE):**
   - Measures prediction accuracy against ground truth
   - Lower values indicate better model performance

2. **Top-K Overlap:**
   - Percentage of overlap between predicted and true top-K compounds
   - Higher values indicate better ranking performance

3. **Average Score of Selected:**
   - Mean target value of newly selected compounds each cycle
   - Trend indicates learning efficiency

4. **Enrichment Factor:**
   - Ratio of active compounds in selected set vs. random selection
   - Values > 1 indicate enrichment above random

### Data Flow Monitoring

The system continuously tracks:
- Number of labeled vs. unlabeled compounds
- Model prediction distributions
- Selection strategy effectiveness
- Computational resource usage

This comprehensive data flow ensures robust, reproducible, and efficient active learning for molecular screening applications.
