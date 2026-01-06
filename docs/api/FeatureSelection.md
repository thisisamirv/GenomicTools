# FeatureSelection

## Description
Perform systematic feature selection for genomic data using machine learning models with automated preprocessing, hyperparameter optimization, and early stopping criteria to identify optimal feature set sizes.

## Arguments
| Argument | Description |
|----------|------------|
| `-m, --metadata` | Path to CSV file containing sample metadata with train/test splits |
| `-c, --counts` | Path to HDF5 or CSV file containing feature data |
| `-r, --response` | Column name for target variable in metadata |
| `-o, --output` | Path to save feature selection results |
| `-i, --indices` | Feature set sizes to evaluate (comma-separated) |
| `-b, --probes` | Path to probe list file for feature filtering |
| `-e, --preprocessing` | Data preprocessing methods to apply |
| `-g, --regularization` | Regularization parameters for noise injection |
| `-a, --evaluation` | Cross-validation evaluation parameters |
| `-t, --training` | Training configuration options |
| `-s, --stopping_criteria` | Early stopping criteria for optimization |
| `-p, --parameters` | Model-specific hyperparameters |
| `-n, --importance_scores` | Path to feature importance file for ordering |
| `-d, --model` | Machine learning algorithm to use |
| `-x, --categorical` | Whether target variable is categorical |

## Options

### `-d, --model`
Machine learning algorithms available for feature selection:
- RF: Random Forest (default)
- EN: Elastic Net
- SVM: Support Vector Machine
- KNN: K-Nearest Neighbors

### `-i, --indices`
Comma-separated list of feature set sizes to evaluate, e.g. "10,50,100,500".

### `-e, --preprocessing`
Data preprocessing pipeline (comma-separated methods):
- scale, center, outlierCapping, knnImpute, YeoJohnson, nzv, NULL

Default: "scale,center,outlierCapping,knnImpute,YeoJohnson,nzv"

### `-g, --regularization`
Regularization through noise injection:
- noise: Noise level (default: 0.1)
- adaptive_noise: Scale noise by feature variance (default: TRUE)

Example: "noise=0.05,adaptive_noise=FALSE"

### `-a, --evaluation`
Cross-validation settings:
- nfolds: Number of CV folds (default: 5)

### `-t, --training`
Training configuration:
- class_weightening: Handle class imbalance (default: TRUE)

### `-s, --stopping_criteria`
Early stopping parameters:
- early_stop: Enable early stopping (default: TRUE)
- monitor: Metrics to monitor (AUROC, AUPRC, RMSE, R2)
- patience: Epochs to wait without improvement (default: 10)
- cut: Minimum improvement threshold (default: 0.005)

Example: "early_stop=TRUE,monitor=AUROC AUPRC,patience=15,cut=0.01"

### `-p, --parameters`
Model-specific hyperparameters, e.g.:
- RF: "n_estimators=100,max_depth=10"
- EN: "alpha=0.1,l1_ratio=0.5"
- SVM: "C=1.0,gamma=scale"
- KNN: "n_neighbors=5,weights=distance"

### `-x, --categorical`
Target variable type:
- True: Classification (default)
- False: Regression

## Usage

```sh
# Classification with Random Forest
GT-FeatureSelection -m metadata.csv -c data.h5 -r disease_status -o results.csv -i "10,50,100,500"

# Regression with Elastic Net
GT-FeatureSelection -m metadata.csv -c data.h5 -r expression_level -o results.csv -i "25,100,250" -d EN -x False

# Custom preprocessing and regularization
GT-FeatureSelection -m metadata.csv -c data.h5 -r phenotype -o results.csv -i "10,50,100,200,500" -e "scale,center,knnImpute" -g "noise=0.05,adaptive_noise=TRUE"

# SVM with custom hyperparameters
GT-FeatureSelection -m metadata.csv -c data.h5 -r disease -o results.csv -i "20,100,500" -d SVM -p "C=0.1,gamma=0.001" -s "early_stop=TRUE,patience=20"

# Use pre-computed feature importance
GT-FeatureSelection -m metadata.csv -c data.h5 -r outcome -o results.csv -i "10,25,50,100" -n feature_importance.csv -d RF
```
