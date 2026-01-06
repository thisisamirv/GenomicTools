# ImportanceRank

## Description
Calculate feature importance rankings for genomic data using Random Forest models with bootstrap sampling to identify the most informative features for classification or regression tasks.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to input data file (HDF5 or CSV format) |
| `-t, --target` | Path to target variable file (CSV format) |
| `-o, --output` | Path to save importance rankings results |
| `-e, --iterations` | Number of bootstrap iterations to perform |
| `-r, --fraction` | Fraction of samples to use in each bootstrap |
| `-c, --count` | Whether input is HDF5 count data (True) or CSV (False) |
| `-a, --categorical` | Whether target variable is categorical (True) or continuous (False) |

## Options

### `-e, --iterations`
Number of bootstrap iterations for importance estimation:
- Default: 100
- More iterations provide more stable estimates but take longer

### `-r, --fraction`
Fraction of samples to use in each bootstrap iteration:
- Default: 0.7
- Range: 0.5-0.9

### `-c, --count`
Specifies input data format:
- True: HDF5 format with genomic count data (default)
- False: CSV format with feature matrix

### `-a, --categorical`
Specifies target variable type:
- True: Categorical target (classification, default)
- False: Continuous target (regression)

## Usage

```sh
# HDF5 methylation data (classification)
GT-ImportanceRank -i methylation.h5 -t targets.csv -o importance_rankings.csv

# CSV feature matrix (regression)
GT-ImportanceRank -i features.csv -t outcomes.csv -o rankings.csv -c False -a False

# High-precision ranking with more iterations
GT-ImportanceRank -i data.h5 -t targets.csv -o rankings.csv -e 500 -r 0.8

# Quick ranking with fewer iterations
GT-ImportanceRank -i data.h5 -t targets.csv -o rankings.csv -e 50 -r 0.6

# Genotype data (HDF5, classification)
GT-ImportanceRank -i genotypes.h5 -t disease_status.csv -o gwas_rankings.csv
```
