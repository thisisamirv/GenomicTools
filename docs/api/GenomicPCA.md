# GenomicPCA

## Description
Perform Principal Component Analysis (PCA) on genomic data to reduce dimensionality and identify major sources of variation for population structure analysis, batch effect detection, and covariate generation.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to input data file (HDF5 or CSV format) |
| `-o, --output` | Path to save PC scores output file |
| `-n, --n_components` | Number of principal components to compute |
| `-d, --data_type` | Type of genomic data for analysis |
| `-r, --residuals` | Process residuals data from CSV file |
| `-b, --batch_size` | Batch size for incremental PCA processing |
| `-s, --scale` | Apply standardization to genotype data |
| `-l, --ld_pruned` | Indicate use of LD-pruned variants (recommended) |

## Options

### `-d, --data_type`
Specifies the type of genomic data for analysis:
- methylation: DNA methylation beta values from EWAS (default)
- genotype: SNP genotype data from GWAS (0, 1, 2 coding)

### `-n, --n_components`
Number of principal components to compute:
- Default: 10
- Range: 1 to min(samples, features)

### `-r, --residuals`
Process residuals data from association analyses:
- False: Process HDF5 genomic data (default)
- True: Process CSV residuals file

### `-b, --batch_size`
Batch size for memory-efficient processing:
- Default: 5000

### `-s, --scale`
Apply standardization to data:
- True: Z-score standardization (default for genotype)
- False: Use raw values (default for methylation)

### `-l, --ld_pruned`
Indicate use of LD-pruned variants:
- False: Using all variants (default)
- True: Using LD-pruned variant set

## Usage

```sh
# Standard methylation PCA (10 components)
GT-GenomicPCA -i methylation_data.h5 -o pca_results.csv -d methylation

# LD-pruned genotype PCA
GT-GenomicPCA -i genotypes_pruned.h5 -o genotype_pca.csv -d genotype -l True -s True -n 15

# PCA on association analysis residuals
GT-GenomicPCA -i residuals.csv -o residual_pca.csv -r True -n 5

# Large dataset with smaller batches
GT-GenomicPCA -i large_data.h5 -o pca.csv -b 1000 -n 10

# All-variant genotype PCA
GT-GenomicPCA -i genotypes_all.h5 -o genotype_pca.csv -d genotype -s True -n 10
```
