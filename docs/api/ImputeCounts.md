# ImputeCounts

## Description
Impute missing values in genomic data using advanced statistical methods. Supports genotype imputation via IMPUTE2 with 1000 Genomes reference panels and methylation data imputation using the K-Nearest Neighbors (KNN) algorithm. Handles HDF5 format, with automatic resource optimization and memory-efficient processing for large datasets.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to input data file (HDF5 format) |
| `-o, --output` | Path to save imputed data file |
| `-dt, --data_type` | Type of genomic data to process (genotype or methylation) |
| `-r, --reference_dir` | Directory containing 1000 Genomes reference files (required for genotype) |
| `-k, --k` | Number of neighbors for KNN imputation (methylation only) |
| `-w, --window` | Window size for IMPUTE2 processing (genotype only) |
| `-bf, --buffer` | Buffer size around imputation windows (genotype only) |
| `-ne, --effective_size` | Effective population size for IMPUTE2 (genotype only) |
| `-th, --threshold` | Minimum INFO score for quality filtering (genotype only) |
| `-s, --samples` | Comma-separated list of samples to include |

## Options

### `-dt, --data_type`
Type of genomic data for processing:
- genotype: SNP genotype data requiring IMPUTE2 imputation
- methylation: DNA methylation beta values using KNN imputation

### `-r, --reference_dir`
Directory containing 1000 Genomes Project Phase 3 reference files (required for genotype imputation):
- Required files: `1000GP_Phase3_chr{N}.hap.gz`, `1000GP_Phase3_chr{N}.legend.gz`, `genetic_map_chr{N}_combined_b37.txt`, `1000GP_Phase3.sample`

### `-k, --k`
Number of neighbors for KNN imputation (methylation only):
- Default: 5

### `-w, --window`
Window size for IMPUTE2 processing (genotype only):
- Default: 5,000,000 bp

### `-bf, --buffer`
Buffer size around imputation windows (genotype only):
- Default: 250,000 bp

### `-ne, --effective_size`
Effective population size for IMPUTE2 (genotype only):
- Default: 20,000

### `-th, --threshold`
Minimum INFO score for quality filtering (genotype only):
- Default: None (no filtering)
- Range: 0.0-1.0

### `-s, --samples`
Comma-separated list of sample IDs to include:
- Default: All samples

## Usage

```sh
# Basic KNN imputation for methylation data
ImputeCounts -i methylation.h5 -o imputed_methylation.h5 -dt methylation

# Custom number of neighbors
ImputeCounts -i methylation.h5 -o imputed_methylation.h5 -dt methylation -k 10

# Basic genotype imputation (requires reference directory)
ImputeCounts -i genotypes.h5 -o imputed_genotypes.h5 -dt genotype -r /path/to/1000GP_reference/

# Imputation with quality filtering
ImputeCounts -i genotypes.h5 -o imputed_genotypes.h5 -dt genotype -r /path/to/1000GP_reference/ -th 0.5

# Subset specific samples
ImputeCounts -i methylation.h5 -o imputed_methylation.h5 -dt methylation -s "sample1,sample2,sample3"
```
