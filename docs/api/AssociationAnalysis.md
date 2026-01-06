# AssociationAnalysis

## Description
Performs genome-wide (GWAS) or epigenome-wide (EWAS) association analyses.

## Arguments
| Argument | Description |
|-----------|-------------|
| `-i, --input` | Path to the HDF5 file containing genotype or methylation data |
| `-m, --metadata` | CSV with sample-level metadata/covariates. If omitted the launcher will attempt to use HDF5 metadata |
| `-s, --sample_id` | Column name in the metadata CSV matching HDF5 sample IDs |
| `-o, --output` | Path where the script should write the CSV results |
| `-f, --formula` | Formula (e.g. `Methylation ~ Trait + Age + Sex`) |
| `-t, --analysis_type` | `Auto` (default), `EWAS`, or `GWAS` |
| `-d, --model` | Regression model: `linear` (default) or `logistic` |
| `-p, --threads` | Number of CPU threads to use |
| `-c, --chunk_size` | Number of features (probes/variants) per processing chunk |
| `-a, --var` | The variable to extract the statistics for |


## Options

### `-t, --analysis_type`
- `Auto`: infer from parsed formula's data variable (default).
- `EWAS`: force methylation analysis.
- `GWAS`: force genotype analysis.
The script validates that the formula's data variable matches the chosen mode and that the dependent variable is appropriate (for EWAS the dependent must reference the methylation data variable; for GWAS the dependent must be a phenotype, not the genotype variable).

### `-d, --model`
- `linear` (default): continuous trait regression.
- `logistic`: binary/case-control regression (mapped to the downstream R test type).

### `-p, --threads`
If unspecified the launcher uses SystemUtils.get_optimal_cores(reserve_cores=1) or the configured reserve in system config. Memory-per-core is computed from available RAM and the selected thread count.

### `-c, --chunk_size`
If not provided, the launcher:
- Attempts to read the number of samples from HDF5.
- Uses conservative per-entry byte estimates (EWAS vs GWAS) and available memory to derive a chunk size bounded by sensible min/max values.

### `-s, --sample_id`
If metadata CSV is provided but the sample ID column is not specified, the launcher will:
- Read the first few sample IDs from the HDF5 metadata group (SampleList/IID).
- Search metadata columns for one containing those IDs and auto-select it if found.
Auto-detection logs a warning when it fails.

### `-a, --var`
Maps to the `var` CLI option in the script. Requested stat variables are validated against covariates parsed from the formula; data-stream variables (Methylation/Genotype) are rejected for stat extraction. If requested variables are not present, the launcher falls back to the first covariate (if available) and emits warnings.

## Usage

```bash
# Automatic analysis type detection (EWAS or GWAS inferred from formula)
GT-AssociationAnalysis -i data.h5 -m metadata.csv -o results.csv -f "Methylation ~ Trait + Age + Sex"

# Force EWAS mode with explicit analysis type
GT-AssociationAnalysis -i methylation.h5 -m metadata.csv -o ewas_results.csv -f "Methylation ~ Smoking + Age + Sex" -a EWAS

# Force GWAS mode with explicit analysis type
GT-AssociationAnalysis -i genotypes.h5 -m metadata.csv -o gwas_results.csv -f "BMI ~ Genotype + Age + Sex + PC1 + PC2" -a GWAS

# Case-control logistic GWAS
GT-AssociationAnalysis -i genotypes.h5 -m metadata.csv -o case_control.csv -f "Disease ~ Genotype + Age + Sex + PC1 + PC2" -a GWAS -d logistic

# EWAS with automatic chunk size
GT-AssociationAnalysis -i methylation.h5 -m metadata.csv -o ewas_results.csv -f "Methylation ~ Smoking + Age + Sex"

# Parallelized GWAS using 8 threads
GT-AssociationAnalysis -i genotypes.h5 -m metadata.csv -o parallel_gwas.csv -f "Height ~ Genotype + Age + Sex" -a GWAS -p 8

# Extract statistics for 'Age' (rather than the default dependent variable)
GT-AssociationAnalysis -i methylation.h5 -m metadata.csv -o ewas_results_age.csv -f "Methylation ~ Age + Sex" -a Age -a EWAS
```
