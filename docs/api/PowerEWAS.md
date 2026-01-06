# PowerEWAS

## Description
Estimate statistical power for Epigenome-Wide Association Studies (EWAS) under realistic methylation distributions, tissue-specific parameters, and user-defined effect sizes. Supports simulation-based power analysis across a range of sample sizes, effect sizes, and detection thresholds, with comprehensive output and visualization.

## Arguments
| Argument | Description |
|----------|-------------|
| `-min, --min_sample_size` | Minimum total sample size to evaluate |
| `-max, --max_sample_size` | Maximum total sample size to evaluate |
| `-step, --sample_size_steps` | Step size for sample size increments |
| `-ctrl, --control_proportion` | Proportion of samples assigned to control group (0–1) |
| `-delta, --target_delta` | Comma-separated effect sizes (mean methylation difference) to target |
| `-sd, --delta_sd` | Comma-separated standard deviations for effect size distribution |
| `-J, --n_cpgs` | Number of CpG sites to simulate per run |
| `-dm, --target_dm_cpgs` | Number of differentially methylated CpGs per simulation |
| `-tissue, --tissue_type` | Tissue type for methylation parameterization |
| `-limit, --detection_limit` | Minimum effect size considered biologically meaningful |
| `-method, --dm_method` | Differential methylation test method |
| `-fdr, --fdr_threshold` | False discovery rate threshold for significance |
| `-sims, --n_simulations` | Number of simulation replicates per scenario |
| `-o, --output` | Output directory or filename for results |

## Options

### `-min, --min_sample_size`
Minimum sample size to consider:
- Integer (e.g., 10)
- Defines the smallest total sample size for power estimation

### `-max, --max_sample_size`
Maximum sample size to consider:
- Integer (e.g., 100)
- Defines the largest total sample size for power estimation

### `-step, --sample_size_steps`
Sample size increment:
- Integer (e.g., 10)
- Step size between evaluated sample sizes

### `-ctrl, --control_proportion`
Proportion of controls:
- Float between 0 and 1 (e.g., 0.5)
- Fraction of samples assigned to the control group

### `-delta, --target_delta`
Target effect sizes:
- Comma-separated floats (e.g., "0.05,0.10,0.15")
- Specifies mean methylation differences to target
- Mutually exclusive with `--delta_sd`

### `-sd, --delta_sd`
Standard deviation(s) for effect size distribution:
- Comma-separated floats (e.g., "0.05,0.10")
- Specifies SD for effect size distribution (tau)
- Mutually exclusive with `--target_delta`

### `-J, --n_cpgs`
Number of CpGs to simulate:
- Integer (e.g., 100000)
- Number of CpG sites per simulation

### `-dm, --target_dm_cpgs`
Number of differentially methylated CpGs:
- Integer (e.g., 100)
- Number of CpGs with simulated effect per run

### `-tissue, --tissue_type`
Tissue type for simulation:
- String (e.g., "Adult (PBMC)", "Saliva", "Sperm", etc.)
- Must match one of the supported tissue types

### `-limit, --detection_limit`
Detection threshold for meaningful effect:
- Float (e.g., 0.01)
- Minimum effect size considered biologically meaningful

### `-method, --dm_method`
Differential methylation test:
- limma: Moderated t-test (default)
- t-test (unequal var): Welch's t-test
- t-test (equal var): Standard t-test
- Wilcox rank sum: Wilcoxon rank-sum test

### `-fdr, --fdr_threshold`
False discovery rate threshold:
- Float (e.g., 0.05)
- FDR cutoff for calling significance

### `-sims, --n_simulations`
Number of simulation replicates:
- Integer (e.g., 50)
- Number of independent simulations per scenario

### `-o, --output`
Output directory or filename:
- Path to directory or CSV file for results and plots

## Usage

```sh
# Basic power analysis for PBMC tissue, default settings
PowerEWAS -tissue "Adult (PBMC)" -delta "0.05,0.10,0.15" -o results/

# Power analysis with custom sample size range and detection limit
PowerEWAS -min 20 -max 200 -step 20 -tissue "Saliva" -delta "0.10" -limit 0.02 -o saliva_power.csv

# Using effect size standard deviation (tau) instead of fixed delta
PowerEWAS -tissue "Placenta" -sd "0.05,0.08" -dm 200 -J 200000 -sims 100 -o placenta_power/

# Welch's t-test and custom FDR threshold
PowerEWAS -tissue "Blood adult" -delta "0.07" -method "t-test (unequal var)" -fdr 0.01 -o blood_power.csv

# Simulate with 70% controls and 30% cases
PowerEWAS -ctrl 0.7 -tissue "Liver" -delta "0.12"
```
