# VariantQC

## Description
Comprehensive variant quality control and analysis tool for genomic data supporting MAF calculation, Hardy-Weinberg equilibrium testing, LD pruning, and MAF filtering with high-performance parallel processing.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to input HDF5 file containing genotype data |
| `-a, --analysis` | Type of variant analysis to perform |
| `-o, --output` | Path to output file (required for filtering/pruning operations) |
| `-t, --threshold` | Significance or frequency threshold for analysis |
| `-p, --pop_code` | Population code for HWE analysis (optional) |
| `-w, --window` | Window size for LD pruning (number of variants) |
| `-s, --step` | Step size for LD pruning window advancement |
| `-r, --r2` | R² threshold for LD pruning |
| `-m, --maf` | MAF threshold for LD pruning pre-filtering |

## Options

### `-a, --analysis`
Type of variant analysis to perform:
- maf: Calculate Minor Allele Frequency for all variants (default)
- hwe: Hardy-Weinberg Equilibrium testing with population stratification
- ld_prune: Linkage Disequilibrium pruning for variant independence
- maf_filter: Filter variants based on minimum MAF threshold

### `-t, --threshold`
Analysis-specific threshold values:

MAF Analysis:
- Default: 0.01 (1% MAF)
- Range: 0.001-0.5

HWE Analysis:
- Default: 1e-6
- Range: 1e-10 to 0.05

LD Pruning:
- Default: 0.2 (r²)
- Range: 0.1-0.9

MAF Filtering:
- Default: 0.01 (1% MAF)
- Range: 0.001-0.1

### `-p, --pop_code`
Population code for HWE analysis:
- Default: None (all samples)
- Integer code or string identifier

### `-w, --window`
Window size for LD pruning:
- Default: 50 variants
- Range: 10-500

### `-s, --step`
Step size for window advancement:
- Default: 5 variants
- Range: 1-50

### `-r, --r2`
R² threshold for LD pruning:
- Default: 0.2
- Range: 0.1-0.9

### `-m, --maf`
MAF threshold for LD pruning pre-filtering:
- Default: 0.01
- Range: 0.0-0.1

## Usage

```sh
# Basic MAF calculation
GT-VariantQC -i genotypes.h5 -a maf

# HWE for all samples
GT-VariantQC -i genotypes.h5 -a hwe -t 1e-6

# Standard LD pruning
GT-VariantQC -i genotypes.h5 -a ld_prune -o pruned_genotypes.h5

# LD pruning with MAF pre-filter
GT-VariantQC -i genotypes.h5 -a ld_prune -o pruned.h5 -m 0.05 -r 0.2

# Filter common variants
GT-VariantQC -i genotypes.h5 -a maf_filter -o common_variants.h5 -t 0.01
```
