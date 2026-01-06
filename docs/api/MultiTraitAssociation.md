# MultiTraitAssociation

## Description
Runs multi-trait association analysis using the CPASSOC implementation (R). Combines per-trait t-values (for SHom/SHet) across multiple summary-statistic files, accounts for trait correlation via a supplied correlation matrix, and reports SHom / SHet results with Bonferroni and FDR (BH) correction.

## Arguments
| Argument | Description |
|----------|-------------|
| `-i, --input` | Comma-separated paths to CSV summary-statistic files (one per trait). |
| `-c, --correlation_matrix` | Path to CSV file containing the trait correlation matrix (square, diagonal = 1). |
| `-s, --sample_sizes` | Comma-separated sample sizes (one value per input file). |
| `-o, --output_file` | Path where the script should write the CSV results. |
| `-m, --marker_col` | Comma-separated list of marker ID column names (one per input file) or a single name to apply to all files. |
| `-t, --t_col` | Comma-separated list of t-value (or z-value) column names (one per input file) or a single name to apply to all files. |
| `-d, --cols_to_add` | Comma-separated list of additional column names to copy from the input files into the final output. |
| `-r, --traits` | Comma-separated list of trait labels (one per input file) used to name per-study columns in the output |
| `-a, --alpha` | Significance level for Bonferroni / FDR threshold (default 0.05). |

## Options

- `--correlation_matrix`
  - Must be square and match the number of input files. Diagonal must be 1 and matrix must be symmetric.

## Usage


```bash
# Single marker/t applied to all files, include per-study p-values via cols_to_add:
GT-MultiTraitAssociation \
  -i a.csv,b.csv,c.csv \
  -c corr.csv \
  -s 1000,900,1200 \
  -m CGID \
  -t T_STAT \
  --cols_to_add P \
  -r TraitA,TraitB,TraitC \
  -o results.csv

# Per-file marker/t columns and include extra columns (e.g. ALLELE, MAF):
GT-MultiTraitAssociation \
  -i trait1.csv,trait2.csv \
  -c corr.csv \
  -s 1500,2000 \
  -m id_col_trait1,id_col_trait2 \
  -t tcol1,tcol2 \
  --cols_to_add P,ALLELE,MAF \
  -r Trait1,Trait2 \
  -o results.csv

# Omit traits to auto-generate labels:
GT-MultiTraitAssociation \
  -i a.csv,b.csv,c.csv \
  -c corr.csv \
  -s 1000,1000,1000 \
  -m SNP \
  -t Z \
  --cols_to_add P \
  -o results_auto_traits.csv
```