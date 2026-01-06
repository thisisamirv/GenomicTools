# MetaAssociation

## Description
Perform meta-analysis on GWAS and EWAS results from multiple studies using fixed-effects and random-effects models with comprehensive heterogeneity assessment and multiple testing correction.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Comma-separated paths to association result files |
| `-n, --names` | Comma-separated study names (same order as input files) |
| `-s, --sample_sizes` | Comma-separated sample sizes (optional; auto-detected if not provided) |
| `-t, --populations` | Comma-separated population/ancestry labels (optional) |
| `-o, --output` | Output filename for meta-analysis results |
| `-m, --method` | Meta-analysis method to perform |
| `-d, --data_type` | Type of association data (auto-detected by default) |
| `-a, --var` | Variable name prefix for multi-variable datasets |

## Options

### `-i, --input`
Paths to input association result files:
- Comma-separated file paths (e.g., "study1.csv,study2.csv,study3.csv")
- Each file must contain required columns (RSID/CGID, COEF, SE, P)
- Column names are automatically standardized

### `-n, --names`
Study identifiers for each input file:
- Comma-separated names (e.g., "AFR_Study,EUR_Study,AMR_Study")
- Must match the number and order of input files

### `-s, --sample_sizes`
Sample sizes for each study:
- Comma-separated integers (e.g., "1000,1200,1500")
- Optional; auto-detected from input files if not provided

### `-t, --populations`
Population/ancestry information:
- Comma-separated labels (e.g., "AFR,EUR,AMR")
- Optional; used for interpretation and output context

### `-o, --output`
Output filename for results:
- CSV file with meta-analysis results

### `-m, --method`
Meta-analysis approach:
- fixed: Fixed-effects only
- random: Random-effects only
- both: Both fixed and random-effects (default)

### `-d, --data_type`
Type of association data:
- auto: Auto-detect GWAS or EWAS (default)
- GWAS: SNP association results (RSID)
- EWAS: Methylation association results (CGID)

### `-a, --var`
Variable name prefix for multi-variable datasets:
- Specify which variable to analyze when multiple variables present
- Auto-detected if not specified

## Usage

```sh
# Basic GWAS Meta-Analysis
GT-MetaAssociation -i "gwas1.csv,gwas2.csv,gwas3.csv" -n "Study1,Study2,Study3" -s "5000,6000,4500"

# Basic EWAS Meta-Analysis
GT-MetaAssociation -i "ewas1.csv,ewas2.csv,ewas3.csv" -n "Cohort1,Cohort2,Cohort3" -s "1000,1200,1500"

# Auto-Detection of Sample Sizes
GT-MetaAssociation -i "study1.csv,study2.csv,study3.csv" -n "Study1,Study2,Study3" -t "AFR,EUR,AMR"

# Fixed-Effects Only
GT-MetaAssociation -i "cohort1.csv,cohort2.csv" -n "MESA,WHI" -s "2000,3000" -m fixed

# Multi-Variable Dataset
GT-MetaAssociation -i "multi_var1.csv,multi_var2.csv" -n "Study1,Study2" -a "Smoking" -o smoking_meta.csv
```
