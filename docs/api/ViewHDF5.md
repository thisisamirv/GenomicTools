# ViewHDF5

## Description
Comprehensive HDF5 file analyzer and viewer for genomic datasets, providing detailed structural analysis, data type detection, element counting, and missing value assessment with support for methylation, genetic, and genotype data formats.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to input HDF5 file to analyze (required) |
| `-m, --missing_analysis` | Enable missing values analysis (optional, default: False) |

## Options

### `-m, --missing_analysis`
Enable comprehensive missing values analysis:
- Default: False (disabled)
- When enabled, performs detailed missing value statistics per chromosome
- Increases analysis time but provides complete data quality assessment
- Recommended for data quality control and preprocessing decisions

## Usage

```sh
# Basic analysis (structure and data summary only)
GT-ViewHDF5 -i methylation_data.h5

# Full analysis including missing values assessment
GT-ViewHDF5 -i methylation_data.h5 -m

# Analyze genotype data with missing values
GT-ViewHDF5 -i genotype_data.h5 --missing_analysis

# Analyze genetic association data (basic mode)
GT-ViewHDF5 -i gwas_data.h5

# Compare datasets with full analysis
GT-ViewHDF5 -i original.h5 -m
GT-ViewHDF5 -i processed.h5 -m

# Document dataset characteristics with missing value analysis
GT-ViewHDF5 -i dataset1.h5 -m > dataset1_full_analysis.txt

# Quick structure check without missing value analysis (faster)
GT-ViewHDF5 -i large_dataset.h5
```
