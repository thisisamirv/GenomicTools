# ConvertCounts

## Description
Convert genomic data between CSV, HDF5, and PLINK formats with automatic format detection and optimized processing for large-scale datasets.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to the input data file or PLINK prefix |
| `-o, --output` | Path to the output file or PLINK prefix |
| `-c, --chip` | Chip type for methylation data annotation |
| `-g, --hg` | Human genome version for coordinate mapping |
| `-t, --transpose` | Transpose data matrix during conversion |

## Options

### `-c, --chip`
Chip type for methylation data. Required if methylation data lacks chromosome info:
- 450k: Illumina HumanMethylation450 BeadChip
- EPIC: Illumina MethylationEPIC BeadChip
- None: For genetic data or data with chromosome info (default)

### `-g, --hg`
Human genome version for coordinate mapping:
- hg38: GRCh38/hg38 (default)
- hg19: GRCh37/hg19

### `-t, --transpose`
Transpose the data matrix during conversion:
- False: Keep original orientation (default)
- True: Transpose samples and features

## Usage

```sh
# CSV to HDF5 (methylation data)
GT-ConvertCounts -i methylation_data.csv -o methylation_data.h5 -c 450k

# HDF5 to CSV
GT-ConvertCounts -i data.h5 -o data.csv

# PLINK to HDF5
GT-ConvertCounts -i genotypes -o genotypes.h5

# CSV to HDF5 with EPIC array annotation
GT-ConvertCounts -i epic_data.csv -o epic_data.h5 -c EPIC -g hg38

# HDF5 to CSV with transposition
GT-ConvertCounts -i data.h5 -o transposed_data.csv -t True
```
