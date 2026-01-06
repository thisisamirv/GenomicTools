# CountsQC

## Description
Calculate quality control metrics for genomic data stored in HDF5 format, including call rates for samples and markers, and variance metrics for methylation probes.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to HDF5 file containing genomic data |
| `-m, --metric` | Type of quality metric to calculate |
| `-o, --output` | Path to save the list of low-quality items |
| `-t, --threshold` | Quality threshold for filtering |
| `-d, --data_type` | Data type (auto-detected if not specified) |

## Options

### `-m, --metric`
Specifies the quality metric to calculate:
- marker_call_rate: Proportion of non-missing values for each marker/probe (default)
- sample_call_rate: Proportion of non-missing values for each sample
- probe_variance: Variance for each methylation probe (methylation data only)

### `-t, --threshold`
Quality threshold for filtering:
- marker_call_rate: Default 0.98 (flag markers with <98% call rate)
- sample_call_rate: Default 0.90 (flag samples with <90% call rate)
- probe_variance: Default 0.05 (flag bottom 5% of probes by variance)
- Custom: 0.0-1.0 for call rates or variance quantile

### `-d, --data_type`
Type of genomic data:
- None: Auto-detect from HDF5 (default)
- genetic: Genetic/genotype data
- methylation: Methylation data

## Usage

```sh
# Calculate marker call rates (default)
GT-CountsQC -i data.h5 -o low_quality_markers.txt

# Calculate sample call rates
GT-CountsQC -i data.h5 -m sample_call_rate -o low_quality_samples.txt

# Calculate probe variance (methylation only)
GT-CountsQC -i methylation.h5 -m probe_variance -o low_variance_probes.txt

# Stricter marker call rate (99%)
GT-CountsQC -i data.h5 -m marker_call_rate -t 0.99 -o strict_markers.txt

# More lenient sample call rate (85%)
GT-CountsQC -i data.h5 -m sample_call_rate -t 0.85 -o lenient_samples.txt
```
