# ProcessHDF5

## Description
Comprehensive HDF5 data processing tool for subsetting, removing, reading, extracting information, and adding metadata to genomic datasets with support for both methylation and genetic data formats.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to input HDF5 file |
| `-o, --output` | Path to output file (HDF5 or CSV depending on operation) |
| `-op, --operation` | Type of operation to perform |
| `-s, --samples` | Sample identifiers to process (file path, comma-separated, or single ID; optional for extract_metadata) |
| `-m, --markers` | Marker identifiers to process (file path, comma-separated, or single ID) |
| `-c, --chromosomes` | Chromosomes to process (comma-separated or single) |
| `-t, --type` | Data type for processing optimization |
| `-cs, --chunk_size` | Number of features to process per chunk |
| `-n, --names` | Type of names to extract (names operation), column names to add (add_metadata operation), or metadata columns to extract (extract_metadata operation) |
| `-md, --metadata` | Path to metadata file for add_metadata operation |

## Options

### `-op, --operation`
Type of processing operation to perform:
- subset: Extract specific samples/markers to new HDF5 file (default)
- remove: Exclude specific samples/markers from HDF5 file
- read: Export data to CSV format
- names: Extract sample or marker lists
- add_metadata: Add external metadata to HDF5 file
- extract_metadata: Extract specified metadata columns from HDF5 file to CSV

### `-t, --type`
Genomic data type for processing optimization:
- Methylation: DNA methylation beta values (EWAS data)
- Genotype: SNP association data (GWAS data)
- None: Auto-detect based on HDF5 structure (default)

### `-s, --samples`
Sample identifiers to process:
- File path: Text file with one sample ID per line
- Comma-separated: "Sample1,Sample2,Sample3"
- Single ID: "Sample123"
- Optional for extract_metadata: If not provided, extracts metadata for all samples

### `-m, --markers`
Marker identifiers to process:
- File path: Text file with one marker ID per line
- Comma-separated: "cg12345,cg67890,rs123456"
- Single ID: "cg12345678"

### `-c, --chromosomes`
Chromosomes to process:
- Comma-separated: "1,2,3,X,Y"
- Single chromosome: "22"
- Accepts "chr1" or "1" format

### `-n, --names`
Dual-purpose parameter depending on operation:

For names operation:
- markers: Extract all marker IDs (probes/SNPs)
- probes: Extract methylation probe IDs
- snps: Extract SNP IDs
- samples: Extract sample IDs

For add_metadata operation:
- Comma-separated column names: "age,sex,phenotype"
- Single column: "age"
- Empty/None: Add all columns from metadata file

For extract_metadata operation:
- Comma-separated column names: "age,sex,phenotype"
- Single column: "age"
- Required: Must specify at least one metadata column to extract

### `-md, --metadata`
Path to external metadata file (add_metadata operation only):
- Supported formats: .csv, .txt, .tsv
- File format automatically detected

### `-cs, --chunk_size`
Number of features to process simultaneously:
- Default: 30,000 features per chunk

## Usage

```sh
# Extract specific samples
GT-ProcessHDF5 -i data.h5 -op subset -s "Sample1,Sample2,Sample3" -o subset_samples.h5

# Extract specific markers
GT-ProcessHDF5 -i data.h5 -op subset -m markers_list.txt -o subset_markers.h5

# Remove specific samples
GT-ProcessHDF5 -i data.h5 -op remove -s "BadSample1,BadSample2" -o cleaned_data.h5

# Export all data to CSV
GT-ProcessHDF5 -i data.h5 -op read -o exported_data.csv

# Add specific metadata columns
GT-ProcessHDF5 -i data.h5 -op add_metadata -md metadata.csv -n "age,sex,phenotype" -o data_with_covariates.h5
```
