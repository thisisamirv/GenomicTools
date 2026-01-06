# Annotator

## Description
Annotate genomic features (SNPs or methylation probes) with comprehensive genomic information including genes, regulatory regions, and functional annotations.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to the input data file (CSV format) |
| `-o, --output` | Path to save the annotated output file |
| `-c, --chip` | Type of annotation to perform |
| `-p, --protein_coding` | Include only protein-coding genes |
| `-g, --genome_version` | Human genome version |
| `-t, --analysis_type` | Type of analysis (auto-detected by default) |
| `-r, --reference` | Path to custom reference file |
| `-s, --standardize_col_names` | Whether to standardize input column names (default: True) |

## Options

### `-c, --chip`
Specifies the type of annotation to perform. Options:

- 450k: Illumina HumanMethylation450 BeadChip annotation with genes and CpG island information.
- EPIC: Illumina MethylationEPIC BeadChip annotation with comprehensive genomic features.
- MethylSeq: Comprehensive methylation sequencing annotation combining genes, genomic regions, and CpG islands (recommended for EWAS).
- Genotype: SNP/variant annotation with gene mapping and regulatory region annotation (used automatically for GWAS).

### `-p, --protein_coding`
- True: Includes only protein-coding genes (default).
- False: Includes all gene types.

### `-g, --genome_version`
Specifies the human genome version for annotation. Options:

- hg38: Human genome version GRCh38/hg38 (default).
- hg19: Human genome version GRCh37/hg19.

### `-t, --analysis_type`
Specifies the type of input data:

- auto: Automatically detect GWAS or EWAS based on column names (default).
- GWAS: SNP association results (requires: RSID, CHR, BP columns).
- EWAS: Methylation association results (requires: CGID column).

### `-r, --reference`
Path to a custom reference file:
- For arrays: Custom manifest file (CSV format).
- For gene annotation: Custom GTF file or processed annotation file.
- For CpG islands: Custom islands annotation file.

## Usage

```sh
# Basic EWAS annotation (recommended)
GT-Annotator -i ewas_results.csv -o annotated_ewas.csv -c MethylSeq

# GWAS annotation (automatically uses Genotype mode)
GT-Annotator -i gwas_results.csv -o annotated_gwas.csv

# 450k array annotation
GT-Annotator -i probe_data.csv -o annotated_450k.csv -c 450k

# EPIC array annotation
GT-Annotator -i probe_data.csv -o annotated_epic.csv -c EPIC

# Include all gene types
GT-Annotator -i results.csv -o annotated.csv -c MethylSeq -p False
```
