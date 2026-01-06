# TransformData

## Description
Comprehensive data transformation pipeline for genomic and tabular data processing, supporting file format conversion, data filtering, scaling, merging, and train/test splitting with stratification.

## Arguments
| Argument | Description |
|----------|------------|
| `-op, --operation` | Type of data processing operation to perform |
| `-df1, --df1` | Path to first input data file |
| `-df2, --df2` | Path to second input data file (for merge operations) |
| `-o, --output` | Path to save processed output file |
| `-tr, --transform` | Transformation parameters for data processing |
| `-f, --filters` | Filter conditions to apply to data |
| `-x, --extract` | Extraction parameters for rows/columns |
| `-cv, --convert` | Value conversion mappings |
| `-sc, --scale` | Scaling parameters for normalization |
| `-sp, --split` | Parameters for train/test splitting |
| `-mg, --merge` | Parameters for merging datasets |

## Options

### `-op, --operation`
Type of data processing operation to perform:
- transform: Apply transformations, filters, scaling, and format conversions
- split: Create stratified train/test splits for machine learning
- merge: Merge two datasets with intelligent overlap handling

### `-tr, --transform`
General transformation parameters:
- sep: Input file separator ("," for CSV, "\t" for TSV)
- change_sep: Change output separator (True/False)
- header: Include header row in output (True/False)
- transpose: Transpose the dataframe (True/False)

Example: "sep=\t,change_sep=True,header=True,transpose=False"

### `-f, --filters`
Filter conditions to apply to data:
- Format: "column1=value1,column2=value2"
- Multiple values: Support for list-based filtering

Examples:
- "sex=female,age=25"
- "treatment=drug,outcome=success"

### `-x, --extract`
Extraction parameters for specific data subsets:
- row: Extract specific rows by index/name
- col: Extract specific columns by name
- unique: Remove duplicate rows (True/False)

Examples:
- "col=sample_id,phenotype,age"
- "row=Sample1,Sample2,Sample3"

### `-cv, --convert`
Value conversion mappings:
- Format: "old_value1=new_value1,old_value2=new_value2"
- Global replacement: Applied across entire dataframe

Examples:
- "case=1,control=0"
- "male=M,female=F"

### `-sc, --scale`
Scaling parameters for data normalization:
- row: Scale specific rows
- col: Scale specific columns
- zero_one: Apply min-max scaling to [0,1] range (True/False)
- z_scale: Apply z-score standardization (True/False)

Examples:
- "col=age,score,zero_one=True"
- "row=expression_values,z_scale=True"

### `-sp, --split`
Parameters for train/test splitting:
- stratify_var: Variable to use for stratification (required)
- train_fraction: Fraction of data for training (default: 0.7)

Examples:
- "stratify_var=disease_status,train_fraction=0.8"
- "stratify_var=outcome,train_fraction=0.7"

### `-mg, --merge`
Parameters for merging datasets:
- join_on: Column(s) to join on (required)
- overlaps: How to handle overlapping columns ("drop" or "fill_na")
- limit: Limit columns from second dataset

Examples:
- "join_on=sample_id,overlaps=drop"
- "join_on=sample_id,overlaps=fill_na,limit=phenotype"

## Usage

```sh
# Convert CSV to TSV with header
GT-TransformData -op transform -df1 data.csv -o data.tsv -tr "sep=',',change_sep=True,header=True"

# Filter and extract specific columns
GT-TransformData -op transform -df1 metadata.csv -o filtered.csv -f "sex=female,age=25" -x "col=sample_id,phenotype"

# Scale biomarker values
GT-TransformData -op transform -df1 biomarkers.csv -o scaled.csv -sc "col=biomarker1,biomarker2,z_scale=True"

# Stratified split for machine learning
GT-TransformData -op split -df1 ml_data.csv -o split_data.csv -sp "stratify_var=disease_status,train_fraction=0.8"

# Basic merge with overlap handling
GT-TransformData -op merge -df1 demographics.csv -df2 clinical.csv -o merged.csv -mg "join_on=sample_id,overlaps=drop"
```
