<img src="config/Logo.jpeg" alt="GenomicTools" width="200" height="200">

# GenomicTools

GenomicTools is a comprehensive toolkit for genomic data processing and analysis at scale, with speed, and at the convenience of the command line.
The project is still ongoing and in its initial stages.

**Note:** Most of the scripts working with genomic data assume your data is stored as an HDF5 file. The function `GT-saveToHDF5` is part of GenomicTools and can be used early in your analysis to convert your data into an HDF5 file.

### Prerequisites

- Python ≥ 3.8 (Python 3.13 is preferred)
- Other dependencies are listed in `requirements.txt`
- Mamba is expected for installation. Note: mamba is most often installed into a conda / Miniconda installation (commonly in the base environment).

## Mamba

If you're on an HPC system with access to internet
- Try loading a miniconda/anaconda module first:
```bash
module load miniconda   # or module load anaconda
```
- Then ensure mamba is available or install it into your base environment:
```bash
conda install -n base -c conda-forge mamba
```

If you do not have conda/miniconda
- Install Miniconda (recommended) and then install mamba as above, or install micromamba:
  - Miniconda installers: https://docs.conda.io/en/latest/miniconda.html
  - Micromamba: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html

If you're on an HPC system without access to internet, you can skip this step.

## Installation
Once mamba is available:

1. Clone the repository:
    ```bash
    git clone https://git.yale.edu/av746/GenomicTools.git
    cd GenomicTools
    ```
2. Run the installation script:
    ```bash
    make install
    ```
3. Reload your shell configuration:
    ```bash
    if [ -n "$ZSH_VERSION" ]; then
        source ~/.zshrc
    else
        source ~/.bashrc
    fi
    ```

## Usage

See the manual at: https://git.yale.edu/pages/av746/GenomicTools/

- A local manual page is also provided and can be accessed using:
  ```bash
  man GenomicTools
  ```

- Each command has a help function accessible with the "-h" flag:
  ```bash
  GT-AssociationAnalysis -h
  ```
  Output:
  ```
  # AssociationAnalysis

  ## Description
  Performs genome-wide (GWAS) or epigenome-wide (EWAS) association analyses.
  
  ## Arguments
  | Argument | Description |
  |-----------|-------------|
  | `-i, --input` | Path to the HDF5 file containing genotype or methylation data |
  | `-m, --metadata` | CSV with sample-level metadata/covariates. If omitted the launcher will attempt to use HDF5 metadata |
  | `-s, --sample_id` | Column name in the metadata CSV matching HDF5 sample IDs |
  | `-o, --output` | Path where the script should write the CSV results |
  | `-f, --formula` | Formula (e.g. `Methylation ~ Trait + Age + Sex`) |
  | `-t, --analysis_type` | `Auto` (default), `EWAS`, or `GWAS` |
  | `-d, --model` | Regression model: `linear` (default) or `logistic` |
  | `-p, --threads` | Number of CPU threads to use |
  | `-c, --chunk_size` | Number of features (probes/variants) per processing chunk |
  | `-a, --var` | The variable to extract the statistics for |
  
  
  ## Options
  
  ### `-t, --analysis_type`
  - `Auto`: infer from parsed formula's data variable (default).
  - `EWAS`: force methylation analysis.
  - `GWAS`: force genotype analysis.
  The script validates that the formula's data variable matches the chosen mode and that the dependent variable is appropriate (for EWAS the dependent must reference the methylation data variable; for GWAS the dependent must be a phenotype, not the genotype variable).
  
  ### `-d, --model`
  - `linear` (default): continuous trait regression.
  - `logistic`: binary/case-control regression (mapped to the downstream R test type).
  
  ### `-p, --threads`
  If unspecified the launcher uses SystemUtils.get_optimal_cores(reserve_cores=1) or the configured reserve in system config. Memory-per-core is computed from available RAM and the selected thread count.
  
  ### `-c, --chunk_size`
  If not provided, the launcher:
  - Attempts to read the number of samples from HDF5.
  - Uses conservative per-entry byte estimates (EWAS vs GWAS) and available memory to derive a chunk size bounded by sensible min/max values.
  
  ### `-s, --sample_id`
  If metadata CSV is provided but the sample ID column is not specified, the launcher will:
  - Read the first few sample IDs from the HDF5 metadata group (SampleList/IID).
  - Search metadata columns for one containing those IDs and auto-select it if found.
  
  ### `-a, --var`
  Maps to the `var` CLI option in the script. Requested stat variables are validated against covariates parsed from the formula; data-stream variables (Methylation/Genotype) are rejected for stat extraction. If requested variables are not present, the launcher falls back to the first covariate (if available) and emits warnings.
  
  ## Usage
  
  # Automatic analysis type detection (EWAS or GWAS inferred from formula)
  GT-AssociationAnalysis -i data.h5 -m metadata.csv -o results.csv -f "Methylation ~ Trait + Age + Sex"
  
  # Force EWAS mode with explicit analysis type
  GT-AssociationAnalysis -i methylation.h5 -m metadata.csv -o ewas_results.csv -f "Methylation ~ Smoking + Age + Sex" -a EWAS
  
  # Force GWAS mode with explicit analysis type
  GT-AssociationAnalysis -i genotypes.h5 -m metadata.csv -o gwas_results.csv -f "BMI ~ Genotype + Age + Sex + PC1 + PC2" -a GWAS
  
  # Case-control logistic GWAS
  GT-AssociationAnalysis -i genotypes.h5 -m metadata.csv -o case_control.csv -f "Disease ~ Genotype + Age + Sex + PC1 + PC2" -a GWAS -d logistic
  
  # EWAS with automatic chunk size
  GT-AssociationAnalysis -i methylation.h5 -m metadata.csv -o ewas_results.csv -f "Methylation ~ Smoking + Age + Sex"
  
  # Parallelized GWAS using 8 threads
  GT-AssociationAnalysis -i genotypes.h5 -m metadata.csv -o parallel_gwas.csv -f "Height ~ Genotype + Age + Sex" -a GWAS -p 8
  
  # Extract statistics for 'Age' (rather than the default dependent variable)
  GT-AssociationAnalysis -i methylation.h5 -m metadata.csv -o ewas_results_age.csv -f "Methylation ~ Age + Sex" -a Age -a EWAS
  ```

## Uninstallation

To uninstall GenomicTools, simply run:
```bash
make uninstall
```

## Citation

**If you use this software, please cite it as follows:**  

Valizadeh, A. (2024). GenomicTools [Computer software]. Yale University. Available from https://git.yale.edu/av746/GenomicTools.git

A citation file in BibTeX format is also available for your convenience. You can find it in the repository as `citation.bib`.

## License

See the `LICENSE` file for more information.

## Acknowledgments

Special thanks to Yale University Department of Psychiatry and to Dr. Ke Xu for the resources, funding, and supportive environment that made development of GenomicTools possible.

## Bug Report

- Use the flags `--verbose INFO` (short log) or `--verbose DEBUG` (extensive log) to produce helpful logs for each function.
- Direct logs to a file by using the `--log <file_name>` flag.
