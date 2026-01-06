# Changelog

All notable changes to this project will be documented in this file.

## [2.2.0] - 2025-10-21
### Added
- CHANGELOG.md to track releases and notable changes.
- Parallel logging tailer helper for EWAS/GWAS.
- Log tailing in EWAS and GWAS to forward worker logs to parent process.
- Worker stdout/stderr forwarding via makeCluster(..., outfile="") to allow parent capture.

### Fixed
- Ensure create_console_appender available and writes to stderr so parent Python captures worker logs.
- Suppress cluster "starting worker pid=..." lines in AssociationAnalysis output.
- Consolidate BACON correction to run once per trait across the whole results, instead of per chunk.
- Removed some noisy BACON/logging messages.

## [2.2.1] - 2025-10-21
### Fixed
- Ensure that multiple test correction is applied on BACON-corrected p values in EWAS, instead of original p values.

## [2.2.2] - 2025-10-21
### Added
- New R helper script PowerLimma.R that runs limma::lmFit + eBayes on a chunked matrix and writes per-feature p-values to CSV for use by PowerEWAS.py.

### Fixed
- Removed a few noisy log messages in EWAS.R and AssociationAnalysis.py.
- Removed in-Python limma/eBayes implementation in PowerEWAS.py; limma analysis is fully outsourced to the R helper.
- In ExperimentHub.py, when a requested ExperimentHub resource is missing from the cache, the code now attempts to download it automatically and stores it in the local cache.
- Removed MetaAssociation.py dependency on the no longer available MultipleTestCorrection.py.
- Added some comments to the scripts and sorted the imported libraries.

## [2.2.3] - 2025-10-21
### Added
- Added SHet multiple test correction to MultiTraitAssociation.py.
- Added BACON-corrected t value extraction to EWAS.R.

## [2.2.4] - 2025-10-22
### Added
- Added dynamic marker column naming to CPASSOC.R.
- Added the ability to cache the downloaded array manifest to Annotator.py.
- Added to Annotator.py the ability to find 450K/EPIC data CHR from the manifest if it was not available in the input data.
- Added the option to reserve the column names of the input file in the output of Annotator.py.
- Added to CLIFramework.py the ability to pass False/True as boolean instead of string.

### Fixed
- Removed the following non-relevant columns from 450K/EPIC annotation in Annotator.py: TSS,TSS_DIST,NEAREST_GENE_DIST,START,END.
- Annotator.py now only returns unique gene names for each probe in 450K/EPIC chips.
- Now, Annotator.py returns Ensembl gene IDs for genes in 450K/EPIC chips.

## [2.2.5] - 2025-10-22
### Added
- Added some comments to the scripts.
- Add a new "cols_to_add" argument to MultiTraitAssociation.py and CPASSOC.R, which gets the list of columns to add to the output file from the input files.
- Added LOG2FC column to the output of EWAS.R.

### Fixed
- Made marker_col and t_col broadcasting more robust in CPASSOC.R.
- Made "traits" arguement optional for MultiTraitAssociation.py and CPASSOC.R.
- Removed the "p_col" argument from MultiTraitAssociation.py and CPASSOC.R.

## [2.2.6] - 2025-10-26
### Fixed
- Fixed a critical bug in EWAS.R that was resulting in a dimension mismatch.
- Fixed AssociationAnalysis.py to not capitalize "eLOG2FC" column names.

## [2.2.7] - 2025-10-26
### Added
- Added scaled λ, empirical bias, and empirical inflation calculation to EWAS.R.

### Fixed
- Removed some noisy logs from EWAS.R.
- Removed BACON correction from EWAS.R, considering that eBayes() in limma already does p/t value moderation.

## [2.2.8] - 2025-10-30
### Added
- Added per-term case/control count computation (n_cases, n_controls) and n_effective in process_chrom_group in EWAS.R and used them when available to scale lambda.
- Updated PlotAssociationAnalysis.py to accept sample_sizes argument.
- Implemented N_effective computation in PlotAssociationAnalysis.py.
- Modified QQ, calibration, and lambda plotting in PlotAssociationAnalysis.py to use the scaled lambda when N_effective > 1000 and otherwise use the raw lambda.

### Fixed
- Fixed per-term p-value adjustment and lambda assignment by using data.table group operations (by Term) to prevent row-length mismatches when multiple stat_vars are present.
- Ensure variable-specific column matching checks both prefix and suffix forms and requires the variable name to appear in matched columns in AliasUtils.py.
- Refactored column detection in PlotAssociationAnalysis to use only determine_variable_columns; removed legacy methods _identify_p_value_column, _identify_effect_column, _get_p_column, and _get_valid_pvalues.
- Improved annotation placement in PlotAssociationAnalysis.lambda_distribution_plot so that annotation bounding boxes are checked in display (pixel) coordinates and shifted iteratively until they no longer overlap other annotation boxes or the axes.
- Corrected scaled genomic inflation calculation in EWAS.R to use effective sample size for binary variables and total sample size for non-binary variables.
- Fixed the QQ plot code in PlotAssociationAnalysis.py to compute lambda from the same null p-value subset used for plotting and correctly exclude the most extreme (smallest) 1%.

## [2.2.9] - 2025-10-30
### Added
- Updated PlotAssociationAnalysis.md to reflect the latest changes.
- Added progress bars to PlotAssociationAnalysis.py.

### Fixed
- Updated the default threshold value for PlotAssociationAnalysis.py to be None.
- In PlotAssociationAnalysis.py removed the empty space between the y-axis and x=1 in Miami plots.
- Completely removed downsampling from PlotAssociationAnalysis.py due to its unreliability.
- Modified the legend position to not overlap with the annotation boxes in PlotAssociationAnalysis.py for Volcano plots.
- Replaced an unbounded/stale overlap-check loop with a bounded retry loop in PlotAssociationAnalysis.py.
- Fixed column standardization in PlotAssociationAnalysis.py retain variable-prefixed P/COEF (e.g. SHom_P) to allow variable-specific column detection.
- Updated the README file.
- Fixed MAKEFILE to also work with install.txt file (For cicumstances where direct transfer of bash files is not feasible).
- Modified MAKEFILE so the 'make clean' command also removes the .pytest_cache dir in the root dir.
