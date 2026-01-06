# PlotAssociationAnalysis

## Description
Generate Manhattan, Miami, QQ, Volcano, and Regional plots for genome-wide association study (GWAS) and epigenome-wide association study (EWAS) results with customizable significance thresholds, annotation of top hits, and publication-ready visualizations.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to association results file (CSV format) |
| `-a, --var` | Variable name prefix for columns in results file |
| `-o, --output` | Path to save the generated plot |
| `-t, --threshold` | P-value significance threshold |
| `-n, --n_annot` | Number of top hits to annotate on the plot |
| `-p, --plot_type` | Type of plot to generate |
| `-w, --width` | Plot width in inches |
| `-e, --height` | Plot height in inches |
| `-m, --max_points` | Maximum number of points when downsampling |
| `-c, --colors` | Custom color palette for chromosomes |
| `-g, --annotate_genes` | Use gene names instead of IDs for annotations |
| `-N, --sample_sizes` | Sample size info as comma-separated key=value pairs (e.g. N=10000,N_cases=5000,N_controls=5000). Use "None" for unknown. |
| `-s, --skip` | Axis skip/break specification for Miami plots as comma-separated key=value pairs (e.g. from=5,to=10,side=unilateral). Use negative values to indicate negative-effect ranges or use side=bilateral to apply symmetric breaks. |

## Options

### `-a, --var`
Variable name prefix for association results columns:
- Identifies the specific association variable in multi-variable results
- If not specified, auto-detected based on available columns

### `-t, --threshold`
P-value significance threshold:
- Default: None
- Used to draw significance lines and highlight significant points

### `-n, --n_annot`
Number of top significant hits to annotate:
- Default: 20
- Selection based on lowest p-values

### `-p, --plot_type`
Type of visualization to generate:
- manhattan: Manhattan plot (default)
- miami: Miami plot by effect direction
- qq: QQ plot for inflation
- volcano: Volcano plot (effect size vs significance)
- region: Regional association plot

### `-g, --annotate_genes`
Use gene names for annotations instead of marker IDs:
- Default: False
- Uses marker IDs if gene names not available

### `-N, --sample_sizes`
Sample size information passed as comma-separated key=value pairs. Recognised keys:
- N: total sample size
- N_cases: number of cases (for case-control studies)
- N_controls: number of controls (for case-control studies)

### `-s, --skip`
Specify axis breaks (skipping ranges) to compress large -log10(p) regions in Miami plots. Provide a comma-separated key=value string with keys:
- from: lower bound of the skip range (numeric; can be positive or negative)
- to: upper bound of the skip range (numeric; must have same sign as 'from' for unilateral mode)
- side: "unilateral" (default) or "bilateral"

## Usage

```sh
# Auto-detected variable Manhattan plot
GT-PlotAssociationAnalysis -i gwas_results.csv -o manhattan_plot.png

# Specify variable for multi-variable results
GT-PlotAssociationAnalysis -i results.csv -a Genotype -o manhattan_plot.png

# Miami plot showing effect direction
GT-PlotAssociationAnalysis -i results.csv -a Methylation -o miami_plot.png -p miami

# QQ plot to check for genomic inflation
GT-PlotAssociationAnalysis -i results.csv -a Genotype -o qq_plot.png -p qq

# Volcano plot showing effect sizes
GT-PlotAssociationAnalysis -i results.csv -a Methylation -o volcano_plot.png -p volcano
```
