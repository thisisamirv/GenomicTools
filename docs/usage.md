# Usage

- A manual page is provided and can be accessed using:
  
```bash
man GenomicTools
```

- Each command has a help function accessible with the "-h" flag:

```bash
GT-PlotAssociationAnalysis -h
```

Output:
```
* Description: Generate Manhattan and Miami plots for genome-wide association study (GWAS) and epigenome-wide association study (EWAS) results with customizable significance thresholds, annotation of top hits, and publication-ready visualizations.

* Details: This script provides comprehensive visualization capabilities for association analysis results, supporting both traditional Manhattan plots and Miami plots that separate results by effect direction. It automatically calculates significance thresholds based on FDR and Holm corrections, annotates top significant hits with gene names, and generates high-quality publication-ready plots with customizable parameters. The tool handles both GWAS and EWAS data formats and includes robust data preprocessing and chromosome positioning algorithms.

* Note:
The script automatically calculates significance thresholds from FDR and Holm corrections and includes intelligent annotation systems with collision avoidance. It supports both GWAS and EWAS data formats with automatic chromosome handling and position calculation.

* Input Format:
- Association Results File (CSV):
  Required columns depend on analysis type:
  GWAS Results:
  SNP,CHR,BP,P,gene_name,chromosome,position,{var}_pvalue,{var}_fdr,{var}_holm,{var}_coef
  rs12345,1,123456,1.2e-8,TP53,chr1,123456,1.2e-8,0.001,0.002,0.45
  rs67890,2,234567,3.4e-6,BRCA1,chr2,234567,3.4e-6,0.012,0.025,0.32
  EWAS Results:
  CGID,chr,position,gene_name,chromosome,{var}_pvalue,{var}_fdr,{var}_holm,{var}_coef
  cg12345,1,123456,TP53,chr1,1.2e-8,0.001,0.002,0.45
  cg67890,2,234567,BRCA1,chr2,3.4e-6,0.012,0.025,0.32
  Required Column Patterns:
  - Position: chromosome, position (or CHR, BP)
  - P-values: {var}_pvalue, {var}_fdr, {var}_holm
  - Effect: {var}_coef
  - Annotation: gene_name (or SNP)

* Output Format:
- Manhattan Plot:
  - X-axis: Chromosome position (concatenated across all chromosomes)
  - Y-axis: -log10(p-value)
  - Colors: Alternating colors by chromosome
  - Significance Lines: Horizontal lines for FDR and Holm thresholds
  - Annotations: Top N hits labeled with gene names

- Miami Plot:
  - Top Panel: Results with positive effect sizes (coefficients > 0)
  - Bottom Panel: Results with negative effect sizes (coefficients < 0)
  - Y-axis: -log10(p-value) (inverted for bottom panel)
  - Colors: Alternating colors by chromosome
  - Annotations: Top N hits from each panel

* Arguments:
  -i, --input  Path to association results file (CSV format)
  -r, --var  Variable name prefix for columns in results file
  -o, --output  Path to save the generated plot
  -t, --threshold  Significance threshold for FDR/Holm corrections
  -n, --n_annot  Number of top hits to annotate on the plot
  -p, --plot_type  Type of plot to generate (manhattan or miami)

* Options:
-r, --var
Variable name prefix for association results columns:
  - Purpose: Identifies the specific association variable in multi-variable results
  - Usage: If results contain multiple variables (e.g., "smoking", "age"), specify which to plot
  - Column Pattern: Script looks for columns with pattern {var}_pvalue, {var}_fdr, {var}_coef, etc.

-t, --threshold
Significance threshold for multiple testing corrections:
  - Default: 0.05 (5% FDR)
  - Range: 0.001-0.1
  - Application: Used for both FDR and Holm correction thresholds
  - Visualization: Draws horizontal lines at calculated p-value thresholds

-n, --n_annot
Number of top significant hits to annotate:
  - Default: 20 top hits
  - Range: 5-100 annotations
  - Selection: Based on lowest p-values
  - Display: Gene names or feature IDs displayed with leader lines

-p, --plot_type
Type of visualization to generate:
  - manhattan: Traditional Manhattan plot showing all results (default)
  - miami: Miami plot separating results by positive/negative effect direction

* Usage:
- Basic Manhattan Plot:
  # Standard Manhattan plot for GWAS results
  GT-PlotAssociationAnalysis -i gwas_results.csv -r genotype -o manhattan_plot.png
  # EWAS Manhattan plot with custom threshold
  GT-PlotAssociationAnalysis -i ewas_results.csv -r methylation -o ewas_plot.png -t 0.01

- Miami Plot Generation:
  # Miami plot showing effect direction
  GT-PlotAssociationAnalysis -i results.csv -r smoking -o miami_plot.png -p miami
  # Miami plot with fewer annotations
  GT-PlotAssociationAnalysis -i results.csv -r genotype -o miami_plot.png -p miami -n 10

- Customized Visualization:
  # High-stringency threshold with many annotations
  GT-PlotAssociationAnalysis -i results.csv -r phenotype -o plot.png -t 0.001 -n 50
  # Relaxed threshold for exploratory analysis
  GT-PlotAssociationAnalysis -i results.csv -r exposure -o plot.png -t 0.1 -n 5

- Multi-variable Results:
  # Plot specific variable from multi-variable analysis
  GT-PlotAssociationAnalysis -i multi_results.csv -r smoking -o smoking_plot.png
  GT-PlotAssociationAnalysis -i multi_results.csv -r age -o age_plot.png
  GT-PlotAssociationAnalysis -i multi_results.csv -r bmi -o bmi_plot.png

- Publication Workflow:
  # Generate publication-ready Manhattan plot
  GT-PlotAssociationAnalysis -i final_results.csv -r genotype -o Figure1_Manhattan.png -t 0.05 -n 25
  # Generate supplementary Miami plot
  GT-PlotAssociationAnalysis -i final_results.csv -r genotype -o SupFig1_Miami.png -p miami -t 0.05 -n 15

- Batch Processing:
  # Generate plots for multiple phenotypes
  for phenotype in height weight bmi; do
  GT-PlotAssociationAnalysis -i ${phenotype}_results.csv -r genotype -o ${phenotype}_manhattan.png
  done
  # Generate both plot types for same data
  GT-PlotAssociationAnalysis -i results.csv -r variable -o manhattan.png -p manhattan
  GT-PlotAssociationAnalysis -i results.csv -r variable -o miami.png -p
```

