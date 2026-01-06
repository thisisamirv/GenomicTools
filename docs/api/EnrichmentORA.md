# EnrichmentORA

## Description
Perform Over-Representation Analysis (ORA) for gene sets using KEGG pathways and Gene Ontology terms with automatic gene ID detection and comprehensive visualization capabilities.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to input file containing gene list and optional statistics |
| `-c, --column` | Column name containing gene identifiers (auto-detected if not specified) |
| `-d, --dataset` | Target pathway/ontology database for enrichment |
| `-o, --output` | Path to save enrichment results |
| `-pv, --pvalue` | P-value cutoff for significance filtering |
| `-u, --include_genes` | Include gene lists in output results |
| `-p, --plot` | Path to save barplot visualization |
| `-n, --top_n` | Number of top results to include in plot |
| `-a, --var` | Variable name for enhanced column detection |

## Options

### `-d, --dataset`
Specifies the pathway or ontology database for enrichment analysis:
- KEGG: KEGG pathway database (default)
- GO_biological: GO Biological Process terms
- GO_molecular: GO Molecular Function terms
- GO_cellular: GO Cellular Component terms

### `-pv, --pvalue`
P-value significance threshold:
- Default: 0.05 (FDR-adjusted p-value ≤ 0.05)
- Range: 0.0-1.0

### `-u, --include_genes`
Include gene lists in enrichment results:
- False: Only statistical summaries (default)
- True: Include comma-separated gene lists

### `-p, --plot`
Generate barplot visualization:
- None: No visualization (default)
- Filename: Save horizontal barplot to specified path

### `-n, --top_n`
Number of top results to include in barplot:
- Default: 20
- Any positive integer

### `-a, --var`
Variable name for enhanced column detection:
- Enables variable-specific p-value and effect size detection
- Useful for multi-variable datasets

## Usage

```bash
# KEGG pathway analysis (auto-detects gene IDs and columns)
GT-EnrichmentORA -i gene_list.csv -o kegg_results.csv

# GO biological process analysis with specific gene column
GT-EnrichmentORA -i genes.csv -c gene_symbol -d GO_biological -o go_bp.csv

# GO molecular function with stricter cutoff
GT-EnrichmentORA -i genes.csv -c gene_id -d GO_molecular -pv 0.01 -o go_mf.csv

# EWAS data with methylation variables
GT-EnrichmentORA -i ewas_results.csv -a Methylation -d GO_biological -o results.csv

# KEGG analysis with barplot
GT-EnrichmentORA -i genes.csv -c gene_symbol -o results.csv -p kegg_plot.png
```
