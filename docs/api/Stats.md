# Stats

## Description
Comprehensive statistical analysis tool for frequency tables, ANOVA, and cross-tabulation analysis with filtering capabilities and publication-ready formatted output tables.

## Arguments
| Argument | Description |
|----------|------------|
| `-i, --input` | Path to input CSV file containing data |
| `-a, --analysis` | Type of statistical analysis to perform |
| `-v, --variable` | Variable name for frequency analysis |
| `-v1, --variable1` | First variable for cross-tabulation |
| `-v2, --variable2` | Second variable for cross-tabulation |
| `-g, --group` | Group variable for ANOVA analysis |
| `-r, --response` | Response variable for ANOVA analysis |
| `-f, --filter` | Filter conditions to apply to data |

## Options

### `-a, --analysis`
Statistical analysis method to perform:
- freq: Frequency table analysis for categorical variables (default)
- anova: One-way ANOVA for continuous variables by groups
- crosstab: Cross-tabulation with chi-squared test for categorical associations

### `-v, --variable`
Column name for frequency analysis (required for freq analysis):
- Calculate frequency counts and percentages for categorical data

### `-v1, --variable1` and `-v2, --variable2`
Variables for cross-tabulation analysis (required for crosstab):
- v1: First categorical variable
- v2: Second categorical variable

### `-g, --group` and `-r, --response`
Variables for ANOVA analysis (required for anova):
- g: Grouping variable (categorical)
- r: Response variable (continuous)

### `-f, --filter`
Apply filter conditions to data before analysis:
- Format: "column1=value1,column2=value2"
- Multiple conditions: Comma-separated key=value pairs

## Usage

```sh
# Basic frequency table
GT-Stats -i data.csv -a freq -v treatment

# Frequency table with filtering
GT-Stats -i data.csv -a freq -v outcome -f "age=25,sex=female"

# Compare continuous variable by group
GT-Stats -i data.csv -a anova -g treatment -r score

# Test association between categorical variables
GT-Stats -i data.csv -a crosstab -v1 treatment -v2 outcome

# Cross-tabulation with filtering
GT-Stats -i data.csv -a crosstab -v1 genotype -v2 phenotype -f "population=EUR"
```
