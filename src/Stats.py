#!/usr/bin/env python
# Import required modules
import numpy as np
import pandas as pd
import sys
from pandas.errors import EmptyDataError
from scipy import stats
from tabulate import tabulate
from typing import Any, Dict, Optional
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.LoggingUtils import log
from utils.ParsingUtils import ParseToKeyValueList


class Stats:
    def __init__(
        self,
        input_file: str,
        analysis_type: str = "freq",
        filter_conditions: Optional[str] = None,
    ) -> None:
        self.input_file = input_file
        self.analysis_type = analysis_type.lower()
        self.filter_conditions = filter_conditions
        valid_analyses = ["freq", "anova", "crosstab"]
        if self.analysis_type not in valid_analyses:
            raise ValueError(
                f"Invalid analysis type '{analysis_type}'. Must be one of: {valid_analyses}"
            )
        self.data = self._load_and_filter_data()

    def _load_and_filter_data(self) -> pd.DataFrame:
        try:
            log.info(f"Reading input data from: {self.input_file}")
            try:
                data = pd.read_csv(self.input_file)
            except EmptyDataError:
                log.warn(
                    f"Input file '{self.input_file}' is empty. Proceeding with empty dataset."
                )
                return pd.DataFrame()
            if self.filter_conditions:
                filter_list = ParseToKeyValueList(self.filter_conditions)
                if filter_list:
                    log.info(
                        f"Filtering data with conditions: {self.filter_conditions}"
                    )
                    for col, val in filter_list:
                        if data.empty:
                            break
                        if col in data.columns:
                            data = data[data[col] == val]
                            log.info(
                                f"Applied filter: {col}={val}, {len(data)} rows remaining"
                            )
                        else:
                            raise ValueError(f"Column {col} not found in the data")
            return data
        except Exception as e:
            log.error(f"Error loading data: {e}")
            raise

    def frequency_analysis(self, variable: str) -> Optional[pd.DataFrame]:
        try:
            log.info(f"Starting frequency analysis for variable: {variable}")
            if variable not in self.data.columns:
                raise ValueError(f"Variable '{variable}' not found in the data")
            freqs = self.data[variable].value_counts()
            percentages = self.data[variable].value_counts(normalize=True) * 100
            result_df = pd.DataFrame(
                {
                    "Value": freqs.index,
                    "Frequency": freqs.values,
                    "Percentage": percentages.values,
                }
            )
            print(f"\nFrequency Table for {variable}:\n")
            print(
                tabulate(
                    result_df,
                    headers="keys",
                    tablefmt="pipe",
                    floatfmt=".2f",
                    showindex=False,
                )
            )
            log.info("Frequency analysis completed successfully.")
            return result_df
        except Exception as e:
            log.error(f"Error in frequency analysis: {e}")
            return None

    def anova_analysis(
        self, group_variable: str, response_variable: str
    ) -> Optional[Dict[str, Any]]:
        try:
            log.info(
                f"Starting ANOVA analysis: {response_variable} by {group_variable}"
            )
            for var in [group_variable, response_variable]:
                if var not in self.data.columns:
                    raise ValueError(f"Variable '{var}' not found in the data")
            clean_data = self.data[[group_variable, response_variable]].dropna()
            if len(clean_data) == 0:
                raise ValueError("No valid data after removing missing values")
            log.info(f"Creating summary statistics for group: {group_variable}")
            summary_stats = (
                clean_data.groupby(group_variable)[response_variable]
                .agg(["count", "mean", "std"])
                .round(3)
                .reset_index()
            )
            summary_stats.columns = [group_variable, "N", "Mean", "Std Dev"]
            print(
                f"\nSummary Statistics for {response_variable} by {group_variable}:\n"
            )
            print(
                tabulate(
                    summary_stats, headers="keys", tablefmt="pipe", showindex=False
                )
            )
            log.info(f"Performing ANOVA for {response_variable} by {group_variable}")
            groups = clean_data[group_variable].unique()
            group_data = [
                clean_data[clean_data[group_variable] == g][response_variable]
                for g in groups
            ]
            if len(groups) < 2:
                log.warn("Need at least 2 groups for ANOVA")
                return None
            f_stat, p_value = stats.f_oneway(*group_data)
            grand_mean = np.mean([val for group in group_data for val in group])
            n_total = sum(len(g) for g in group_data)
            n_groups = len(groups)
            df_between = n_groups - 1
            df_within = n_total - n_groups
            ss_between = sum(
                len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data
            )
            ss_total = sum(
                (val - grand_mean) ** 2 for group in group_data for val in group
            )
            ss_within = ss_total - ss_between
            ms_between = ss_between / df_between if df_between > 0 else 0
            ms_within = ss_within / df_within if df_within > 0 else 0
            anova_table = pd.DataFrame(
                {
                    "Source": [group_variable, "Residuals", "Total"],
                    "Df": [df_between, df_within, df_between + df_within],
                    "Sum Sq": [ss_between, ss_within, ss_total],
                    "Mean Sq": [ms_between, ms_within, np.nan],
                    "F value": [f_stat, np.nan, np.nan],
                    "Pr(>F)": [p_value, np.nan, np.nan],
                }
            )
            print("\nANOVA Table:\n")
            print(
                tabulate(
                    anova_table,
                    headers="keys",
                    tablefmt="pipe",
                    floatfmt=".4f",
                    showindex=False,
                )
            )
            alpha = 0.05
            if p_value < alpha:
                print(
                    f"\nResult: Significant difference between groups (p = {p_value:.4f})"
                )
            else:
                print(
                    f"\nResult: No significant difference between groups (p = {p_value:.4f})"
                )
            log.info("ANOVA analysis completed successfully.")
            return {
                "summary_stats": summary_stats,
                "anova_table": anova_table,
                "f_statistic": f_stat,
                "p_value": p_value,
            }
        except Exception as e:
            log.error(f"Error in ANOVA analysis: {e}")
            return None

    def crosstab_analysis(
        self, variable1: str, variable2: str
    ) -> Optional[Dict[str, Any]]:
        try:
            log.info(f"Starting cross-tabulation analysis: {variable1} vs {variable2}")
            for var in [variable1, variable2]:
                if var not in self.data.columns:
                    raise ValueError(f"Variable '{var}' not found in the data")
            clean_data = self.data[[variable1, variable2]].dropna()
            if len(clean_data) == 0:
                raise ValueError("No valid data after removing missing values")
            clean_data[variable1] = clean_data[variable1].astype("category")
            clean_data[variable2] = clean_data[variable2].astype("category")
            log.info(
                f"Creating cross tabulation for variables: {variable1}, {variable2}"
            )
            contingency_table = pd.crosstab(
                clean_data[variable1], clean_data[variable2]
            )
            row_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
            combined_table = pd.DataFrame(
                index=contingency_table.index, columns=contingency_table.columns
            )
            for idx in combined_table.index:
                for col in combined_table.columns:
                    count = contingency_table.loc[idx, col]
                    pct = row_pct.loc[idx, col]
                    combined_table.loc[idx, col] = f"{count} ({pct:.1f}%)"
            print(f"\nCross-tabulation of {variable1} by {variable2}:\n")
            print(
                tabulate(
                    combined_table, headers="keys", tablefmt="pipe", showindex=True
                )
            )
            log.info("Performing chi-squared test")
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(
                contingency_table, correction=False
            )
            min_expected = np.min(expected)
            if min_expected < 5:
                log.info(
                    "Expected cell count < 5, results should be interpreted with caution"
                )
                if contingency_table.shape == (2, 2):
                    log.info("Using Fisher's exact test for 2x2 table")
                    odds_ratio, fisher_p = stats.fisher_exact(contingency_table)
                    print(f"\nFisher's Exact Test p-value: {fisher_p:.4f}")
            print("\nChi-squared Test Results:\n")
            chi2_results = pd.DataFrame(
                {
                    "Test": ["Pearson's Chi-squared"],
                    "Chi-squared": [f"{chi2_stat:.4f}"],
                    "df": [dof],
                    "p-value": [f"{p_value:.4f}"],
                }
            )
            print(
                tabulate(
                    chi2_results,
                    headers="keys",
                    tablefmt="pipe",
                    showindex=False,
                )
            )
            print(f"\nMinimum expected frequency: {min_expected:.2f}")
            alpha = 0.05
            if p_value < alpha:
                print(
                    f"\nResult: Significant association between variables (p = {p_value:.4f})"
                )
            else:
                print(
                    f"\nResult: No significant association between variables (p = {p_value:.4f})"
                )
            log.info("Cross-tabulation analysis completed successfully.")
            return {
                "contingency_table": contingency_table,
                "row_percentages": row_pct,
                "combined_table": combined_table,
                "chi2_statistic": chi2_stat,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "expected_frequencies": expected,
                "min_expected": min_expected,
            }
        except Exception as e:
            log.error(f"Error in cross-tabulation analysis: {e}")
            return None

    def run_analysis(self, **kwargs: Any) -> None:
        try:
            result: Optional[Any] = None
            if self.analysis_type == "freq":
                if "variable" not in kwargs:
                    raise ValueError(
                        "Variable parameter required for frequency analysis"
                    )
                result = self.frequency_analysis(kwargs["variable"])
            elif self.analysis_type == "anova":
                if "group_variable" not in kwargs or "response_variable" not in kwargs:
                    raise ValueError(
                        "Group_variable and response_variable parameters required for ANOVA"
                    )
                result = self.anova_analysis(
                    kwargs["group_variable"], kwargs["response_variable"]
                )
            elif self.analysis_type == "crosstab":
                if "variable1" not in kwargs or "variable2" not in kwargs:
                    raise ValueError(
                        "Variable1 and variable2 parameters required for cross-tabulation"
                    )
                result = self.crosstab_analysis(
                    kwargs["variable1"], kwargs["variable2"]
                )
            else:
                raise ValueError(f"Unknown analysis type: {self.analysis_type}")

            if result is None:
                raise ValueError("Analysis did not produce any results")

            sys.exit(0)
        except Exception as e:
            log.error(f"Error running {self.analysis_type} analysis: {e}")
            sys.exit(1)

    def run(self, opt: Optional[Any] = None) -> None:
        try:
            if self.analysis_type == "freq":
                variable = getattr(opt, "variable", None) if opt is not None else None
                if not variable:
                    log.error("For frequency analysis, --variable must be provided")
                    sys.exit(1)
                self.run_analysis(variable=variable)
            elif self.analysis_type == "anova":
                group = getattr(opt, "group", None) if opt is not None else None
                response = getattr(opt, "response", None) if opt is not None else None
                if not group or not response:
                    log.error(
                        "For ANOVA analysis, --group and --response must be provided"
                    )
                    sys.exit(1)
                self.run_analysis(group_variable=group, response_variable=response)
            elif self.analysis_type == "crosstab":
                v1 = getattr(opt, "variable1", None) if opt is not None else None
                v2 = getattr(opt, "variable2", None) if opt is not None else None
                if not v1 or not v2:
                    log.error(
                        "For cross-tabulation analysis, --variable1 and --variable2 must be provided"
                    )
                    sys.exit(1)
                self.run_analysis(variable1=v1, variable2=v2)
            else:
                log.error(f"Invalid analysis type: {self.analysis_type}")
                sys.exit(1)
        except Exception as e:
            log.error(f"Error in Stats.run: {e}")
            sys.exit(1)


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(
        flags=["-a", "--analysis"],
        type=str,
        required=True,
        choices=["freq", "anova", "crosstab"],
    ),
    OptionConfig(flags=["-v", "--variable"], type=str, default=None, required=False),
    OptionConfig(flags=["-v1", "--variable1"], type=str, default=None, required=False),
    OptionConfig(flags=["-v2", "--variable2"], type=str, default=None, required=False),
    OptionConfig(flags=["-g", "--group"], type=str, default=None, required=False),
    OptionConfig(flags=["-r", "--response"], type=str, default=None, required=False),
    OptionConfig(flags=["-f", "--filter"], type=str, default=None, required=False),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="Stats")
    opt = framework.run()
    analyzer = Stats(opt.input, opt.analysis, opt.filter)
    analyzer.run(opt)
