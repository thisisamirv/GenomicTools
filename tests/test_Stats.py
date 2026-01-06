#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from unittest.mock import patch

from Stats import Stats


@pytest.fixture
def sample_data():
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"] * 10,
            "score": np.random.normal(50, 10, 100),
            "category": np.random.choice(["X", "Y", "Z"], 100),
            "treatment": np.random.choice(["Control", "Treatment"], 100),
            "age_group": np.random.choice(["Young", "Old"], 100),
            "response": np.random.choice([0, 1], 100),
            "continuous_var": np.random.normal(25, 5, 100),
        }
    )

    data.loc[data["group"] == "A", "score"] += 5
    data.loc[data["group"] == "C", "score"] -= 3

    return data


@pytest.fixture
def temp_csv_file(sample_data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def small_dataset():
    return pd.DataFrame(
        {
            "treatment": ["A", "A", "B", "B", "B", "C"],
            "outcome": [10, 12, 15, 16, 18, 20],
            "status": ["Pass", "Pass", "Fail", "Pass", "Pass", "Fail"],
            "gender": ["M", "F", "M", "F", "M", "F"],
        }
    )


@pytest.fixture
def small_csv_file(small_dataset):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        small_dataset.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.mark.unit
def test_stats_initialization(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "freq")
    assert stats_analyzer.input_file == temp_csv_file
    assert stats_analyzer.analysis_type == "freq"
    assert stats_analyzer.filter_conditions is None
    assert stats_analyzer.data is not None
    assert len(stats_analyzer.data) == 100

    stats_analyzer_filtered = Stats(temp_csv_file, "anova", "group=A")
    assert stats_analyzer_filtered.filter_conditions == "group=A"

    with pytest.raises(ValueError, match="Invalid analysis type"):
        Stats(temp_csv_file, "invalid_analysis")


@pytest.mark.unit
def test_load_and_filter_data(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "freq")
    assert len(stats_analyzer.data) == 100

    with patch("Stats.ParseToKeyValueList") as mock_parse:
        mock_parse.return_value = [("group", "A")]

        stats_analyzer = Stats(temp_csv_file, "freq", "group=A")
        assert len(stats_analyzer.data) < 100


@pytest.mark.unit
def test_frequency_analysis(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "freq")

    result = stats_analyzer.frequency_analysis("group")

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "Value" in result.columns
    assert "Frequency" in result.columns
    assert "Percentage" in result.columns
    assert len(result) == 3

    assert abs(result["Percentage"].sum() - 100.0) < 0.01

    result_invalid = stats_analyzer.frequency_analysis("nonexistent")
    assert result_invalid is None


@pytest.mark.unit
def test_anova_analysis(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "anova")

    result = stats_analyzer.anova_analysis("group", "score")

    assert result is not None
    assert isinstance(result, dict)
    assert "summary_stats" in result
    assert "anova_table" in result
    assert "f_statistic" in result
    assert "p_value" in result

    summary_stats = result["summary_stats"]
    assert len(summary_stats) == 3
    assert "N" in summary_stats.columns
    assert "Mean" in summary_stats.columns
    assert "Std Dev" in summary_stats.columns

    anova_table = result["anova_table"]
    assert len(anova_table) == 3
    assert "F value" in anova_table.columns
    assert "Pr(>F)" in anova_table.columns

    result_invalid = stats_analyzer.anova_analysis("nonexistent", "score")
    assert result_invalid is None

    result_invalid2 = stats_analyzer.anova_analysis("group", "nonexistent")
    assert result_invalid2 is None


@pytest.mark.unit
def test_anova_insufficient_groups(small_csv_file):
    single_group_data = pd.DataFrame(
        {"group": ["A"] * 5, "score": [10, 12, 14, 16, 18]}
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        single_group_data.to_csv(f.name, index=False)
        temp_file = f.name

    try:
        stats_analyzer = Stats(temp_file, "anova")
        result = stats_analyzer.anova_analysis("group", "score")
        assert result is None
    finally:
        os.unlink(temp_file)


@pytest.mark.unit
def test_crosstab_analysis(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "crosstab")

    result = stats_analyzer.crosstab_analysis("treatment", "age_group")

    assert result is not None
    assert isinstance(result, dict)
    assert "contingency_table" in result
    assert "row_percentages" in result
    assert "combined_table" in result
    assert "chi2_statistic" in result
    assert "p_value" in result
    assert "degrees_of_freedom" in result
    assert "expected_frequencies" in result
    assert "min_expected" in result

    contingency_table = result["contingency_table"]
    assert isinstance(contingency_table, pd.DataFrame)
    assert contingency_table.sum().sum() == 100

    result_invalid = stats_analyzer.crosstab_analysis("nonexistent", "age_group")
    assert result_invalid is None


@pytest.mark.unit
def test_crosstab_2x2_table(small_csv_file):
    stats_analyzer = Stats(small_csv_file, "crosstab")

    with patch("Stats.stats.fisher_exact") as mock_fisher:
        mock_fisher.return_value = (2.0, 0.05)

        result = stats_analyzer.crosstab_analysis("status", "gender")

        assert result is not None
        assert "chi2_statistic" in result


@pytest.mark.unit
def test_run_analysis_frequency(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "freq")

    with pytest.raises(SystemExit) as exc:
        stats_analyzer.run_analysis(variable="group")
    assert exc.value.code in (0, None)

    with pytest.raises(SystemExit) as exc_invalid:
        stats_analyzer.run_analysis()
    assert exc_invalid.value.code not in (0, None)


@pytest.mark.unit
def test_run_analysis_anova(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "anova")

    with pytest.raises(SystemExit) as exc:
        stats_analyzer.run_analysis(group_variable="group", response_variable="score")
    assert exc.value.code in (0, None)

    with pytest.raises(SystemExit) as exc_invalid:
        stats_analyzer.run_analysis(group_variable="group")
    assert exc_invalid.value.code not in (0, None)


@pytest.mark.unit
def test_run_analysis_crosstab(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "crosstab")

    with pytest.raises(SystemExit) as exc:
        stats_analyzer.run_analysis(variable1="treatment", variable2="age_group")
    assert exc.value.code in (0, None)

    with pytest.raises(SystemExit) as exc_invalid:
        stats_analyzer.run_analysis(variable1="treatment")
    assert exc_invalid.value.code not in (0, None)


@pytest.mark.unit
def test_run_analysis_invalid_type(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "freq")
    stats_analyzer.analysis_type = "invalid"

    with pytest.raises(SystemExit) as exc:
        stats_analyzer.run_analysis(variable="group")
    assert exc.value.code not in (0, None)


@pytest.mark.unit
def test_freq_convenience_function(temp_csv_file):
    stats_obj = Stats(temp_csv_file, "freq")
    result = stats_obj.frequency_analysis("group")

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


@pytest.mark.unit
def test_anovaTab_convenience_function(temp_csv_file):
    stats_obj = Stats(temp_csv_file, "anova")
    result = stats_obj.anova_analysis("group", "score")

    assert result is not None
    assert isinstance(result, dict)
    assert "f_statistic" in result
    assert "p_value" in result


@pytest.mark.unit
def test_crossTab_convenience_function(temp_csv_file):
    stats_obj = Stats(temp_csv_file, "crosstab")
    result = stats_obj.crosstab_analysis("treatment", "age_group")

    assert result is not None
    assert isinstance(result, dict)
    assert "chi2_statistic" in result
    assert "p_value" in result


@pytest.mark.integration
def test_complete_frequency_workflow(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "freq")

    for var in ["group", "category", "treatment", "age_group"]:
        result = stats_analyzer.frequency_analysis(var)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert all(
            col in result.columns for col in ["Value", "Frequency", "Percentage"]
        )


@pytest.mark.integration
def test_complete_anova_workflow(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "anova")

    combinations = [
        ("group", "score"),
        ("treatment", "continuous_var"),
        ("age_group", "score"),
    ]

    for group_var, response_var in combinations:
        result = stats_analyzer.anova_analysis(group_var, response_var)
        assert result is not None
        assert isinstance(result, dict)

        assert isinstance(result["f_statistic"], (int, float))
        assert isinstance(result["p_value"], (int, float))
        assert 0 <= result["p_value"] <= 1


@pytest.mark.integration
def test_complete_crosstab_workflow(temp_csv_file):
    stats_analyzer = Stats(temp_csv_file, "crosstab")

    combinations = [
        ("treatment", "age_group"),
        ("group", "category"),
        ("treatment", "category"),
    ]

    for var1, var2 in combinations:
        result = stats_analyzer.crosstab_analysis(var1, var2)
        assert result is not None
        assert isinstance(result, dict)

        assert isinstance(result["chi2_statistic"], (int, float))
        assert isinstance(result["p_value"], (int, float))
        assert 0 <= result["p_value"] <= 1
        assert result["degrees_of_freedom"] > 0


@pytest.mark.integration
def test_filtered_analysis(temp_csv_file):
    with patch("Stats.ParseToKeyValueList") as mock_parse:
        mock_parse.return_value = [("group", "A")]

        stats_analyzer = Stats(temp_csv_file, "freq", "group=A")

        assert len(stats_analyzer.data) < 100
        assert all(stats_analyzer.data["group"] == "A")

        result = stats_analyzer.frequency_analysis("treatment")
        assert result is not None


@pytest.mark.integration
def test_error_handling_missing_file():
    with pytest.raises(Exception):
        Stats("nonexistent_file.csv", "freq")


@pytest.mark.integration
def test_error_handling_empty_data():
    empty_data = pd.DataFrame()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        empty_data.to_csv(f.name, index=False)
        temp_file = f.name

    try:
        stats_analyzer = Stats(temp_file, "freq")
        result = stats_analyzer.frequency_analysis("any_column")
        assert result is None
    finally:
        os.unlink(temp_file)


@pytest.mark.integration
def test_missing_values_handling():
    data_with_na = pd.DataFrame(
        {
            "group": ["A", "B", "A", "B", np.nan, "A"],
            "score": [10, 20, np.nan, 25, 15, 30],
            "category": ["X", "Y", "X", np.nan, "Y", "X"],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        data_with_na.to_csv(f.name, index=False)
        temp_file = f.name

    try:
        stats_analyzer = Stats(temp_file, "anova")

        result = stats_analyzer.anova_analysis("group", "score")
        if result is not None:
            assert isinstance(result, dict)

        stats_analyzer_cross = Stats(temp_file, "crosstab")
        result_cross = stats_analyzer_cross.crosstab_analysis("group", "category")
        if result_cross is not None:
            assert isinstance(result_cross, dict)

    finally:
        os.unlink(temp_file)


@pytest.mark.integration
def test_statistical_accuracy():
    np.random.seed(123)

    group_val = ["A"] * 50 + ["B"] * 50
    score_val = list(np.random.normal(100, 10, 50))
    score_val += list(np.random.normal(90, 10, 50))
    known_data = pd.DataFrame(
        {
            "group": group_val,
            "score": score_val,
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        known_data.to_csv(f.name, index=False)
        temp_file = f.name

    try:
        stats_analyzer = Stats(temp_file, "anova")
        result = stats_analyzer.anova_analysis("group", "score")

        if result is not None:
            summary_stats = result["summary_stats"]
            mean_a = summary_stats[summary_stats["group"] == "A"]["Mean"].iloc[0]
            mean_b = summary_stats[summary_stats["group"] == "B"]["Mean"].iloc[0]

            assert mean_a > mean_b, f"Expected A ({mean_a}) > B ({mean_b})"
            assert result["p_value"] < 0.1

    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
