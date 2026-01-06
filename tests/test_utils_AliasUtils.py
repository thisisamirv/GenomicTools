#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import pytest
from scipy import stats
from utils.AliasUtils import AliasUtils
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.fixture
def ewas_data(data_dir):
    file_path = os.path.join(data_dir, "ewas_450k_linear.csv")
    if not os.path.exists(file_path):
        pytest.skip(f"EWAS test file not found at {file_path}")
    return pd.read_csv(file_path)


@pytest.fixture
def gwas_data(data_dir):
    file_path = os.path.join(data_dir, "annotated_gwas.csv")
    if not os.path.exists(file_path):
        pytest.skip(f"GWAS test file not found at {file_path}")
    return pd.read_csv(file_path)


@pytest.fixture
def mock_alias_data():
    return {
        "RSID": ["rs123", "rs456", "rs789"],
        "chromosome": [1, 2, 3],
        "bp_hg38": [1000, 2000, 3000],
        "t.stat": [2.5, 3.1, 1.8],
        "pvalue": [0.01, 0.001, 0.05],
        "se": [0.2, 0.3, 0.15],
        "beta": [0.5, 0.8, 0.3],
        "methylation_se": [0.1, 0.2, 0.3],
        "methylation_coef": [0.4, 0.5, 0.6],
    }


@pytest.mark.unit
def test_get_field():
    assert AliasUtils.get_field("rs_id") == "RSID"
    assert AliasUtils.get_field("snp") == "RSID"
    assert AliasUtils.get_field("chromosome") == "CHR"
    assert AliasUtils.get_field("p.value") == "P"
    assert AliasUtils.get_field("standard_error") == "SE"

    assert AliasUtils.get_field("nonexistent_alias") is None


@pytest.mark.unit
def test_get_aliases():
    chr_aliases = AliasUtils.get_aliases("CHR")
    assert isinstance(chr_aliases, list)
    assert "chromosome" in chr_aliases

    p_aliases = AliasUtils.get_aliases("P")
    assert isinstance(p_aliases, list)
    assert "pvalue" in p_aliases

    multiple_aliases = AliasUtils.get_aliases(["CHR", "P"])
    assert isinstance(multiple_aliases, list)
    assert "chromosome" in multiple_aliases
    assert "pvalue" in multiple_aliases
    assert len(multiple_aliases) > len(chr_aliases)

    nonexistent_aliases = AliasUtils.get_aliases("NONEXISTENT")
    assert isinstance(nonexistent_aliases, list)
    assert len(nonexistent_aliases) == 0


@pytest.mark.unit
def test_find_keys():
    columns = ["chr", "position", "pval", "beta", "standard_error"]

    found = AliasUtils.find_keys(columns, "CHR")
    assert found == "chr"

    found = AliasUtils.find_keys(columns, "P")
    assert found == "pval"

    found = AliasUtils.find_keys(columns, "COEF")
    assert found == "beta"

    found = AliasUtils.find_keys(columns, "SE")
    assert found == "standard_error"

    found = AliasUtils.find_keys(columns, "NONEXISTENT")
    assert found is None

    found = AliasUtils.find_keys([], "CHR")
    assert found is None


@pytest.mark.unit
def test_get_complex_field():
    assert AliasUtils.get_complex_field("COEF", "SE") == "COEF_SE"
    assert AliasUtils.get_complex_field("P", "FDR") == "P_FDR"
    assert AliasUtils.get_complex_field("TSS", "DIST") == "TSS_DIST"

    assert AliasUtils.get_complex_field("SE", "COEF") == "COEF_SE"
    assert AliasUtils.get_complex_field("FDR", "P") == "P_FDR"


@pytest.mark.unit
def test_find_complex_keys_with_mock_data():
    p_aliases = AliasUtils.get_aliases("P")
    fdr_aliases = AliasUtils.get_aliases("P_FDR")

    print(f"P aliases: {p_aliases[:10]}...")
    print(f"P_FDR aliases: {fdr_aliases[:10]}...")

    columns = ["coef_se", "p_fdr", "tss_distance", "pvalue", "beta_standard_error"]

    found = AliasUtils.find_complex_keys(columns, "COEF", "SE")
    assert found in [
        "coef_se",
        "beta_standard_error",
    ], f"Expected COEF+SE match, got {found}"

    found = AliasUtils.find_complex_keys(columns, "P", "FDR")
    assert found == "p_fdr"

    found = AliasUtils.find_complex_keys(columns, "TSS", "DIST")
    assert found == "tss_distance"

    found = AliasUtils.find_complex_keys(columns, "NONEXISTENT", "FIELD")
    assert found is None


@pytest.mark.integration
def test_find_complex_keys_in_ewas_data(ewas_data):
    coef_se_field = AliasUtils.find_complex_keys(ewas_data.columns, "COEF", "SE")
    assert coef_se_field is None

    coef_field = AliasUtils.find_keys(ewas_data.columns, "COEF")
    se_field = AliasUtils.find_keys(ewas_data.columns, "SE")
    assert coef_field == "Methylation_COEF"
    assert se_field == "Methylation_SE"

    p_fdr_field = AliasUtils.find_complex_keys(ewas_data.columns, "P", "FDR")
    assert p_fdr_field is None

    p_fdr_aliases = AliasUtils.get_aliases("P_FDR")
    print(f"P_FDR aliases: {p_fdr_aliases[:10]}...")

    mock_columns = ["coef_se", "beta_stderr", "p_fdr", "methylation_p", "P_FDR", "fdr"]

    coef_se_mock = AliasUtils.find_complex_keys(mock_columns, "COEF", "SE")
    assert coef_se_mock is not None, "Should find a COEF+SE combination"
    assert (
        coef_se_mock in mock_columns
    ), f"Found field should be in the mock columns: {coef_se_mock}"

    coef_se_lower = coef_se_mock.lower()
    has_coef_alias = any(
        alias.lower() in coef_se_lower for alias in ["coef", "beta", "coefficient"]
    )
    has_se_alias = any(
        alias.lower() in coef_se_lower
        for alias in ["se", "stderr", "standard_error", "error"]
    )
    assert (
        has_coef_alias and has_se_alias
    ), f"Found field '{coef_se_mock}' should contain both COEF and SE aliases"

    p_fdr_mock = AliasUtils.find_complex_keys(mock_columns, "P", "FDR")
    assert (
        p_fdr_mock is not None
    ), f"Should find P_FDR field. Available: {mock_columns}, P_FDR aliases: {p_fdr_aliases[:5]}"
    assert p_fdr_mock in [
        "p_fdr",
        "P_FDR",
        "fdr",
    ], f"Expected P_FDR match from {mock_columns}, got {p_fdr_mock}"


@pytest.mark.unit
def test_get_complex_aliases():
    aliases = AliasUtils.get_complex_aliases("COEF", "SE")

    assert isinstance(aliases, list)
    assert len(aliases) > 10

    aliases_lower = [alias.lower() for alias in aliases]
    aliases_str = " ".join(aliases_lower)

    coef_terms = ["coef", "beta", "coefficient"]
    se_terms = ["se", "standard", "error"]

    has_coef_term = any(term in aliases_str for term in coef_terms)
    has_se_term = any(term in aliases_str for term in se_terms)

    assert (
        has_coef_term
    ), f"Should contain COEF-related terms. Aliases: {aliases[:10]}..."
    assert has_se_term, f"Should contain SE-related terms. Aliases: {aliases[:10]}..."

    forward_combos = [
        alias
        for alias in aliases_lower
        if any(alias.startswith(term) for term in coef_terms)
    ]
    reverse_combos = [
        alias
        for alias in aliases_lower
        if any(alias.startswith(term) for term in se_terms)
    ]

    assert len(forward_combos) > 0, "Should have COEF-first combinations"
    assert len(reverse_combos) > 0, "Should have SE-first combinations"


@pytest.mark.unit
def test_validate_p_column_with_mock_data():
    valid_data = pd.DataFrame(
        {"p1": [0.01, 0.05, 0.001, 0.9], "p2": [1.0, 0.5, 0.0, 0.3]}
    )

    invalid_data = pd.DataFrame(
        {
            "not_p1": [1.5, 2.0, -0.5, 10.0],
            "not_p2": [1, 1, 1, 1],
        }
    )

    assert AliasUtils._validate_p_column(valid_data, "p1") is True
    assert AliasUtils._validate_p_column(valid_data, "p2") is True

    assert AliasUtils._validate_p_column(invalid_data, "not_p1") is False
    assert AliasUtils._validate_p_column(invalid_data, "not_p2") is False


@pytest.mark.unit
def test_validate_se_column_with_mock_data():
    valid_data = pd.DataFrame(
        {
            "se1": [0.1, 0.2, 0.3, 0.4],
            "se2": [1.0, 2.0, 1.5, 0.8],
            "se3": [1000, 2000, 5000, 10000],
        }
    )

    invalid_data = pd.DataFrame(
        {
            "not_se1": [-0.1, -0.2, 0.0, -0.5],
            "not_se2": [1e-15, 1e-15, 1e-15, 1e-15],
            "not_se3": [1e7, 2e7, 5e7, 1e8],
            "not_se4": [0.5, 0.5, 0.5, 0.5],
        }
    )

    assert AliasUtils._validate_se_column(valid_data, "se1") is True
    assert AliasUtils._validate_se_column(valid_data, "se2") is True
    assert AliasUtils._validate_se_column(valid_data, "se3") is True

    assert AliasUtils._validate_se_column(invalid_data, "not_se1") is False
    assert AliasUtils._validate_se_column(invalid_data, "not_se2") is False
    assert AliasUtils._validate_se_column(invalid_data, "not_se3") is False
    assert AliasUtils._validate_se_column(invalid_data, "not_se4") is False


@pytest.mark.unit
def test_find_p_column_comprehensive_with_mock_data():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "p.value": [0.01, 0.05, 0.001, 0.5],
            "methylation_p": [0.02, 0.03, 0.002, 0.4],
            "not_p": [1.5, 2.0, 3.0, 4.0],
        }
    )

    p_col = AliasUtils.find_p_column_comprehensive(data)
    assert p_col == "p.value"

    p_col = AliasUtils.find_p_column_comprehensive(data, target_variable="methylation")
    assert p_col == "methylation_p"

    existing_mappings = {"p.value": "P"}
    p_col = AliasUtils.find_p_column_comprehensive(
        data, existing_mappings=existing_mappings
    )
    assert p_col == "methylation_p"

    data_no_p = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "not_p1": [1.5, 2.0, 3.0, 4.0],
            "not_p2": [5.0, 6.0, 7.0, 8.0],
        }
    )
    p_col = AliasUtils.find_p_column_comprehensive(data_no_p)
    assert p_col is None


@pytest.mark.unit
def test_find_se_column_comprehensive_with_mock_data():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "standard_error": [0.1, 0.2, 0.3, 0.4],
            "methylation_se": [0.15, 0.25, 0.35, 0.45],
            "not_related": [1000, 2000, 3000, 4000],
        }
    )

    se_col = AliasUtils.find_se_column_comprehensive(data)
    assert se_col == "standard_error"

    se_col = AliasUtils.find_se_column_comprehensive(
        data, target_variable="methylation"
    )
    assert se_col == "methylation_se"

    existing_mappings = {"standard_error": "SE"}
    se_col = AliasUtils.find_se_column_comprehensive(
        data, existing_mappings=existing_mappings
    )
    assert se_col == "methylation_se"

    data_no_se = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "coefficient": [
                -0.1,
                -0.2,
                -0.3,
                -0.4,
            ],
            "pvalue": [0.01, 0.02, 0.03, 0.04],
            "sample_size": [100, 200, 300, 400],
            "bad_values": [1e8, 2e8, 3e8, 4e8],
        }
    )
    se_col = AliasUtils.find_se_column_comprehensive(data_no_se)
    assert se_col is None

    data_invalid_se = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "fake_se": [0.5, 0.5, 0.5, 0.5],
            "bad_stderr": [-1, -2, -3, -4],
            "huge_se": [1e7, 1e7, 1e7, 1e7],
        }
    )
    se_col = AliasUtils.find_se_column_comprehensive(data_invalid_se)
    assert se_col is None

    data_priority = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "se": [0.1, 0.2, 0.3, 0.4],
            "standard_error": [0.15, 0.25, 0.35, 0.45],
        }
    )
    se_col = AliasUtils.find_se_column_comprehensive(data_priority)
    assert se_col in ["se", "standard_error"]


@pytest.mark.unit
def test_get_all_p_value_columns():
    data = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "p.value": [0.01, 0.05, 0.001, 0.5],
            "p_fdr": [0.02, 0.1, 0.003, 0.6],
            "p_bonf": [0.03, 0.15, 0.005, 0.7],
            "not_p": [1.5, 2.0, 3.0, 4.0],
        }
    )

    p_found = AliasUtils.find_keys(data.columns, "P")
    fdr_found = AliasUtils.find_keys(data.columns, "P_FDR")
    bonf_found = AliasUtils.find_keys(data.columns, "P_BONFERRONI")

    print(f"P found: {p_found}")
    print(f"P_FDR found: {fdr_found}")
    print(f"P_BONFERRONI found: {bonf_found}")

    p_columns = AliasUtils.get_all_p_value_columns(data)

    assert isinstance(p_columns, dict)

    expected_mappings = {
        "P": "p.value",
        "P_FDR": "p_fdr",
        "P_BONFERRONI": "p_bonf",
    }

    for field, expected_col in expected_mappings.items():
        if field in p_columns:
            assert (
                p_columns[field] == expected_col
            ), f"Expected {field} -> {expected_col}, got {p_columns[field]}"

    assert "P" in p_columns, f"Should find basic P column, found: {p_columns}"


@pytest.mark.unit
def test_edge_cases():
    empty_df = pd.DataFrame()

    assert AliasUtils.find_keys(empty_df.columns, "CHR") is None
    assert AliasUtils.find_p_column_comprehensive(empty_df) is None
    assert AliasUtils.find_se_column_comprehensive(empty_df) is None

    assert AliasUtils.get_field(None) is None
    assert AliasUtils.get_aliases(None) == []

    assert AliasUtils.find_keys(None, "CHR") is None
    assert AliasUtils.find_keys(["chr", "bp"], None) is None

    p_values, method = AliasUtils.calculate_p_from_other_stats(empty_df)
    assert p_values is None
    assert method is None

    se_values, method = AliasUtils.calculate_se_from_other_stats(empty_df)
    assert se_values is None
    assert method is None


@pytest.mark.unit
def test_calculate_p_from_t_stat_with_mock_data():
    data = pd.DataFrame(
        {
            "COEF": [0.5, 0.8, -0.3, 1.2],
            "T-STAT": [2.5, 4.0, -1.5, 6.0],
            "N": [100, 100, 100, 100],
        }
    )

    p_values, method = AliasUtils.calculate_p_from_other_stats(data)

    assert p_values is not None
    assert method is not None
    assert "t-statistic" in method
    assert len(p_values) == 4
    assert all(0 <= p <= 1 for p in p_values)

    expected_p = 2 * (1 - stats.t.cdf(abs(2.5), 99))
    np.testing.assert_almost_equal(p_values.iloc[0], expected_p, decimal=3)


@pytest.mark.unit
def test_calculate_se_from_t_stat_with_mock_data():
    data = pd.DataFrame(
        {"COEF": [0.5, 0.8, -0.3, 1.2], "T-STAT": [2.5, 4.0, -1.5, 6.0]}
    )

    se_values, method = AliasUtils.calculate_se_from_other_stats(data)

    assert se_values is not None
    assert method is not None
    assert "COEF/T-STAT" in method
    assert len(se_values) == 4
    assert all(se > 0 for se in se_values)

    expected_se = [0.5 / 2.5, 0.8 / 4.0, 0.3 / 1.5, 1.2 / 6.0]
    for i, expected in enumerate(expected_se):
        assert abs(se_values.iloc[i] - expected) < 1e-10


@pytest.mark.unit
def test_alias_caching():
    AliasUtils._alias_cache.clear()

    aliases1 = AliasUtils.get_aliases("CHR")
    assert len(AliasUtils._alias_cache) > 0

    aliases2 = AliasUtils.get_aliases("CHR")
    assert aliases1 == aliases2

    assert any("CHR" in str(key) for key in AliasUtils._alias_cache.keys())


@pytest.mark.unit
def test_lazy_loading():
    original_aliases = AliasUtils.ALIASES.copy()
    original_cache = AliasUtils._alias_cache.copy()

    try:
        AliasUtils.ALIASES["CHR"] = None
        AliasUtils._alias_cache.clear()

        assert AliasUtils.ALIASES["CHR"] is None

        aliases = AliasUtils.get_aliases("CHR")
        assert isinstance(aliases, list)
        assert len(aliases) > 0

        assert len(AliasUtils._alias_cache) > 0

    finally:
        AliasUtils.ALIASES.update(original_aliases)
        AliasUtils._alias_cache.update(original_cache)


@pytest.mark.unit
def test_debug_complex_aliases():
    p_fdr_aliases = AliasUtils.get_aliases("P_FDR")
    print(f"\nP_FDR aliases ({len(p_fdr_aliases)} total):")
    for i, alias in enumerate(p_fdr_aliases[:20]):
        print(f"  {i}: '{alias}'")

    test_strings = ["pval_adjusted", "p_fdr_corrected", "p_fdr", "P_FDR"]
    for test_string in test_strings:
        if test_string in p_fdr_aliases:
            print(f"✓ '{test_string}' found in P_FDR aliases")
        else:
            print(f"✗ '{test_string}' NOT found in P_FDR aliases")

    complex_aliases = AliasUtils.get_complex_aliases("COEF", "SE")
    print(f"\nCOEF+SE complex aliases ({len(complex_aliases)} total):")
    for i, alias in enumerate(complex_aliases[:20]):
        print(f"  {i}: '{alias}'")

    mock_columns = ["p_fdr", "pval_adjusted", "p_fdr_corrected"]
    for col in mock_columns:
        found = AliasUtils.find_complex_keys([col], "P", "FDR")
        print(f"find_complex_keys(['{col}'], 'P', 'FDR') -> {found}")


@pytest.mark.integration
def test_find_keys_in_ewas_data(ewas_data):
    chr_field = AliasUtils.find_keys(ewas_data.columns, "CHR")
    assert chr_field == "CHR"

    coef_field = AliasUtils.find_keys(ewas_data.columns, "COEF")
    assert coef_field == "Methylation_COEF"

    p_field = AliasUtils.find_keys(ewas_data.columns, "P")
    assert p_field == "Methylation_P"

    se_field = AliasUtils.find_keys(ewas_data.columns, "SE")
    assert se_field == "Methylation_SE"

    t_field = AliasUtils.find_keys(ewas_data.columns, "T-STAT")
    assert t_field == "Methylation_T-STAT"


@pytest.mark.integration
def test_find_keys_in_gwas_data(gwas_data):
    chr_field = AliasUtils.find_keys(gwas_data.columns, "CHR")
    assert chr_field == "CHR"

    bp_field = AliasUtils.find_keys(gwas_data.columns, "BP")
    assert bp_field == "BP"

    gene_field = AliasUtils.find_keys(gwas_data.columns, "GENE")
    assert gene_field == "GENE"

    info_field = AliasUtils.find_keys(gwas_data.columns, "INFO")
    assert info_field == "INFO"

    a1_field = AliasUtils.find_keys(gwas_data.columns, "A1")
    a2_field = AliasUtils.find_keys(gwas_data.columns, "A2")
    assert a1_field == "A1"
    assert a2_field == "A2"


@pytest.mark.integration
def test_get_aliases_for_basic_fields():
    chr_aliases = AliasUtils.get_aliases("CHR")
    assert isinstance(chr_aliases, list)
    assert len(chr_aliases) > 5
    assert "chromosome" in chr_aliases
    assert "chr" in chr_aliases

    p_aliases = AliasUtils.get_aliases("P")
    assert isinstance(p_aliases, list)
    assert len(p_aliases) > 5
    assert "pvalue" in p_aliases
    assert "p.value" in p_aliases

    rsid_aliases = AliasUtils.get_aliases("RSID")
    assert isinstance(rsid_aliases, list)
    assert len(rsid_aliases) > 5
    assert "rs_id" in rsid_aliases
    assert "snp" in rsid_aliases


@pytest.mark.unit
def test_get_aliases_chr():
    aliases = AliasUtils.get_aliases("CHR")

    assert isinstance(aliases, list)
    assert len(aliases) > 5

    assert "chr" in aliases
    assert "chromosome" in aliases
    assert "CHR" in aliases
    assert "CHROMOSOME" in aliases


@pytest.mark.unit
def test_get_aliases_p():
    aliases = AliasUtils.get_aliases("P")

    assert isinstance(aliases, list)
    assert len(aliases) > 5

    assert "p" in aliases
    assert "pvalue" in aliases
    assert "p.value" in aliases
    assert "pval" in aliases


@pytest.mark.unit
def test_get_aliases_se():
    aliases = AliasUtils.get_aliases("SE")

    assert isinstance(aliases, list)
    assert len(aliases) > 5

    assert "se" in aliases
    assert "SE" in aliases
    assert "standard_error" in aliases
    assert "StandardError" in aliases


@pytest.mark.integration
def test_find_variable_specific_fields_in_ewas_data(ewas_data):
    methylation_fields = AliasUtils.get_all_fields_for_variable(
        ewas_data.columns, "Methylation"
    )
    assert len(methylation_fields) >= 4
    assert "Methylation_COEF" in methylation_fields
    assert "Methylation_SE" in methylation_fields
    assert "Methylation_P" in methylation_fields
    assert "Methylation_T-STAT" in methylation_fields

    hiv_fields = AliasUtils.get_all_fields_for_variable(ewas_data.columns, "hiv")
    assert len(hiv_fields) >= 4
    assert "hiv_COEF" in hiv_fields
    assert "hiv_SE" in hiv_fields
    assert "hiv_P" in hiv_fields
    assert "hiv_T-STAT" in hiv_fields


@pytest.mark.integration
def test_find_p_column_comprehensive(ewas_data):
    columns = ["CGID", "CHR", "Methylation_P", "hiv_P"]
    p_col = AliasUtils.find_p_column_comprehensive(ewas_data[columns])

    assert p_col == "Methylation_P"

    p_col = AliasUtils.find_p_column_comprehensive(
        ewas_data[columns], target_variable="hiv"
    )
    assert p_col == "hiv_P"


@pytest.mark.integration
def test_find_se_column_comprehensive(ewas_data):
    columns = ["CGID", "CHR", "Methylation_SE", "hiv_SE"]
    se_col = AliasUtils.find_se_column_comprehensive(ewas_data[columns])

    assert se_col == "Methylation_SE"

    se_col = AliasUtils.find_se_column_comprehensive(
        ewas_data[columns], target_variable="hiv"
    )
    assert se_col == "hiv_SE"


@pytest.mark.integration
def test_validate_p_column(ewas_data):
    is_valid = AliasUtils._validate_p_column(ewas_data, "Methylation_P")
    assert is_valid is True

    is_valid = AliasUtils._validate_p_column(ewas_data, "CHR")
    assert is_valid is False


@pytest.mark.integration
def test_validate_se_column(ewas_data):
    is_valid = AliasUtils._validate_se_column(ewas_data, "Methylation_SE")
    assert is_valid is True

    is_valid = AliasUtils._validate_se_column(ewas_data, "CHR")
    assert is_valid is False


@pytest.mark.integration
def test_calculate_p_from_other_stats(ewas_data):
    df = ewas_data[["CGID", "Methylation_COEF", "Methylation_T-STAT"]].copy()

    df.rename(
        columns={"Methylation_COEF": "COEF", "Methylation_T-STAT": "T-STAT"},
        inplace=True,
    )

    p_values, method = AliasUtils.calculate_p_from_other_stats(df)

    assert p_values is not None
    assert method is not None
    assert "t-statistic" in method
    assert len(p_values) == len(df)
    assert all(0 <= p <= 1 for p in p_values.dropna())


@pytest.mark.integration
def test_calculate_se_from_other_stats(ewas_data):
    df = ewas_data[["CGID", "Methylation_COEF", "Methylation_T-STAT"]].copy()

    df.rename(
        columns={"Methylation_COEF": "COEF", "Methylation_T-STAT": "T-STAT"},
        inplace=True,
    )

    se_values, method = AliasUtils.calculate_se_from_other_stats(df)

    assert se_values is not None
    assert method is not None
    assert "COEF/T-STAT" in method
    assert len(se_values) == len(df)
    assert all(se > 0 for se in se_values.dropna())


if __name__ == "__main__":
    pytest.main(["-v", __file__])
