#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import pytest
import tempfile
from utils.LoggingUtils import log
from scipy.stats import norm
from MetaAssociation import MetaAssociation
from MetaAssociation import (
    _validate_study_data,
    _fast_fixed_effects_meta,
    _fast_random_effects_meta,
    _fast_heterogeneity_stats,
    calculate_precise_p_value,
)

log.setup(level="DEBUG")


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def sample_ewas_files(output_dir):
    study1 = pd.DataFrame(
        {
            "CGID": [
                "cg00000001",
                "cg00000002",
                "cg00000003",
                "cg00000004",
                "cg00000005",
            ],
            "CHR": [1, 1, 2, 2, 3],
            "BP": [100, 200, 300, 400, 500],
            "COEF": [0.1, 0.2, 0.3, 0.4, 0.5],
            "SE": [0.01, 0.02, 0.03, 0.04, 0.05],
            "P": [0.05, 0.01, 0.001, 0.0001, 0.00001],
            "N": [100, 100, 100, 100, 100],
        }
    )

    study2 = pd.DataFrame(
        {
            "CGID": [
                "cg00000001",
                "cg00000002",
                "cg00000003",
                "cg00000004",
                "cg00000005",
            ],
            "CHR": [1, 1, 2, 2, 3],
            "BP": [100, 200, 300, 400, 500],
            "COEF": [0.15, 0.25, 0.35, 0.45, 0.55],
            "SE": [0.015, 0.025, 0.035, 0.045, 0.055],
            "P": [0.04, 0.009, 0.0009, 0.00009, 0.000009],
            "N": [150, 150, 150, 150, 150],
        }
    )

    study3 = pd.DataFrame(
        {
            "CGID": [
                "cg00000001",
                "cg00000002",
                "cg00000003",
                "cg00000004",
                "cg00000005",
            ],
            "CHR": [1, 1, 2, 2, 3],
            "BP": [100, 200, 300, 400, 500],
            "COEF": [-0.12, -0.22, -0.32, -0.42, -0.52],
            "SE": [0.012, 0.022, 0.032, 0.042, 0.052],
            "P": [0.045, 0.0095, 0.00095, 0.000095, 0.0000095],
            "N": [120, 120, 120, 120, 120],
        }
    )

    file_paths = []
    for i, df in enumerate([study1, study2, study3]):
        file_path = os.path.join(output_dir, f"ewas_study{i + 1}.csv")
        df.to_csv(file_path, index=False)
        file_paths.append(file_path)

    return file_paths


@pytest.fixture
def sample_gwas_files(output_dir):
    study1 = pd.DataFrame(
        {
            "RSID": [
                "rs00000001",
                "rs00000002",
                "rs00000003",
                "rs00000004",
                "rs00000005",
            ],
            "CHR": [1, 1, 2, 2, 3],
            "BP": [100, 200, 300, 400, 500],
            "A1": ["A", "C", "G", "T", "A"],
            "A2": ["G", "T", "C", "A", "G"],
            "COEF": [0.1, 0.2, 0.3, 0.4, 0.5],
            "SE": [0.01, 0.02, 0.03, 0.04, 0.05],
            "P": [0.05, 0.01, 0.001, 0.0001, 0.00001],
            "N": [100, 100, 100, 100, 100],
        }
    )

    study2 = pd.DataFrame(
        {
            "RSID": [
                "rs00000001",
                "rs00000002",
                "rs00000003",
                "rs00000004",
                "rs00000005",
            ],
            "CHR": [1, 1, 2, 2, 3],
            "BP": [100, 200, 300, 400, 500],
            "A1": ["A", "C", "G", "T", "A"],
            "A2": ["G", "T", "C", "A", "G"],
            "COEF": [0.15, 0.25, 0.35, 0.45, 0.55],
            "SE": [0.015, 0.025, 0.035, 0.045, 0.055],
            "P": [0.04, 0.009, 0.0009, 0.00009, 0.000009],
            "N": [150, 150, 150, 150, 150],
        }
    )

    file_paths = []
    for i, df in enumerate([study1, study2]):
        file_path = os.path.join(output_dir, f"gwas_study{i + 1}.csv")
        df.to_csv(file_path, index=False)
        file_paths.append(file_path)

    return file_paths


@pytest.fixture
def create_test_study_files(tmp_path):
    study1_data = pd.DataFrame(
        {
            "CGID": [
                "cg00000001",
                "cg00000002",
                "cg00000003",
                "cg00000004",
                "cg00000005",
            ],
            "CHR": [1, 2, 3, 4, 5],
            "BP": [1000, 2000, 3000, 4000, 5000],
            "COEF": [0.1, 0.2, 0.3, 0.4, 0.5],
            "SE": [0.01, 0.02, 0.03, 0.04, 0.05],
            "P": [0.01, 0.02, 0.03, 0.04, 0.05],
            "N": [100] * 5,
        }
    )

    study2_data = pd.DataFrame(
        {
            "CGID": [
                "cg00000001",
                "cg00000002",
                "cg00000003",
                "cg00000004",
                "cg00000005",
            ],
            "CHR": [1, 2, 3, 4, 5],
            "BP": [1000, 2000, 3000, 4000, 5000],
            "COEF": [0.15, 0.25, 0.35, 0.45, 0.55],
            "SE": [0.015, 0.025, 0.035, 0.045, 0.055],
            "P": [0.008, 0.015, 0.025, 0.035, 0.045],
            "N": [150] * 5,
        }
    )

    study3_data = pd.DataFrame(
        {
            "CGID": [
                "cg00000001",
                "cg00000002",
                "cg00000003",
                "cg00000004",
                "cg00000005",
            ],
            "CHR": [1, 2, 3, 4, 5],
            "BP": [1000, 2000, 3000, 4000, 5000],
            "COEF": [0.12, 0.22, 0.32, 0.42, 0.52],
            "SE": [0.012, 0.022, 0.032, 0.042, 0.052],
            "P": [0.009, 0.018, 0.028, 0.038, 0.048],
            "N": [120] * 5,
        }
    )

    file_paths = []
    for i, data in enumerate([study1_data, study2_data, study3_data], 1):
        file_path = tmp_path / f"study{i}.csv"
        data.to_csv(file_path, index=False)
        file_paths.append(str(file_path))

    return file_paths


@pytest.mark.unit
def test_fast_fixed_effects_meta():
    coefs = np.array([0.1, 0.2, 0.15], dtype=np.float64)
    ses = np.array([0.05, 0.06, 0.04], dtype=np.float64)

    pooled_coef, pooled_se, pooled_z, q_stat, sum_weights = _fast_fixed_effects_meta(
        coefs, ses
    )

    assert isinstance(pooled_coef, float), "Pooled coefficient should be float"
    assert isinstance(pooled_se, float), "Pooled SE should be float"
    assert isinstance(pooled_z, float), "Pooled Z should be float"
    assert isinstance(q_stat, float), "Q statistic should be float"
    assert isinstance(sum_weights, float), "Sum of weights should be float"

    assert pooled_se > 0, "Pooled SE should be positive"
    assert pooled_se < min(ses), "Pooled SE should be smaller than individual SEs"

    assert q_stat >= 0, "Q statistic should be non-negative"


@pytest.mark.unit
def test_fast_random_effects_meta():
    coefs = np.array([0.1, 0.2, 0.15], dtype=np.float64)
    ses = np.array([0.05, 0.06, 0.04], dtype=np.float64)

    pooled_coef_re, pooled_se_re, pooled_z_re, tau_squared = _fast_random_effects_meta(
        coefs, ses
    )

    assert isinstance(pooled_coef_re, float), "Pooled coefficient should be float"
    assert isinstance(pooled_se_re, float), "Pooled SE should be float"
    assert isinstance(pooled_z_re, float), "Pooled Z should be float"
    assert isinstance(tau_squared, float), "Tau squared should be float"

    assert pooled_se_re > 0, "Pooled SE should be positive"

    assert tau_squared >= 0, "Tau squared should be non-negative"


@pytest.mark.unit
def test_fast_heterogeneity_stats():
    q_stat = 10.0
    df_q = 2

    i_squared = _fast_heterogeneity_stats(q_stat, df_q)

    assert isinstance(i_squared, float), "I-squared should be float"
    assert 0 <= i_squared <= 1, f"I-squared should be between 0 and 1, got {i_squared}"

    q_stat_zero = 0.0
    i_squared_zero = _fast_heterogeneity_stats(q_stat_zero, df_q)
    assert i_squared_zero == 0.0, "I-squared should be 0 when Q=0"

    q_small = 1.0
    df_large = 5
    i_squared_small = _fast_heterogeneity_stats(q_small, df_large)
    assert i_squared_small == 0.0, "I-squared should be 0 when Q < df"


@pytest.mark.unit
def test_fast_p_value_calculation():
    z_moderate = 2.0
    p_moderate = calculate_precise_p_value(z_moderate, precision_mode="standard")
    expected_moderate = 2 * (1 - norm.cdf(2.0))
    assert (
        abs(p_moderate - expected_moderate) < 1e-6
    ), f"Expected {expected_moderate}, got {p_moderate}"

    z_extreme = 10.0
    p_extreme = calculate_precise_p_value(z_extreme, precision_mode="ultra")

    assert p_extreme < 1e-20, f"Expected very small p-value, got {p_extreme}"
    assert p_extreme > 0, "P-value should be positive"

    z_zero = 0.0
    p_zero = calculate_precise_p_value(z_zero, precision_mode="standard")
    assert abs(p_zero - 1.0) < 0.01, f"Expected p≈1 for z=0, got {p_zero}"


@pytest.mark.unit
def test_validate_study_data():
    coefs = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    ses = np.array([0.05, 0.06, 0.07], dtype=np.float64)
    pvals = np.array([0.05, 0.01, 0.001], dtype=np.float64)

    valid_mask = _validate_study_data(coefs, ses, pvals)
    assert np.all(valid_mask), "All valid data should pass validation"

    coefs_invalid = np.array([0.1, 0.2], dtype=np.float64)
    ses_invalid = np.array([0.05, -0.06], dtype=np.float64)
    pvals_invalid = np.array([0.05, 0.01], dtype=np.float64)

    valid_mask_invalid = _validate_study_data(coefs_invalid, ses_invalid, pvals_invalid)
    assert valid_mask_invalid[0], "First entry should be valid"
    assert not valid_mask_invalid[1], "Second entry should be invalid (negative SE)"

    pvals_invalid2 = np.array([0.05, 1.5], dtype=np.float64)
    ses_valid = np.array([0.05, 0.06], dtype=np.float64)

    valid_mask_invalid2 = _validate_study_data(coefs_invalid, ses_valid, pvals_invalid2)
    assert valid_mask_invalid2[0], "First entry should be valid"
    assert not valid_mask_invalid2[1], "Second entry should be invalid (p-value > 1)"

    coefs_nan = np.array([0.1, np.nan], dtype=np.float64)
    ses_valid = np.array([0.05, 0.06], dtype=np.float64)
    pvals_valid = np.array([0.05, 0.01], dtype=np.float64)

    valid_mask_nan = _validate_study_data(coefs_nan, ses_valid, pvals_valid)
    assert valid_mask_nan[0], "First entry should be valid"
    assert not valid_mask_nan[1], "Second entry should be invalid (NaN coefficient)"


@pytest.mark.unit
def test_determine_target_variable():
    df = pd.DataFrame(
        {
            "CGID": ["cg00000001", "cg00000002"],
            "Methylation_COEF": [0.1, 0.2],
            "Methylation_SE": [0.01, 0.02],
            "Methylation_P": [0.05, 0.01],
            "Age_COEF": [0.3, 0.4],
            "Age_SE": [0.03, 0.04],
            "Age_P": [0.003, 0.004],
        }
    )

    meta = MetaAssociation(output=".", var="Age")
    target_var = meta.determine_target_variable(df)
    assert target_var == "Age"

    meta = MetaAssociation(output=".", data_type="EWAS")
    target_var = meta.determine_target_variable(df)
    assert target_var == "Methylation"


@pytest.mark.unit
def test_standardize_columns():
    df = pd.DataFrame(
        {
            "CGID": ["cg00000001", "cg00000002"],
            "chromosome": [1, 2],
            "COEF": [0.1, 0.2],
            "SE": [0.01, 0.02],
            "pvalue": [0.05, 0.01],
            "N": [100, 100],
        }
    )

    meta = MetaAssociation(output=".", data_type="EWAS")
    std_df = meta.standardize_columns(df)

    assert std_df is not None
    assert "CHR" in std_df.columns
    assert "P" in std_df.columns


@pytest.mark.unit
def test_precision_modes():
    z_moderate = 5.0
    z_extreme = 15.0

    p_std = calculate_precise_p_value(z_moderate, precision_mode="standard")
    assert p_std > 0 and p_std < 1

    p_high = calculate_precise_p_value(z_moderate, precision_mode="high")
    assert p_high > 0 and p_high < 1

    p_ultra = calculate_precise_p_value(z_extreme, precision_mode="ultra")
    assert p_ultra >= 0

    p_auto = calculate_precise_p_value(z_extreme, precision_mode="auto")
    assert p_auto >= 0


@pytest.mark.unit
def test_error_handling(output_dir):
    with pytest.raises(ValueError, match="Invalid method"):
        MetaAssociation(
            output=os.path.join(output_dir, "invalid_method.csv"),
            method="invalid_method",
        )

    with pytest.raises(ValueError, match="Invalid precision_mode"):
        MetaAssociation(
            output=os.path.join(output_dir, "invalid_precision.csv"),
            precision_mode="invalid_precision",
        )

    meta_no_files = MetaAssociation(
        input="nonexistent_file.csv",
        names="single_study",
        sample_sizes="100",
        output="test_output.csv",
    )
    result = meta_no_files.run()
    assert result is None, "Should return None when input files don't exist"
    assert (
        len(meta_no_files.studies) == 0
    ), "Should have no studies when files don't exist"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("CGID,CHR,BP,COEF,SE,P,N\n")
        f.write("cg00000001,1,100,0.1,0.01,0.05,100\n")
        f.write("cg00000002,1,200,0.2,0.02,0.01,100\n")
        temp_file1 = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("CGID,CHR,BP,COEF,SE,P,N\n")
        f.write("cg00000001,1,100,0.15,0.015,0.04,150\n")
        f.write("cg00000002,1,200,0.25,0.025,0.008,150\n")
        temp_file2 = f.name

    try:
        meta_single = MetaAssociation(
            input=temp_file1,
            names="study1",
            sample_sizes="100",
            output="test_single.csv",
            data_type="EWAS",
        )
        result_single = meta_single.run()
        assert result_single is None, "Should return None with insufficient studies"

        meta_mismatch = MetaAssociation(
            input=f"{temp_file1},{temp_file2}",
            names="study1",
            sample_sizes="100,150",
            output="test_mismatch.csv",
            data_type="EWAS",
        )
        result_mismatch = meta_mismatch.run()
        assert result_mismatch is None, "Should return None with mismatched counts"

        meta_sample_mismatch = MetaAssociation(
            input=f"{temp_file1},{temp_file2}",
            names="study1,study2",
            sample_sizes="100",
            output="test_sample_mismatch.csv",
            data_type="EWAS",
        )
        result_sample = meta_sample_mismatch.run()
        assert result_sample is None, "Should return None with mismatched sample sizes"

    finally:
        for temp_file in [temp_file1, temp_file2]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


@pytest.mark.unit
def test_invalid_input_validation():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("CGID,CHR,BP,COEF,SE,P,N\n")
        f.write("cg00000001,1,100,0.1,-0.01,0.05,100\n")
        f.write("cg00000002,1,200,0.2,0.02,1.5,100\n")
        temp_file = f.name

    try:
        meta = MetaAssociation(
            input=f"{temp_file},{temp_file}",
            names="study1,study2",
            sample_sizes="100,100",
            output="test_invalid.csv",
            data_type="EWAS",
        )

        meta.run()

    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


@pytest.mark.integration
def test_meta_analysis_fixed_effects(create_test_study_files, tmp_path):
    output_file = str(tmp_path / "fixed_effects_result.csv")

    meta = MetaAssociation(
        input=",".join(create_test_study_files),
        names="study1,study2,study3",
        sample_sizes="100,150,120",
        output=output_file,
        method="fixed",
        data_type="EWAS",
    )

    result = meta.run()
    assert result is not None, "Meta-analysis should return results"
    assert len(result) > 0, "Should have analyzed some markers"

    expected_columns = ["CGID", "COEF_FIXED", "SE_FIXED", "P_FIXED"]
    for col in expected_columns:
        assert col in result.columns, f"Result should contain {col} column"

    assert os.path.exists(output_file), "Output file should be created"


@pytest.mark.integration
def test_meta_analysis_random_effects(sample_ewas_files, output_dir):
    output_file = os.path.join(output_dir, "meta_random.csv")

    meta = MetaAssociation(
        input=",".join(sample_ewas_files),
        names="study1,study2,study3",
        sample_sizes="100,150,120",
        output=output_file,
        method="random",
        data_type="EWAS",
    )

    result = meta.run()

    assert result is not None
    assert os.path.exists(output_file)

    results_df = pd.read_csv(output_file)

    assert "CGID" in results_df.columns
    assert "COEF_RANDOM" in results_df.columns
    assert "SE_RANDOM" in results_df.columns
    assert "P_RANDOM" in results_df.columns
    assert "TAU_SQUARED" in results_df.columns

    assert len(results_df) == 5


@pytest.mark.integration
def test_meta_analysis_both_methods(sample_ewas_files, output_dir):
    output_file = os.path.join(output_dir, "meta_both.csv")

    meta = MetaAssociation(
        input=",".join(sample_ewas_files),
        names="study1,study2,study3",
        sample_sizes="100,150,120",
        output=output_file,
        method="both",
        data_type="EWAS",
    )

    result = meta.run()

    assert result is not None
    assert os.path.exists(output_file)

    results_df = pd.read_csv(output_file)

    assert "CGID" in results_df.columns
    assert "COEF_FIXED" in results_df.columns
    assert "SE_FIXED" in results_df.columns
    assert "P_FIXED" in results_df.columns
    assert "COEF_RANDOM" in results_df.columns
    assert "SE_RANDOM" in results_df.columns
    assert "P_RANDOM" in results_df.columns
    assert "TAU_SQUARED" in results_df.columns


@pytest.mark.integration
def test_gwas_meta_analysis(sample_gwas_files, output_dir):
    output_file = os.path.join(output_dir, "gwas_meta.csv")

    meta = MetaAssociation(
        input=",".join(sample_gwas_files),
        names="gwas1,gwas2",
        sample_sizes="100,150",
        output=output_file,
        method="both",
        data_type="GWAS",
    )

    result = meta.run()

    assert result is not None
    assert os.path.exists(output_file)

    results_df = pd.read_csv(output_file)

    assert "RSID" in results_df.columns
    assert "CHR" in results_df.columns
    assert "BP" in results_df.columns

    assert len(results_df) == 5
    assert all(results_df["N_STUDIES"] == 2)


@pytest.mark.integration
def test_heterogeneity_detection(sample_ewas_files, output_dir):
    output_file = os.path.join(output_dir, "meta_heterogeneity.csv")

    meta = MetaAssociation(
        input=",".join(sample_ewas_files),
        names="study1,study2,study3",
        sample_sizes="100,150,120",
        output=output_file,
        method="both",
        data_type="EWAS",
    )

    result = meta.run()

    assert result is not None
    assert os.path.exists(output_file)

    results_df = pd.read_csv(output_file)

    assert "I_SQUARED" in results_df.columns
    assert "Q_STAT" in results_df.columns
    assert "Q_P" in results_df.columns

    i_squared_values = pd.to_numeric(results_df["I_SQUARED"], errors="coerce")
    assert any(i_squared_values > 0.5)


@pytest.mark.integration
def test_auto_sample_size_detection(output_dir):
    study_df = pd.DataFrame(
        {
            "CGID": ["cg00000001", "cg00000002"],
            "CHR": [1, 2],
            "BP": [100, 200],
            "COEF": [0.1, 0.2],
            "SE": [0.01, 0.02],
            "P": [0.05, 0.01],
            "N": [123, 123],
        }
    )

    file_path = os.path.join(output_dir, "auto_n_study.csv")
    study_df.to_csv(file_path, index=False)

    output_file = os.path.join(output_dir, "auto_n_result.csv")

    meta = MetaAssociation(
        input=file_path,
        names="auto_study",
        output=output_file,
        method="fixed",
        data_type="EWAS",
    )

    if len(meta.studies) > 0:
        assert meta.studies[0]["sample_size"] == 123


@pytest.mark.integration
def test_data_type_autodetection(output_dir):
    ewas_df = pd.DataFrame(
        {
            "CGID": ["cg00000001", "cg00000002"],
            "CHR": [1, 2],
            "COEF": [0.1, 0.2],
            "SE": [0.01, 0.02],
            "P": [0.05, 0.01],
        }
    )

    ewas_path = os.path.join(output_dir, "auto_ewas.csv")
    ewas_df.to_csv(ewas_path, index=False)

    gwas_df = pd.DataFrame(
        {
            "RSID": ["rs00000001", "rs00000002"],
            "CHR": [1, 2],
            "COEF": [0.1, 0.2],
            "SE": [0.01, 0.02],
            "P": [0.05, 0.01],
        }
    )

    gwas_path = os.path.join(output_dir, "auto_gwas.csv")
    gwas_df.to_csv(gwas_path, index=False)

    ewas_output = os.path.join(output_dir, "auto_ewas_result.csv")
    ewas_meta = MetaAssociation(
        input=ewas_path,
        names="ewas_study",
        output=ewas_output,
        method="fixed",
        data_type="auto",
    )

    gwas_output = os.path.join(output_dir, "auto_gwas_result.csv")
    gwas_meta = MetaAssociation(
        input=gwas_path,
        names="gwas_study",
        output=gwas_output,
        method="fixed",
        data_type="auto",
    )

    assert ewas_meta.id_col == "CGID"
    assert gwas_meta.id_col == "RSID"


@pytest.mark.integration
def test_handle_non_standard_column_names(output_dir):
    study1_df = pd.DataFrame(
        {
            "CGID": ["cg00000001", "cg00000002"],
            "chromosome": [1, 2],
            "position": [100, 200],
            "COEF": [0.1, 0.2],
            "SE": [0.01, 0.02],
            "pvalue": [0.05, 0.01],
            "N": [100, 100],
        }
    )

    study2_df = pd.DataFrame(
        {
            "CGID": ["cg00000001", "cg00000002"],
            "chromosome": [1, 2],
            "position": [100, 200],
            "COEF": [0.15, 0.25],
            "SE": [0.015, 0.025],
            "pvalue": [0.04, 0.009],
            "N": [150, 150],
        }
    )

    file_path1 = os.path.join(output_dir, "nonstandard_study1.csv")
    file_path2 = os.path.join(output_dir, "nonstandard_study2.csv")
    study1_df.to_csv(file_path1, index=False)
    study2_df.to_csv(file_path2, index=False)

    output_file = os.path.join(output_dir, "nonstandard_result.csv")

    meta = MetaAssociation(
        input=f"{file_path1},{file_path2}",
        names="nonstandard_study1,nonstandard_study2",
        sample_sizes="100,150",
        output=output_file,
        method="fixed",
        data_type="EWAS",
    )

    result = meta.run()
    assert result is not None
    assert os.path.exists(output_file)


@pytest.mark.integration
def test_handle_missing_values(output_dir):
    study1 = pd.DataFrame(
        {
            "CGID": ["cg00000001", "cg00000002", "cg00000003"],
            "CHR": [1, 1, 2],
            "BP": [100, 200, 300],
            "COEF": [0.1, 0.2, 0.3],
            "SE": [0.01, 0.02, 0.03],
            "P": [0.05, 0.01, 0.001],
            "N": [100, 100, 100],
        }
    )

    study2 = pd.DataFrame(
        {
            "CGID": ["cg00000001", "cg00000002", "cg00000004"],
            "CHR": [1, 1, 3],
            "BP": [100, 200, 400],
            "COEF": [0.15, 0.25, 0.35],
            "SE": [0.015, 0.025, 0.035],
            "P": [0.04, 0.009, 0.0009],
            "N": [150, 150, 150],
        }
    )

    file_paths = []
    for i, df in enumerate([study1, study2]):
        file_path = os.path.join(output_dir, f"missing_values_study{i + 1}.csv")
        df.to_csv(file_path, index=False)
        file_paths.append(file_path)

    output_file = os.path.join(output_dir, "missing_values_result.csv")

    meta = MetaAssociation(
        input=",".join(file_paths),
        names="study1,study2",
        sample_sizes="100,150",
        output=output_file,
        method="fixed",
        data_type="EWAS",
    )

    result = meta.run()
    assert result is not None
    assert os.path.exists(output_file)

    results_df = pd.read_csv(output_file)
    assert len(results_df) >= 1


@pytest.mark.integration
def test_memory_optimization(sample_ewas_files, output_dir):
    output_file = os.path.join(output_dir, "memory_test.csv")

    meta = MetaAssociation(
        input=",".join(sample_ewas_files),
        names="study1,study2,study3",
        sample_sizes="100,150,120",
        output=output_file,
        method="both",
        data_type="EWAS",
    )

    meta.chunk_size = 2

    result = meta.run()

    assert result is not None
    assert os.path.exists(output_file)


@pytest.mark.integration
def test_parallel_processing(sample_ewas_files, output_dir):
    output_file = os.path.join(output_dir, "parallel_test.csv")

    meta = MetaAssociation(
        input=",".join(sample_ewas_files),
        names="study1,study2,study3",
        sample_sizes="100,150,120",
        output=output_file,
        method="both",
        data_type="EWAS",
    )

    meta.threads = 2
    meta.parallel_params = {"n_jobs": 2, "verbose": 0, "backend": "threading"}

    result = meta.run()

    assert result is not None
    assert os.path.exists(output_file)


@pytest.mark.integration
def test_metaassociation_with_real_mcseq(data_dir, output_dir):
    data_file = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    if not os.path.exists(data_file):
        pytest.skip(f"Test data file not found: {data_file}")

    output_file = os.path.join(output_dir, "meta_mcseq.csv")

    meta = MetaAssociation(
        input=f"{data_file},{data_file}",
        names="study1,study2",
        sample_sizes="100,100",
        output=output_file,
        method="fixed",
        data_type="EWAS",
        var="Methylation",
    )

    result = meta.run()
    assert result is not None
    assert os.path.exists(output_file)

    df = pd.read_csv(output_file)
    assert "CGID" in df.columns
    assert "COEF_FIXED" in df.columns
    assert "SE_FIXED" in df.columns
    assert "P_FIXED" in df.columns
    assert len(df) > 0


@pytest.mark.integration
def test_metaassociation_with_real_450k(data_dir, output_dir):
    data_file = os.path.join(data_dir, "ewas_450k_linear.csv")

    if not os.path.exists(data_file):
        pytest.skip(f"Test data file not found: {data_file}")

    output_file = os.path.join(output_dir, "meta_450k.csv")

    meta = MetaAssociation(
        input=f"{data_file},{data_file}",
        names="study1,study2",
        sample_sizes="100,100",
        output=output_file,
        method="fixed",
        data_type="EWAS",
        var="Methylation",
    )

    result = meta.run()
    assert result is not None
    assert os.path.exists(output_file)

    df = pd.read_csv(output_file)
    assert "CGID" in df.columns
    assert "COEF_FIXED" in df.columns
    assert "SE_FIXED" in df.columns
    assert "P_FIXED" in df.columns
    assert len(df) > 0


@pytest.mark.integration
def test_multiple_testing_corrections(sample_ewas_files, output_dir):
    output_file = os.path.join(output_dir, "corrections_test.csv")

    meta = MetaAssociation(
        input=",".join(sample_ewas_files),
        names="study1,study2,study3",
        sample_sizes="100,150,120",
        output=output_file,
        method="both",
        data_type="EWAS",
    )

    result = meta.run()
    assert result is not None
    assert os.path.exists(output_file)

    results_df = pd.read_csv(output_file)
    assert "FDR_FIXED" in results_df.columns
    assert "HOLM_FIXED" in results_df.columns
    assert "FDR_RANDOM" in results_df.columns
    assert "HOLM_RANDOM" in results_df.columns

    p_fixed = pd.to_numeric(results_df["P_FIXED"], errors="coerce")
    fdr_fixed = pd.to_numeric(results_df["FDR_FIXED"], errors="coerce")
    holm_fixed = pd.to_numeric(results_df["HOLM_FIXED"], errors="coerce")

    valid_idx = ~(p_fixed.isna() | fdr_fixed.isna() | holm_fixed.isna())
    if valid_idx.any():
        assert all(
            fdr_fixed[valid_idx] >= p_fixed[valid_idx]
        ), "FDR should be >= original p-values"
        assert all(
            holm_fixed[valid_idx] >= p_fixed[valid_idx]
        ), "Holm should be >= original p-values"
