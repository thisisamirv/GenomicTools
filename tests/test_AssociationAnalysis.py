#!/usr/bin/env python
import h5py
import numpy as np
import os
import pandas as pd
import pytest
import tempfile

from AssociationAnalysis import AssociationAnalysisLauncher
from utils.AliasUtils import AliasUtils
from utils.LoggingUtils import log

log.setup(level="DEBUG")


def get_hdf5_sample_ids(h5_file_path, analysis_type="GWAS"):
    with h5py.File(h5_file_path, "r") as h5f:
        metadata_key = AliasUtils.find_keys(h5f, "Metadata")
        if metadata_key is None:
            print(f"DEBUG: No metadata group found in {h5_file_path}")
            return []

        if analysis_type == "EWAS":
            sample_key = AliasUtils.find_keys(h5f[metadata_key], "SampleList")
        else:
            sample_key = AliasUtils.find_keys(h5f[metadata_key], "IID")

        if sample_key is None:
            print(
                f"DEBUG: Available keys in {metadata_key}: {list(h5f[metadata_key].keys())}"
            )
            return []

        sample_ids = h5f[metadata_key][sample_key][:]
        return [
            s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in sample_ids
        ]


@pytest.mark.unit
def test_invalid_model_type_raises_error():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            meta = hf.create_group("Metadata")
            meta.create_dataset("IID", data=["s1", "s2", "s3"], dtype="S10")
            meta.create_dataset("phenotype", data=[1.0, 2.0, 3.0], dtype=np.float64)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_output:
        pass

    with pytest.raises(ValueError, match="Invalid model type 'invalid'"):
        AssociationAnalysisLauncher(
            h5_file=tmp_hdf5.name,
            metadata=None,
            output=tmp_output.name,
            formula="phenotype ~ Genotype",
            model="invalid",
            analysis_type="Auto",
        )

    os.remove(tmp_hdf5.name)
    os.remove(tmp_output.name)


@pytest.mark.unit
def test_gwas_structure_and_formula_parsing():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            meta = hf.create_group("Metadata")
            meta.create_dataset("IID", data=[f"s{i}" for i in range(1, 7)], dtype="S50")
            meta.create_dataset("phenotype", data=[1, 2, 3, 4, 5, 6], dtype=np.float64)
            meta.create_dataset(
                "category", data=["A", "B", "A", "B", "A", "B"], dtype="S10"
            )

            chr21 = hf.create_group("CHR21")
            chr21.create_dataset("RSID", data=["rs1"], dtype="S20")
            chr21.create_dataset("BP", data=[100], dtype=np.int32)
            chr21.create_dataset("A1", data=["A"], dtype="S5")
            chr21.create_dataset("A2", data=["G"], dtype="S5")
            chr21.create_dataset(
                "Genotype", data=np.array([[0, 1, 2, 0, 1, 2]]), dtype=np.int8
            )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_output:
        pass

    analysis = AssociationAnalysisLauncher(
        h5_file=tmp_hdf5.name,
        metadata=None,
        output=tmp_output.name,
        formula="phenotype ~ Genotype + category",
        model="linear",
        analysis_type="GWAS",
    )

    assert analysis.formula_components["dependent_var"] == "phenotype"
    assert "category" in analysis.formula_components["covariates"]
    assert analysis.analysis_type == "GWAS"
    assert analysis.model == "linear"

    assert analysis.data_variable == "Genotype"
    assert analysis.dependent_var == "phenotype"
    os.remove(tmp_hdf5.name)
    os.remove(tmp_output.name)


@pytest.mark.unit
def test_ewas_requires_methylation_outcome():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_h5:
        pass

    with pytest.raises(
        ValueError, match="dependent variable must be the methylation data"
    ):
        AssociationAnalysisLauncher(
            h5_file=tmp_h5.name,
            metadata=None,
            output="out.csv",
            formula="age ~ Methylation + sex",
            model="linear",
            analysis_type="EWAS",
        )

    os.remove(tmp_h5.name)


@pytest.mark.unit
def test_gwas_rejects_genotype_as_outcome():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_h5:
        pass

    with pytest.raises(ValueError, match="dependent variable must be the phenotype"):
        AssociationAnalysisLauncher(
            h5_file=tmp_h5.name,
            metadata=None,
            output="out.csv",
            formula="Genotype ~ Phenotype + Sex",
            model="linear",
            analysis_type="GWAS",
        )

    os.remove(tmp_h5.name)


@pytest.mark.unit
def test_auto_analysis_type_detection():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_h5:
        with h5py.File(tmp_h5.name, "w") as hf:
            meta = hf.create_group("Metadata")
            meta.create_dataset("SampleList", data=["s1", "s2"], dtype="S10")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_meta:
        tmp_meta.write(b"sample_id,age,sex\ns1,30,M\ns2,40,F\n")
        tmp_meta.flush()

        launcher = AssociationAnalysisLauncher(
            h5_file=tmp_h5.name,
            metadata=tmp_meta.name,
            output="out.csv",
            formula="Methylation ~ age + sex",
            model="linear",
            analysis_type="Auto",
        )

        assert launcher.analysis_type == "EWAS"

    os.remove(tmp_h5.name)
    os.remove(tmp_meta.name)


@pytest.mark.integration
def test_ewas_var_parameter(data_dir, output_dir):
    h5_file = os.path.join(data_dir, "mcseq.h5")
    metadata_file = os.path.join(data_dir, "mcseq_metadata.csv")

    output_file = os.path.join(output_dir, "ewas_var_age.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=h5_file,
        metadata=metadata_file,
        output=output_file,
        formula="Methylation ~ Age + Sex",
        model="linear",
        analysis_type="EWAS",
        var="Age",
    )

    analysis.run()
    df = pd.read_csv(output_file)

    assert any("Age" in c for c in df.columns), "Age columns not found"
    assert all("Sex" not in c for c in df.columns), "Sex columns should not be present"

    output_file = os.path.join(output_dir, "ewas_var_both.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=h5_file,
        metadata=metadata_file,
        output=output_file,
        formula="Methylation ~ Age + Sex",
        model="linear",
        analysis_type="EWAS",
        var="Age,Sex",
    )

    analysis.run()
    df = pd.read_csv(output_file)

    assert any("Age" in c for c in df.columns), "Age columns not found"
    assert any("Sex" in c for c in df.columns), "Sex columns not found"

    output_file = os.path.join(output_dir, "ewas_var_invalid.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=h5_file,
        metadata=metadata_file,
        output=output_file,
        formula="Methylation ~ Age + Sex",
        model="linear",
        analysis_type="EWAS",
        var="NonExistent",
    )

    analysis.run()
    assert os.path.exists(output_file)


@pytest.mark.integration
def test_gwas_linear_association(data_dir, output_dir):
    h5_file = os.path.join(data_dir, "gen_data.h5")
    assert os.path.exists(h5_file), f"{h5_file} not found"

    output_file = os.path.join(output_dir, "gwas_linear.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=h5_file,
        metadata=None,
        output=output_file,
        formula="Phenotype ~ Genotype + Sex",
        model="logistic",
        analysis_type="GWAS",
    )

    result = analysis.run()
    assert result == output_file
    assert os.path.exists(output_file)

    df = pd.read_csv(output_file)
    assert not df.empty, "GWAS output is empty"
    assert "P" in df.columns or any("P" in c for c in df.columns)


@pytest.mark.integration
def test_gwas_linear_association_with_var(data_dir, output_dir):
    h5_file = os.path.join(data_dir, "gen_data.h5")
    assert os.path.exists(h5_file), f"{h5_file} not found"

    output_file = os.path.join(output_dir, "gwas_linear_with_var.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=h5_file,
        metadata=None,
        output=output_file,
        formula="Phenotype ~ Genotype + Sex",
        model="logistic",
        analysis_type="GWAS",
        var="Sex",
    )

    result = analysis.run()
    assert result == output_file
    assert os.path.exists(output_file)

    df = pd.read_csv(output_file)
    assert not df.empty, "GWAS output is empty"
    assert "P" in df.columns or any("P" in c for c in df.columns)


@pytest.mark.integration
def test_ewas_mcseq_linear_association(data_dir, output_dir):
    h5_file = os.path.join(data_dir, "mcseq.h5")
    metadata_file = os.path.join(data_dir, "mcseq_metadata.csv")
    assert os.path.exists(h5_file)
    assert os.path.exists(metadata_file)

    output_file = os.path.join(output_dir, "ewas_mcseq_linear.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=h5_file,
        metadata=metadata_file,
        output=output_file,
        formula="Methylation ~ Age + Sex",
        model="linear",
        analysis_type="EWAS",
        var="Age,Sex",
    )

    result = analysis.run()
    assert result == output_file
    assert os.path.exists(output_file)

    df = pd.read_csv(output_file)
    assert any("P_Age" in c for c in df.columns), "P_Age column not found"
    assert any("P_Sex" in c for c in df.columns), "P_Sex column not found"

    p_age_col = next((c for c in df.columns if "P_Age" in c), None)
    p_sex_col = next((c for c in df.columns if "P_Sex" in c), None)

    if p_age_col:
        assert not df[p_age_col].isna().all(), "All Age p-values are NaN"
    if p_sex_col:
        assert not df[p_sex_col].isna().all(), "All Sex p-values are NaN"


@pytest.mark.integration
def test_ewas_450k_linear_association(data_dir, output_dir):
    h5_file = os.path.join(data_dir, "450k.h5")
    assert os.path.exists(h5_file)

    output_file = os.path.join(output_dir, "ewas_450k_linear.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=h5_file,
        metadata=None,
        output=output_file,
        formula="Methylation ~ agebl + hiv",
        model="linear",
        analysis_type="EWAS",
    )

    result = analysis.run()
    assert result == output_file
    assert os.path.exists(output_file)

    df = pd.read_csv(output_file)
    assert not df.empty, "EWAS output is empty"
    assert "P" in df.columns or any("P" in c for c in df.columns)


@pytest.mark.integration
def test_missing_metadata_column():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            meta = hf.create_group("Metadata")
            meta.create_dataset("IID", data=[f"s{i}" for i in range(1, 6)], dtype="S20")
            meta.create_dataset("Phenotype", data=[1, 2, 3, 4, 5], dtype=np.float64)
            chr21 = hf.create_group("CHR21")
            chr21.create_dataset("RSID", data=["rs1"], dtype="S20")
            chr21.create_dataset(
                "Genotype", data=np.array([[0, 1, 2, 0, 1]]), dtype=np.int8
            )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_output:
        pass

    with pytest.raises(ValueError):
        analysis = AssociationAnalysisLauncher(
            h5_file=tmp_hdf5.name,
            metadata=None,
            output=tmp_output.name,
            formula="Phenotype ~ Genotype + covariate",
            model="linear",
            analysis_type="GWAS",
        )
        analysis.run()

    os.remove(tmp_hdf5.name)


@pytest.mark.integration
def test_invalid_formula(data_dir, output_dir):
    h5_file = os.path.join(data_dir, "gen_data.h5")
    assert os.path.exists(h5_file)

    samples = get_hdf5_sample_ids(h5_file, "GWAS")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_metadata:
        metadata_df = pd.DataFrame(
            {"Phenotype": np.arange(1, len(samples) + 1)}, index=samples
        )
        metadata_df.index.name = "sample_id"
        metadata_df.to_csv(tmp_metadata.name)

    output_file = os.path.join(output_dir, "invalid_formula.csv")
    with pytest.raises(ValueError):
        analysis = AssociationAnalysisLauncher(
            h5_file=h5_file,
            metadata=tmp_metadata.name,
            output=output_file,
            formula="Phenotype ~ Genotype + NonexistentCovariate",
            model="linear",
            analysis_type="GWAS",
        )
        analysis.run()

    os.remove(tmp_metadata.name)


@pytest.mark.integration
def test_low_variance_data(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            meta = hf.create_group("Metadata")
            meta.create_dataset(
                "IID", data=[f"s{i}" for i in range(1, 11)], dtype="S20"
            )
            chr21 = hf.create_group("CHR21")
            chr21.create_dataset("RSID", data=["rs1"], dtype="S20")
            chr21.create_dataset("Genotype", data=np.zeros((10, 1), dtype=np.int8))

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_metadata:
        metadata_df = pd.DataFrame(
            {"phenotype": np.arange(1, 11)}, index=[f"s{i}" for i in range(1, 11)]
        )
        metadata_df.index.name = "sample_id"
        metadata_df.to_csv(tmp_metadata.name)

    output_file = os.path.join(output_dir, "low_variance.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=tmp_hdf5.name,
        metadata=tmp_metadata.name,
        output=output_file,
        formula="phenotype ~ Genotype",
        model="linear",
        analysis_type="GWAS",
    )

    result = analysis.run()
    assert result == output_file

    assert os.path.exists(
        output_file
    ), "Output file should be created even for empty results"

    df = pd.read_csv(output_file)
    assert isinstance(df, pd.DataFrame), "Output should be a valid DataFrame"

    expected_cols = [
        "RSID",
        "CHR",
        "BP",
        "EFFECT_ALLELE",
        "N",
        "EAF",
        "MAF",
        "COEF",
        "COEF_SE",
        "U_STAT",
        "U_SE",
        "P",
        "R2",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Column '{col}' should be present in output"

    assert len(df) == 0, "Should have 0 rows after filtering monomorphic variants"

    log.info("✓ Test passed: Low-variance data handled gracefully with empty output")

    os.remove(tmp_metadata.name)
    os.remove(tmp_hdf5.name)


@pytest.mark.integration
def test_gwas_small_dataset(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            meta = hf.create_group("Metadata")
            meta.create_dataset("IID", data=["s1", "s2", "s3", "s4"], dtype="S20")
            chr21 = hf.create_group("CHR21")
            chr21.create_dataset("RSID", data=["rs1"], dtype="S20")
            chr21.create_dataset("BP", data=[100], dtype=np.int32)
            chr21.create_dataset("A1", data=["A"], dtype="S5")
            chr21.create_dataset("A2", data=["G"], dtype="S5")
            chr21.create_dataset(
                "Genotype", data=np.array([[0, 1, 2, 0]]).T, dtype=np.int8
            )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_metadata:
        metadata_df = pd.DataFrame(
            {"phenotype": [1.0, 2.0, 3.0, 1.5]}, index=["s1", "s2", "s3", "s4"]
        )
        metadata_df.index.name = "sample_id"
        metadata_df.to_csv(tmp_metadata.name)

    output_file = os.path.join(output_dir, "small_gwas.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=tmp_hdf5.name,
        metadata=tmp_metadata.name,
        output=output_file,
        formula="phenotype ~ Genotype",
        model="linear",
        analysis_type="GWAS",
    )

    result = analysis.run()
    assert result == output_file
    assert os.path.exists(output_file)

    df = pd.read_csv(output_file)
    assert "P" in df.columns or any("P" in c for c in df.columns)
    assert not df["P"].isna().all()

    os.remove(tmp_hdf5.name)
    os.remove(tmp_metadata.name)


@pytest.mark.integration
def test_formula_parsing_integration(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            meta = hf.create_group("Metadata")
            meta.create_dataset("IID", data=[f"s{i}" for i in range(1, 9)], dtype="S20")
            meta.create_dataset(
                "outcome", data=[1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64
            )
            meta.create_dataset(
                "age", data=[25, 30, 35, 40, 45, 50, 55, 60], dtype=np.float64
            )
            chr21 = hf.create_group("CHR21")
            chr21.create_dataset("RSID", data=["rs1"], dtype="S20")
            chr21.create_dataset("BP", data=[100], dtype=np.int32)
            chr21.create_dataset("A1", data=["A"], dtype="S5")
            chr21.create_dataset("A2", data=["G"], dtype="S5")
            chr21.create_dataset(
                "Genotype", data=np.array([[0, 1, 2, 1, 0, 2, 1, 0]]).T, dtype=np.int8
            )

    output_file = os.path.join(output_dir, "formula_parsing.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=tmp_hdf5.name,
        metadata=None,
        output=output_file,
        formula="outcome ~ Genotype + age",
        model="linear",
        analysis_type="GWAS",
    )

    assert analysis.formula_components["dependent_var"] == "outcome"
    assert analysis.formula_components["data_variable"] == "Genotype"
    assert "age" in analysis.formula_components["covariates"]

    result = analysis.run()
    assert result == output_file
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert not df.empty
    assert "P" in df.columns or any("P" in c for c in df.columns)

    os.remove(tmp_hdf5.name)


@pytest.mark.integration
def test_interaction_terms_large_dataset(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        n_samples = 30
        with h5py.File(tmp_hdf5.name, "w") as hf:
            meta = hf.create_group("Metadata")
            meta.create_dataset(
                "IID", data=[f"s{i}" for i in range(1, n_samples + 1)], dtype="S20"
            )
            np.random.seed(42)
            ages = np.linspace(20, 80, n_samples)
            genotypes = np.tile([0, 1, 2], n_samples // 3 + 1)[:n_samples]
            var1 = 0.1 * ages
            var2 = 0.5 * genotypes
            var3 = 0.01 * ages * genotypes
            var4 = np.random.normal(0, 2, n_samples)
            var1_2 = var1 + var2
            var3_4 = var3 + var4
            outcomes = var1_2 + var3_4
            meta.create_dataset("outcome", data=outcomes, dtype=np.float64)
            meta.create_dataset("age", data=ages, dtype=np.float64)
            chr21 = hf.create_group("CHR21")
            chr21.create_dataset("RSID", data=["rs1"], dtype="S20")
            chr21.create_dataset(
                "Genotype", data=genotypes.reshape(-1, 1), dtype=np.int8
            )

    output_file = os.path.join(output_dir, "interaction_terms.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=tmp_hdf5.name,
        metadata=None,
        output=output_file,
        formula="outcome ~ Genotype + age + Genotype:age",
        model="linear",
        analysis_type="GWAS",
    )

    assert analysis.formula_components["dependent_var"] == "outcome"
    assert analysis.formula_components["data_variable"] == "Genotype"
    assert "age" in analysis.formula_components["covariates"]

    result = analysis.run()
    assert result == output_file
    assert os.path.exists(output_file)

    df = pd.read_csv(output_file)
    assert not df.empty
    assert "P" in df.columns or any("P" in c for c in df.columns)

    os.remove(tmp_hdf5.name)


@pytest.mark.integration
def test_interaction_parsing_simple(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            meta = hf.create_group("Metadata")
            n_samples = 50
            sample_ids = [f"s{i}" for i in range(1, n_samples + 1)]
            meta.create_dataset("IID", data=sample_ids, dtype="S20")

            np.random.seed(123)
            ages = np.random.uniform(30, 70, n_samples)
            genotypes = np.random.choice([0, 1, 2], n_samples, p=[0.25, 0.5, 0.25])
            var1 = 0.5 * ages
            var2 = 2 * genotypes
            var3 = 0.01 * ages * genotypes
            var4 = np.random.normal(0, 1, n_samples)
            var1_2 = var1 + var2
            var3_4 = var3 + var4
            outcomes = 10 + var1_2 + var3_4

            meta.create_dataset("outcome", data=outcomes, dtype=np.float64)
            meta.create_dataset("age", data=ages, dtype=np.float64)
            chr21 = hf.create_group("CHR21")
            chr21.create_dataset("RSID", data=["rs_interaction"], dtype="S20")
            chr21.create_dataset(
                "Genotype", data=genotypes.reshape(-1, 1), dtype=np.int8
            )

    output_file = os.path.join(output_dir, "interaction_parsing_simple.csv")

    analysis_main = AssociationAnalysisLauncher(
        h5_file=tmp_hdf5.name,
        metadata=None,
        output=output_file,
        formula="outcome ~ Genotype + age",
        model="linear",
        analysis_type="GWAS",
    )

    analysis_main.run()
    assert os.path.exists(output_file)

    df_main = pd.read_csv(output_file)
    assert not df_main.empty
    assert "P" in df_main.columns or any("P" in c for c in df_main.columns)

    analysis_int = AssociationAnalysisLauncher(
        h5_file=tmp_hdf5.name,
        metadata=None,
        output=output_file.replace(".csv", "_int.csv"),
        formula="outcome ~ Genotype + age + Genotype:age",
        model="linear",
        analysis_type="GWAS",
    )

    assert analysis_int.formula_components["dependent_var"] == "outcome"
    assert analysis_int.formula_components["data_variable"] == "Genotype"
    assert "age" in analysis_int.formula_components["covariates"]

    print("✓ Interaction term parsing successful")
    os.remove(tmp_hdf5.name)


@pytest.mark.integration
def test_gwas_logistic_odds_ratios(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            meta = hf.create_group("Metadata")
            meta.create_dataset(
                "IID", data=["s1", "s2", "s3", "s4", "s5", "s6"], dtype="S20"
            )
            meta.create_dataset("case_control", data=[0, 1, 0, 1, 0, 1], dtype=np.int32)
            chr21 = hf.create_group("CHR21")
            chr21.create_dataset("RSID", data=["rs1"], dtype="S20")
            chr21.create_dataset(
                "Genotype", data=np.array([[0, 1, 2, 1, 0, 2]]).T, dtype=np.int8
            )

    output_file = os.path.join(output_dir, "logistic_odds_ratios.csv")
    analysis = AssociationAnalysisLauncher(
        h5_file=tmp_hdf5.name,
        metadata=None,
        output=output_file,
        formula="case_control ~ Genotype",
        model="logistic",
        analysis_type="GWAS",
    )

    result = analysis.run()
    assert result == output_file
    assert os.path.exists(output_file)

    df = pd.read_csv(output_file)
    assert not df.empty
    assert any("P" in col for col in df.columns)
    assert any("OR" in col.upper() for col in df.columns), "Expected odds ratio column"

    os.remove(tmp_hdf5.name)
