#!/usr/bin/env python
import h5py
import numpy as np
import os
import pandas as pd
import pytest
import shutil
import tempfile
import time
from GenomicPCA import GenomicPCA
from utils.AliasUtils import AliasUtils
from utils.H5Utils import CachedH5Utils
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.fixture
def mock_methylation_h5():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    n_samples = 10
    n_probes = 100
    sample_ids = [f"sample_{i}" for i in range(n_samples)]

    with h5py.File(temp_path, "w") as h5f:
        h5f.create_group("metadata")
        h5f["metadata"].create_dataset(
            "sampleList", data=np.array(sample_ids, dtype="S10")
        )

        for chr_num in range(1, 3):
            chr_group = f"chr{chr_num}"
            h5f.create_group(chr_group)

            betas = np.random.random((n_samples, n_probes))
            h5f[chr_group].create_dataset("beta", data=betas)

            positions = np.arange(1, n_probes + 1) * 1000
            h5f[chr_group].create_dataset("position", data=positions)

            probe_ids = [f"probe_chr{chr_num}_{i}" for i in range(n_probes)]
            h5f[chr_group].create_dataset(
                "probeID", data=np.array(probe_ids, dtype="S20")
            )

    yield temp_path

    os.unlink(temp_path)


@pytest.fixture
def mock_genotype_h5():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    n_samples = 10
    n_variants = 100
    sample_ids = [f"sample_{i}" for i in range(n_samples)]

    with h5py.File(temp_path, "w") as h5f:
        h5f.create_group("Metadata")
        h5f["Metadata"].create_dataset(
            "sample_list", data=np.array(sample_ids, dtype="S10")
        )

        for chr_num in range(1, 3):
            chr_group = f"chr{chr_num}"
            h5f.create_group(chr_group)

            genotypes = np.random.randint(0, 3, size=(n_variants, n_samples))
            h5f[chr_group].create_dataset("Genotype", data=genotypes)

            positions = np.arange(1, n_variants + 1) * 1000
            h5f[chr_group].create_dataset("position", data=positions)

            snp_ids = [f"rs{chr_num}_{i}" for i in range(n_variants)]
            h5f[chr_group].create_dataset("rsid", data=np.array(snp_ids, dtype="S20"))

    yield temp_path

    os.unlink(temp_path)


@pytest.fixture
def mock_alternative_h5():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = f.name

    n_samples = 8
    n_probes = 80
    sample_ids = [f"sample_{i}" for i in range(n_samples)]

    with h5py.File(temp_path, "w") as h5f:
        h5f.create_dataset("samples", data=np.array(sample_ids, dtype="S10"))

        for chr_num in range(1, 3):
            chr_group = f"chr{chr_num}"
            h5f.create_group(chr_group)

            betas = np.random.random((n_samples, n_probes))
            h5f[chr_group].create_dataset("Methylation", data=betas)

    yield temp_path

    os.unlink(temp_path)


@pytest.fixture
def mock_residuals_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        temp_path = f.name

    n_samples = 10
    n_features = 50

    data = {"IID": [f"sample_{i}" for i in range(n_samples)]}

    for i in range(n_features):
        data[f"feature_{i}"] = np.random.normal(0, 1, n_samples)

    df = pd.DataFrame(data)
    df.to_csv(temp_path, index=False)

    yield temp_path

    os.unlink(temp_path)


@pytest.fixture
def output_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def real_methylation_h5():
    return os.path.join(os.path.dirname(__file__), "data", "450k.h5")


@pytest.fixture
def real_genotype_h5():
    return os.path.join(os.path.dirname(__file__), "data", "gen_data.h5")


@pytest.mark.unit
def test_sample_list_detection_methylation(mock_methylation_h5):
    with h5py.File(mock_methylation_h5, "r") as h5f:
        metadata_aliases = AliasUtils.generate_metadata_aliases()

        metadata_path = None
        for meta_alias in metadata_aliases:
            if meta_alias in h5f:
                metadata_path = meta_alias
                break

        assert metadata_path == "metadata", "Failed to find metadata group"

        sample_list_aliases = AliasUtils.generate_samplelist_aliases()
        sample_path = None
        for sample_alias in sample_list_aliases:
            if sample_alias in h5f[metadata_path]:
                sample_path = sample_alias
                break

        assert sample_path == "sampleList", "Failed to find sample list"

        samples = [s.decode("utf-8") for s in h5f[metadata_path][sample_path]]
        assert len(samples) == 10, f"Expected 10 samples, got {len(samples)}"


@pytest.mark.unit
def test_sample_list_detection_genotype(mock_genotype_h5):
    with h5py.File(mock_genotype_h5, "r") as h5f:
        metadata_aliases = AliasUtils.generate_metadata_aliases()

        metadata_path = None
        for meta_alias in metadata_aliases:
            if meta_alias in h5f:
                metadata_path = meta_alias
                break

        assert metadata_path == "Metadata", "Failed to find metadata group"

        sample_list_aliases = AliasUtils.generate_samplelist_aliases()
        sample_path = None
        for sample_alias in sample_list_aliases:
            if sample_alias in h5f[metadata_path]:
                sample_path = sample_alias
                break

        assert sample_path == "sample_list", "Failed to find sample list"

        samples = [s.decode("utf-8") for s in h5f[metadata_path][sample_path]]
        assert len(samples) == 10, f"Expected 10 samples, got {len(samples)}"


@pytest.mark.unit
def test_chromosome_detection(mock_methylation_h5):
    with h5py.File(mock_methylation_h5, "r") as h5f:
        h5_utils = CachedH5Utils(h5f)
        chromosomes = h5_utils.get_chromosomes()

        assert set(chromosomes) == {
            "chr1",
            "chr2",
        }, f"Expected chr1, chr2; got {chromosomes}"


@pytest.mark.unit
def test_data_field_detection_methylation(mock_methylation_h5):
    with h5py.File(mock_methylation_h5, "r") as h5f:
        chr_group = "chr1"

        beta_field = AliasUtils.find_keys(list(h5f[chr_group].keys()), "Methylation")

        if not beta_field:
            for field in [
                "betas",
                "beta",
                "Beta",
                "BETA",
                "Methylation",
                "Methylation",
            ]:
                if field in h5f[chr_group]:
                    beta_field = field
                    break

        assert beta_field == "beta", f"Expected beta field, got {beta_field}"

        assert h5f[chr_group][beta_field].shape == (
            10,
            100,
        ), f"Expected shape (10, 100), got {h5f[chr_group][beta_field].shape}"


@pytest.mark.unit
def test_data_field_detection_genotype(mock_genotype_h5):
    with h5py.File(mock_genotype_h5, "r") as h5f:
        chr_group = "chr1"

        genotype_field = AliasUtils.find_keys(list(h5f[chr_group].keys()), "Genotype")

        if not genotype_field:
            for field in ["genotypes", "Genotype", "Genotypes", "Genotype", "GT", "gt"]:
                if field in h5f[chr_group]:
                    genotype_field = field
                    break

        assert (
            genotype_field == "Genotype"
        ), f"Expected genotype field, got {genotype_field}"

        assert h5f[chr_group][genotype_field].shape == (
            100,
            10,
        ), f"Expected shape (100, 10), got {h5f[chr_group][genotype_field].shape}"


@pytest.mark.unit
def test_alternative_structure_detection(mock_alternative_h5):
    with h5py.File(mock_alternative_h5, "r") as h5f:
        sample_key = AliasUtils.find_keys(list(h5f.keys()), "SampleList")
        assert sample_key == "samples", f"Expected to find 'samples', got {sample_key}"

        chr_group = "chr1"
        beta_field = AliasUtils.find_keys(list(h5f[chr_group].keys()), "Methylation")
        assert (
            beta_field == "Methylation"
        ), f"Expected to find 'Methylation', got {beta_field}"


@pytest.mark.unit
def test_residuals_column_detection(mock_residuals_csv):
    df = pd.read_csv(mock_residuals_csv)

    sample_id_col = AliasUtils.find_keys(df.columns, "IID")
    assert sample_id_col == "IID", f"Expected to find 'IID', got {sample_id_col}"

    assert (
        len(df.columns) == 51
    ), f"Expected 51 columns (IID + 50 features), got {len(df.columns)}"

    for col in df.columns:
        if col != "IID":
            assert np.issubdtype(
                df[col].dtype, np.number
            ), f"Column {col} is not numeric: {df[col].dtype}"


@pytest.mark.unit
def test_unsupported_data_type(mock_methylation_h5, output_dir):
    output_file = os.path.join(output_dir, "error_pca.csv")

    GenomicPCA(
        input=mock_methylation_h5,
        output=output_file,
        data_type="unsupported",
    )

    assert not os.path.exists(output_file), "Output file should not be created on error"


@pytest.mark.unit
def test_missing_input_file(output_dir):
    output_file = os.path.join(output_dir, "error_pca.csv")

    GenomicPCA(input="nonexistent_file.h5", output=output_file)

    assert not os.path.exists(output_file), "Output file should not be created on error"


@pytest.mark.integration
def test_methylation_pca_integration(mock_methylation_h5, output_dir):
    output_file = os.path.join(output_dir, "methylation_pca.csv")

    GenomicPCA(
        input=mock_methylation_h5,
        output=output_file,
        n_components=3,
        data_type="Methylation",
        batch_size=50,
    )

    assert os.path.exists(output_file), "Output file not created"

    variance_file = os.path.join(output_dir, "methylation_pca_ExplainedVariance.csv")
    assert os.path.exists(variance_file), "Variance file not created"

    pca_results = pd.read_csv(output_file, index_col=0)
    assert pca_results.shape == (
        10,
        3,
    ), f"Expected shape (10, 3), got {pca_results.shape}"
    assert all(
        col in pca_results.columns for col in ["PC1", "PC2", "PC3"]
    ), "Missing PC columns"

    variance_results = pd.read_csv(variance_file)
    assert variance_results.shape == (
        3,
        2,
    ), f"Expected shape (3, 2), got {variance_results.shape}"
    assert (
        "ExplainedVariance" in variance_results.columns
    ), "Missing ExplainedVariance column"

    assert (
        0.0 < variance_results["ExplainedVariance"].sum() <= 1.0
    ), f"Explained variance sum: {variance_results['ExplainedVariance'].sum()}"


@pytest.mark.integration
def test_genotype_pca_integration(mock_genotype_h5, output_dir):
    output_file = os.path.join(output_dir, "genotype_pca.csv")

    GenomicPCA(
        input=mock_genotype_h5,
        output=output_file,
        n_components=3,
        data_type="Genotype",
        batch_size=50,
    )

    assert os.path.exists(output_file), "Output file not created"

    variance_file = os.path.join(output_dir, "genotype_pca_ExplainedVariance.csv")
    assert os.path.exists(variance_file), "Variance file not created"

    pca_results = pd.read_csv(output_file, index_col=0)
    assert (
        pca_results.shape[0] > 0
    ), f"No samples in output, got shape {pca_results.shape}"
    assert (
        pca_results.shape[1] > 0
    ), f"No components in output, got shape {pca_results.shape}"

    variance_results = pd.read_csv(variance_file)
    assert (
        "ExplainedVariance" in variance_results.columns
    ), "Missing ExplainedVariance column"

    assert all(
        0 <= v <= 1 for v in variance_results["ExplainedVariance"]
    ), "Invalid variance values"
    assert (
        0 < variance_results["ExplainedVariance"].sum() <= 1.0
    ), f"Invalid variance sum: {variance_results['ExplainedVariance'].sum()}"


@pytest.mark.integration
def test_residuals_pca_integration(mock_residuals_csv, output_dir):
    output_file = os.path.join(output_dir, "residuals_pca.csv")

    GenomicPCA(
        input=mock_residuals_csv, output=output_file, n_components=3, residuals=True
    )

    assert os.path.exists(output_file), "Output file not created"

    variance_file = os.path.join(output_dir, "residuals_pca_ExplainedVariance.csv")
    assert os.path.exists(variance_file), "Variance file not created"

    pca_results = pd.read_csv(output_file, index_col=0)
    assert (
        pca_results.shape[0] == 10
    ), f"Expected 10 samples, got {pca_results.shape[0]}"
    assert (
        pca_results.shape[1] <= 3
    ), f"Expected ≤3 components, got {pca_results.shape[1]}"

    variance_results = pd.read_csv(variance_file)
    assert (
        variance_results.shape[1] == 2
    ), f"Expected 2 columns, got {variance_results.shape[1]}"
    assert (
        "ExplainedVariance" in variance_results.columns
    ), "Missing ExplainedVariance column"

    assert (
        0 < variance_results["ExplainedVariance"].sum() <= 1.0
    ), f"Invalid variance: {variance_results['ExplainedVariance'].sum()}"


@pytest.mark.integration
def test_alternative_structure_pca(mock_alternative_h5, output_dir):
    output_file = os.path.join(output_dir, "alternative_pca.csv")

    GenomicPCA(
        input=mock_alternative_h5,
        output=output_file,
        n_components=2,
        data_type="Methylation",
        batch_size=40,
    )

    assert os.path.exists(output_file), "Output file not created"

    pca_results = pd.read_csv(output_file, index_col=0)
    assert pca_results.shape == (
        8,
        2,
    ), f"Expected shape (8, 2), got {pca_results.shape}"
    assert all(
        col in pca_results.columns for col in ["PC1", "PC2"]
    ), "Missing PC columns"


@pytest.mark.integration
def test_batch_size_effect(mock_methylation_h5, output_dir):
    output_small = os.path.join(output_dir, "small_batch.csv")
    start_time = time.time()
    GenomicPCA(input=mock_methylation_h5, output=output_small, batch_size=10)
    small_batch_time = time.time() - start_time

    output_large = os.path.join(output_dir, "large_batch.csv")
    start_time = time.time()
    GenomicPCA(input=mock_methylation_h5, output=output_large, batch_size=1000)
    large_batch_time = time.time() - start_time

    small_results = pd.read_csv(output_small, index_col=0)
    large_results = pd.read_csv(output_large, index_col=0)

    pd.testing.assert_frame_equal(small_results, large_results, rtol=1e-3, atol=1e-3)

    print(f"\nSmall batch time: {small_batch_time:.4f}s")
    print(f"Large batch time: {large_batch_time:.4f}s")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "data", "450k.h5")),
    reason="Real Methylation data file not found",
)
def test_real_methylation_data(real_methylation_h5, output_dir):
    output_file = os.path.join(output_dir, "real_methyl_pca.csv")

    GenomicPCA(
        input=real_methylation_h5,
        output=output_file,
        n_components=5,
        data_type="Methylation",
        batch_size=1000,
    )

    assert os.path.exists(output_file), "Output file not created"

    variance_file = os.path.join(output_dir, "real_methyl_pca_ExplainedVariance.csv")
    assert os.path.exists(variance_file), "Variance file not created"

    pca_results = pd.read_csv(output_file, index_col=0)
    assert (
        len(pca_results.columns) == 5
    ), f"Expected 5 PC columns, got {len(pca_results.columns)}"
    assert all(
        f"PC{i + 1}" in pca_results.columns for i in range(5)
    ), "Missing PC columns"

    variance_results = pd.read_csv(variance_file)
    assert (
        len(variance_results) == 5
    ), f"Expected 5 components in variance file, got {len(variance_results)}"
    assert (
        "ExplainedVariance" in variance_results.columns
    ), "Missing ExplainedVariance column"

    print(f"\nReal Methylation data: {len(pca_results)} samples processed")
    print(
        f"Total variance explained: {variance_results['ExplainedVariance'].sum() * 100:.2f}%"
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "data", "gen_data.h5")),
    reason="Real genotype data file not found",
)
def test_real_genotype_data(real_genotype_h5, output_dir):
    output_file = os.path.join(output_dir, "real_geno_pca.csv")

    GenomicPCA(
        input=real_genotype_h5,
        output=output_file,
        n_components=5,
        data_type="Genotype",
        batch_size=1000,
    )

    assert os.path.exists(output_file), "Output file not created"

    variance_file = os.path.join(output_dir, "real_geno_pca_ExplainedVariance.csv")
    assert os.path.exists(variance_file), "Variance file not created"

    pca_results = pd.read_csv(output_file, index_col=0)
    assert (
        len(pca_results.columns) == 5
    ), f"Expected 5 PC columns, got {len(pca_results.columns)}"
    assert all(
        f"PC{i + 1}" in pca_results.columns for i in range(5)
    ), "Missing PC columns"

    variance_results = pd.read_csv(variance_file)
    assert (
        len(variance_results) == 5
    ), f"Expected 5 components in variance file, got {len(variance_results)}"
    assert (
        "ExplainedVariance" in variance_results.columns
    ), "Missing ExplainedVariance column"

    print(f"\nReal genotype data: {len(pca_results)} samples processed")
    print(
        f"Total variance explained: {variance_results['ExplainedVariance'].sum() * 100:.2f}%"
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "data", "450k.h5")),
    reason="Real Methylation data file not found",
)
def test_pca_on_subset_chromosomes(real_methylation_h5, output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_h5 = f.name

    try:
        with h5py.File(real_methylation_h5, "r") as src, h5py.File(temp_h5, "w") as dst:
            if "metadata" in src:
                src.copy("metadata", dst)

            for chr_name in ["chr1", "chr22"]:
                if chr_name in src:
                    src.copy(chr_name, dst)

        output_file = os.path.join(output_dir, "subset_pca.csv")
        GenomicPCA(
            input=temp_h5,
            output=output_file,
            n_components=3,
            data_type="Methylation",
            batch_size=1000,
        )

        assert os.path.exists(output_file), "Output file not created"
        pca_results = pd.read_csv(output_file, index_col=0)
        print(f"\nSubset chromosome PCA: {len(pca_results)} samples processed")

    finally:
        if os.path.exists(temp_h5):
            os.unlink(temp_h5)
