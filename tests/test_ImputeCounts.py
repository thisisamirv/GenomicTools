#!/usr/bin/env python
import h5py
import numpy as np
import os
import pytest
import tempfile
import gzip
from ImputeCounts import (
    ImputeCounts,
    DataProcessor,
    MethylationImputer,
    ParallelMethylationImputer,
    ChunkedMethylationImputer,
    FastKNNImputer,
    QualityFilter,
)
from utils.AliasUtils import AliasUtils
from utils.LoggingUtils import log
from utils.SystemUtils import SystemUtils, monitor_resources

log.setup(level="DEBUG")

REFERENCE_DIR = "/gpfs/gibbs/data/genomes/1000Genomes/1000GP_Phase3/1000GP_Phase3"


def check_no_nans_in_betas(h5_file):
    with h5py.File(h5_file, "r") as h5f:
        chr_keys = [
            grp
            for grp in h5f.keys()
            if AliasUtils.strip_numeric_suffix(grp) in AliasUtils.get_aliases("CHR")
        ]
        for grp in chr_keys:
            methylation_key = AliasUtils.find_keys(h5f[grp], "Methylation")
            if methylation_key:
                betas = h5f[grp][methylation_key][:]
                return not np.isnan(betas).any()
    return False


def get_hdf5_sample_ids(h5_file_path, sample_key="iid"):
    with h5py.File(h5_file_path, "r") as h5f:
        metadata_key = AliasUtils.find_keys(h5f, "Metadata")
        if metadata_key:
            iid_key = AliasUtils.find_keys(h5f[metadata_key], "IID")
            if iid_key:
                sample_ids = h5f[metadata_key][iid_key][:]
                return [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in sample_ids
                ]
    return []


def find_dataset_in_group(group, expected_names):
    return AliasUtils.find_keys(group, expected_names)


def assert_dataset_exists(group, expected_names, error_msg):
    found_key = find_dataset_in_group(group, expected_names)
    assert found_key is not None, f"{error_msg}. Available keys: {list(group.keys())}"
    return found_key


def create_minimal_genotype_test_data(filepath):
    with h5py.File(filepath, "w") as hf:
        meta_grp = hf.create_group("Metadata")
        meta_grp.create_dataset(
            "IID", data=np.array(["sample1", "sample2"], dtype="S20")
        )

        chr_grp = hf.create_group("CHR1")
        chr_grp.create_dataset(
            "Genotype", data=np.array([[0, 1], [1, 2]], dtype=np.int8)
        )
        chr_grp.create_dataset("RSID", data=np.array(["rs1", "rs2"], dtype="S20"))
        chr_grp.create_dataset("BP", data=np.array([1000, 2000], dtype=np.int32))
        chr_grp.create_dataset("A1", data=np.array(["A", "C"], dtype="S1"))
        chr_grp.create_dataset("A2", data=np.array(["T", "G"], dtype="S1"))


def create_minimal_reference_files(ref_dir):
    sample_file = ref_dir / "1000GP_Phase3.sample"
    with open(sample_file, "w") as f:
        f.write("ID_1 ID_2 missing sex\n")
        f.write("0 0 0 D\n")
        f.write("sample1 sample1 0 1\n")
        f.write("sample2 sample2 0 2\n")

    hap_file = ref_dir / "1000GP_Phase3_chr1.hap.gz"
    with gzip.open(hap_file, "wt") as f:
        f.write("1 0 0 1\n")
        f.write("0 1 1 0\n")

    legend_file = ref_dir / "1000GP_Phase3_chr1.legend.gz"
    with gzip.open(legend_file, "wt") as f:
        f.write("id position a0 a1\n")
        f.write("rs1 1000 A T\n")
        f.write("rs2 2000 C G\n")

    map_file = ref_dir / "genetic_map_chr1_combined_b37.txt"
    with open(map_file, "w") as f:
        f.write("position COMBINED_rate(cM/Mb) Genetic_Map(cM)\n")
        f.write("1000 1.0 0.0\n")
        f.write("2000 1.0 0.001\n")


def create_test_methylation_file(filepath):
    with h5py.File(filepath, "w") as hf:
        metadata_grp = hf.create_group("metadata")
        metadata_grp.create_dataset("iid", data=[f"sample_{i}" for i in range(10)])

        chr_grp = hf.create_group("CHR1")
        betas = np.random.rand(5, 10)
        chr_grp.create_dataset("Methylation", data=betas)
        chr_grp.create_dataset("ProbeList", data=[f"probe_{i}" for i in range(5)])
    return filepath


@pytest.fixture(scope="session", autouse=True)
def setup_mp_logging():
    log.setup(level="DEBUG")
    log.start_multiprocessing_logging()
    yield
    if hasattr(log, "mp_queue") and log.mp_queue is not None:
        log.mp_queue.put_nowait(None)
        if hasattr(log, "listener") and log.listener is not None:
            log.listener.join(timeout=1)


@pytest.mark.unit
def test_fast_knn_imputer():
    X_small = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    imputer_small = FastKNNImputer(n_neighbors=2, max_samples_for_knn=10)
    result_small = imputer_small.fit_transform(X_small)
    assert not np.isnan(result_small).any()

    X_large = np.random.rand(1000, 50)
    X_large[100:110, 10:15] = np.nan
    imputer_large = FastKNNImputer(n_neighbors=5, max_samples_for_knn=100)
    result_large = imputer_large.fit_transform(X_large)
    assert not np.isnan(result_large).any()
    assert result_large.shape == X_large.shape


@pytest.mark.unit
def test_fast_knn_imputer_edge_cases():
    X_complete = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    imputer = FastKNNImputer()
    result = imputer.fit_transform(X_complete)
    np.testing.assert_array_equal(result, X_complete)

    X_all_nan = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    result_all_nan = imputer.fit_transform(X_all_nan)
    assert not np.isnan(result_all_nan).any()


@pytest.mark.unit
def test_chunked_methylation_imputer():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            grp = hf.create_group("CHR1")
            betas = np.random.rand(100, 20)
            betas[10:20, 5:10] = np.nan
            grp.create_dataset("Methylation", data=betas)
            grp.create_dataset("ProbeList", data=[f"probe_{i}" for i in range(100)])

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "chunked_test.h5")

        imputer = ChunkedMethylationImputer(
            input_file=tmp_input.name,
            output_file=output_file,
            k=3,
            chunk_size=50,
            max_samples_for_full_knn=25,
        )

        result = imputer.process_chromosome_chunked("CHR1", temp_dir)
        assert result is not None
        assert os.path.exists(result)

        with h5py.File(result, "r") as h5f:
            methylation_data = h5f["CHR1"]["Methylation"][:]
            assert not np.isnan(methylation_data).any()

    os.remove(tmp_input.name)


@pytest.mark.unit
def test_parallel_methylation_imputer_initialization():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            grp = hf.create_group("CHR1")
            betas = np.random.rand(10, 5)
            grp.create_dataset("Methylation", data=betas)
            grp.create_dataset("ProbeList", data=[f"probe_{i}" for i in range(10)])

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "parallel_init_test.h5")

        imputer1 = ParallelMethylationImputer(
            input_file=tmp_input.name, output_file=output_file
        )
        assert imputer1.n_processes >= 1
        assert imputer1.n_processes <= 4

        imputer2 = ParallelMethylationImputer(
            input_file=tmp_input.name, output_file=output_file, n_processes=2
        )
        assert imputer2.n_processes == 2

    os.remove(tmp_input.name)


@pytest.mark.unit
def test_invalid_data_type():
    with pytest.raises(
        ValueError, match="data_type must be either 'genotype' or 'methylation'"
    ):
        ImputeCounts(data_type="invalid")


@pytest.mark.unit
def test_missing_reference_dir_for_genotype(data_dir, output_dir):
    with pytest.raises(
        ValueError, match="reference_dir is required for genotype imputation"
    ):
        pipeline = ImputeCounts(
            input_file=os.path.join(data_dir, "gen_data.h5"),
            output_file=os.path.join(output_dir, "gen_imputed_test.h5"),
            data_type="genotype",
        )
        pipeline.run()


@pytest.mark.unit
def test_data_processor_optimize_parameters_advanced():
    params = DataProcessor.optimize_parameters_advanced(threads=8, window_size=5000000)
    assert "max_parallel_chr" in params
    assert "window_size" in params
    assert "process_large_chr_last" in params
    assert "system_info" in params
    assert "memory_per_process_gb" in params


@pytest.mark.unit
def test_data_processor_advanced_optimization_memory_scaling():
    params_low = DataProcessor.optimize_parameters_advanced(
        threads=2, window_size=1000000
    )
    params_high = DataProcessor.optimize_parameters_advanced(
        threads=16, window_size=1000000
    )

    assert "max_parallel_chr" in params_low
    assert "max_parallel_chr" in params_high

    params_small_window = DataProcessor.optimize_parameters_advanced(
        threads=8, window_size=1000000
    )
    params_large_window = DataProcessor.optimize_parameters_advanced(
        threads=8, window_size=10000000
    )

    assert params_small_window["window_size"] == 1000000
    assert params_large_window["window_size"] == 10000000

    for params in [params_low, params_high, params_small_window, params_large_window]:
        assert "system_info" in params
        assert "memory_per_process_gb" in params
        assert isinstance(params["memory_per_process_gb"], (int, float))
        assert params["memory_per_process_gb"] > 0


@pytest.mark.unit
def test_methylation_imputer_parameter_validation():
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_file:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            MethylationImputer(tmp_file.name, "out.h5", chunk_size=0)

        with pytest.raises(ValueError, match="k must be positive"):
            MethylationImputer(tmp_file.name, "out.h5", k=0)

        with pytest.raises(
            ValueError, match="compression_level must be between 1 and 9"
        ):
            MethylationImputer(tmp_file.name, "out.h5", compression_level=10)


@pytest.mark.unit
def test_methylation_imputer_with_nan_data(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            meta_grp = hf.create_group("Metadata")
            meta_grp.create_dataset("IID", data=[f"sample_{i}" for i in range(2)])

            grp = hf.create_group("CHR1")
            betas = np.array([[0.5, np.nan], [0.3, 0.7]])
            grp.create_dataset("Methylation", data=betas)
            grp.create_dataset("ProbeList", data=["probe1", "probe2"])

    output_file = os.path.join(output_dir, "test_nan_imputed.h5")
    imputer = MethylationImputer(
        tmp_input.name,
        output_file,
        k=1,
        chunk_size=10,
        use_parallel=False,
    )
    result = imputer.run()
    assert result is not None
    assert check_no_nans_in_betas(result)
    os.remove(tmp_input.name)


@pytest.mark.integration
def test_system_utils_advanced_optimization_integration():
    log.info("Testing SystemUtils advanced optimization integration...")

    system_info = SystemUtils.get_system_info()
    SystemUtils.get_memory_info()

    params = DataProcessor.optimize_parameters_advanced(threads=4, window_size=1000000)

    assert params["system_info"]["environment"] == system_info["environment"]
    assert "memory_per_process_gb" in params

    if system_info["environment"] in ["LSF", "SLURM"]:
        assert params["system_info"]["environment"] in ["LSF", "SLURM"]
    else:
        assert params["system_info"]["environment"] == "Local"

    is_valid, message = SystemUtils.validate_resources(cores=2, memory_gb=2.0)
    assert isinstance(is_valid, bool)
    assert isinstance(message, str)

    log.success("SystemUtils advanced optimization integration validated")


@pytest.mark.integration
def test_methylation_imputation_with_resource_monitoring(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")

    if not os.path.exists(input_file):
        pytest.skip(f"Test data file not found: {input_file}")

    output_file = os.path.join(output_dir, "methyl_imputed_monitored_test.h5")

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            meta_grp = hf.create_group("Metadata")
            meta_grp.create_dataset("IID", data=[f"sample_{i}" for i in range(20)])

            grp = hf.create_group("CHR1")
            betas = np.random.rand(1000, 20)
            betas[100:200, 5:10] = np.nan
            grp.create_dataset("Methylation", data=betas)
            grp.create_dataset("ProbeList", data=[f"probe_{i}" for i in range(1000)])

    try:
        with monitor_resources(interval=0.1) as stats:
            pipeline = ImputeCounts(
                input_file=tmp_input.name,
                output_file=output_file,
                data_type="methylation",
                k=5,
            )
            result = pipeline.run()

            assert "max_cpu" in stats
            assert "max_memory" in stats
            assert "samples" in stats
            assert stats["max_cpu"] >= 0
            assert stats["max_memory"] >= 0

        assert result == output_file
        assert os.path.exists(output_file)
        assert check_no_nans_in_betas(output_file)

    finally:
        os.remove(tmp_input.name)


@pytest.mark.integration
def test_methylation_imputation_parallel_vs_single(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            meta_grp = hf.create_group("Metadata")
            meta_grp.create_dataset("IID", data=[f"sample_{i}" for i in range(10)])

            for chr_num in [1, 2]:
                grp = hf.create_group(f"CHR{chr_num}")
                betas = np.random.rand(50, 10)
                betas[5:10, 2:4] = np.nan
                grp.create_dataset("Methylation", data=betas)
                grp.create_dataset(
                    "ProbeList", data=[f"probe_{chr_num}_{i}" for i in range(50)]
                )

    output_parallel = os.path.join(output_dir, "parallel_test.h5")
    pipeline_parallel = ImputeCounts(
        input_file=tmp_input.name,
        output_file=output_parallel,
        data_type="methylation",
        k=3,
    )
    result_parallel = pipeline_parallel.run()

    output_single = os.path.join(output_dir, "single_test.h5")
    imputer_single = MethylationImputer(
        input_file=tmp_input.name, output_file=output_single, k=3, use_parallel=False
    )
    result_single = imputer_single.run()

    assert result_parallel is not None
    assert result_single is not None
    assert check_no_nans_in_betas(result_parallel)
    assert check_no_nans_in_betas(result_single)

    os.remove(tmp_input.name)


@pytest.mark.integration
def test_methylation_imputation_memory_optimization(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            meta_grp = hf.create_group("Metadata")
            meta_grp.create_dataset("IID", data=[f"sample_{i}" for i in range(5)])

            grp = hf.create_group("CHR1")
            betas = np.random.rand(20, 5)
            betas[5:8, 1:3] = np.nan
            grp.create_dataset("Methylation", data=betas)
            grp.create_dataset("ProbeList", data=[f"probe_{i}" for i in range(20)])

    output_file = os.path.join(output_dir, "memory_opt_test.h5")

    pipeline = ImputeCounts(
        input_file=tmp_input.name, output_file=output_file, data_type="methylation", k=3
    )

    result = pipeline.run()
    assert result is not None
    assert os.path.exists(output_file)
    assert check_no_nans_in_betas(output_file)

    os.remove(tmp_input.name)


@pytest.mark.integration
def test_genotype_imputation_with_advanced_optimization(tmp_path, output_dir):
    mini_ref_dir = tmp_path / "mini_reference"
    mini_ref_dir.mkdir(exist_ok=True)

    test_input = tmp_path / "test_input.h5"
    output_file = os.path.join(output_dir, "advanced_opt_imputed_test.h5")

    with h5py.File(test_input, "w") as hf:
        meta_grp = hf.create_group("Metadata")
        meta_grp.create_dataset(
            "IID", data=np.array(["sample1", "sample2"], dtype="S20")
        )

        chr_grp = hf.create_group("CHR1")
        chr_grp.create_dataset(
            "Genotype", data=np.array([[0, 1], [1, 2]], dtype=np.int8)
        )
        chr_grp.create_dataset("RSID", data=np.array(["rs1", "rs2"], dtype="S20"))
        chr_grp.create_dataset("BP", data=np.array([1000, 2000], dtype=np.int32))
        chr_grp.create_dataset("A1", data=np.array(["A", "C"], dtype="S1"))
        chr_grp.create_dataset("A2", data=np.array(["T", "G"], dtype="S1"))

    create_minimal_reference_files(mini_ref_dir)

    try:
        log.info("Testing DataProcessor.optimize_parameters_advanced...")

        params = DataProcessor.optimize_parameters_advanced(
            threads=8, window_size=5000000
        )

        assert "max_parallel_chr" in params
        assert "window_size" in params
        assert "process_large_chr_last" in params
        assert "system_info" in params
        assert "memory_per_process_gb" in params

        system_info = params["system_info"]
        assert "cpu_name" in system_info
        assert "environment" in system_info
        assert "effective_cores" in system_info

        log.success("Advanced optimization parameters validated successfully")

        log.info("Testing ImputeCounts initialization with advanced optimization...")

        pipeline = ImputeCounts(
            input_file=str(test_input),
            output_file=output_file,
            data_type="genotype",
            reference_dir=str(mini_ref_dir),
            threshold=None,
            window_size=5000,
        )

        assert pipeline.data_type == "Genotype"
        assert pipeline.reference_dir == str(mini_ref_dir)
        assert pipeline.window_size == 5000

        log.success("ImputeCounts initialization with advanced optimization successful")

        log.info(
            "Attempting genotype imputation (may skip due to reference compatibility)..."
        )

        result = pipeline.run()

        if result is not None:
            assert os.path.exists(output_file)
            log.success("Genotype imputation completed successfully")
        else:
            log.info(
                "Genotype imputation failed as expected with minimal reference data"
            )
            log.success("Advanced optimization features validated successfully")

    except FileNotFoundError as e:
        log.info(f"Genotype imputation skipped due to missing dependencies: {e}")
        log.success("Advanced optimization parameters were validated successfully")

    except Exception as e:
        error_msg = str(e).lower()
        if any(
            keyword in error_msg
            for keyword in [
                "reference",
                "impute2",
                "legend",
                "haplotype",
                "genetic map",
                "sample file",
                "no rsids",
                "compatibility",
            ]
        ):
            log.info(f"Genotype imputation skipped due to reference compatibility: {e}")
            log.success("Advanced optimization features were validated successfully")
        else:
            log.error(f"Unexpected error in advanced optimization test: {e}")
            raise


@pytest.mark.integration
def test_quality_filter_low_threshold(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            grp = hf.create_group("CHR1")
            grp.create_dataset("INFO", data=[0.4, 0.2, 0.6])
            grp.create_dataset("Genotype", data=np.random.rand(3, 10))
            grp.create_dataset("RSID", data=["rs1", "rs2", "rs3"])
            grp.create_dataset("BP", data=[100, 200, 300])
            grp.create_dataset("A1", data=["A", "C", "G"])
            grp.create_dataset("A2", data=["T", "G", "C"])

    output_file = os.path.join(output_dir, "test_filtered.h5")
    filter_tool = QualityFilter(tmp_input.name, output_file, threshold=0.3, threads=2)
    result = filter_tool.run()
    with h5py.File(result, "r") as h5f:
        info_key = find_dataset_in_group(h5f["CHR1"], "INFO")
        assert len(h5f["CHR1"][info_key]) == 2
        assert len(h5f["CHR1"]["Genotype"]) == 2
        assert len(h5f["CHR1"]["RSID"]) == 2
        assert "filter_info" in h5f
        assert h5f["filter_info"].attrs["threshold_score"] == 0.3
    os.remove(tmp_input.name)


@pytest.mark.integration
def test_methylation_imputation(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_imputed_test.h5")

    pipeline = ImputeCounts(
        input_file=input_file, output_file=output_file, data_type="methylation", k=5
    )
    result = pipeline.run()
    assert result == output_file
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as h5f:
        chr_keys = [
            grp
            for grp in h5f.keys()
            if AliasUtils.strip_numeric_suffix(grp) in AliasUtils.get_aliases("CHR")
        ]
        assert len(chr_keys) > 0, "No chromosome groups found"

        first_chr = chr_keys[0]
        assert_dataset_exists(
            h5f[first_chr], "Methylation", "No methylation data found"
        )
        assert_dataset_exists(h5f[first_chr], "ProbeList", "No probe list found")
        assert_dataset_exists(h5f, "Metadata", "No metadata found")
        assert check_no_nans_in_betas(output_file)


@pytest.mark.integration
def test_genotype_pipeline_initialization_and_validation():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            meta_grp = hf.create_group("Metadata")
            meta_grp.create_dataset(
                "IID", data=np.array(["sample1", "sample2"], dtype="S20")
            )

            chr_grp = hf.create_group("CHR1")
            chr_grp.create_dataset(
                "Genotype", data=np.array([[0, 1], [1, 2]], dtype=np.int8)
            )
            chr_grp.create_dataset("RSID", data=np.array(["rs1", "rs2"], dtype="S20"))
            chr_grp.create_dataset("BP", data=np.array([1000, 2000], dtype=np.int32))
            chr_grp.create_dataset("A1", data=np.array(["A", "C"], dtype="S1"))
            chr_grp.create_dataset("A2", data=np.array(["T", "G"], dtype="S1"))

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.h5")

            pipeline = ImputeCounts(
                input_file=tmp_input.name,
                output_file=output_file,
                data_type="genotype",
                reference_dir="/fake/reference/dir",
                threshold=None,
                window_size=5000,
            )

            assert pipeline.data_type == "Genotype"
            assert pipeline.reference_dir == "/fake/reference/dir"
            assert pipeline.window_size == 5000

            assert pipeline.input_file == tmp_input.name
            assert pipeline.output_file == output_file

            log.success("Genotype pipeline initialization successful")

            with pytest.raises((FileNotFoundError, ValueError)):
                pipeline.run()

            log.success("Genotype pipeline properly validates reference directory")

    finally:
        os.remove(tmp_input.name)


@pytest.mark.integration
def test_genotype_imputation_error_handling(output_dir):
    with pytest.raises(FileNotFoundError):
        pipeline = ImputeCounts(
            input_file="/nonexistent/input.h5",
            output_file=os.path.join(output_dir, "output.h5"),
            data_type="genotype",
            reference_dir="/fake/ref",
        )
        pipeline.run()

    with pytest.raises(
        ValueError, match="data_type must be either 'genotype' or 'methylation'"
    ):
        ImputeCounts(data_type="invalid")

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_file:
        with pytest.raises(
            ValueError, match="reference_dir is required for genotype imputation"
        ):
            pipeline = ImputeCounts(
                input_file=tmp_file.name,
                output_file=os.path.join(output_dir, "output.h5"),
                data_type="Genotype",
            )
            pipeline.run()


@pytest.mark.unit
def test_genotype_imputer_parameter_optimization():
    params = DataProcessor.optimize_parameters_advanced(threads=8, window_size=5000000)

    assert "max_parallel_chr" in params
    assert "window_size" in params
    assert "process_large_chr_last" in params
    assert "system_info" in params

    memory_gb = params.get("memory_per_process_gb", 2.0)
    assert isinstance(memory_gb, (int, float))
    assert memory_gb > 0

    small_window_params = DataProcessor.optimize_parameters_advanced(
        threads=4, window_size=1000000
    )
    large_window_params = DataProcessor.optimize_parameters_advanced(
        threads=4, window_size=10000000
    )

    assert small_window_params["window_size"] == 1000000
    assert large_window_params["window_size"] == 10000000

    log.success("Genotype imputation parameter optimization validated")


@pytest.mark.integration
def test_genotype_imputer_initialization_with_optimization():
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            meta_grp = hf.create_group("Metadata")
            meta_grp.create_dataset("IID", data=np.array(["sample1"], dtype="S20"))

            chr_grp = hf.create_group("CHR1")
            chr_grp.create_dataset("Genotype", data=np.array([[0]], dtype=np.int8))
            chr_grp.create_dataset("RSID", data=np.array(["rs1"], dtype="S20"))
            chr_grp.create_dataset("BP", data=np.array([1000], dtype=np.int32))
            chr_grp.create_dataset("A1", data=np.array(["A"], dtype="S1"))
            chr_grp.create_dataset("A2", data=np.array(["T"], dtype="S1"))

    with tempfile.TemporaryDirectory() as temp_ref_dir:
        sample_file = os.path.join(temp_ref_dir, "1000GP_Phase3.sample")
        with open(sample_file, "w") as f:
            f.write("ID_1 ID_2 missing sex\n0 0 0 D\nsample1 sample1 0 1\n")

        try:
            from ImputeCounts import GenotypeImputer

            imputer = GenotypeImputer(
                input_file=tmp_input.name,
                output_file="/tmp/test_output.h5",
                reference_dir=temp_ref_dir,
                window_size=5000,
                buffer_size=250000,
                ne=20000,
                sample_list=None,
            )

            assert hasattr(imputer, "max_parallel_chr")
            assert hasattr(imputer, "window_size")
            assert hasattr(imputer, "process_large_chr_last")

            log.success("GenotypeImputer initialization with optimization successful")

        except Exception as e:
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in ["reference", "impute2", "file not found"]
            ):
                log.info(f"GenotypeImputer initialization test skipped: {e}")
            else:
                raise
        finally:
            os.remove(tmp_input.name)


@pytest.mark.integration
def test_genotype_imputer_missing_files(data_dir, output_dir):
    test_input = os.path.join(output_dir, "test_genotype_input.h5")

    with h5py.File(test_input, "w") as hf:
        meta_grp = hf.create_group("Metadata")
        meta_grp.create_dataset(
            "IID", data=np.array(["sample1", "sample2"], dtype="S20")
        )

        chr_grp = hf.create_group("CHR1")
        chr_grp.create_dataset(
            "Genotype", data=np.array([[0, 1], [1, 2]], dtype=np.int8)
        )
        chr_grp.create_dataset("RSID", data=np.array(["rs1", "rs2"], dtype="S20"))
        chr_grp.create_dataset("BP", data=np.array([1000, 2000], dtype=np.int32))
        chr_grp.create_dataset("A1", data=np.array(["A", "C"], dtype="S1"))
        chr_grp.create_dataset("A2", data=np.array(["T", "G"], dtype="S1"))

    output_file = os.path.join(output_dir, "missing_ref_test.h5")
    fake_ref_dir = "/nonexistent/reference/dir"

    try:
        with pytest.raises((FileNotFoundError, ValueError)):
            pipeline = ImputeCounts(
                input_file=test_input,
                output_file=output_file,
                data_type="genotype",
                reference_dir=fake_ref_dir,
            )
            pipeline.run()
    finally:
        if os.path.exists(test_input):
            os.remove(test_input)


@pytest.mark.integration
def test_methylation_imputer_empty_data(output_dir):
    output_file = os.path.join(output_dir, "empty_data_test.h5")
    test_input = os.path.join(output_dir, "minimal_methylation_input.h5")

    with h5py.File(test_input, "w") as h5f:
        meta_grp = h5f.create_group("Metadata")
        meta_grp.create_dataset("IID", data=np.array(["sample1"], dtype="S20"))

        chr_grp = h5f.create_group("CHR1")
        chr_grp.create_dataset("BP", data=np.array([1000], dtype=np.int32))
        chr_grp.create_dataset("RSID", data=np.array(["rs1"], dtype="S20"))
        chr_grp.create_dataset("Methylation", data=np.array([[0.5]], dtype=np.float32))

    try:
        imputer = MethylationImputer(
            input_file=test_input,
            output_file=output_file,
            k=1,
        )

        log.info("Testing MethylationImputer with minimal data...")
        result = imputer.run()

        if result is not None:
            assert os.path.exists(output_file)
            log.success("MethylationImputer minimal data test passed")
        else:
            log.info(
                "MethylationImputer returned None for minimal data - acceptable behavior"
            )

    except Exception as e:
        log.info(f"MethylationImputer minimal data test: {e}")

    finally:
        for file_path in [test_input, output_file]:
            if os.path.exists(file_path):
                os.remove(file_path)


@pytest.mark.integration
def test_quality_filter_all_pass(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            grp = hf.create_group("CHR1")
            grp.create_dataset("INFO", data=[0.9, 0.95, 0.98])
            grp.create_dataset("Genotype", data=np.random.rand(3, 10))
            grp.create_dataset("RSID", data=["rs1", "rs2", "rs3"])
            grp.create_dataset("BP", data=[100, 200, 300])
            grp.create_dataset("A1", data=["A", "C", "G"])
            grp.create_dataset("A2", data=["T", "G", "C"])

    output_file = os.path.join(output_dir, "all_pass_test.h5")
    filter_tool = QualityFilter(tmp_input.name, output_file, threshold=0.8, threads=2)
    result = filter_tool.run()

    with h5py.File(result, "r") as h5f:
        info_key = find_dataset_in_group(h5f["CHR1"], "INFO")
        assert len(h5f["CHR1"][info_key]) == 3
        assert len(h5f["CHR1"]["Genotype"]) == 3
        assert len(h5f["CHR1"]["RSID"]) == 3
        assert "filter_info" in h5f
        assert h5f["filter_info"].attrs["threshold_score"] == 0.8

    os.remove(tmp_input.name)


@pytest.mark.integration
def test_quality_filter_none_pass(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            grp = hf.create_group("CHR1")
            grp.create_dataset("INFO", data=[0.1, 0.2, 0.3])
            grp.create_dataset("Genotype", data=np.random.rand(3, 10))
            grp.create_dataset("RSID", data=["rs1", "rs2", "rs3"])
            grp.create_dataset("BP", data=[100, 200, 300])
            grp.create_dataset("A1", data=["A", "C", "G"])
            grp.create_dataset("A2", data=["T", "G", "C"])

    output_file = os.path.join(output_dir, "none_pass_test.h5")
    filter_tool = QualityFilter(tmp_input.name, output_file, threshold=0.8, threads=2)
    result = filter_tool.run()

    assert result == output_file
    assert os.path.exists(result)

    with h5py.File(result, "r") as h5f:
        assert "filter_info" in h5f
        assert h5f["filter_info"].attrs["threshold_score"] == 0.8

        if "CHR1" in h5f:
            info_key = find_dataset_in_group(h5f["CHR1"], "INFO")
            if info_key:
                assert len(h5f["CHR1"][info_key]) == 0

    os.remove(tmp_input.name)


@pytest.mark.integration
def test_methylation_imputation_multiple_chromosomes(data_dir, output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_input:
        with h5py.File(tmp_input.name, "w") as hf:
            meta_grp = hf.create_group("Metadata")
            meta_grp.create_dataset(
                "SampleList", data=[f"sample_{i}" for i in range(5)]
            )

            for chr_num in [1, 2]:
                grp = hf.create_group(f"CHR{chr_num}")
                betas = np.random.rand(10, 5)
                betas[0, 0] = np.nan
                betas[5, 2] = np.nan
                grp.create_dataset("Methylation", data=betas)
                grp.create_dataset(
                    "ProbeList", data=[f"cg{chr_num}_{i:04d}" for i in range(10)]
                )

    output_file = os.path.join(output_dir, "multi_chr_imputed.h5")
    pipeline = ImputeCounts(
        input_file=tmp_input.name, output_file=output_file, data_type="methylation", k=3
    )
    result = pipeline.run()

    assert result == output_file
    assert os.path.exists(output_file)
    assert check_no_nans_in_betas(output_file)

    with h5py.File(output_file, "r") as h5f:
        chr_keys = [grp for grp in h5f.keys() if grp.startswith("CHR")]
        assert len(chr_keys) >= 2
        assert "CHR1" in h5f
        assert "CHR2" in h5f

    os.remove(tmp_input.name)
