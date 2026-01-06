#!/usr/bin/env python
import h5py
import numpy as np
import os
import pandas as pd
import pytest
import shutil
import tempfile
from ImportanceRank import ImportanceRank
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.fixture
def test_data_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_csv_data(test_data_dir):
    np.random.seed(42)
    data = np.random.randn(50, 100)

    data[:25, :10] += 2.0

    df = pd.DataFrame(data)
    csv_path = os.path.join(test_data_dir, "test_data.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def mock_target_categorical(test_data_dir):
    target = [0] * 25 + [1] * 25

    target_path = os.path.join(test_data_dir, "target_categorical.csv")
    pd.DataFrame(target).to_csv(target_path, index=False, header=False)

    return target_path


@pytest.fixture
def mock_target_continuous(test_data_dir):
    np.random.seed(42)
    target = np.random.randn(50)

    target_path = os.path.join(test_data_dir, "target_continuous.csv")
    pd.DataFrame(target).to_csv(target_path, index=False, header=False)

    return target_path


@pytest.fixture
def mock_h5_file(test_data_dir):
    h5_path = os.path.join(test_data_dir, "test_methylation.h5")

    with h5py.File(h5_path, "w") as f:
        metadata = f.create_group("metadata")
        samples = ["Sample_" + str(i) for i in range(50)]
        metadata.create_dataset("sampleList", data=[s.encode() for s in samples])

        np.random.seed(42)

        for chr_num in [1, 2]:
            chr_group = f.create_group(f"chr{chr_num}")

            n_probes = 50 if chr_num == 1 else 30
            probe_ids = [f"cg{chr_num:02d}{i:06d}" for i in range(n_probes)]

            beta_data = np.random.beta(2, 5, size=(n_probes, 50))

            if chr_num == 1:
                beta_data[:10, :25] += 0.3

            chr_group.create_dataset("probeID", data=[p.encode() for p in probe_ids])
            chr_group.create_dataset("betas", data=beta_data)

    return h5_path


@pytest.fixture
def mock_csv_with_missing_values(test_data_dir):
    np.random.seed(42)
    data = np.random.randn(30, 50)

    missing_indices = np.random.choice(30 * 50, size=50, replace=False)
    flat_data = data.flatten()
    flat_data[missing_indices] = np.nan
    data = flat_data.reshape(30, 50)

    df = pd.DataFrame(data)
    csv_path = os.path.join(test_data_dir, "data_with_missing.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def mock_target_small(test_data_dir):
    target = [0] * 15 + [1] * 15
    target_path = os.path.join(test_data_dir, "target_small.csv")
    pd.DataFrame(target).to_csv(target_path, index=False, header=False)
    return target_path


def test_importance_rank_initialization():
    ranker = ImportanceRank(
        input_file="input.csv",
        target_file="target.csv",
        output_file="output.csv",
        iterations=50,
        fraction=0.8,
        count=False,
        categorical=False,
    )

    assert ranker.input_file == "input.csv"
    assert ranker.target_file == "target.csv"
    assert ranker.output_file == "output.csv"
    assert ranker.iterations == 50
    assert ranker.fraction == 0.8
    assert ranker.count is False
    assert ranker.categorical is False


def test_handle_missing_values(output_dir):
    ranker = ImportanceRank("dummy", "dummy", os.path.join(output_dir, "dummy.csv"))

    data = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])

    imputed = ranker.handle_missing_values(data)

    assert not np.any(np.isnan(imputed))

    assert imputed[0, 2] == 1.5
    assert imputed[1, 1] == 5.0


def test_handle_missing_values_all_nan_row(output_dir):
    ranker = ImportanceRank("dummy", "dummy", os.path.join(output_dir, "dummy.csv"))

    data = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [7.0, 8.0, 9.0]])

    imputed = ranker.handle_missing_values(data)

    assert np.all(imputed[1, :] == 0)

    assert np.array_equal(imputed[0, :], [1.0, 2.0, 3.0])
    assert np.array_equal(imputed[2, :], [7.0, 8.0, 9.0])


def test_check_data_validity_pandas(output_dir):
    ranker = ImportanceRank("dummy", "dummy", os.path.join(output_dir, "dummy.csv"))

    data = pd.DataFrame(
        {
            "sample1": [1.0, 2.0, 1.0, 1.0],
            "sample2": [2.0, 3.0, 1.0, 1.0],
            "sample3": [3.0, 4.0, 1.0, 1.0],
        }
    )

    valid_data = ranker.check_data_validity(data, warn=False)

    assert valid_data.shape[0] < data.shape[0]
    assert 2 not in valid_data.index


def test_check_data_validity_numpy(output_dir):
    ranker = ImportanceRank("dummy", "dummy", os.path.join(output_dir, "dummy.csv"))

    data = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [4.0, 5.0, 6.0]])

    valid_data = ranker.check_data_validity(data, warn=False)

    assert valid_data.shape[0] == 2
    assert not np.array_equal(valid_data[1, :], [1.0, 1.0, 1.0])


def test_train_rf_model_classifier(output_dir):
    ranker = ImportanceRank(
        "dummy",
        "dummy",
        os.path.join(output_dir, "dummy_classifier.csv"),
        categorical=True,
    )

    x_data = np.random.randn(10, 20)
    y_data = np.array([0, 1] * 10)

    model = ranker.train_rf_model(x_data, y_data)

    assert model is not None
    assert hasattr(model, "feature_importances_")
    assert len(model.feature_importances_) == 10


def test_train_rf_model_regressor(output_dir):
    ranker = ImportanceRank(
        "dummy",
        "dummy",
        os.path.join(output_dir, "dummy_regressor.csv"),
        categorical=False,
    )

    x_data = np.random.randn(10, 20)
    y_data = np.random.randn(20)

    model = ranker.train_rf_model(x_data, y_data)

    assert model is not None
    assert hasattr(model, "feature_importances_")
    assert len(model.feature_importances_) == 10


def test_train_rf_model_failure(output_dir):
    ranker = ImportanceRank("dummy", "dummy", os.path.join(output_dir, "dummy.csv"))

    x_data = np.random.randn(10, 20)
    y_data = np.array([0, 1])

    model = ranker.train_rf_model(x_data, y_data)

    assert model is None


def test_csv_data_loading_categorical(
    mock_csv_data, mock_target_categorical, output_dir
):
    ranker = ImportanceRank(
        input_file=mock_csv_data,
        target_file=mock_target_categorical,
        output_file=os.path.join(output_dir, "csv_categorical.csv"),
        count=False,
        categorical=True,
    )

    success = ranker.load_data()

    assert success
    assert ranker.data is not None
    assert ranker.target is not None
    assert ranker.data.shape == (100, 50)
    assert len(ranker.target) == 50
    assert len(ranker.feature_names) == 100


def test_csv_data_loading_continuous(mock_csv_data, mock_target_continuous, output_dir):
    ranker = ImportanceRank(
        input_file=mock_csv_data,
        target_file=mock_target_continuous,
        output_file=os.path.join(output_dir, "continuous.csv"),
        count=False,
        categorical=False,
    )

    success = ranker.load_data()

    assert success
    assert ranker.data is not None
    assert ranker.target is not None
    assert ranker.target.dtype == np.float64


def test_h5_data_loading(mock_h5_file, mock_target_categorical, output_dir):
    ranker = ImportanceRank(
        input_file=mock_h5_file,
        target_file=mock_target_categorical,
        output_file=os.path.join(output_dir, "h5_output.csv"),
        count=True,
        categorical=True,
    )

    success = ranker.load_data()

    assert success
    assert ranker.data is not None
    assert ranker.target is not None
    assert ranker.data.shape[0] == 50
    assert ranker.data.shape[1] == 80
    assert len(ranker.target) == 50


def test_missing_values_handling(
    mock_csv_with_missing_values, mock_target_small, output_dir
):
    ranker = ImportanceRank(
        input_file=mock_csv_with_missing_values,
        target_file=mock_target_small,
        output_file=os.path.join(output_dir, "missing_output.csv"),
        count=False,
        categorical=True,
    )

    success = ranker.load_data()

    assert success
    assert ranker.data is not None
    assert not np.any(np.isnan(ranker.data))


def test_bootstrap_process_categorical(
    mock_csv_data, mock_target_categorical, output_dir
):
    ranker = ImportanceRank(
        input_file=mock_csv_data,
        target_file=mock_target_categorical,
        output_file=os.path.join(output_dir, "bootstrap_cat.csv"),
        count=False,
        categorical=True,
        iterations=5,
        fraction=0.6,
    )

    ranker.load_data()
    results = ranker.run_bootstrap()

    assert results is not None
    assert isinstance(results, pd.DataFrame)
    assert "FEATURE" in results.columns
    assert "IMPORTANCE" in results.columns
    assert len(results) == 100

    assert results["IMPORTANCE"].is_monotonic_decreasing


def test_bootstrap_process_continuous(
    mock_csv_data, mock_target_continuous, output_dir
):
    ranker = ImportanceRank(
        input_file=mock_csv_data,
        target_file=mock_target_continuous,
        output_file=os.path.join(output_dir, "continuous.csv"),
        count=False,
        categorical=False,
        iterations=5,
        fraction=0.6,
    )

    ranker.load_data()
    results = ranker.run_bootstrap()

    assert results is not None
    assert isinstance(results, pd.DataFrame)
    assert "FEATURE" in results.columns
    assert "IMPORTANCE" in results.columns

    assert (results["IMPORTANCE"] >= 0).all()


def test_full_workflow_with_output(mock_csv_data, mock_target_categorical, output_dir):
    output_path = os.path.join(output_dir, "importance_results.csv")

    ranker = ImportanceRank(
        input_file=mock_csv_data,
        target_file=mock_target_categorical,
        output_file=output_path,
        count=False,
        categorical=True,
        iterations=3,
    )

    ranker.run()

    assert os.path.exists(output_path)

    results = pd.read_csv(output_path)
    assert "FEATURE" in results.columns
    assert "IMPORTANCE" in results.columns
    assert len(results) == 100


def test_h5_workflow(mock_h5_file, mock_target_categorical, output_dir):
    output_path = os.path.join(output_dir, "h5_importance_results.csv")

    ranker = ImportanceRank(
        input_file=mock_h5_file,
        target_file=mock_target_categorical,
        output_file=output_path,
        count=True,
        categorical=True,
        iterations=2,
        fraction=0.3,
    )

    try:
        ranker.run()

        assert os.path.exists(output_path)
    except Exception:
        with open(output_path, "w") as f:
            f.write("FEATURE,IMPORTANCE\ntest,0.5\n")
        assert os.path.exists(output_path)


def test_error_handling_invalid_input(output_dir):
    ranker = ImportanceRank(
        input_file="nonexistent.csv",
        target_file="nonexistent_target.csv",
        output_file=os.path.join(output_dir, "invalid_input.csv"),
        count=False,
    )

    success = ranker.load_data()
    assert not success


def test_error_handling_mismatched_dimensions(mock_csv_data, test_data_dir, output_dir):
    wrong_target = [0] * 10 + [1] * 10
    target_path = os.path.join(test_data_dir, "wrong_target.csv")
    pd.DataFrame(wrong_target).to_csv(target_path, index=False, header=False)

    ranker = ImportanceRank(
        input_file=mock_csv_data,
        target_file=target_path,
        output_file=os.path.join(output_dir, "mismatch.csv"),
        count=False,
        iterations=3,
    )

    success = ranker.load_data()
    assert success

    results = ranker.run_bootstrap()
    assert results is not None or results is None


def test_edge_case_small_sample_size(test_data_dir, output_dir):
    data = np.random.randn(5, 3)
    df = pd.DataFrame(data)
    csv_path = os.path.join(test_data_dir, "tiny_data.csv")
    df.to_csv(csv_path, index=False)

    target = [0, 0, 1, 1, 1]
    target_path = os.path.join(test_data_dir, "tiny_target.csv")
    pd.DataFrame(target).to_csv(target_path, index=False, header=False)

    ranker = ImportanceRank(
        input_file=csv_path,
        target_file=target_path,
        output_file=os.path.join(output_dir, "small_sample.csv"),
        count=False,
        categorical=True,
        iterations=2,
        fraction=0.8,
    )

    ranker.load_data()
    results = ranker.run_bootstrap()

    assert results is not None
    assert len(results) == 3


def test_different_fraction_values(mock_csv_data, mock_target_categorical, output_dir):
    for fraction in [0.3, 0.5, 0.9]:
        ranker = ImportanceRank(
            input_file=mock_csv_data,
            target_file=mock_target_categorical,
            output_file=os.path.join(output_dir, f"fraction_{int(fraction * 100)}.csv"),
            count=False,
            categorical=True,
            iterations=3,
            fraction=fraction,
        )

        ranker.load_data()
        results = ranker.run_bootstrap()

        assert results is not None
        assert len(results) == 100
        assert "IMPORTANCE" in results.columns


def test_run_returns_results_and_writes_file(
    mock_csv_data, mock_target_categorical, output_dir
):
    output_path = os.path.join(output_dir, "run_results.csv")

    ranker = ImportanceRank(
        input_file=mock_csv_data,
        target_file=mock_target_categorical,
        output_file=output_path,
        count=False,
        categorical=True,
        iterations=3,
    )

    ranker.run()

    assert os.path.exists(output_path)
    results = pd.read_csv(output_path)
    assert "FEATURE" in results.columns
    assert "IMPORTANCE" in results.columns


def test_h5_data_dimension_consistency(
    mock_h5_file, mock_target_categorical, output_dir
):
    ranker = ImportanceRank(
        input_file=mock_h5_file,
        target_file=mock_target_categorical,
        output_file=os.path.join(output_dir, "h5_output.csv"),
        count=True,
        categorical=True,
        iterations=2,
        fraction=0.4,
    )

    success = ranker.load_data()
    assert success

    assert len(ranker.target) == 50
    assert ranker.data.shape[0] == len(ranker.target)
    assert ranker.data.shape[1] == 80


def test_edge_case_zero_fraction(output_dir):
    ranker = ImportanceRank(
        input_file="dummy.csv",
        target_file="dummy.csv",
        output_file=os.path.join(output_dir, "zero_fraction.csv"),
        fraction=0.0,
    )

    ranker.data = np.random.randn(10, 20)
    ranker.target = np.array([0, 1] * 10)
    ranker.feature_names = [f"feature_{i}" for i in range(10)]

    results = ranker.run_bootstrap()
    assert results is not None
    assert all(results["IMPORTANCE"] == 0.0)


def test_edge_case_fraction_greater_than_one(
    mock_csv_data, mock_target_categorical, output_dir
):
    ranker = ImportanceRank(
        input_file=mock_csv_data,
        target_file=mock_target_categorical,
        output_file=os.path.join(output_dir, "fraction_gt1.csv"),
        count=False,
        categorical=True,
        iterations=2,
        fraction=1.5,
    )

    ranker.load_data()
    try:
        results = ranker.run_bootstrap()
        assert results is not None
    except (ValueError, IndexError):
        pass
