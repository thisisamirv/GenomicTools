#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import pytest
import tempfile
from unittest.mock import MagicMock, patch
from FeatureSelection import FeatureSelection
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.fixture
def mock_metadata_file():
    data = {
        "sample_id": ["sample1", "sample2", "sample3", "sample4"],
        "set": ["train", "train", "test", "test"],
        "response": ["A", "B", "A", "B"],
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_counts_csv():
    data = {
        "sample_id": ["sample1", "sample2", "sample3", "sample4"],
        "feature1": [0.1, 0.2, 0.3, 0.4],
        "feature2": [0.5, 0.6, 0.7, 0.8],
        "feature3": [0.9, 1.0, 1.1, 1.2],
        "feature4": [1.3, 1.4, 1.5, 1.6],
        "feature5": [1.7, 1.8, 1.9, 2.0],
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_importance_scores():
    data = {
        "feature": ["feature3", "feature1", "feature4", "feature2", "feature5"],
        "importance": [0.3, 0.25, 0.2, 0.15, 0.1],
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_output_file():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def basic_feature_selection_unparsed(
    mock_metadata_file, mock_counts_csv, mock_output_file
):
    return FeatureSelection(
        metadata=mock_metadata_file,
        counts=mock_counts_csv,
        response="response",
        output=mock_output_file,
        indices="1,2,3",
        preprocessing="scale,center",
        regularization="noise=0.1",
        evaluation="nfolds=3",
        model="RF",
        categorical=True,
    )


@pytest.fixture
def basic_feature_selection(mock_metadata_file, mock_counts_csv, mock_output_file):
    fs = FeatureSelection(
        metadata=mock_metadata_file,
        counts=mock_counts_csv,
        response="response",
        output=mock_output_file,
        indices="1,2,3",
        preprocessing="scale,center",
        regularization="noise=0.1",
        evaluation="nfolds=3",
        model="RF",
        categorical=True,
    )

    fs.indices = [1, 2, 3]
    fs.preprocessing = ["scale", "center"]
    fs.regularization = {"noise": "0.1"}
    fs.evaluation = {"nfolds": "3"}
    fs.stopping_criteria = {}
    fs.training = {}
    fs.parameters = {}

    return fs


@pytest.mark.unit
def test_parse_parameters(basic_feature_selection_unparsed):
    result = basic_feature_selection_unparsed._parse_parameters()
    assert result is True
    assert basic_feature_selection_unparsed.indices == [1, 2, 3]
    assert basic_feature_selection_unparsed.preprocessing == ["scale", "center"]
    assert basic_feature_selection_unparsed.regularization == {"noise": "0.1"}
    assert basic_feature_selection_unparsed.evaluation == {"nfolds": "3"}


@pytest.mark.unit
def test_parse_parameters_with_empty_values():
    fs = FeatureSelection(
        metadata="test.csv",
        counts="test.csv",
        response="test",
        output="out.csv",
        indices="1,2,3",
        preprocessing=None,
        regularization="",
        evaluation="",
        stopping_criteria="",
    )
    result = fs._parse_parameters()
    assert result is True
    assert fs.preprocessing is None
    assert fs.regularization == {}
    assert fs.evaluation == {}
    assert fs.stopping_criteria == {}


@pytest.mark.unit
def test_parse_parameters_with_invalid_indices():
    fs = FeatureSelection(
        metadata="test.csv",
        counts="test.csv",
        response="test",
        output="out.csv",
        indices="a,b,c",
    )
    with patch("utils.LoggingUtils.log.error") as mock_log:
        result = fs._parse_parameters()
        assert result is False
        mock_log.assert_called()


@pytest.mark.unit
def test_calculate_feature_scale():
    fs = FeatureSelection(
        metadata="test.csv",
        counts="test.csv",
        response="test",
        output="out.csv",
        indices="1",
        model="RF",
    )

    scale = fs._calculate_feature_scale(100, 50)
    assert scale == 10.0

    fs.model = "KNN"
    scale = fs._calculate_feature_scale(2000, 50)
    assert np.isclose(scale, np.sqrt(0.5))

    fs.model = "SVM"
    scale = fs._calculate_feature_scale(2000, 50)
    assert np.isclose(scale, np.sqrt(0.25))

    fs.model = "EN"
    scale = fs._calculate_feature_scale(2000, 50)
    assert scale is None


@pytest.mark.unit
def test_calculate_feature_scale_with_invalid_model():
    fs = FeatureSelection(
        metadata="test.csv",
        counts="test.csv",
        response="test",
        output="out.csv",
        indices="1",
        model="INVALID",
    )
    with pytest.raises(ValueError):
        fs._calculate_feature_scale(100, 50)


@pytest.mark.unit
@patch("pandas.read_csv")
def test_load_csv_data(mock_read_csv, basic_feature_selection):
    mock_counts = pd.DataFrame(
        {
            "sample_id": ["sample1", "sample2", "sample3", "sample4"],
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "feature2": [0.5, 0.6, 0.7, 0.8],
        }
    )
    mock_read_csv.return_value = mock_counts

    basic_feature_selection.train_samples = ["sample1", "sample2"]
    basic_feature_selection.test_samples = ["sample3", "sample4"]

    basic_feature_selection._load_csv_data()

    assert basic_feature_selection.train_data.shape == (2, 2)
    assert basic_feature_selection.test_data.shape == (2, 2)
    assert "sample_id" not in basic_feature_selection.train_data.columns
    assert "sample_id" not in basic_feature_selection.test_data.columns
    assert "feature1" in basic_feature_selection.train_data.columns
    assert "feature2" in basic_feature_selection.train_data.columns


@pytest.mark.unit
def test_sort_features_by_importance(mock_importance_scores):
    fs = FeatureSelection(
        metadata="test.csv",
        counts="test.csv",
        response="test",
        output="out.csv",
        indices="1",
        importance_scores=mock_importance_scores,
    )

    fs.train_data = pd.DataFrame(
        {
            "feature1": [0.1, 0.2],
            "feature2": [0.3, 0.4],
            "feature3": [0.5, 0.6],
            "feature4": [0.7, 0.8],
            "feature5": [0.9, 1.0],
        }
    )
    fs.test_data = pd.DataFrame(
        {
            "feature1": [1.1, 1.2],
            "feature2": [1.3, 1.4],
            "feature3": [1.5, 1.6],
            "feature4": [1.7, 1.8],
            "feature5": [1.9, 2.0],
        }
    )

    fs._sort_features_by_importance()

    expected_order = ["feature3", "feature1", "feature4", "feature2", "feature5"]
    assert list(fs.train_data.columns) == expected_order
    assert list(fs.test_data.columns) == expected_order


@pytest.mark.unit
@patch("h5py.File")
@patch("FeatureSelection.CachedH5Utils")
def test_load_hdf5_data(mock_h5_utils_class, mock_h5py_file):
    fs = FeatureSelection(
        metadata="test.csv",
        counts="test.h5",
        response="test",
        output="out.csv",
        indices="1",
    )

    mock_h5py_file.return_value
    mock_h5_utils = mock_h5_utils_class.return_value

    mock_h5_utils.validate_file_structure.return_value = True
    mock_h5_utils.get_data_info.return_value = {
        "data_type": "methylation",
        "n_chromosomes": 2,
        "n_samples": 4,
    }
    mock_h5_utils.get_chromosomes.return_value = ["chr1", "chr2"]
    mock_h5_utils.get_sample_indices.side_effect = lambda samples: (
        [0, 1] if samples == ["sample1", "sample2"] else [2, 3]
    )

    chr1_data = pd.DataFrame(
        {
            "probe_id": ["probe1", "probe2"],
            "sample1": [0.1, 0.2],
            "sample2": [0.3, 0.4],
            "sample3": [0.5, 0.6],
            "sample4": [0.7, 0.8],
        }
    )
    chr2_data = pd.DataFrame(
        {
            "probe_id": ["probe3", "probe4"],
            "sample1": [1.1, 1.2],
            "sample2": [1.3, 1.4],
            "sample3": [1.5, 1.6],
            "sample4": [1.7, 1.8],
        }
    )

    mock_h5_utils.read_chromosome.side_effect = lambda chr_name, data_type: (
        chr1_data if chr_name == "chr1" else chr2_data
    )

    fs.train_samples = ["sample1", "sample2"]
    fs.test_samples = ["sample3", "sample4"]

    fs._load_hdf5_data()

    assert fs.h5_file is not None
    assert fs.h5_utils is not None
    assert fs.train_data.shape == (2, 4)
    assert fs.test_data.shape == (2, 4)
    assert list(fs.train_data.columns) == ["probe1", "probe2", "probe3", "probe4"]
    assert list(fs.test_data.columns) == ["probe1", "probe2", "probe3", "probe4"]


@pytest.mark.unit
def test_parse_parameters_parsing_logic():
    from utils.ParsingUtils import ParseToList, ParseToKeyValueDict

    preprocessing_string = "scale,center"
    parsed_list = ParseToList(preprocessing_string)
    assert parsed_list == ["scale", "center"]

    regularization_string = "noise=0.1"
    parsed_dict = ParseToKeyValueDict(regularization_string)
    assert parsed_dict == {"noise": "0.1"}

    evaluation_string = "nfolds=3"
    parsed_evaluation = ParseToKeyValueDict(evaluation_string)
    assert parsed_evaluation == {"nfolds": "3"}


@pytest.mark.integration
@patch("FeatureSelection.DataPreprocessor")
@patch("FeatureSelection.ParameterGrid")
@patch("FeatureSelection.NoiseInjector")
@patch("FeatureSelection.TrainML")
def test_process_feature_sets(
    mock_train_ml_class,
    mock_noise_injector_class,
    mock_param_grid_class,
    mock_preprocessor_class,
    basic_feature_selection,
):
    basic_feature_selection.train_data = pd.DataFrame(
        {
            "feature1": [0.1, 0.2],
            "feature2": [0.3, 0.4],
            "feature3": [0.5, 0.6],
            "feature4": [0.7, 0.8],
            "feature5": [0.9, 1.0],
        }
    )
    basic_feature_selection.test_data = pd.DataFrame(
        {
            "feature1": [1.1, 1.2],
            "feature2": [1.3, 1.4],
            "feature3": [1.5, 1.6],
            "feature4": [1.7, 1.8],
            "feature5": [1.9, 2.0],
        }
    )
    basic_feature_selection.train_y = pd.Series(["A", "B"])
    basic_feature_selection.test_y = pd.Series(["A", "B"])
    basic_feature_selection.indices = [2, 3]
    basic_feature_selection.weights = None

    mock_preprocessor = mock_preprocessor_class.return_value

    def mock_preprocess_side_effect(train_data, test_data, methods):
        expected_methods = ["scale", "center"]
        assert (
            methods == expected_methods
        ), f"Expected {expected_methods}, got {methods}"
        return {"train": train_data, "test": test_data}

    mock_preprocessor.preprocess.side_effect = mock_preprocess_side_effect

    mock_noise_injector = mock_noise_injector_class.return_value
    mock_noise_injector.add_noise.return_value = (
        basic_feature_selection.train_data.iloc[:, :2]
    )

    mock_param_grid = mock_param_grid_class.return_value
    mock_param_grid.generate.return_value = pd.DataFrame(
        {"param1": [1, 2], "param2": [3, 4]}
    )

    mock_train_ml = mock_train_ml_class.return_value
    mock_train_ml.train.return_value = {
        "train_metrics": {"roc_value": 0.8, "prc_value": 0.7},
        "test_metrics": {"roc_value": 0.75, "prc_value": 0.65},
        "final_model": MagicMock(),
        "feature_importance": None,
    }

    results = basic_feature_selection._process_feature_sets()

    assert len(results) == 2
    assert "feature_set" in results.columns
    assert "test_roc" in results.columns
    assert "test_prc" in results.columns
    assert list(results["feature_set"]) == [2, 3]
    assert list(results["test_roc"]) == [
        0.75,
        0.75,
    ]


@pytest.mark.integration
@patch.object(FeatureSelection, "_parse_parameters", return_value=True)
@patch.object(FeatureSelection, "_load_data", return_value=True)
@patch.object(FeatureSelection, "_process_feature_sets")
def test_run_integration(
    mock_process_feature_sets,
    mock_load_data,
    mock_parse_parameters,
    basic_feature_selection,
    mock_output_file,
):
    mock_results = pd.DataFrame(
        {
            "feature_set": [1, 2, 3],
            "test_roc": [0.7, 0.75, 0.8],
            "test_prc": [0.6, 0.65, 0.7],
            "train_roc": [0.75, 0.8, 0.85],
            "train_prc": [0.65, 0.7, 0.75],
        }
    )
    mock_process_feature_sets.return_value = mock_results

    result = basic_feature_selection.run()

    assert result is True
    mock_parse_parameters.assert_called_once()
    mock_load_data.assert_called_once()
    mock_process_feature_sets.assert_called_once()

    assert os.path.exists(mock_output_file)
    output_df = pd.read_csv(mock_output_file)
    assert output_df.equals(mock_results)


@pytest.mark.integration
def test_run_with_parameter_parsing_failure(basic_feature_selection):
    with patch.object(FeatureSelection, "_parse_parameters", return_value=False):
        result = basic_feature_selection.run()
        assert result is False


@pytest.mark.integration
def test_run_with_data_loading_failure(basic_feature_selection):
    with patch.object(FeatureSelection, "_parse_parameters", return_value=True):
        with patch.object(FeatureSelection, "_load_data", return_value=False):
            result = basic_feature_selection.run()
            assert result is False


@pytest.mark.integration
def test_run_with_feature_processing_exception(basic_feature_selection):
    with patch.object(FeatureSelection, "_parse_parameters", return_value=True):
        with patch.object(FeatureSelection, "_load_data", return_value=True):
            with patch.object(
                FeatureSelection,
                "_process_feature_sets",
                side_effect=Exception("Test exception"),
            ):
                with patch("utils.LoggingUtils.log.error") as mock_log:
                    result = basic_feature_selection.run()
                    assert result is False
                    mock_log.assert_called()


@pytest.mark.integration
def test_feature_selection_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdirname:
        metadata = pd.DataFrame(
            {
                "sample_id": [f"sample{i}" for i in range(10)],
                "set": ["train"] * 7 + ["test"] * 3,
                "response": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )
        metadata_file = os.path.join(tmpdirname, "metadata.csv")
        metadata.to_csv(metadata_file, index=False)

        counts = pd.DataFrame({"sample_id": [f"sample{i}" for i in range(10)]})
        for i in range(20):
            counts[f"feature{i}"] = np.random.normal(0, 1, 10)
        counts_file = os.path.join(tmpdirname, "counts.csv")
        counts.to_csv(counts_file, index=False)

        output_file = os.path.join(tmpdirname, "output.csv")

        with patch("FeatureSelection.TrainML") as mock_train_ml_class:
            mock_train_ml = mock_train_ml_class.return_value
            mock_train_ml.train.return_value = {
                "train_metrics": {"roc_value": 0.8, "prc_value": 0.7},
                "test_metrics": {"roc_value": 0.75, "prc_value": 0.65},
                "final_model": MagicMock(),
                "feature_importance": None,
            }

            fs = FeatureSelection(
                metadata=metadata_file,
                counts=counts_file,
                response="response",
                output=output_file,
                indices="5,10,15",
                preprocessing="scale,center",
                regularization="noise=0.1",
                evaluation="nfolds=2",
                model="RF",
                categorical=True,
            )

            result = fs.run()

            assert result is True
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            assert len(output_df) == 3
            assert "feature_set" in output_df.columns
            assert "test_roc" in output_df.columns
            assert list(output_df["feature_set"]) == [5, 10, 15]


@pytest.mark.integration
def test_feature_selection_regression_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdirname:
        metadata = pd.DataFrame(
            {
                "sample_id": [f"sample{i}" for i in range(10)],
                "set": ["train"] * 7 + ["test"] * 3,
                "response": np.random.normal(0, 1, 10),
            }
        )
        metadata_file = os.path.join(tmpdirname, "metadata.csv")
        metadata.to_csv(metadata_file, index=False)

        counts = pd.DataFrame({"sample_id": [f"sample{i}" for i in range(10)]})
        for i in range(20):
            counts[f"feature{i}"] = np.random.normal(0, 1, 10)
        counts_file = os.path.join(tmpdirname, "counts.csv")
        counts.to_csv(counts_file, index=False)

        output_file = os.path.join(tmpdirname, "output.csv")

        with patch("FeatureSelection.TrainML") as mock_train_ml_class:
            mock_train_ml = mock_train_ml_class.return_value
            mock_train_ml.train.return_value = {
                "train_metrics": {"rmse": 0.5, "r2": 0.7, "mse": 0.25, "mae": 0.4},
                "test_metrics": {"rmse": 0.6, "r2": 0.65, "mse": 0.36, "mae": 0.5},
                "final_model": MagicMock(),
                "feature_importance": None,
            }

            fs = FeatureSelection(
                metadata=metadata_file,
                counts=counts_file,
                response="response",
                output=output_file,
                indices="5,10",
                preprocessing="scale,center",
                regularization="noise=0.1",
                model="RF",
                categorical=False,
            )

            result = fs.run()

            assert result is True
            assert os.path.exists(output_file)
            output_df = pd.read_csv(output_file)
            assert len(output_df) == 2
            assert "feature_set" in output_df.columns
            assert "test_rmse" in output_df.columns
            assert "test_r2" in output_df.columns
            assert list(output_df["feature_set"]) == [5, 10]
