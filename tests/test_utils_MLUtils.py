#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from unittest.mock import MagicMock, patch
from utils.LoggingUtils import log
from utils.MLUtils import (
    BaseMLUtils,
    CrossValidation,
    DataPreprocessor,
    DataProcessor,
    EarlyStopping,
    ModelFactory,
    NoiseInjector,
    ParameterGrid,
    PerformanceMetrics,
    SampleWeights,
    TrainML,
)

log.setup(level="DEBUG")


@pytest.fixture
def sample_classification_data():
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=123, n_informative=5
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)]), pd.Series(y)


@pytest.fixture
def sample_regression_data():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=123)
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)]), pd.Series(y)


@pytest.fixture
def sample_multiclass_data():
    X, y = make_classification(
        n_samples=120, n_features=8, n_classes=3, random_state=123, n_informative=5
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)]), pd.Series(y)


@pytest.fixture
def sample_data_with_missing():
    X, y = make_classification(n_samples=50, n_features=5, random_state=123)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    df.iloc[0:5, 0] = np.nan
    df.iloc[10:15, 1] = np.nan
    return df, pd.Series(y)


@pytest.mark.unit
def test_to_numpy():
    base_utils = BaseMLUtils()

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = base_utils._to_numpy(df)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)

    series = pd.Series([1, 2, 3])
    result = base_utils._to_numpy(series)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)

    arr = np.array([1, 2, 3])
    result = base_utils._to_numpy(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


@pytest.mark.unit
def test_validate_data_match():
    base_utils = BaseMLUtils()

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    base_utils._validate_data_match(X, y)

    X_mismatch = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="X and y must have the same length"):
        base_utils._validate_data_match(X_mismatch, y)


@pytest.mark.unit
def test_validate_numeric_data():
    base_utils = BaseMLUtils()

    numeric_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
    assert base_utils._validate_numeric_data(numeric_df) is True

    mixed_df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with pytest.raises(ValueError, match="Non-numeric columns detected"):
        base_utils._validate_numeric_data(mixed_df)


@pytest.mark.unit
def test_safe_execute():
    base_utils = BaseMLUtils()

    def success_func():
        return "success"

    result = base_utils._safe_execute(success_func, "Test error")
    assert result == "success"

    def error_func():
        raise ValueError("Test exception")

    result = base_utils._safe_execute(error_func, "Test error")
    assert result is None


@pytest.mark.unit
def test_create_transformer_dataframe():
    base_utils = BaseMLUtils()

    mock_transformer = MagicMock()
    mock_transformer.fit_transform.return_value = np.array([[1, 2], [3, 4]])

    data = np.array([[1, 2, 3], [4, 5, 6]])

    result = base_utils._create_transformer_dataframe(
        mock_transformer, data, prefix="PC"
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["PC1", "PC2"]
    assert result.shape == (2, 2)

    custom_cols = ["comp1", "comp2"]
    result = base_utils._create_transformer_dataframe(
        mock_transformer, data, columns=custom_cols
    )
    assert list(result.columns) == custom_cols


@pytest.mark.unit
def test_apply_transformer_yeo_johnson(sample_classification_data):
    data, _ = sample_classification_data

    result = DataProcessor.apply_transformer(data, "YeoJohnson")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == data.shape
    assert list(result.columns) == list(data.columns)


@pytest.mark.unit
def test_remove_correlated_features():
    np.random.seed(123)
    base_features = np.random.randn(50, 3)
    correlated_feature = base_features[:, 0] + 0.01 * np.random.randn(50)

    data = pd.DataFrame(
        np.column_stack([base_features, correlated_feature]),
        columns=["feat1", "feat2", "feat3", "feat1_corr"],
    )

    result = DataProcessor.remove_correlated_features(data, threshold=0.9)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] < data.shape[1]


@pytest.mark.unit
def test_apply_outlier_capping():
    data = pd.DataFrame(
        {
            "normal": np.random.normal(0, 1, 100),
            "with_outliers": np.concatenate(
                [np.random.normal(0, 1, 95), [10, -10, 15, -15, 20]]
            ),
        }
    )

    result = DataProcessor.apply_outlier_capping(data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == data.shape
    assert result["with_outliers"].max() < data["with_outliers"].max()
    assert result["with_outliers"].min() > data["with_outliers"].min()


@pytest.mark.unit
def test_apply_zero_variance():
    preprocessor = DataPreprocessor()

    data = pd.DataFrame(
        {
            "varying": [1, 2, 3, 4, 5],
            "constant": [1, 1, 1, 1, 1],
            "varying2": [5, 4, 3, 2, 1],
        }
    )

    result = preprocessor._apply_zero_variance(data)
    assert isinstance(result, pd.DataFrame)
    assert "constant" not in result.columns
    assert "varying" in result.columns
    assert "varying2" in result.columns


@pytest.mark.unit
def test_apply_near_zero_variance():
    preprocessor = DataPreprocessor()

    data = pd.DataFrame(
        {
            "good_feature": np.random.randn(100),
            "bad_feature": np.concatenate([[1] * 95, [2] * 5]),
        }
    )

    result = preprocessor._apply_near_zero_variance(data)
    assert isinstance(result, pd.DataFrame)
    assert "good_feature" in result.columns


@pytest.mark.unit
def test_preprocess_success(sample_classification_data):
    data, _ = sample_classification_data
    train_df = data.iloc[:80]
    test_df = data.iloc[80:]

    preprocessor = DataPreprocessor()
    methods = ["center", "scale"]

    result = preprocessor.preprocess(train_df, test_df, methods)
    assert result is not None
    assert "train" in result
    assert "test" in result
    assert isinstance(result["train"], pd.DataFrame)
    assert isinstance(result["test"], pd.DataFrame)


@pytest.mark.unit
def test_preprocess_mismatched_columns(sample_classification_data):
    data, _ = sample_classification_data
    train_df = data.iloc[:80]
    test_df = data.iloc[80:].drop(columns=[data.columns[0]])

    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess(train_df, test_df, ["center"])
    assert result is None


@pytest.mark.unit
def test_preprocess_invalid_methods(sample_classification_data):
    data, _ = sample_classification_data
    train_df = data.iloc[:80]
    test_df = data.iloc[80:]

    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess(train_df, test_df, ["invalid_method"])
    assert result is None


@pytest.mark.unit
def test_apply_near_zero_variance_single_value_column():
    preprocessor = DataPreprocessor()

    data = pd.DataFrame(
        {
            "good_feature": np.random.randn(100),
            "single_value": [5] * 100,
        }
    )

    result = preprocessor._apply_near_zero_variance(data)
    assert isinstance(result, pd.DataFrame)
    assert "good_feature" in result.columns
    assert "single_value" not in result.columns


@pytest.mark.unit
def test_apply_near_zero_variance_high_frequency_ratio():
    preprocessor = DataPreprocessor()

    data_values = [1] * 96 + [2] * 2 + [3] * 2

    data = pd.DataFrame(
        {"good_feature": np.random.randn(100), "bad_feature": data_values}
    )

    result = preprocessor._apply_near_zero_variance(data)
    assert isinstance(result, pd.DataFrame)
    assert "good_feature" in result.columns
    assert "bad_feature" not in result.columns


@pytest.mark.unit
def test_create_range():
    param_grid = ParameterGrid()

    result = param_grid._create_range(0, 10, step=2)
    expected = np.array([0, 2, 4, 6, 8, 10])
    np.testing.assert_array_equal(result, expected)

    result = param_grid._create_range(0, 1, num=5)
    expected = np.linspace(0, 1, 5)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.unit
def test_generate_default_params_en():
    param_grid = ParameterGrid()
    result = param_grid.generate("EN", n_features=10, n_samples=100, feature_scale=1.0)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "alpha" in result.columns
    assert "lambda" in result.columns
    assert len(result) > 0


@pytest.mark.unit
def test_generate_default_params_knn():
    param_grid = ParameterGrid()
    result = param_grid.generate("KNN", n_features=10, n_samples=100, feature_scale=1.0)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "k" in result.columns
    assert "algorithm" in result.columns


@pytest.mark.unit
def test_generate_default_params_rf():
    param_grid = ParameterGrid()
    result = param_grid.generate("RF", n_features=10, n_samples=100, feature_scale=1.0)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "mtry" in result.columns
    assert "ntree" in result.columns
    assert "nodesize" in result.columns
    assert "maxdepth" in result.columns
    assert "sampsize" in result.columns


@pytest.mark.unit
def test_generate_default_params_svm():
    param_grid = ParameterGrid()
    result = param_grid.generate("SVM", n_features=10, n_samples=100, feature_scale=1.0)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "cost" in result.columns
    assert "gamma" in result.columns
    assert "kernel" in result.columns


@pytest.mark.unit
def test_generate_custom_params():
    param_grid = ParameterGrid()
    custom_params = {
        "alpha_step": 0.2,
        "lambda_max": 2.0,
        "lambda_min": 0.001,
        "n_lambda": 50,
    }

    result = param_grid.generate(
        "EN", params=custom_params, n_features=10, n_samples=100, feature_scale=1.0
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


@pytest.mark.unit
def test_generate_unsupported_model():
    param_grid = ParameterGrid()
    result = param_grid.generate(
        "UNSUPPORTED", n_features=10, n_samples=100, feature_scale=1.0
    )
    assert result is None


@pytest.mark.unit
def test_generate_empty_param_dict():
    param_grid = ParameterGrid()

    with patch.object(param_grid, "_get_default_params", return_value={}):
        with patch.object(param_grid, "_get_custom_params", return_value={}):
            result = param_grid.generate(
                "RF", n_features=10, n_samples=100, feature_scale=1.0
            )
            assert result is None


@pytest.mark.unit
def test_generate_empty_param_grid():
    param_grid = ParameterGrid()

    empty_params = {
        "ntree": [],
        "mtry": [2, 3],
    }

    with patch.object(param_grid, "_get_default_params", return_value=empty_params):
        result = param_grid.generate(
            "RF", n_features=10, n_samples=100, feature_scale=1.0
        )
        assert result is None


@pytest.mark.unit
def test_add_noise_basic(sample_classification_data):
    data, _ = sample_classification_data
    injector = NoiseInjector(random_state=123)

    result = injector.add_noise(data, noise_level=0.1, adaptive=False)
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert result.shape == data.shape
    assert list(result.columns) == list(data.columns)
    assert not np.allclose(result.values, data.values)


@pytest.mark.unit
def test_add_noise_adaptive(sample_classification_data):
    data, _ = sample_classification_data
    injector = NoiseInjector(random_state=123)

    result = injector.add_noise(data, noise_level=0.1, adaptive=True)
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert result.shape == data.shape


@pytest.mark.unit
def test_add_noise_negative_level(sample_classification_data):
    data, _ = sample_classification_data
    injector = NoiseInjector()

    result = injector.add_noise(data, noise_level=-0.1)
    assert result is None


@pytest.mark.unit
def test_calculate_balanced_weights(sample_classification_data):
    data, y = sample_classification_data
    weights = SampleWeights(strategy="balanced")

    result = weights.calculate(data, y)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result) == len(y)
    assert np.all(result > 0)


@pytest.mark.unit
def test_calculate_balanced_subsample_weights(sample_classification_data):
    data, y = sample_classification_data
    weights = SampleWeights(strategy="balanced_subsample", random_state=123)

    result = weights.calculate(data, y)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result) == len(y)
    assert np.all(result > 0)


@pytest.mark.unit
def test_calculate_custom_weights(sample_classification_data):
    data, y = sample_classification_data
    weights = SampleWeights(strategy="custom")
    custom_weights = {0: 1.0, 1: 2.0}

    result = weights.calculate(data, y, custom_weights=custom_weights)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result) == len(y)


@pytest.mark.unit
def test_calculate_custom_weights_no_dict(sample_classification_data):
    data, y = sample_classification_data
    weights = SampleWeights(strategy="custom")

    result = weights.calculate(data, y)
    assert result is None


@pytest.mark.unit
def test_calculate_unsupported_strategy(sample_classification_data):
    data, y = sample_classification_data
    weights = SampleWeights(strategy="unsupported")

    result = weights.calculate(data, y)
    assert result is None


@pytest.mark.unit
def test_calculate_invalid_train_data_type():
    weights = SampleWeights(strategy="balanced")

    invalid_data = "not_a_dataframe_or_array"
    y = np.array([0, 1, 0, 1])

    result = weights.calculate(invalid_data, y)
    assert result is None


@pytest.mark.unit
def test_calculate_invalid_train_y_type():
    weights = SampleWeights(strategy="balanced")

    data = np.array([[1, 2], [3, 4], [5, 6]])
    invalid_y = {"not": "valid"}

    result = weights.calculate(data, invalid_y)
    assert result is None


@pytest.mark.unit
def test_detect_task_type_classification():
    metrics = PerformanceMetrics()
    labels = np.array([0, 1, 0, 1, 0])
    task_type = metrics._detect_task_type(labels)
    assert task_type == "classification"


@pytest.mark.unit
def test_detect_task_type_regression():
    metrics = PerformanceMetrics()
    labels = np.array([1.5, 2.3, 0.8, 3.1, 2.7])
    task_type = metrics._detect_task_type(labels)
    assert task_type == "regression"


@pytest.mark.unit
def test_calculate_regression_metrics(sample_regression_data):
    data, y_true = sample_regression_data
    y_pred = y_true + np.random.normal(0, 0.1, len(y_true))

    metrics = PerformanceMetrics()
    result = metrics.calculate(
        y_pred.values.reshape(-1, 1), y_true, task_type="regression"
    )

    assert result is not None
    assert "mse" in result
    assert "mae" in result
    assert "r2" in result
    assert "rmse" in result
    assert "explained_variance" in result


@pytest.mark.unit
def test_calculate_binary_classification_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred_proba = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]]
    )

    metrics = PerformanceMetrics()
    result = metrics.calculate(y_pred_proba, y_true, task_type="classification")

    assert result is not None
    assert "roc_value" in result
    assert "prc_value" in result
    assert 0 <= result["roc_value"] <= 1
    assert 0 <= result["prc_value"] <= 1


@pytest.mark.unit
def test_calculate_multiclass_classification_metrics(sample_multiclass_data):
    data, y_true = sample_multiclass_data
    np.random.seed(123)
    y_pred_proba = np.random.dirichlet([1, 1, 1], len(y_true))

    metrics = PerformanceMetrics()
    result = metrics.calculate(y_pred_proba, y_true, task_type="classification")

    assert result is not None
    assert "roc_value" in result
    assert "prc_value" in result


@pytest.mark.unit
def test_preprocess_inputs_1d_array():
    metrics = PerformanceMetrics()

    pred_prob_1d = np.array([0.7, 0.3, 0.8, 0.2])
    true_labels = np.array([1, 0, 1, 0])

    pred_prob_processed, true_labels_processed = metrics._preprocess_inputs(
        pred_prob_1d, true_labels
    )

    assert pred_prob_processed.ndim == 2
    assert pred_prob_processed.shape == (4, 1)


@pytest.mark.unit
def test_detect_task_type_with_predefined_type():
    metrics = PerformanceMetrics(task_type="classification")
    labels = np.array([1.5, 2.3, 0.8, 3.1, 2.7])

    task_type = metrics._detect_task_type(labels)
    assert task_type == "classification"


@pytest.mark.unit
def test_calculate_multiclass_with_class_error():
    metrics = PerformanceMetrics()

    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred_proba = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.3, 0.7, 0.0],
            [0.1, 0.9, 0.0],
            [0.9, 0.1, 0.0],
            [0.2, 0.8, 0.0],
            [0.4, 0.6, 0.0],
            [0.7, 0.3, 0.0],
            [0.3, 0.7, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )

    call_count = [0]

    def failing_roc_auc_score(y_true_class, y_scores):
        call_count[0] += 1
        if call_count[0] == 3:
            raise ValueError("Simulated AUC calculation failure")
        from sklearn.metrics import roc_auc_score

        return roc_auc_score(y_true_class, y_scores)

    with patch("utils.MLUtils.roc_auc_score", side_effect=failing_roc_auc_score):
        result = metrics.calculate(y_pred_proba, y_true, task_type="classification")

        assert result is not None


@pytest.mark.unit
def test_calculate_unsupported_task_type():
    metrics = PerformanceMetrics()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])

    result = metrics.calculate(y_pred, y_true, task_type="unsupported_task")
    assert result is None


@pytest.mark.unit
def test_calculate_regression_multiple_columns_warning():
    metrics = PerformanceMetrics()

    y_true = np.array([1.5, 2.3, 0.8, 3.1, 2.7])
    y_pred_multi = np.array(
        [
            [1.4, 1.6, 1.3],
            [2.2, 2.4, 2.1],
            [0.9, 0.7, 0.8],
            [3.0, 3.2, 2.9],
            [2.8, 2.6, 2.7],
        ]
    )

    result = metrics.calculate(y_pred_multi, y_true, task_type="regression")

    assert result is not None


@pytest.mark.unit
def test_create_elastic_net_regression():
    params = {"lambda": 0.1, "alpha": 0.5}
    model = ModelFactory.create_model("EN", params, categorical=False, random_state=123)

    assert isinstance(model, ElasticNet)
    assert model.alpha == 0.1
    assert model.l1_ratio == 0.5


@pytest.mark.unit
def test_create_elastic_net_binary_classification():
    params = {"lambda": 0.1, "alpha": 0.5, "n_classes": 2}
    model = ModelFactory.create_model("EN", params, categorical=True, random_state=123)

    assert isinstance(model, LogisticRegression)
    assert model.C == 1 / 0.1
    assert model.l1_ratio == 0.5


@pytest.mark.unit
def test_create_elastic_net_multiclass():
    params = {"lambda": 0.1, "alpha": 0.5, "n_classes": 3}
    model = ModelFactory.create_model("EN", params, categorical=True, random_state=123)

    assert isinstance(model, LogisticRegression)
    assert model.multi_class == "multinomial"


@pytest.mark.unit
def test_create_knn_classification():
    params = {"k": 5, "algorithm": "kd_tree"}
    model = ModelFactory.create_model("KNN", params, categorical=True, random_state=123)

    assert isinstance(model, KNeighborsClassifier)
    assert model.n_neighbors == 5
    assert model.algorithm == "kd_tree"


@pytest.mark.unit
def test_create_knn_regression():
    params = {"k": 5, "algorithm": "kd_tree"}
    model = ModelFactory.create_model(
        "KNN", params, categorical=False, random_state=123
    )

    assert isinstance(model, KNeighborsRegressor)
    assert model.n_neighbors == 5


@pytest.mark.unit
def test_create_random_forest_classification():
    params = {"ntree": 100, "mtry": 3, "nodesize": 5, "maxdepth": 10}
    model = ModelFactory.create_model("RF", params, categorical=True, random_state=123)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.max_features == 3
    assert model.min_samples_split == 5
    assert model.max_depth == 10


@pytest.mark.unit
def test_create_random_forest_regression():
    params = {"ntree": 100, "mtry": 3, "nodesize": 5, "maxdepth": 10}
    model = ModelFactory.create_model("RF", params, categorical=False, random_state=123)

    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 100


@pytest.mark.unit
def test_create_svm_classification():
    params = {"cost": 1.0, "gamma": 0.1, "kernel": "rbf"}
    model = ModelFactory.create_model("SVM", params, categorical=True, random_state=123)

    assert isinstance(model, SVC)
    assert model.C == 1.0
    assert model.gamma == 0.1
    assert model.kernel == "rbf"
    assert model.probability is True


@pytest.mark.unit
def test_create_svm_regression():
    params = {"cost": 1.0, "gamma": 0.1, "kernel": "rbf"}
    model = ModelFactory.create_model(
        "SVM", params, categorical=False, random_state=123
    )

    assert isinstance(model, SVR)
    assert model.C == 1.0
    assert model.gamma == 0.1


@pytest.mark.unit
def test_create_invalid_model():
    with pytest.raises(ValueError, match="Invalid model type"):
        ModelFactory.create_model("INVALID", {}, categorical=True, random_state=123)


@pytest.mark.unit
def test_create_random_forest_with_sampsize():
    params = {
        "ntree": 100,
        "mtry": 3,
        "nodesize": 5,
        "maxdepth": 10,
        "sampsize": 80,
        "fold_size": 100,
    }

    model = ModelFactory.create_model("RF", params, categorical=True, random_state=123)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.max_samples == 0.8


@pytest.mark.unit
def test_create_random_forest_with_sampsize_no_fold_size():
    params = {
        "ntree": 100,
        "mtry": 3,
        "nodesize": 5,
        "maxdepth": 10,
        "sampsize": 80,
    }

    model = ModelFactory.create_model("RF", params, categorical=True, random_state=123)

    assert isinstance(model, RandomForestClassifier)
    assert model.max_samples == 80.0


@pytest.mark.unit
def test_process_sample_weights_none():
    cv = CrossValidation()
    result = cv._process_sample_weights(
        None, np.array([0, 1, 2]), np.array([0, 1, 0]), "RF", True
    )
    assert result is None


@pytest.mark.unit
def test_process_sample_weights_svm_classification():
    cv = CrossValidation()
    sample_weights = np.array([1.0, 2.0, 1.5, 2.5])
    train_idx = np.array([0, 1, 2])
    fold_train_y = np.array([0, 1, 0])

    result = cv._process_sample_weights(
        sample_weights, train_idx, fold_train_y, "SVM", True
    )
    assert isinstance(result, dict)
    assert 0 in result
    assert 1 in result


@pytest.mark.unit
def test_get_predictions_classification():
    cv = CrossValidation()

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
    test_set = np.array([[1, 2], [3, 4]])

    result = cv._get_predictions(mock_model, test_set, categorical=True)
    assert result.shape == (2, 2)
    mock_model.predict_proba.assert_called_once_with(test_set)


@pytest.mark.unit
def test_get_predictions_regression():
    cv = CrossValidation()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1.5, 2.3])
    test_set = np.array([[1, 2], [3, 4]])

    result = cv._get_predictions(mock_model, test_set, categorical=False)
    assert result.shape == (2, 1)
    mock_model.predict.assert_called_once_with(test_set)


@pytest.mark.unit
def test_run_regression_success(sample_regression_data):
    data, y = sample_regression_data
    cv = CrossValidation(nfolds=3, random_state=123, n_jobs=1)

    params = {"ntree": 10, "mtry": 2, "nodesize": 5, "maxdepth": 5}

    result = cv.run(data.values, y.values, params, model="RF", categorical=False)

    assert result is not None
    assert "fold_results" in result
    assert "cv_rmse_mean" in result
    assert "cv_r2_mean" in result
    assert len(result["fold_results"]) == 3


@pytest.mark.unit
def test_run_unsupported_model(sample_classification_data):
    data, y = sample_classification_data
    cv = CrossValidation()

    result = cv.run(data.values, y.values, {}, model="UNSUPPORTED", categorical=True)
    assert result is None


@pytest.mark.unit
def test_get_predictions_classification_without_predict_proba():
    cv = CrossValidation()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 0])
    mock_model.classes_ = np.array([0, 1])
    del mock_model.predict_proba

    test_set = np.array([[1, 2], [3, 4], [5, 6]])

    result = cv._get_predictions(mock_model, test_set, categorical=True)

    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result[0], [1, 0])
    np.testing.assert_array_equal(result[1], [0, 1])


@pytest.mark.unit
def test_run_single_fold_with_sample_weights_non_svm():
    cv = CrossValidation(nfolds=2, random_state=123, n_jobs=1)

    train_set = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    train_y = np.array([0, 1, 0, 1])
    train_idx = np.array([0, 1])
    test_idx = np.array([2, 3])

    sample_weights = np.array([1.5, 2.0, 1.0, 1.8])

    params = {"ntree": 10, "mtry": 1, "nodesize": 2, "maxdepth": 5}

    result = cv._run_single_fold(
        0,
        train_idx,
        test_idx,
        train_set,
        train_y,
        params,
        "RF",
        True,
        sample_weights,
    )

    assert result is not None
    assert "roc" in result
    assert "prc" in result


@pytest.mark.unit
def test_run_single_fold_exception_handling():
    cv = CrossValidation(nfolds=2, random_state=123, n_jobs=1)

    train_set = np.array([[1, 2], [3, 4]])
    train_y = np.array([0, 1])
    train_idx = np.array([0])
    test_idx = np.array([1])

    invalid_params = {"invalid_param": "will_cause_error"}

    result = cv._run_single_fold(
        0, train_idx, test_idx, train_set, train_y, invalid_params, "RF", True, None
    )

    assert result is None


@pytest.mark.unit
def test_run_all_folds_fail():
    cv = CrossValidation(nfolds=2, random_state=123, n_jobs=1)

    train_set = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    train_y = np.array([0, 1, 0, 1])

    with patch.object(cv, "_run_single_fold", return_value=None):
        result = cv.run(train_set, train_y, {}, model="RF", categorical=True)

        assert result is None


@pytest.mark.unit
def test_run_single_fold_result_structure():
    cv = CrossValidation(nfolds=2, random_state=123, n_jobs=1)

    train_set = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    train_y = np.array([0, 1, 0, 1])
    train_idx = np.array([0, 1])
    test_idx = np.array([2, 3])

    params = {"ntree": 10, "mtry": 1, "nodesize": 2, "maxdepth": 5}

    result = cv._run_single_fold(
        0, train_idx, test_idx, train_set, train_y, params, "RF", True, None
    )

    assert result is not None
    assert "fold" in result
    assert "model" in result
    assert result["fold"] == 0
    assert result["model"] is not None


@pytest.mark.unit
def test_run_single_fold_metrics_none_specific():
    cv = CrossValidation(nfolds=2, random_state=123, n_jobs=1)

    train_set = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    train_y = np.array([0, 1, 0, 1])
    train_idx = np.array([0, 1])
    test_idx = np.array([2, 3])

    params = {"ntree": 10, "mtry": 1, "nodesize": 2, "maxdepth": 5}

    with patch.object(cv.metrics_calculator, "calculate") as mock_calculate:
        mock_calculate.return_value = None

        result = cv._run_single_fold(
            0, train_idx, test_idx, train_set, train_y, params, "RF", True, None
        )

        assert result is None


@pytest.mark.unit
def test_check_metric_improvement_higher_better():
    early_stop = EarlyStopping(patience=3, min_improvement=0.01)

    results_df = pd.DataFrame({"cv_roc": [0.5, 0.6, 0.65, 0.7, 0.75, 0.76]})

    improvement = early_stop._check_metric_improvement(results_df, "auroc")
    assert improvement


@pytest.mark.unit
def test_check_metric_improvement_lower_better():
    early_stop = EarlyStopping(patience=3, min_improvement=0.01, monitor="rmse")

    results_df = pd.DataFrame({"cv_rmse": [1.0, 0.9, 0.85, 0.8, 0.75, 0.74]})

    improvement = early_stop._check_metric_improvement(results_df, "rmse")
    assert improvement


@pytest.mark.unit
def test_should_stop_early_iterations():
    early_stop = EarlyStopping(patience=5)
    results_df = pd.DataFrame({"cv_roc": [0.5, 0.6, 0.65]})

    should_stop = early_stop.should_stop(results_df, current_iteration=3)
    assert should_stop is False


@pytest.mark.unit
def test_should_stop_with_improvement():
    early_stop = EarlyStopping(patience=3, min_improvement=0.01)

    results_df = pd.DataFrame({"cv_roc": [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]})

    should_stop = early_stop.should_stop(results_df, current_iteration=6)
    assert should_stop is False


@pytest.mark.unit
def test_should_stop_without_improvement():
    early_stop = EarlyStopping(patience=3, min_improvement=0.01)

    results_df = pd.DataFrame({"cv_roc": [0.5, 0.6, 0.7, 0.7, 0.7, 0.7]})

    should_stop = early_stop.should_stop(results_df, current_iteration=6)
    assert should_stop is True


@pytest.mark.unit
def test_get_best_iteration_higher_better():
    early_stop = EarlyStopping(monitor="auroc")

    results_df = pd.DataFrame(
        {
            "iteration": [1, 2, 3, 4, 5],
            "cv_roc": [0.5, 0.7, 0.6, 0.8, 0.75],
        }
    )

    best_iter = early_stop.get_best_iteration(results_df)
    assert best_iter == 4


@pytest.mark.unit
def test_get_best_iteration_lower_better():
    early_stop = EarlyStopping(monitor="rmse")

    results_df = pd.DataFrame(
        {
            "iteration": [1, 2, 3, 4, 5],
            "cv_rmse": [1.0, 0.7, 0.9, 0.5, 0.8],
        }
    )

    best_iter = early_stop.get_best_iteration(results_df, metric="rmse")
    assert best_iter == 4


@pytest.mark.unit
def test_check_metric_improvement_unknown_metric():
    early_stop = EarlyStopping(patience=3, min_improvement=0.01)

    results_df = pd.DataFrame({"cv_roc": [0.5, 0.6, 0.65, 0.7, 0.75, 0.76]})

    improvement = early_stop._check_metric_improvement(results_df, "unknown_metric")

    assert improvement is False


@pytest.mark.unit
def test_check_metric_improvement_missing_column():
    early_stop = EarlyStopping(patience=3, min_improvement=0.01)

    results_df = pd.DataFrame({"some_other_column": [0.5, 0.6, 0.65, 0.7, 0.75, 0.76]})

    improvement = early_stop._check_metric_improvement(results_df, "auroc")

    assert improvement is False


@pytest.mark.unit
def test_check_metric_improvement_with_previous_values():
    early_stop = EarlyStopping(patience=3, min_improvement=0.01)

    results_df = pd.DataFrame({"cv_roc": [0.5, 0.6, 0.65, 0.7, 0.68, 0.69, 0.72]})

    improvement = early_stop._check_metric_improvement(results_df, "auroc")

    assert improvement


@pytest.mark.unit
def test_should_stop_required_columns_calculation():
    early_stop = EarlyStopping(patience=3, monitor=["auroc", "auprc", "unknown"])

    results_df = pd.DataFrame(
        {
            "iteration": [1, 2, 3, 4, 5],
            "cv_roc": [0.5, 0.6, 0.65, 0.7, 0.75],
            "cv_prc": [0.4, 0.5, 0.55, 0.6, 0.65],
        }
    )

    should_stop = early_stop.should_stop(results_df, current_iteration=5)

    assert should_stop is False


@pytest.mark.unit
def test_should_stop_improvements_calculation():
    early_stop = EarlyStopping(
        patience=2, min_improvement=0.01, monitor=["auroc", "auprc"]
    )

    results_df = pd.DataFrame(
        {
            "iteration": [1, 2, 3, 4],
            "cv_roc": [0.5, 0.6, 0.61, 0.62],
            "cv_prc": [0.4, 0.5, 0.51, 0.52],
        }
    )

    should_stop = early_stop.should_stop(results_df, current_iteration=4)

    assert should_stop is False


@pytest.mark.unit
def test_get_best_iteration_metric_parameter_handling():
    early_stop = EarlyStopping(monitor="auroc")

    results_df = pd.DataFrame(
        {
            "iteration": [1, 2, 3, 4, 5],
            "cv_rmse": [1.0, 0.8, 0.9, 0.7, 0.85],
        }
    )

    best_iter = early_stop.get_best_iteration(results_df, metric="RMSE")
    assert best_iter == 4

    results_df["cv_roc"] = [0.5, 0.7, 0.6, 0.8, 0.75]
    best_iter_default = early_stop.get_best_iteration(results_df, metric=None)
    assert best_iter_default == 4


@pytest.mark.unit
def test_get_best_iteration_column_mapping():
    early_stop = EarlyStopping(monitor="r2")

    results_df = pd.DataFrame(
        {
            "iteration": [1, 2, 3, 4, 5],
            "cv_r2": [0.5, 0.7, 0.6, 0.8, 0.75],
        }
    )

    best_iter = early_stop.get_best_iteration(results_df, metric="r2")
    assert best_iter == 4


@pytest.mark.unit
def test_get_best_iteration_idxmax_idxmin():
    early_stop = EarlyStopping(monitor="auroc")

    results_df_roc = pd.DataFrame(
        {
            "iteration": [1, 2, 3, 4, 5],
            "cv_roc": [0.5, 0.7, 0.6, 0.8, 0.75],
        }
    )

    best_iter_roc = early_stop.get_best_iteration(results_df_roc, metric="auroc")
    assert best_iter_roc == 4

    results_df_rmse = pd.DataFrame(
        {
            "iteration": [1, 2, 3, 4, 5],
            "cv_rmse": [1.0, 0.8, 0.9, 0.6, 0.85],
        }
    )

    best_iter_rmse = early_stop.get_best_iteration(results_df_rmse, metric="rmse")
    assert best_iter_rmse == 4


@pytest.mark.unit
def test_get_best_iteration_without_iteration_column():
    early_stop = EarlyStopping(monitor="auroc")

    results_df = pd.DataFrame({"cv_roc": [0.5, 0.7, 0.6, 0.8, 0.75]})

    best_iter = early_stop.get_best_iteration(results_df, metric="auroc")
    assert best_iter == 4


@pytest.mark.unit
def test_check_metric_improvement_early_iterations():
    early_stop = EarlyStopping(patience=5, min_improvement=0.01)

    results_df = pd.DataFrame({"cv_roc": [0.5, 0.6, 0.65]})

    improvement = early_stop._check_metric_improvement(results_df, "auroc")
    assert improvement is True


@pytest.mark.unit
def test_should_stop_non_dataframe():
    early_stop = EarlyStopping(patience=3)

    not_df = [1, 2, 3]
    result = early_stop.should_stop(not_df, current_iteration=5)
    assert result is None


@pytest.mark.unit
def test_should_stop_missing_required_columns():
    early_stop = EarlyStopping(patience=3, monitor=["auroc", "auprc"])

    results_df = pd.DataFrame(
        {
            "iteration": [1, 2, 3, 4, 5],
            "some_other_metric": [0.5, 0.6, 0.65, 0.7, 0.75],
        }
    )

    result = early_stop.should_stop(results_df, current_iteration=5)
    assert result is None


@pytest.mark.unit
def test_get_best_iteration_non_dataframe():
    early_stop = EarlyStopping(monitor="auroc")

    not_df = {"not": "dataframe"}
    result = early_stop.get_best_iteration(not_df)
    assert result is None


@pytest.mark.unit
def test_get_best_iteration_unknown_metric():
    early_stop = EarlyStopping(monitor="auroc")

    results_df = pd.DataFrame(
        {"iteration": [1, 2, 3, 4, 5], "cv_roc": [0.5, 0.7, 0.6, 0.8, 0.75]}
    )

    result = early_stop.get_best_iteration(results_df, metric="unknown_metric")
    assert result is None


@pytest.mark.unit
def test_get_best_iteration_missing_column():
    early_stop = EarlyStopping(monitor="auroc")

    results_df = pd.DataFrame(
        {
            "iteration": [1, 2, 3, 4, 5],
            "some_other_column": [0.5, 0.7, 0.6, 0.8, 0.75],
        }
    )

    result = early_stop.get_best_iteration(results_df, metric="auroc")
    assert result is None


@pytest.mark.unit
def test_prepare_data_classification(sample_classification_data):
    trainer = TrainML()
    data, y = sample_classification_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    result = trainer._prepare_data(
        train_data, train_y, test_data, test_y, categorical=True
    )

    assert len(result) == 7
    (
        train_set_values,
        train_y_values,
        test_data_values,
        test_y_values,
        train_columns,
        label_encoder,
        class_labels,
    ) = result

    assert isinstance(train_set_values, np.ndarray)
    assert isinstance(train_y_values, np.ndarray)
    assert label_encoder is not None
    assert class_labels is not None


@pytest.mark.unit
def test_prepare_data_regression(sample_regression_data):
    trainer = TrainML()
    data, y = sample_regression_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    result = trainer._prepare_data(
        train_data, train_y, test_data, test_y, categorical=False
    )

    assert len(result) == 7
    (
        train_set_values,
        train_y_values,
        test_data_values,
        test_y_values,
        train_columns,
        label_encoder,
        class_labels,
    ) = result

    assert isinstance(train_set_values, np.ndarray)
    assert isinstance(train_y_values, np.ndarray)
    assert label_encoder is None
    assert class_labels is None


@pytest.mark.unit
def test_get_model_predictions_regression():
    trainer = TrainML()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1.5, 2.3])

    data = np.array([[1, 2], [3, 4]])
    result = trainer._get_model_predictions(mock_model, data, categorical=False)

    assert result.shape == (2, 1)
    mock_model.predict.assert_called_once_with(data)


@pytest.mark.unit
def test_create_feature_importance_with_rf():
    trainer = TrainML(model="RF")

    mock_model = MagicMock()
    mock_model.feature_importances_ = np.array([0.3, 0.7, 0.1])

    train_columns = ["feat1", "feat2", "feat3"]
    result = trainer._create_feature_importance(mock_model, train_columns)

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "feature" in result.columns
    assert "importance" in result.columns
    assert len(result) == 3


@pytest.mark.unit
def test_create_feature_importance_without_rf():
    trainer = TrainML(model="EN")

    mock_model = MagicMock()
    train_columns = ["feat1", "feat2", "feat3"]
    result = trainer._create_feature_importance(mock_model, train_columns)

    assert result is None


@pytest.mark.unit
def test_update_best_metrics_classification():
    trainer = TrainML()

    best_metrics = {"cv_roc": 0.7, "cv_prc": 0.6, "params": {}, "iteration": 1}
    cv_results = {"cv_roc_mean": 0.8, "cv_prc_mean": 0.75}
    params = {"test": "param"}

    updated = trainer._update_best_metrics(
        best_metrics, cv_results, params, 2, categorical=True
    )

    assert updated is True
    assert best_metrics["cv_roc"] == 0.8
    assert best_metrics["cv_prc"] == 0.75
    assert best_metrics["params"] == params
    assert best_metrics["iteration"] == 2


@pytest.mark.unit
def test_update_best_metrics_regression():
    trainer = TrainML()

    best_metrics = {
        "cv_roc": 1.0,
        "cv_prc": 0.6,
        "params": {},
        "iteration": 1,
    }
    cv_results = {"cv_rmse_mean": 0.8, "cv_r2_mean": 0.75}
    params = {"test": "param"}

    updated = trainer._update_best_metrics(
        best_metrics, cv_results, params, 2, categorical=False
    )

    assert updated is True
    assert best_metrics["cv_roc"] == 0.8
    assert best_metrics["cv_prc"] == 0.75


@pytest.mark.unit
def test_train_unsupported_model(sample_classification_data):
    trainer = TrainML(model="UNSUPPORTED")
    data, y = sample_classification_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    param_grid = pd.DataFrame({"param": [1]})

    result = trainer.train(
        train_data, train_y, test_data, test_y, param_grid, categorical=True
    )
    assert result is None


@pytest.mark.unit
def test_train_data_mismatch(sample_classification_data):
    trainer = TrainML()
    data, y = sample_classification_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:70]
    test_y = y.iloc[80:]

    param_grid = pd.DataFrame({"param": [1]})

    result = trainer.train(
        train_data, train_y, test_data, test_y, param_grid, categorical=True
    )
    assert result is None


@pytest.mark.unit
def test_get_model_predictions_classification_without_predict_proba():
    trainer = TrainML()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 0])
    mock_model.classes_ = np.array([0, 1])
    del mock_model.predict_proba

    data = np.array([[1, 2], [3, 4], [5, 6]])
    result = trainer._get_model_predictions(mock_model, data, categorical=True)

    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result[0], [1, 0])
    np.testing.assert_array_equal(result[1], [0, 1])
    np.testing.assert_array_equal(result[2], [1, 0])


@pytest.mark.unit
def test_train_with_sample_weights_non_svm_non_knn(sample_classification_data):
    trainer = TrainML(nfolds=2, model="RF", early_stopping=False, random_state=123)
    data, y = sample_classification_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    sample_weights = np.ones(len(train_y)) * 1.5

    param_grid = pd.DataFrame(
        {"ntree": [10], "mtry": [2], "nodesize": [5], "maxdepth": [5]}
    )

    result = trainer.train(
        train_data,
        train_y,
        test_data,
        test_y,
        param_grid,
        categorical=True,
        sample_weights=sample_weights,
    )

    assert result is not None
    assert "final_model" in result


@pytest.mark.unit
def test_train_with_non_dataframe_param_grid(sample_classification_data):
    trainer = TrainML(nfolds=2, model="RF", early_stopping=False, random_state=123)
    data, y = sample_classification_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    param_grid_dict = {"ntree": [10], "mtry": [2], "nodesize": [5], "maxdepth": [5]}

    result = trainer.train(
        train_data, train_y, test_data, test_y, param_grid_dict, categorical=True
    )

    assert result is not None
    assert "final_model" in result


@pytest.mark.unit
def test_train_with_early_stopping_triggered(sample_classification_data):
    trainer = TrainML(
        nfolds=2,
        model="RF",
        early_stopping=True,
        patience=1,
        min_improvement=0.9999,
        random_state=123,
    )
    data, y = sample_classification_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    param_combinations = [
        {"ntree": 10, "mtry": 2, "nodesize": 5, "maxdepth": 5},
        {"ntree": 20, "mtry": 2, "nodesize": 5, "maxdepth": 5},
        {"ntree": 30, "mtry": 2, "nodesize": 5, "maxdepth": 5},
        {"ntree": 40, "mtry": 2, "nodesize": 5, "maxdepth": 5},
    ]
    param_grid = pd.DataFrame(param_combinations)

    result = trainer.train(
        train_data, train_y, test_data, test_y, param_grid, categorical=True
    )

    assert result is not None
    assert "final_model" in result


@pytest.mark.unit
def test_train_no_valid_model_found(sample_classification_data):
    trainer = TrainML(nfolds=2, model="RF", early_stopping=False, random_state=123)
    data, y = sample_classification_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    param_grid = pd.DataFrame(
        {"ntree": [10], "mtry": [2], "nodesize": [5], "maxdepth": [5]}
    )

    with patch.object(trainer.cv, "run", return_value=None):
        result = trainer.train(
            train_data, train_y, test_data, test_y, param_grid, categorical=True
        )

        assert result is None


@pytest.mark.unit
def test_prepare_data_missing_values_check(sample_data_with_missing):
    trainer = TrainML()
    data, y = sample_data_with_missing

    train_data = data.iloc[:40]
    test_data = data.iloc[40:]
    train_y = y.iloc[:40]
    test_y = y.iloc[40:]

    with patch.object(trainer, "logger") as mock_logger:
        result = trainer._prepare_data(
            train_data, train_y, test_data, test_y, categorical=True
        )

        assert result is not None
        mock_logger.warn.assert_called()


@pytest.mark.unit
def test_prepare_data_column_mismatch():
    trainer = TrainML()

    train_data = pd.DataFrame({"A": [1, 2], "B": [4, 5]})
    test_data = pd.DataFrame({"A": [7, 8], "C": [9, 10]})
    train_y = pd.Series([0, 1])
    test_y = pd.Series([1, 0])

    result = trainer._prepare_data(
        train_data, train_y, test_data, test_y, categorical=True
    )

    assert result is None


@pytest.mark.unit
def test_prepare_data_length_mismatch():
    trainer = TrainML()

    train_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    test_data = pd.DataFrame({"A": [7, 8], "B": [9, 10]})
    train_y = pd.Series([0, 1])
    test_y = pd.Series([1, 0])

    result = trainer._prepare_data(
        train_data, train_y, test_data, test_y, categorical=True
    )

    assert result is None


@pytest.mark.unit
def test_get_model_predictions_exception_handling():
    trainer = TrainML()

    mock_model = MagicMock()
    mock_model.predict_proba.side_effect = Exception("Prediction failed")

    data = np.array([[1, 2], [3, 4]])

    result = trainer._get_model_predictions(mock_model, data, categorical=True)

    assert result is None


@pytest.mark.unit
def test_create_feature_importance_exception_handling():
    trainer = TrainML(model="RF")

    mock_model = MagicMock()
    del mock_model.feature_importances_

    train_columns = ["feat1", "feat2", "feat3"]

    result = trainer._create_feature_importance(mock_model, train_columns)

    assert result is None


@pytest.mark.unit
def test_update_best_metrics_no_improvement():
    trainer = TrainML()

    best_metrics = {"cv_roc": 0.8, "cv_prc": 0.7, "params": {}, "iteration": 1}
    cv_results = {"cv_roc_mean": 0.75, "cv_prc_mean": 0.65}
    params = {"test": "param"}

    updated = trainer._update_best_metrics(
        best_metrics, cv_results, params, 2, categorical=True
    )

    assert updated is False
    assert best_metrics["cv_roc"] == 0.8
    assert best_metrics["cv_prc"] == 0.7
    assert best_metrics["iteration"] == 1


@pytest.mark.unit
def test_train_data_validation_failure():
    trainer = TrainML()

    train_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    test_data = pd.DataFrame({"C": [7, 8], "D": [9, 10]})
    train_y = pd.Series([0, 1, 0])
    test_y = pd.Series([1, 0])

    param_grid = pd.DataFrame({"param": [1]})

    result = trainer.train(
        train_data,
        train_y,
        test_data,
        test_y,
        param_grid,
        categorical=True,
    )

    assert result is None


@pytest.mark.unit
def test_train_empty_param_grid():
    trainer = TrainML()
    data = pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})
    y = pd.Series([0, 1, 0, 1])

    train_data = data.iloc[:3]
    test_data = data.iloc[3:]
    train_y = y.iloc[:3]
    test_y = y.iloc[3:]

    param_grid = pd.DataFrame()

    result = trainer.train(
        train_data, train_y, test_data, test_y, param_grid, categorical=True
    )

    assert result is None


@pytest.mark.unit
def test_train_final_model_creation_failure(sample_classification_data):
    trainer = TrainML(nfolds=2, model="RF", early_stopping=False, random_state=123)
    data, y = sample_classification_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    invalid_param_grid = pd.DataFrame(
        {"ntree": [-10], "mtry": [2], "nodesize": [5], "maxdepth": [5]}
    )

    with patch("utils.MLUtils.ModelFactory.create_model") as mock_create:

        def smart_side_effect(*args, **kwargs):
            from sklearn.ensemble import RandomForestClassifier

            params = args[1] if len(args) > 1 else kwargs.get("params", {})

            if params.get("ntree", 0) < 0:
                raise ValueError("Invalid parameter: ntree must be positive")

            return RandomForestClassifier(
                n_estimators=max(10, params.get("ntree", 10)),
                max_features=2,
                min_samples_split=5,
                max_depth=5,
            )

        mock_create.side_effect = smart_side_effect

        result = trainer.train(
            train_data, train_y, test_data, test_y, invalid_param_grid, categorical=True
        )

        assert result is None


@pytest.mark.unit
def test_train_metrics_calculation_failure(sample_classification_data):
    trainer = TrainML(nfolds=2, model="RF", early_stopping=False, random_state=123)
    data, y = sample_classification_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    param_grid = pd.DataFrame(
        {"ntree": [10], "mtry": [2], "nodesize": [5], "maxdepth": [5]}
    )

    with patch.object(trainer.metrics_calculator, "calculate") as mock_calculate:
        original_call_count = [0]
        cv_calls = []
        final_calls = []

        def side_effect(*args, **kwargs):
            original_call_count[0] += 1
            call_info = (args, kwargs)

            if len(args) >= 3 or "task_type" in kwargs:
                final_calls.append(call_info)
                if len(final_calls) >= 2:
                    return None
                else:
                    return {"roc_value": 0.8, "prc_value": 0.7}
            else:
                cv_calls.append(call_info)
                return {"roc_value": 0.8, "prc_value": 0.7}

        mock_calculate.side_effect = side_effect

        result = trainer.train(
            train_data, train_y, test_data, test_y, param_grid, categorical=True
        )

        assert result is not None
        assert result["test_metrics"] is None or result["train_metrics"] is None


@pytest.mark.unit
def test_train_cv_result_append(sample_classification_data):
    trainer = TrainML(nfolds=2, model="RF", early_stopping=False, random_state=123)
    data, y = sample_classification_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    param_combinations = [
        {"ntree": 10, "mtry": 2, "nodesize": 5, "maxdepth": 5},
        {"ntree": 20, "mtry": 2, "nodesize": 5, "maxdepth": 5},
    ]
    param_grid = pd.DataFrame(param_combinations)

    result = trainer.train(
        train_data, train_y, test_data, test_y, param_grid, categorical=True
    )

    assert result is not None
    assert "cv_results" in result
    cv_results_df = result["cv_results"]
    assert len(cv_results_df) == 2
    assert "iteration" in cv_results_df.columns
    assert "cv_roc" in cv_results_df.columns


@pytest.mark.integration
def test_train_classification_small_grid(sample_classification_data):
    trainer = TrainML(nfolds=2, model="RF", early_stopping=False, random_state=123)
    data, y = sample_classification_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    param_combinations = [
        {"ntree": 10, "mtry": 2, "nodesize": 5, "maxdepth": 5},
        {"ntree": 20, "mtry": 3, "nodesize": 5, "maxdepth": 5},
    ]
    param_grid = pd.DataFrame(param_combinations)

    result = trainer.train(
        train_data, train_y, test_data, test_y, param_grid, categorical=True
    )

    assert result is not None
    assert "final_model" in result
    assert "train_metrics" in result
    assert "test_metrics" in result
    assert "best_params" in result
    assert "cv_results" in result


@pytest.mark.integration
def test_train_regression_small_grid(sample_regression_data):
    trainer = TrainML(nfolds=2, model="RF", early_stopping=False, random_state=123)
    data, y = sample_regression_data

    train_data = data.iloc[:80]
    test_data = data.iloc[80:]
    train_y = y.iloc[:80]
    test_y = y.iloc[80:]

    param_grid = pd.DataFrame(
        {"ntree": [10], "mtry": [2], "nodesize": [5], "maxdepth": [5]}
    )

    result = trainer.train(
        train_data, train_y, test_data, test_y, param_grid, categorical=False
    )

    assert result is not None
    assert "final_model" in result
    assert "train_metrics" in result
    assert "test_metrics" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
