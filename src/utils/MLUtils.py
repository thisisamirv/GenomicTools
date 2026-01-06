#!/usr/bin/env python
# Import required modules
import gc
import importlib
import itertools
import numpy as np
import os
import pandas as pd
import time
import warnings
from joblib import Parallel, delayed
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    auc,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import (
    LabelEncoder,
    PowerTransformer,
    StandardScaler,
    label_binarize,
)
from sklearn.svm import SVC, SVR
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from .LoggingUtils import log

try:
    importlib.import_module("sklearn.experimental.enable_iterative_imputer")
    from sklearn.impute import IterativeImputer

    log.debug("IterativeImputer is available for use.")
except Exception:
    IterativeImputer = SimpleImputer
    log.warning("IterativeImputer is not available; using SimpleImputer as fallback.")


class BaseMLUtils:
    def __init__(self, random_state: int = 123) -> None:
        self.random_state: int = random_state

    def _to_numpy(self, data: Any) -> np.ndarray:
        """Convert input data to a numpy array."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return np.array(data)

    def _validate_data_match(self, X: Any, y: Any) -> None:
        """Validate that X and y have matching lengths."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

    def _validate_numeric_data(self, data: Any, name: str = "data") -> bool:
        """Validate that the data contains only numeric values."""
        if isinstance(data, pd.DataFrame):
            non_numeric_cols = data.select_dtypes(exclude=["number"]).columns
            if len(non_numeric_cols) > 0:
                raise ValueError(
                    f"Non-numeric columns detected in {name}: {list(non_numeric_cols)}"
                )
        return True

    def _safe_execute(self, func: Callable[[], Any], error_msg: str) -> Any:
        """Execute a function safely, logging errors if they occur."""
        try:
            return func()
        except Exception as e:
            log.error(f"{error_msg}: {str(e)}")
            return None

    def _create_transformer_dataframe(
        self,
        transformer: Any,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        prefix: str = "",
    ) -> pd.DataFrame:
        """Apply a transformer and return a DataFrame with appropriate column names."""
        transformed_data = transformer.fit_transform(data)
        if columns is None:
            columns = [f"{prefix}{i + 1}" for i in range(transformed_data.shape[1])]
        return pd.DataFrame(transformed_data, columns=columns)


class DataProcessor:
    @staticmethod
    def apply_imputer(
        data: pd.DataFrame, imputer_type: str, random_state: int = 123
    ) -> pd.DataFrame:
        """Apply specified imputation method to the data."""
        imputers: Dict[str, Any] = {
            "medianImpute": SimpleImputer(strategy="median"),
            "knnImpute": KNNImputer(n_neighbors=5),
            "bagImpute": IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=10, random_state=random_state
                ),
                random_state=random_state,
                max_iter=10,
            ),
        }
        log.debug(f"Applying {imputer_type} imputation.")
        imputer = imputers[imputer_type]
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    @staticmethod
    def apply_transformer(data: pd.DataFrame, transformer_type: str) -> pd.DataFrame:
        """Apply specified transformation to the data."""
        transformers: Dict[str, Any] = {
            "YeoJohnson": PowerTransformer(method="yeo-johnson"),
            "center": StandardScaler(with_mean=True, with_std=False),
            "scale": StandardScaler(with_mean=False, with_std=True),
        }
        log.debug(f"Applying {transformer_type} transformation.")
        transformer = transformers[transformer_type]
        return pd.DataFrame(transformer.fit_transform(data), columns=data.columns)

    @staticmethod
    def remove_correlated_features(
        data: pd.DataFrame, threshold: float = 0.9
    ) -> pd.DataFrame:
        """Remove features that are highly correlated above the specified threshold."""
        log.debug("Removing highly correlated features.")
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        log.debug(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
        return data.drop(columns=to_drop)

    @staticmethod
    def apply_outlier_capping(data: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers in the data using robust scaling."""
        log.debug("Applying outlier capping.")

        def rob_scale(x: np.ndarray) -> np.ndarray:
            q05, q95 = np.nanpercentile(x, [5, 95])
            iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
            lb, ub = q05 - 1.5 * iqr, q95 + 1.5 * iqr
            return np.clip(x, lb, ub)

        return data.apply(rob_scale)


class DataPreprocessor(BaseMLUtils):
    def __init__(self, random_state: int = 123) -> None:
        super().__init__(random_state)
        self.method_handlers: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
            "medianImpute": lambda data: DataProcessor.apply_imputer(
                data, "medianImpute", self.random_state
            ),
            "knnImpute": lambda data: DataProcessor.apply_imputer(
                data, "knnImpute", self.random_state
            ),
            "bagImpute": lambda data: DataProcessor.apply_imputer(
                data, "bagImpute", self.random_state
            ),
            "corr": DataProcessor.remove_correlated_features,
            "YeoJohnson": lambda data: DataProcessor.apply_transformer(
                data, "YeoJohnson"
            ),
            "center": lambda data: DataProcessor.apply_transformer(data, "center"),
            "scale": lambda data: DataProcessor.apply_transformer(data, "scale"),
            "pca": self._apply_pca,
            "ica": self._apply_ica,
            "zv": self._apply_zero_variance,
            "nzv": self._apply_near_zero_variance,
            "outlierCapping": DataProcessor.apply_outlier_capping,
        }

    def _apply_pca(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA to reduce dimensionality while retaining 95% variance."""
        log.debug("Applying PCA.")
        return self._create_transformer_dataframe(
            PCA(n_components=0.95), data, prefix="PC"
        )

    def _apply_ica(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply ICA to the data."""
        log.debug("Applying ICA.")
        n_components = min(data.shape)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FastICA did not converge")
            return self._create_transformer_dataframe(
                FastICA(n_components=n_components, random_state=self.random_state),
                data,
                prefix="IC",
            )

    def _apply_zero_variance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove features with zero variance."""
        log.debug("Applying zero variance filter.")
        selector = VarianceThreshold(threshold=0)
        transformed = selector.fit_transform(data)
        selected_cols = [
            col for i, col in enumerate(data.columns) if selector.get_support()[i]
        ]
        return pd.DataFrame(transformed, columns=selected_cols)

    def _apply_near_zero_variance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove features with near zero variance."""
        log.debug("Applying near zero variance filter.")
        to_drop: List[str] = []
        for col in data.columns:
            counts = data[col].value_counts()
            if len(counts) <= 1:
                to_drop.append(col)
                continue
            freq_ratio = (
                counts.iloc[0] / counts.iloc[1] if len(counts) > 1 else float("inf")
            )
            unique_pct = (len(counts) / len(data)) * 100
            if freq_ratio > 19 and unique_pct < 10:
                to_drop.append(col)
        log.debug(f"Dropping {len(to_drop)} near-zero variance features.")
        return data.drop(columns=to_drop)

    def preprocess(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, methods: List[str]
    ) -> Any:
        """Preprocess train and test datasets using specified methods."""

        def _preprocess() -> Dict[str, pd.DataFrame]:
            if list(train_df.columns) != list(test_df.columns):
                raise ValueError(
                    "Train and test datasets must have identical column names"
                )
            self._validate_numeric_data(train_df, "train_df")
            self._validate_numeric_data(test_df, "test_df")
            invalid_methods = [m for m in methods if m not in self.method_handlers]
            if invalid_methods:
                raise ValueError(
                    f"Invalid methods: {invalid_methods}. Valid: {list(self.method_handlers.keys())}"
                )
            train_rows = train_df.shape[0]
            data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
            log.debug(f"Applying methods: {methods}")
            for method in methods:
                data = self.method_handlers[method](data)
            return {
                "train": data.iloc[:train_rows].copy(),
                "test": data.iloc[train_rows:].copy(),
            }

        return self._safe_execute(_preprocess, "Error in DataPreprocessor.preprocess")


class ParameterGrid(BaseMLUtils):
    def __init__(self, random_state: int = 123) -> None:
        super().__init__(random_state)
        self.supported_models: List[str] = ["EN", "KNN", "RF", "SVM"]

    def _create_range(
        self,
        start: float,
        stop: float,
        step: Optional[float] = None,
        num: Optional[int] = None,
    ) -> np.ndarray:
        """Create a range of values either by step or by number of points."""
        if num:
            return np.linspace(start, stop, num)
        return np.arange(start, stop + (step or 1), step or 1)

    def _get_default_params(
        self, model: str, n_features: int, n_samples: int, feature_scale: float
    ) -> Dict[str, Any]:
        """Get default hyperparameter configurations for the specified model."""
        configs: Dict[str, Dict[str, Any]] = {
            "EN": {
                "alpha": self._create_range(0.1, 1.0, 0.1),
                "lambda": np.exp(np.linspace(np.log(1), np.log(0.0001), 100)),
            },
            "KNN": {
                "k": np.ceil(self._create_range(5, 30, 2) * feature_scale).astype(int),
                "algorithm": ["kd_tree", "cover_tree"],
            },
            "RF": {
                "mtry": np.unique(
                    np.minimum(
                        np.maximum(
                            np.round(np.array([0.15, 0.25, 0.33]) * feature_scale), 1
                        ),
                        n_features,
                    )
                ).astype(int),
                "ntree": [500, 1000],
                "nodesize": [30, 40],
                "maxdepth": [10, 15],
                "sampsize": [round(0.6 * n_samples)],
            },
            "SVM": {
                "cost": np.array([0.1, 0.25, 0.5, 1]) * feature_scale,
                "gamma": self._create_range(0.001, 0.075, 0.005) * feature_scale,
                "kernel": ["radial"],
            },
        }
        return configs.get(model, {})

    def _get_custom_params(
        self,
        params: Dict[str, Any],
        model: str,
        n_features: int,
        n_samples: int,
        feature_scale: float,
    ) -> Dict[str, Any]:
        """Get custom hyperparameter configurations for the specified model."""
        gamma_range = self._create_range(
            params.get("gamma_min", 0.001),
            params.get("gamma_max", 0.075),
            params.get("gamma_step", 0.005),
        )
        cost = params.get("cost_constants", [0.1, 0.25, 0.5, 1])
        mtry = np.array(params.get("mtry_constant", [0.15, 0.25, 0.33]))
        k_range = self._create_range(
            params.get("kmin", 5),
            params.get("kmax", 30),
            params.get("k_step", 2),
        )
        generators: Dict[str, Callable[[], Dict[str, Any]]] = {
            "EN": lambda: {
                "alpha": self._create_range(0.1, 1.0, params.get("alpha_step", 0.1)),
                "lambda": np.exp(
                    np.linspace(
                        np.log(params.get("lambda_max", 1)),
                        np.log(params.get("lambda_min", 0.0001)),
                        params.get("n_lambda", 100),
                    )
                ),
            },
            "KNN": lambda: {
                "k": np.ceil(k_range * feature_scale).astype(int),
                "algorithm": params.get("algorithm", ["kd_tree", "cover_tree"]),
            },
            "RF": lambda: {
                "mtry": np.unique(
                    np.minimum(
                        np.maximum(
                            np.round(mtry * feature_scale),
                            1,
                        ),
                        n_features,
                    )
                ).astype(int),
                "ntree": params.get("ntree", [500, 1000]),
                "nodesize": params.get("nodesize", [30, 40]),
                "maxdepth": params.get("maxdepth", [10, 15]),
                "sampsize": [round(params.get("sampsize_constant", 0.6) * n_samples)],
            },
            "SVM": lambda: {
                "cost": np.array(cost) * feature_scale,
                "gamma": gamma_range * feature_scale,
                "kernel": [params.get("kernel", "radial")],
            },
        }
        return generators.get(model, lambda: {})()

    def generate(
        self,
        model: str,
        params: Optional[Dict[str, Any]] = None,
        n_features: Optional[int] = None,
        n_samples: Optional[int] = None,
        feature_scale: Optional[float] = None,
    ) -> Any:
        """Generate a parameter grid for the specified model."""

        def _generate() -> pd.DataFrame:
            if model not in self.supported_models:
                raise ValueError(
                    f"Model {model} not supported. Supported: {self.supported_models}"
                )
            log.debug(f"Generating parameter grid for model: {model}")
            param_dict = (
                self._get_custom_params(
                    params, model, n_features, n_samples, feature_scale
                )
                if params
                else self._get_default_params(
                    model, n_features, n_samples, feature_scale
                )
            )
            if not param_dict:
                raise ValueError(f"No parameters defined for model {model}")
            param_grid = pd.DataFrame(
                list(itertools.product(*param_dict.values())),
                columns=list(param_dict.keys()),
            )
            if len(param_grid) == 0:
                raise ValueError(f"Parameter grid is empty for model {model}")
            log.debug(f"Generated {len(param_grid)} parameter combinations")
            return param_grid

        return self._safe_execute(_generate, "Error in ParameterGrid.generate")


class NoiseInjector(BaseMLUtils):
    def add_noise(
        self, train_set: pd.DataFrame, noise_level: float, adaptive: bool = True
    ) -> Any:
        """Add Gaussian noise to the training dataset."""

        def _add_noise() -> pd.DataFrame:
            if noise_level < 0:
                raise ValueError("noise_level must be a non-negative value")
            self._validate_numeric_data(train_set, "train_set")
            np.random.seed(self.random_state)
            log.debug(f"Adding noise - level: {noise_level}, adaptive: {adaptive}")
            adjusted_noise_level = noise_level
            if adaptive:
                n_features = train_set.shape[1]
                adjusted_noise_level = noise_level * np.log10(n_features / 100 + 1)
                log.debug(f"Adjusted noise level: {adjusted_noise_level}")
            noise_matrix = np.random.normal(0, adjusted_noise_level, train_set.shape)
            noisy_train_set = train_set + pd.DataFrame(
                noise_matrix, index=train_set.index, columns=train_set.columns
            )
            return noisy_train_set

        return self._safe_execute(_add_noise, "Error in NoiseInjector.add_noise")


class SampleWeights(BaseMLUtils):
    def __init__(self, strategy: str = "balanced", random_state: int = 123) -> None:
        super().__init__(random_state)
        self.strategy: str = strategy
        self.supported_strategies: List[str] = [
            "balanced",
            "balanced_subsample",
            "custom",
        ]

    def _calculate_balanced_weights(self, train_y: Iterable[Any]) -> np.ndarray:
        """Calculate balanced sample weights based on class distribution."""
        classes, counts = np.unique(train_y, return_counts=True)
        n_samples, n_classes = len(train_y), len(classes)
        class_weights = n_samples / (n_classes * counts)
        class_weights = (class_weights / np.sum(class_weights)) * n_classes
        weight_dict = dict(zip(classes, class_weights))
        log.debug(f"Class distribution: {dict(zip(classes, counts))}")
        log.debug(f"Class weights: {dict(zip(classes, class_weights))}")
        return np.array([weight_dict[y] for y in train_y])

    def calculate(
        self,
        train_data: Union[pd.DataFrame, np.ndarray],
        train_y: Iterable[Any],
        custom_weights: Optional[Dict[Any, float]] = None,
    ) -> Any:
        """Calculate sample weights based on the specified strategy."""

        def _calculate() -> np.ndarray:
            if not isinstance(train_data, (pd.DataFrame, np.ndarray)):
                raise ValueError("train_data must be a DataFrame or numpy array")
            if not isinstance(train_y, (list, np.ndarray, pd.Series)):
                raise ValueError(
                    "train_y must be a list, numpy array, or pandas Series"
                )
            train_y_values = self._to_numpy(train_y)
            self._validate_data_match(train_data, train_y_values)
            if self.strategy not in self.supported_strategies:
                raise ValueError(f"Unsupported strategy: {self.strategy}")
            log.debug(f"Calculating sample weights using strategy: {self.strategy}")
            if self.strategy == "balanced":
                sample_weights = self._calculate_balanced_weights(train_y_values)
            elif self.strategy == "balanced_subsample":
                np.random.seed(self.random_state)
                sample_weights = self._calculate_balanced_weights(train_y_values)
                noise = np.random.normal(0, 0.01, len(sample_weights))
                sample_weights = np.maximum(sample_weights + noise, 0.01)
            elif self.strategy == "custom":
                if custom_weights is None:
                    raise ValueError(
                        "custom_weights must be provided for 'custom' strategy"
                    )
                sample_weights = np.array(
                    [custom_weights.get(y, 1.0) for y in train_y_values]
                )
                log.debug(f"Applied custom weights: {custom_weights}")
            log.debug(
                f"Sample weights range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]"
            )
            return sample_weights

        return self._safe_execute(_calculate, "Error in SampleWeights.calculate")


class PerformanceMetrics(BaseMLUtils):
    def __init__(
        self, task_type: Optional[str] = None, random_state: int = 123
    ) -> None:
        super().__init__(random_state)
        self.task_type: Optional[str] = task_type
        self.supported_tasks: List[str] = ["classification", "regression"]

    def _preprocess_inputs(
        self, pred_prob: Any, true_labels: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess prediction probabilities and true labels into numpy arrays."""
        pred_prob = self._to_numpy(pred_prob)
        true_labels = self._to_numpy(true_labels)
        if pred_prob.ndim == 1:
            pred_prob = pred_prob.reshape(-1, 1)
        return pred_prob, true_labels

    def _detect_task_type(self, true_labels: Any) -> str:
        """Detect whether the task is classification or regression based on true labels."""
        if self.task_type is not None:
            return self.task_type
        unique_values = np.unique(true_labels)
        condition1 = len(unique_values) <= 10
        condition2 = np.all(np.isclose(true_labels, np.round(true_labels)))
        return (
            "classification"
            if condition1 and condition2 and len(unique_values) > 1
            else "regression"
        )

    def _calculate_regression_metrics(
        self, pred_values: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression performance metrics."""
        metrics: Dict[str, float] = {
            "mse": mean_squared_error(true_labels, pred_values),
            "mae": mean_absolute_error(true_labels, pred_values),
            "r2": r2_score(true_labels, pred_values),
            "explained_variance": explained_variance_score(true_labels, pred_values),
        }
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        log.debug(
            f"Regression metrics - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}"
        )
        return metrics

    def _calculate_classification_metrics(
        self, pred_prob: np.ndarray, true_labels: np.ndarray, class_levels: Any
    ) -> Dict[str, float]:
        """Calculate classification performance metrics."""
        log.debug(
            f"{'Binary' if len(class_levels) == 2 else 'Multi-class'} classification detected"
        )
        if len(class_levels) == 2:
            if pred_prob.shape[1] == 2:
                pred_prob_1d = pred_prob[:, 1]
            else:
                pred_prob_1d = pred_prob.ravel()
            true_labels_binary = (
                (true_labels == class_levels[1]).astype(int)
                if not np.all(np.isin(true_labels, [0, 1]))
                else true_labels
            )
            roc_value = roc_auc_score(true_labels_binary, pred_prob_1d)
            precision, recall, _ = precision_recall_curve(
                true_labels_binary, pred_prob_1d
            )
            prc_value = auc(recall, precision)
        else:
            true_labels_onehot = label_binarize(true_labels, classes=class_levels)
            roc_values: List[float] = []
            prc_values: List[float] = []
            for i, class_name in enumerate(class_levels):
                try:
                    class_probs = pred_prob[:, i]
                    roc = roc_auc_score(true_labels_onehot[:, i], class_probs)
                    precision, recall, _ = precision_recall_curve(
                        true_labels_onehot[:, i], class_probs
                    )
                    prc = auc(recall, precision)
                    roc_values.append(roc)
                    prc_values.append(prc)
                    log.debug(
                        f"Class {class_name}: ROC AUC = {roc:.4f}, PRC AUC = {prc:.4f}"
                    )
                except Exception as e:
                    log.warn(f"Error calculating metrics for class {class_name}: {e}")
                    continue
            roc_value, prc_value = (
                (np.mean(roc_values), np.mean(prc_values))
                if roc_values and prc_values
                else (0.0, 0.0)
            )
        return {"roc_value": float(roc_value), "prc_value": float(prc_value)}

    def calculate(
        self, pred_prob: Any, true_labels: Any, task_type: Optional[str] = None
    ) -> Any:
        """Calculate performance metrics based on task type."""

        def _calculate() -> Dict[str, float]:
            pred_prob_processed, true_labels_processed = self._preprocess_inputs(
                pred_prob, true_labels
            )
            current_task_type = task_type or self._detect_task_type(
                true_labels_processed
            )
            log.debug(f"Task type detected: {current_task_type}")
            if current_task_type not in self.supported_tasks:
                raise ValueError(f"Unsupported task type: {current_task_type}")
            if current_task_type == "regression":
                if pred_prob_processed.ndim > 1 and pred_prob_processed.shape[1] > 1:
                    log.warn(
                        "Multiple prediction columns for regression task. Using first column."
                    )
                    pred_values = pred_prob_processed[:, 0]
                else:
                    pred_values = pred_prob_processed.ravel()
                return self._calculate_regression_metrics(
                    pred_values, true_labels_processed
                )
            else:
                class_levels = (
                    true_labels_processed.categories.tolist()
                    if hasattr(true_labels_processed, "categories")
                    else np.unique(true_labels_processed)
                )
                metrics = self._calculate_classification_metrics(
                    pred_prob_processed, true_labels_processed, class_levels
                )
                log.debug(
                    f"Overall ROC AUC: {metrics['roc_value']:.4f}, PRC AUC: {metrics['prc_value']:.4f}"
                )
                return metrics

        return self._safe_execute(_calculate, "Error in PerformanceMetrics.calculate")


class ModelFactory:
    @staticmethod
    def create_model(
        model_type: str,
        params: Dict[str, Any],
        categorical: bool,
        random_state: int,
        fold_weights: Optional[Any] = None,
    ) -> Any:
        """Create a machine learning model instance based on the specified type and parameters."""
        creators = {
            "EN": ModelFactory._create_elastic_net,
            "KNN": ModelFactory._create_knn,
            "RF": ModelFactory._create_random_forest,
            "SVM": ModelFactory._create_svm,
        }
        if model_type not in creators:
            raise ValueError(f"Invalid model type: {model_type}")
        return creators[model_type](params, categorical, random_state, fold_weights)

    @staticmethod
    def _create_elastic_net(
        params: Dict[str, Any],
        categorical: bool,
        random_state: int,
        fold_weights: Optional[Any] = None,
    ) -> Any:
        """Create an Elastic Net or Logistic Regression model based on parameters."""
        common_params = {
            "max_iter": 10000,
            "random_state": random_state,
        }
        if categorical:
            common_params.update(
                {
                    "C": 1 / params["lambda"],
                    "l1_ratio": params["alpha"],
                }
            )
            if params.get("n_classes", 2) > 2:
                return LogisticRegression(
                    **common_params, multi_class="multinomial", solver="saga"
                )
            else:
                return LogisticRegression(
                    **common_params, penalty="elasticnet", solver="saga"
                )
        else:
            return ElasticNet(
                alpha=params["lambda"], l1_ratio=params["alpha"], **common_params
            )

    @staticmethod
    def _create_knn(
        params: Dict[str, Any],
        categorical: bool,
        random_state: int,
        fold_weights: Optional[Any] = None,
    ) -> Any:
        """Create a K-Nearest Neighbors model based on parameters."""
        common_params = {
            "n_neighbors": params["k"],
            "algorithm": params.get("algorithm", "auto").lower(),
            "weights": "uniform",
        }
        return (
            KNeighborsClassifier(**common_params)
            if categorical
            else KNeighborsRegressor(**common_params)
        )

    @staticmethod
    def _create_random_forest(
        params: Dict[str, Any],
        categorical: bool,
        random_state: int,
        fold_weights: Optional[Any] = None,
    ) -> Any:
        """Create a Random Forest model based on parameters."""
        common_params = {
            "n_estimators": params["ntree"],
            "max_features": params["mtry"],
            "min_samples_split": params["nodesize"],
            "max_depth": params["maxdepth"],
            "random_state": random_state,
            "n_jobs": 1,
        }
        if "sampsize" in params:
            common_params["max_samples"] = params["sampsize"] / params.get(
                "fold_size", 1
            )
        if categorical:
            return RandomForestClassifier(
                **common_params,
                class_weight="balanced" if fold_weights is None else None,
            )
        else:
            return RandomForestRegressor(**common_params)

    @staticmethod
    def _create_svm(
        params: Dict[str, Any],
        categorical: bool,
        random_state: int,
        fold_weights: Optional[Any] = None,
    ) -> Any:
        """Create a Support Vector Machine model based on parameters."""
        if categorical:
            common_params = {
                "C": params["cost"],
                "gamma": params["gamma"],
                "kernel": params.get("kernel", "rbf"),
                "random_state": random_state,
            }
            return SVC(**common_params, class_weight=fold_weights, probability=True)
        else:
            common_params = {
                "C": params["cost"],
                "gamma": params["gamma"],
                "kernel": params.get("kernel", "rbf"),
            }
            return SVR(**common_params)


class CrossValidation(BaseMLUtils):
    def __init__(
        self, nfolds: int = 5, random_state: int = 123, n_jobs: Optional[int] = None
    ) -> None:
        super().__init__(random_state)
        self.nfolds: int = nfolds
        self.n_jobs: int = n_jobs if n_jobs is not None else max(1, os.cpu_count() - 1)
        self.supported_models: List[str] = ["EN", "KNN", "RF", "SVM"]
        self.metrics_calculator: PerformanceMetrics = PerformanceMetrics()

    def _process_sample_weights(
        self,
        sample_weights: Optional[np.ndarray],
        train_idx: np.ndarray,
        fold_train_y: np.ndarray,
        model: str,
        categorical: bool,
    ) -> Optional[Union[np.ndarray, Dict[Any, float]]]:
        """Process sample weights for the current fold."""
        if sample_weights is None:
            return None

        if model == "SVM" and categorical:
            classes = np.unique(fold_train_y)
            class_weights: Dict[Any, float] = {}
            for cls in classes:
                class_mask = fold_train_y == cls
                if np.any(class_mask):
                    class_weights[cls] = np.mean(sample_weights[train_idx][class_mask])
            return class_weights
        elif model != "KNN":
            return sample_weights[train_idx]

    def _get_predictions(
        self, model_instance: Any, test_set: np.ndarray, categorical: bool
    ) -> np.ndarray:
        """Get prediction probabilities or values from the model instance."""
        if categorical:
            if hasattr(model_instance, "predict_proba"):
                return model_instance.predict_proba(test_set)
            else:
                pred = model_instance.predict(test_set)
                classes = model_instance.classes_
                pred_prob = np.zeros((len(pred), len(classes)))
                for i, cls in enumerate(classes):
                    pred_prob[:, i] = (pred == cls).astype(float)
                return pred_prob
        else:
            return model_instance.predict(test_set).reshape(-1, 1)

    def _run_single_fold(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        train_set: np.ndarray,
        train_y: np.ndarray,
        params: Dict[str, Any],
        model: str,
        categorical: bool,
        sample_weights: Optional[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        """Run a single fold of cross-validation."""
        try:
            log.debug(f"Processing fold {fold_idx + 1}/{self.nfolds}")

            fold_train_set, fold_test_set = train_set[train_idx], train_set[test_idx]
            fold_train_y, fold_test_y = train_y[train_idx], train_y[test_idx]

            fold_weights = self._process_sample_weights(
                sample_weights, train_idx, fold_train_y, model, categorical
            )

            params_copy = params.copy()
            params_copy.update(
                {
                    "fold_size": len(train_idx),
                    "n_classes": len(np.unique(fold_train_y)) if categorical else 2,
                }
            )

            model_instance = ModelFactory.create_model(
                model, params_copy, categorical, self.random_state, fold_weights
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if fold_weights is not None and model != "SVM":
                    model_instance.fit(
                        fold_train_set, fold_train_y, sample_weight=fold_weights
                    )
                else:
                    model_instance.fit(fold_train_set, fold_train_y)

            fold_pred_prob = self._get_predictions(
                model_instance, fold_test_set, categorical
            )
            task_type = "classification" if categorical else "regression"
            metrics = self.metrics_calculator.calculate(
                fold_pred_prob, fold_test_y, task_type=task_type
            )

            if metrics is None:
                raise ValueError("Performance metrics calculation failed")

            result: Dict[str, Any] = {"fold": fold_idx, "model": model_instance}

            if categorical:
                result.update(
                    {"roc": metrics["roc_value"], "prc": metrics["prc_value"]}
                )
            else:
                result.update(
                    {metric: metrics[metric] for metric in ["rmse", "r2", "mse", "mae"]}
                )

            return result

        except Exception as e:
            log.error(f"Error in fold {fold_idx + 1}: {str(e)}")
            return None

    def run(
        self,
        train_set: Union[pd.DataFrame, np.ndarray],
        train_y: Union[pd.Series, List[Any], np.ndarray],
        params: Dict[str, Any],
        model: str = "RF",
        categorical: bool = True,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Any:
        """Run cross-validation on the training dataset."""

        def _run() -> Dict[str, Any]:
            train_set_values = self._to_numpy(train_set)
            train_y_values = self._to_numpy(train_y)

            if model not in self.supported_models:
                raise ValueError(
                    f"Unsupported model type: {model}. Supported: {self.supported_models}"
                )

            np.random.seed(self.random_state)

            log.debug("Creating folds for cross-validation.")
            cv_splitter = (
                StratifiedKFold(
                    n_splits=self.nfolds, shuffle=True, random_state=self.random_state
                )
                if categorical
                else KFold(
                    n_splits=self.nfolds, shuffle=True, random_state=self.random_state
                )
            )

            log.debug(f"Using {self.n_jobs} CPU cores for parallel processing.")

            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self._run_single_fold)(
                    i,
                    train_idx,
                    test_idx,
                    train_set_values,
                    train_y_values,
                    params,
                    model,
                    categorical,
                    sample_weights,
                )
                for i, (train_idx, test_idx) in enumerate(
                    cv_splitter.split(
                        train_set_values,
                        (
                            train_y_values
                            if categorical
                            else np.zeros(len(train_y_values))
                        ),
                    )
                )
            )

            valid_results = [r for r in results if r is not None]
            if not valid_results:
                raise ValueError("All folds failed in cross-validation")

            cv_result: Dict[str, Any] = {"fold_results": valid_results}

            if categorical:
                cv_result.update(
                    {
                        "cv_roc_mean": np.mean([r["roc"] for r in valid_results]),
                        "cv_prc_mean": np.mean([r["prc"] for r in valid_results]),
                    }
                )
                log.debug(
                    f"CV completed: ROC AUC = {cv_result['cv_roc_mean']:.4f}, PRC AUC = {cv_result['cv_prc_mean']:.4f}"
                )
            else:
                for metric in ["rmse", "r2", "mse", "mae"]:
                    cv_result[f"cv_{metric}_mean"] = np.mean(
                        [r[metric] for r in valid_results]
                    )
                log.debug(
                    f"CV completed: RMSE = {cv_result['cv_rmse_mean']:.4f}, R² = {cv_result['cv_r2_mean']:.4f}"
                )

            return cv_result

        return self._safe_execute(_run, "Error in CrossValidation.run")


class EarlyStopping(BaseMLUtils):
    def __init__(
        self,
        patience: int = 10,
        min_improvement: float = 0.0005,
        monitor: Union[str, List[str]] = "auroc",
        random_state: int = 123,
    ) -> None:
        super().__init__(random_state)
        self.patience: int = patience
        self.min_improvement: float = min_improvement
        self.monitor: List[str] = (
            [monitor.lower()]
            if isinstance(monitor, str)
            else [m.lower() for m in monitor]
        )
        self.column_mapping: Dict[str, str] = {
            "auroc": "cv_roc",
            "auprc": "cv_prc",
            "rmse": "cv_rmse",
            "r2": "cv_r2",
            "mse": "cv_mse",
            "mae": "cv_mae",
        }
        self.higher_better: set = {"auroc", "auprc", "r2"}

    def _check_metric_improvement(self, results_df: pd.DataFrame, metric: str) -> bool:
        """Check if there is improvement in the specified metric over the patience period."""
        if metric not in self.column_mapping:
            log.warn(f"Unknown metric: {metric}. Skipping.")
            return False

        column_name = self.column_mapping[metric]
        if column_name not in results_df.columns:
            log.warn(f"Column {column_name} not found. Skipping metric {metric}.")
            return False

        recent_values = results_df[column_name].tail(self.patience).values

        if len(results_df) <= self.patience:
            return True

        previous_values = results_df[column_name].iloc[: -self.patience]

        if metric in self.higher_better:
            previous_best = previous_values.max()
            improvement = np.any(recent_values > (previous_best + self.min_improvement))
            recent_best = recent_values.max()
        else:
            previous_best = previous_values.min()
            improvement = np.any(recent_values < (previous_best - self.min_improvement))
            recent_best = recent_values.min()

        log.debug(
            f"Metric {metric}: previous_best={previous_best:.4f}, recent_best={recent_best:.4f}, improved={improvement}"
        )
        return improvement

    def should_stop(self, results_df: pd.DataFrame, current_iteration: int) -> Any:
        """Determine whether training should stop based on early stopping criteria."""

        def _should_stop() -> bool:
            if current_iteration <= self.patience:
                log.debug(
                    f"Iteration {current_iteration} <= patience {self.patience}. Continuing."
                )
                return False

            if not isinstance(results_df, pd.DataFrame) or len(results_df) == 0:
                raise ValueError("results_df must be a non-empty pandas DataFrame")

            required_columns = [
                self.column_mapping[metric]
                for metric in self.monitor
                if metric in self.column_mapping
            ]
            missing_columns = [
                col for col in required_columns if col not in results_df.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in results_df: {missing_columns}"
                )

            improvements = [
                self._check_metric_improvement(results_df, metric)
                for metric in self.monitor
            ]
            should_stop = not any(improvements)

            if should_stop:
                log.info(
                    f"Early stopping triggered after {self.patience} iterations without "
                    f"{self.min_improvement} improvement in {self.monitor}"
                )
            else:
                log.debug(
                    "Continue training - improvement detected in monitored metrics"
                )

            return should_stop

        return self._safe_execute(_should_stop, "Error in EarlyStopping.should_stop")

    def get_best_iteration(
        self, results_df: pd.DataFrame, metric: Optional[str] = None
    ) -> Any:
        """Get the best iteration based on the specified metric."""

        def _get_best() -> int:
            if not isinstance(results_df, pd.DataFrame) or len(results_df) == 0:
                raise ValueError("results_df must be a non-empty pandas DataFrame")

            metric_to_use = metric.lower() if metric is not None else self.monitor[0]

            if metric_to_use not in self.column_mapping:
                raise ValueError(f"Unknown metric: {metric_to_use}")

            column_name = self.column_mapping[metric_to_use]
            if column_name not in results_df.columns:
                raise ValueError(f"Column {column_name} not found in results_df")

            best_idx = (
                results_df[column_name].idxmax()
                if metric_to_use in self.higher_better
                else results_df[column_name].idxmin()
            )

            best_iteration = (
                results_df.loc[best_idx, "iteration"]
                if "iteration" in results_df.columns
                else best_idx + 1
            )
            log.debug(f"Best iteration for {metric_to_use}: {best_iteration}")
            return int(best_iteration)

        return self._safe_execute(
            _get_best, "Error in EarlyStopping.get_best_iteration"
        )


class TrainML(BaseMLUtils):
    def __init__(
        self,
        nfolds: int = 5,
        model: str = "EN",
        early_stopping: bool = True,
        early_stopping_monitor: Union[str, List[str]] = "auroc",
        patience: int = 10,
        min_improvement: float = 0.0005,
        random_state: int = 123,
    ) -> None:
        super().__init__(random_state)
        self.nfolds: int = nfolds
        self.model: str = model
        self.early_stopping: bool = early_stopping
        self.patience: int = patience
        self.min_improvement: float = min_improvement
        self.logger = log

        self.cv: CrossValidation = CrossValidation(
            nfolds=nfolds, random_state=random_state
        )
        self.metrics_calculator: PerformanceMetrics = PerformanceMetrics()

        if early_stopping:
            self.early_stopper: EarlyStopping = EarlyStopping(
                patience=patience,
                min_improvement=min_improvement,
                monitor=early_stopping_monitor,
                random_state=random_state,
            )

        self.supported_models: List[str] = ["EN", "KNN", "RF", "SVM"]

    def _prepare_data(
        self,
        train_set: pd.DataFrame,
        train_y: Union[pd.Series, List[Any], np.ndarray],
        test_data: pd.DataFrame,
        test_y: Union[pd.Series, List[Any], np.ndarray],
        categorical: bool,
    ) -> Optional[
        Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Optional[pd.Index],
            Optional[LabelEncoder],
            Optional[np.ndarray],
        ]
    ]:
        """Prepare and validate training and test datasets."""
        if list(train_set.columns) != list(test_data.columns):
            self.logger.error(
                "Train and test datasets must have identical column names"
            )
            return None

        if len(train_set) != len(train_y):
            self.logger.error("Training data and labels must have the same length")
            return None

        if len(test_data) != len(test_y):
            self.logger.error("Test data and labels must have the same length")
            return None

        if train_set.isnull().any().any() or test_data.isnull().any().any():
            self.logger.warn("Missing values detected in training or test data")

        train_set_values = self._to_numpy(train_set)
        test_data_values = self._to_numpy(test_data)
        train_y_values = self._to_numpy(train_y)
        test_y_values = self._to_numpy(test_y)

        train_columns = (
            train_set.columns if isinstance(train_set, pd.DataFrame) else None
        )
        label_encoder: Optional[LabelEncoder] = None
        class_labels: Optional[np.ndarray] = None

        if categorical:
            label_encoder = LabelEncoder()
            train_y_values = label_encoder.fit_transform(train_y_values)
            test_y_values = label_encoder.transform(test_y_values)
            class_labels = label_encoder.classes_

        return (
            train_set_values,
            train_y_values,
            test_data_values,
            test_y_values,
            train_columns,
            label_encoder,
            class_labels,
        )

    def _get_model_predictions(
        self, model: Any, data: np.ndarray, categorical: bool
    ) -> Optional[np.ndarray]:
        """Get predictions from the trained model."""
        try:
            if categorical:
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(data)
                else:
                    pred = model.predict(data)
                    classes = model.classes_
                    pred_prob = np.zeros((len(pred), len(classes)))
                    for i, cls in enumerate(classes):
                        pred_prob[:, i] = (pred == cls).astype(float)
                    return pred_prob
            else:
                return model.predict(data).reshape(-1, 1)
        except Exception as e:
            log.error(f"Error getting model predictions: {e}")
            return None

    def _create_feature_importance(
        self, model: Any, train_columns: Optional[pd.Index]
    ) -> Optional[pd.DataFrame]:
        """Create a DataFrame of feature importances for Random Forest models."""
        condition1 = self.model == "RF"
        condition2 = train_columns is not None and hasattr(
            model, "feature_importances_"
        )
        if condition1 and condition2:
            return pd.DataFrame(
                {
                    "feature": train_columns,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
        return None

    def _update_best_metrics(
        self,
        best_metrics: Dict[str, Any],
        cv_results: Dict[str, Any],
        params: Dict[str, Any],
        iteration: int,
        categorical: bool,
    ) -> bool:
        """Update the best metrics if the current model is better."""
        if categorical:
            if cv_results and cv_results["cv_roc_mean"] > best_metrics["cv_roc"]:
                best_metrics.update(
                    {
                        "cv_roc": cv_results["cv_roc_mean"],
                        "cv_prc": cv_results["cv_prc_mean"],
                        "params": params,
                        "iteration": iteration,
                    }
                )
                log.debug(
                    f"New best model found (iter {iteration}): "
                    f"ROC = {round(best_metrics['cv_roc'], 3)}, "
                    f"PRC = {round(best_metrics['cv_prc'], 3)}"
                )
                return True
        else:
            if cv_results and cv_results["cv_rmse_mean"] < best_metrics["cv_roc"]:
                best_metrics.update(
                    {
                        "cv_roc": cv_results["cv_rmse_mean"],
                        "cv_prc": cv_results["cv_r2_mean"],
                        "params": params,
                        "iteration": iteration,
                    }
                )
                log.debug(
                    f"New best model found (iter {iteration}): "
                    f"RMSE = {round(best_metrics['cv_roc'], 3)}, "
                    f"R² = {round(best_metrics['cv_prc'], 3)}"
                )
                return True
        return False

    def train(
        self,
        train_set: pd.DataFrame,
        train_y: Union[pd.Series, List[Any], np.ndarray],
        test_data: pd.DataFrame,
        test_y: Union[pd.Series, List[Any], np.ndarray],
        param_grid: Union[pd.DataFrame, List[Dict[str, Any]]],
        sample_weights: Optional[np.ndarray] = None,
        categorical: bool = False,
    ) -> Any:
        """Train a machine learning model with hyperparameter tuning and early stopping."""

        def _train() -> Dict[str, Any]:
            if not isinstance(param_grid, pd.DataFrame):
                param_grid_df = pd.DataFrame(param_grid)
            else:
                param_grid_df = param_grid

            self._validate_data_match(train_set, train_y)
            self._validate_data_match(test_data, test_y)

            if self.model not in self.supported_models:
                raise ValueError(
                    f"Unsupported model type: {self.model}. Supported: {self.supported_models}"
                )

            prepared = self._prepare_data(
                train_set, train_y, test_data, test_y, categorical
            )
            if prepared is None:
                raise ValueError("Data preparation failed")

            (
                train_set_values,
                train_y_values,
                test_data_values,
                test_y_values,
                train_columns,
                label_encoder,
                class_labels,
            ) = prepared

            total_iters = len(param_grid_df)
            results_list: List[Dict[str, Any]] = []
            best_metrics: Dict[str, Any] = {
                "cv_roc": float("-inf") if categorical else float("inf"),
                "cv_prc": float("-inf") if categorical else float("inf"),
                "params": None,
                "iteration": 0,
            }

            for i in range(total_iters):
                params = param_grid_df.iloc[i].to_dict()
                param_str = ", ".join(
                    [f"{name} = {value}" for name, value in params.items()]
                )
                log.debug(f"Iteration {i + 1}/{total_iters}: {param_str}")

                start_time = time.time()
                cv_results = self.cv.run(
                    train_set_values,
                    train_y_values,
                    params,
                    model=self.model,
                    categorical=categorical,
                    sample_weights=sample_weights,
                )
                train_time = time.time() - start_time

                if categorical:
                    result = {
                        "iteration": i + 1,
                        "params": params,
                        "train_time": train_time,
                        "cv_roc": (
                            cv_results["cv_roc_mean"] if cv_results else float("nan")
                        ),
                        "cv_prc": (
                            cv_results["cv_prc_mean"] if cv_results else float("nan")
                        ),
                    }
                else:
                    result = {
                        "iteration": i + 1,
                        "params": params,
                        "train_time": train_time,
                        "cv_rmse": (
                            cv_results["cv_rmse_mean"] if cv_results else float("nan")
                        ),
                        "cv_r2": (
                            cv_results["cv_r2_mean"] if cv_results else float("nan")
                        ),
                        "cv_mse": (
                            cv_results["cv_mse_mean"] if cv_results else float("nan")
                        ),
                        "cv_mae": (
                            cv_results["cv_mae_mean"] if cv_results else float("nan")
                        ),
                    }

                self._update_best_metrics(
                    best_metrics, cv_results, params, i + 1, categorical
                )
                results_list.append(result)

                if self.early_stopping and i >= self.patience:
                    results_df = pd.DataFrame(results_list)
                    if self.early_stopper.should_stop(results_df, i + 1):
                        log.info("Early stopping triggered.")
                        break

            results_df = pd.DataFrame(results_list)

            if best_metrics["params"] is None:
                raise ValueError("No valid model was found during training.")

            log.info("Training final model with best parameters")

            final_model = ModelFactory.create_model(
                self.model,
                best_metrics["params"],
                categorical,
                self.random_state,
                sample_weights,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if sample_weights is not None and self.model not in ["SVM", "KNN"]:
                    final_model.fit(
                        train_set_values, train_y_values, sample_weight=sample_weights
                    )
                else:
                    final_model.fit(train_set_values, train_y_values)

            train_pred = self._get_model_predictions(
                final_model, train_set_values, categorical
            )
            test_pred = self._get_model_predictions(
                final_model, test_data_values, categorical
            )

            log.debug("Evaluating final model on training and test data")

            task_type = "classification" if categorical else "regression"
            train_metrics = self.metrics_calculator.calculate(
                train_pred, train_y_values, task_type=task_type
            )
            test_metrics = self.metrics_calculator.calculate(
                test_pred, test_y_values, task_type=task_type
            )

            log.debug("Training completed")

            feature_importance = self._create_feature_importance(
                final_model, train_columns
            )

            return {
                "final_model": final_model,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "best_params": best_metrics["params"],
                "cv_results": results_df,
                "feature_importance": feature_importance,
                "label_encoder": label_encoder,
            }

        result = self._safe_execute(_train, "Error in TrainML.train")
        gc.collect()
        return result
