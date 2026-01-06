#!/usr/bin/env python
# Import required modules
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.H5Utils import CachedH5Utils
from utils.LoggingUtils import log
from utils.MLUtils import (
    DataPreprocessor,
    ParameterGrid,
    NoiseInjector,
    SampleWeights,
    TrainML,
)
from utils.ParsingUtils import ParseToKeyValueDict, ParseToList


class FeatureSelection:
    def __init__(
        self,
        metadata: str,
        counts: str,
        response: str,
        output: str,
        indices: Union[str, List[int]],
        probes: Optional[Union[str, List[str]]] = None,
        preprocessing: Union[
            str, List[str]
        ] = "scale,center,outlierCapping,knnImpute,YeoJohnson,nzv",
        regularization: Union[str, Dict[str, Any]] = "noise=0.1,adaptive_noise=TRUE",
        evaluation: Union[str, Dict[str, Any]] = "nfolds=5",
        training: Union[str, Dict[str, Any]] = "class_weightening=TRUE",
        stopping_criteria: Union[
            str, Dict[str, Any]
        ] = "early_stop=TRUE,monitor=AUROC AUPRC,patience=10,cut=0.005",
        parameters: Optional[Union[str, Dict[str, Any]]] = None,
        importance_scores: Optional[str] = None,
        model: str = "RF",
        categorical: bool = True,
    ) -> None:
        self.metadata_path: str = metadata
        self.counts_path: str = counts
        self.response: str = response
        self.output_path: str = output
        self.indices: Union[str, List[int]] = indices
        self.probes: Optional[Union[str, List[str]]] = probes
        self.preprocessing: Union[str, List[str]] = preprocessing
        self.regularization: Union[str, Dict[str, Any]] = regularization
        self.evaluation: Union[str, Dict[str, Any]] = evaluation
        self.training: Union[str, Dict[str, Any]] = training
        self.stopping_criteria: Union[str, Dict[str, Any]] = stopping_criteria
        self.parameters: Optional[Union[str, Dict[str, Any]]] = parameters
        self.importance_scores: Optional[str] = importance_scores
        self.model: str = model
        self.categorical: bool = categorical
        self.data_preprocessor: DataPreprocessor = DataPreprocessor()
        self.param_generator: ParameterGrid = ParameterGrid()
        self.noise_injector: NoiseInjector = NoiseInjector()
        self.sample_weights_calc: SampleWeights = SampleWeights(strategy="balanced")
        self.metadata: Optional[pd.DataFrame] = None
        self.h5_file: Optional[h5py.File] = None
        self.h5_utils: Optional[CachedH5Utils] = None
        self.chr_list: Optional[List[str]] = None
        self.sample_indices: Optional[List[int]] = None
        self.sample_ids: Optional[List[str]] = None
        self.features: Optional[List[str]] = None
        self.feature_importance: Optional[pd.DataFrame] = None

    def _parse_parameters(self) -> bool:
        try:
            if isinstance(self.indices, str):
                indices_list = ParseToList(self.indices)
                self.indices = [int(idx) for idx in indices_list]
            elif isinstance(self.indices, list):
                self.indices = [int(idx) for idx in self.indices]

            if isinstance(self.preprocessing, str) and self.preprocessing.strip():
                self.preprocessing = ParseToList(self.preprocessing)
                log.debug(f"Preprocessing methods: {', '.join(self.preprocessing)}")
            elif isinstance(self.preprocessing, list):
                log.debug(f"Preprocessing methods: {', '.join(self.preprocessing)}")

            kv_attrs = [
                ("regularization", "Regularization parameters", False),
                ("evaluation", "Evaluation parameters", False),
                ("stopping_criteria", "Stopping criteria", True),
                ("training", "Training parameters", False),
                ("parameters", "Hyperparameters", False),
            ]

            for attr, display, has_monitor in kv_attrs:
                val = getattr(self, attr)
                is_dict = isinstance(val, dict)
                is_dict_nonempty = is_dict and bool(val)
                is_str = isinstance(val, str)
                is_not_none = val is not None
                is_str_nonempty = is_not_none and is_str and bool(val.strip())

                if is_dict_nonempty:
                    log.debug(
                        f"{display}: {', '.join([f'{k}={v}' for k, v in val.items()])}"
                    )
                elif is_str_nonempty:
                    parsed = ParseToKeyValueDict(val)
                    if has_monitor and "monitor" in parsed:
                        parsed["monitor"] = parsed["monitor"].split()
                    setattr(self, attr, parsed)
                    log.debug(
                        f"{display}: {', '.join([f'{k}={v}' for k, v in parsed.items()])}"
                    )
                else:
                    setattr(self, attr, {})

            return True
        except Exception as e:
            log.error(f"Error parsing parameters: {e}")
            return False

    def _load_data(self) -> bool:
        try:
            log.debug("Loading the metadata.")
            metadata = pd.read_csv(self.metadata_path)
            if self.probes:
                log.debug("Loading the probe IDs.")
                if isinstance(self.probes, str):
                    self.probes = (
                        pd.read_csv(self.probes, header=None).iloc[:, 0].tolist()
                    )
            log.info("Getting the training and test sample IDs.")
            self.train_samples = metadata[metadata["set"] == "train"][
                "sample_id"
            ].tolist()
            self.test_samples = metadata[metadata["set"] == "test"][
                "sample_id"
            ].tolist()
            log.info("Getting the training and test response values.")
            if self.categorical:
                self.train_y = pd.Categorical(
                    metadata[metadata["sample_id"].isin(self.train_samples)][
                        self.response
                    ]
                )
                self.test_y = pd.Categorical(
                    metadata[metadata["sample_id"].isin(self.test_samples)][
                        self.response
                    ]
                )
            else:
                self.train_y = metadata[metadata["sample_id"].isin(self.train_samples)][
                    self.response
                ].values
                self.test_y = metadata[metadata["sample_id"].isin(self.test_samples)][
                    self.response
                ].values
            if self.counts_path.endswith((".h5", ".hdf5", ".HDF5")):
                self._load_hdf5_data()
            else:
                self._load_csv_data()
            if self.importance_scores:
                self._sort_features_by_importance()
            if str(self.training.get("class_weightening", "TRUE")).upper() == "TRUE":
                if self.model != "KNN":
                    log.info("Getting the sample weights for class imbalance.")
                    train_y_for_weights = self.train_y
                    if isinstance(self.train_y, pd.Categorical):
                        train_y_for_weights = pd.Series(self.train_y)
                    self.weights = self.sample_weights_calc.calculate(
                        self.train_data, train_y_for_weights
                    )
                    if self.weights is None:
                        log.error("Class imbalance handling failed.")
                        return False
                    log.debug("Sample weights calculated.")
                else:
                    self.weights = None
            else:
                self.weights = None
            return True
        except Exception as e:
            log.error(f"Error loading data: {e}")
            return False

    def _load_hdf5_data(self) -> None:
        log.info("Opening the input HDF5 file.")
        log.debug(f"Count data path: {self.counts_path}")
        self.h5_file = h5py.File(self.counts_path, "r")
        self.h5_utils = CachedH5Utils(self.h5_file)
        if not self.h5_utils.validate_file_structure():
            raise ValueError("Invalid HDF5 file structure")
        file_info = self.h5_utils.get_data_info()
        log.info(
            f"HDF5 file contains {file_info['data_type']} data with "
            f"{file_info['n_chromosomes']} chromosomes and "
            f"{file_info['n_samples']} samples"
        )
        chr_list = self.h5_utils.get_chromosomes()
        if not chr_list:
            raise ValueError("No chromosomes found in HDF5 file")
        train_indices = self.h5_utils.get_sample_indices(self.train_samples)
        test_indices = self.h5_utils.get_sample_indices(self.test_samples)
        if train_indices is None:
            raise ValueError("Failed to get training sample indices")
        if test_indices is None:
            raise ValueError("Failed to get test sample indices")
        for data_type, indices, samples in [
            ("training", train_indices, self.train_samples),
            ("test", test_indices, self.test_samples),
        ]:
            log.info(f"Reading the {data_type} data.")
            data_list: List[pd.DataFrame] = []
            for chromosome in tqdm(chr_list, desc=f"Loading {data_type} data"):
                chr_data = self.h5_utils.read_chromosome(
                    chromosome, data_type="methylation"
                )
                if chr_data is not None and not chr_data.empty:
                    if self.probes:
                        chr_data = chr_data[chr_data["probe_id"].isin(self.probes)]
                    if len(indices) > 0:
                        cols_to_keep = ["probe_id"] + [
                            chr_data.columns[i + 1]
                            for i in indices
                            if i + 1 < len(chr_data.columns)
                        ]
                        chr_data = chr_data[cols_to_keep]
                        if not chr_data.empty:
                            data_list.append(chr_data)
            if data_list:
                data_combined = pd.concat(data_list, axis=0, ignore_index=True)
                if data_combined.empty:
                    raise ValueError(f"No {data_type} data found after filtering")
                probe_ids = data_combined["probe_id"]
                data_matrix = data_combined.drop("probe_id", axis=1).T
                data_matrix.columns = probe_ids
                data_matrix = data_matrix.loc[:, ~data_matrix.columns.duplicated()]
                if data_type == "training":
                    self.train_data = data_matrix
                else:
                    self.test_data = data_matrix
            else:
                raise ValueError(f"No {data_type} data loaded from HDF5 file")
        if not all(self.train_data.columns.isin(self.test_data.columns)):
            log.warn("Some features in training data are not in test data")
        if not all(self.test_data.columns.isin(self.train_data.columns)):
            log.warn("Some features in test data are not in training data")
        common_features = self.train_data.columns.intersection(self.test_data.columns)
        if len(common_features) == 0:
            raise ValueError("No common features between training and test data")
        self.train_data = self.train_data[common_features]
        self.test_data = self.test_data[common_features]
        log.debug(f"Final training data shape: {self.train_data.shape}")
        log.debug(f"Final test data shape: {self.test_data.shape}")

    def _load_csv_data(self) -> None:
        log.info("Reading the training and test data.")
        counts = pd.read_csv(self.counts_path)
        self.train_data = counts[counts["sample_id"].isin(self.train_samples)].copy()
        self.test_data = counts[counts["sample_id"].isin(self.test_samples)].copy()
        for data in [self.train_data, self.test_data]:
            if "sample_id" in data.columns:
                data.drop("sample_id", axis=1, inplace=True)
        log.debug("Training and test data read.")

    def _sort_features_by_importance(self) -> None:
        log.debug("Sorting features based on importance.")
        importance = pd.read_csv(self.importance_scores)
        importance = importance.sort_values("importance", ascending=False)
        feature_order = importance["feature"].tolist()
        available_features = [
            f
            for f in feature_order
            if f in self.train_data.columns and f in self.test_data.columns
        ]
        if len(available_features) == 0:
            log.warn(
                "No features from importance file found in data. Using original order."
            )
            return
        self.train_data = self.train_data[available_features]
        self.test_data = self.test_data[available_features]
        log.debug(
            f"Features sorted based on importance. Using {len(available_features)} features."
        )

    def _calculate_feature_scale(
        self, n_features: int, n_samples: int
    ) -> Optional[float]:
        feature_scales = {
            "EN": None,
            "KNN": np.sqrt(min(1, 1000 / n_features)),
            "RF": np.sqrt(n_features),
            "SVM": min(1, np.sqrt(500 / n_features)),
        }
        if self.model not in feature_scales:
            raise ValueError(f"Invalid model specified: {self.model}")
        return feature_scales[self.model]

    def _process_feature_sets(self) -> pd.DataFrame:
        results_list: List[Dict[str, Any]] = []
        max_features = min(self.train_data.shape[1], self.test_data.shape[1])
        valid_indices = [idx for idx in self.indices if idx <= max_features]
        if len(valid_indices) != len(self.indices):
            invalid_indices = [idx for idx in self.indices if idx > max_features]
            log.warn(
                f"Some indices exceed available features ({max_features}). "
                f"Skipping indices: {invalid_indices}"
            )
            self.indices = valid_indices
        if not self.indices:
            raise ValueError("No valid feature indices to process")
        if self.categorical:
            print("| Features | Train AUROC | Train AUPRC | Test AUROC | Test AUPRC |")
            print("|----------|-------------|-------------|------------|------------|")
        else:
            print("| Features | Train RMSE | Train R² | Test RMSE | Test R² |")
            print("|----------|------------|----------|-----------|---------|")
        for idx in self.indices:
            try:
                log.debug(f"Processing feature set: {idx}")
                train_set_subset = self.train_data.iloc[:, :idx].copy()
                test_set_subset = self.test_data.iloc[:, :idx].copy()
                log.debug(f"Training data subset shape: {train_set_subset.shape}")
                log.debug(f"Test data subset shape: {test_set_subset.shape}")
                if not all(train_set_subset.columns == test_set_subset.columns):
                    raise ValueError("Feature mismatch between training and test data")
                condition1 = isinstance(self.preprocessing, list)
                condition2 = "NULL" in self.preprocessing
                condition = condition1 and condition2
                if self.preprocessing is None or condition:
                    log.info("No preprocessing specified. Using raw data.")
                    train_set_processed = train_set_subset
                    test_set_processed = test_set_subset
                else:
                    log.debug("Preprocessing the data.")
                    processed_data = self.data_preprocessor.preprocess(
                        train_set_subset, test_set_subset, self.preprocessing
                    )
                    if processed_data is None:
                        raise ValueError("Data preprocessing failed")
                    train_set_processed = processed_data["train"]
                    test_set_processed = processed_data["test"]
                    log.debug("Data preprocessed.")
                log.info("Adding noise to the training set.")
                noise_level = float(self.regularization.get("noise", 0.1))
                regularization_condition = str(
                    self.regularization.get("adaptive_noise", "TRUE")
                ).upper()
                adaptive_noise = regularization_condition == "TRUE"
                log.debug(f"Noise level: {noise_level}")
                log.debug(f"Adaptive noise: {adaptive_noise}")
                train_set_noisy = self.noise_injector.add_noise(
                    train_set_processed, noise_level, adaptive_noise
                )
                if train_set_noisy is None:
                    raise ValueError("Noise injection failed")
                log.debug("Noise added to the training set.")
                log.info("Scaling the hyperparameters.")
                n_samples = len(train_set_noisy)
                n_features = train_set_noisy.shape[1]
                feature_scale = self._calculate_feature_scale(n_features, n_samples)
                log.debug("Creating a grid of hyperparameters to search over.")
                param_grid = self.param_generator.generate(
                    self.model, self.parameters, n_features, n_samples, feature_scale
                )
                if param_grid is None:
                    raise ValueError("Parameter grid generation failed")
                log.info("Training the model.")
                early_stop_condition = str(
                    self.stopping_criteria.get("early_stop", "TRUE")
                ).upper()
                early_stop = early_stop_condition == "TRUE"
                early_stop_monitor = self.stopping_criteria.get(
                    "monitor", ["AUROC", "AUPRC"]
                )
                if isinstance(early_stop_monitor, list):
                    early_stop_monitor = [m.lower() for m in early_stop_monitor]
                else:
                    early_stop_monitor = early_stop_monitor.lower()
                trainer = TrainML(
                    nfolds=int(self.evaluation.get("nfolds", 5)),
                    model=self.model,
                    early_stopping=early_stop,
                    early_stopping_monitor=early_stop_monitor,
                    patience=int(self.stopping_criteria.get("patience", 10)),
                    min_improvement=float(self.stopping_criteria.get("cut", 0.005)),
                )
                model_results = trainer.train(
                    train_set=train_set_noisy,
                    train_y=self.train_y,
                    test_data=test_set_processed,
                    test_y=self.test_y,
                    param_grid=param_grid,
                    sample_weights=self.weights,
                    categorical=self.categorical,
                )
                if model_results is None:
                    raise ValueError("Model training failed")
                else:
                    log.debug("Model trained.")
                log.debug("Saving the results.")
                if self.categorical:
                    result: Dict[str, Any] = {
                        "feature_set": idx,
                        "test_roc": model_results["test_metrics"]["roc_value"],
                        "test_prc": model_results["test_metrics"]["prc_value"],
                        "train_roc": model_results["train_metrics"]["roc_value"],
                        "train_prc": model_results["train_metrics"]["prc_value"],
                    }
                    print(
                        f"| {idx:<8} | {model_results['train_metrics']['roc_value']:<11.2f} | "
                        f"{model_results['train_metrics']['prc_value']:<11.2f} | "
                        f"{model_results['test_metrics']['roc_value']:<10.2f} | "
                        f"{model_results['test_metrics']['prc_value']:<10.2f} |"
                    )
                else:
                    result = {
                        "feature_set": idx,
                        "test_rmse": model_results["test_metrics"]["rmse"],
                        "test_r2": model_results["test_metrics"]["r2"],
                        "train_rmse": model_results["train_metrics"]["rmse"],
                        "train_r2": model_results["train_metrics"]["r2"],
                    }
                    print(
                        f"| {idx:<8} | {model_results['train_metrics']['rmse']:<10.2f} | "
                        f"{model_results['train_metrics']['r2']:<8.2f} | "
                        f"{model_results['test_metrics']['rmse']:<9.2f} | "
                        f"{model_results['test_metrics']['r2']:<7.2f} |"
                    )
                results_list.append(result)
            except Exception as e:
                log.error(f"Error processing feature set {idx}: {e}")
        return pd.DataFrame(results_list)

    def run(self) -> bool:
        try:
            log.info("Starting feature selection.")
            if not self._parse_parameters():
                log.error("Failed to parse parameters")
                return False
            if not self._load_data():
                log.error("Failed to load data")
                return False
            results_df = self._process_feature_sets()
            if results_df is None:
                log.error("Feature processing returned no results")
                return False
            log.info(f"Writing results to {self.output_path}")
            results_df.to_csv(self.output_path, index=False)
            log.success(
                f"{self.model} {'classification' if self.categorical else 'regression'} completed."
            )
            log.success(f"Results saved to: {self.output_path}")
            return True
        except Exception as e:
            log.error(f"Error in feature selection: {e}")
            return False
        finally:
            if hasattr(self, "h5_file") and self.h5_file is not None:
                self.h5_file.close()


options = [
    OptionConfig(flags=["-m", "--metadata"], type=str),
    OptionConfig(flags=["-c", "--counts"], type=str),
    OptionConfig(flags=["-r", "--response"], type=str),
    OptionConfig(flags=["-o", "--output"], type=str),
    OptionConfig(flags=["-i", "--indices"], type=str),
    OptionConfig(flags=["-b", "--probes"], type=str, default=None),
    OptionConfig(
        flags=["-e", "--preprocessing"],
        type=str,
        default="scale,center,outlierCapping,knnImpute,YeoJohnson,nzv",
    ),
    OptionConfig(
        flags=["-g", "--regularization"],
        type=str,
        default="noise=0.1,adaptive_noise=TRUE",
    ),
    OptionConfig(flags=["-a", "--evaluation"], type=str, default="nfolds=5"),
    OptionConfig(
        flags=["-t", "--training"], type=str, default="class_weightening=TRUE"
    ),
    OptionConfig(
        flags=["-s", "--stopping_criteria"],
        type=str,
        default="early_stop=TRUE,monitor=AUROC AUPRC,patience=10,cut=0.005",
    ),
    OptionConfig(flags=["-p", "--parameters"], type=str, default=None),
    OptionConfig(flags=["-n", "--importance_scores"], type=str, default=None),
    OptionConfig(flags=["-d", "--model"], type=str, default="RF"),
    OptionConfig(flags=["-x", "--categorical"], type=bool, default=True),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="FeatureSelection")
    opt = framework.run()
    feature_selector = FeatureSelection(
        metadata=opt.metadata,
        counts=opt.counts,
        response=opt.response,
        output=opt.output,
        indices=opt.indices,
        probes=opt.probes,
        preprocessing=opt.preprocessing,
        regularization=opt.regularization,
        evaluation=opt.evaluation,
        training=opt.training,
        stopping_criteria=opt.stopping_criteria,
        parameters=opt.parameters,
        importance_scores=opt.importance_scores,
        model=opt.model,
        categorical=opt.categorical,
    )
    feature_selector.run()
