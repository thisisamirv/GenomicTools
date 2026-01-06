#!/usr/bin/env python
# Import required modules
import h5py
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tqdm import tqdm
from typing import Optional, Union
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.H5Utils import CachedH5Utils
from utils.LoggingUtils import log


class ImportanceRank:
    def __init__(
        self,
        input_file: str,
        target_file: str,
        output_file: str,
        iterations: int = 100,
        fraction: float = 0.7,
        count: bool = True,
        categorical: bool = True,
    ) -> None:
        self.input_file = input_file
        self.target_file = target_file
        self.output_file = output_file
        self.iterations = iterations
        self.fraction = fraction
        self.count = count
        self.categorical = categorical
        self.data = None
        self.target = None
        self.h5_file = None

    def handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        try:
            imputed_data = data.copy()
            valid_counts = np.sum(~np.isnan(imputed_data), axis=1)
            row_sums = np.nansum(imputed_data, axis=1)
            row_means = np.divide(
                row_sums,
                valid_counts,
                out=np.zeros_like(row_sums),
                where=valid_counts > 0,
            )
            all_nan_rows = np.isnan(row_means)
            row_means[all_nan_rows] = 0
            nan_indices = np.where(np.isnan(imputed_data))
            for i, j in zip(nan_indices[0], nan_indices[1]):
                imputed_data[i, j] = row_means[i]
            return imputed_data
        except Exception as e:
            log.error(f"Error handling missing values: {e}")
            return data

    def check_data_validity(
        self, data: Union[pd.DataFrame, np.ndarray], warn: bool = True
    ) -> Union[pd.DataFrame, np.ndarray]:
        try:
            if isinstance(data, pd.DataFrame):
                if data.shape[1] <= 1:
                    return data
                variance = data.var(axis=1, skipna=True)
                epsilon = np.finfo(float).eps
                invalid_rows = variance[(variance.isna()) | (variance < epsilon)].index
                if len(invalid_rows) > 0:
                    if warn:
                        reasons = []
                        for idx in invalid_rows:
                            if np.isnan(variance[idx]):
                                reasons.append("NA variance")
                            elif variance[idx] == 0:
                                reasons.append("zero variance")
                            else:
                                reasons.append("unknown reason")
                        log.warn(
                            f"Invalid rows detected. Reasons: {', '.join(set(reasons))}"
                        )
                        log.warn(
                            f"Examples: {', '.join(map(str, list(invalid_rows)[:6]))}"
                        )
                        log.warn("Removing invalid rows...")
                    valid_data = data.drop(invalid_rows)
                    return valid_data
                else:
                    log.info("Data validity check passed.")
                    return data
            else:
                if data.ndim != 2 or data.shape[1] <= 1:
                    return data
                variance = np.var(data, axis=1, ddof=1)
                epsilon = np.finfo(float).eps
                invalid_rows = np.where(np.isnan(variance) | (variance < epsilon))[0]
                if len(invalid_rows) > 0:
                    if warn:
                        reasons = []
                        for idx in invalid_rows:
                            if np.isnan(variance[idx]):
                                reasons.append("NA variance")
                            elif variance[idx] == 0:
                                reasons.append("zero variance")
                            else:
                                reasons.append("unknown reason")
                        log.warn(
                            f"Invalid rows detected. Reasons: {', '.join(set(reasons))}"
                        )
                        log.warn(
                            f"Examples: {', '.join(map(str, list(invalid_rows)[:6]))}"
                        )
                        log.warn("Removing invalid rows...")
                    valid_rows = np.setdiff1d(np.arange(data.shape[0]), invalid_rows)
                    valid_data = data[valid_rows]
                    return valid_data
                else:
                    log.info("Data validity check passed.")
                    return data
        except Exception as e:
            log.error(f"Error checking data validity: {e}")
            return data

    def train_rf_model(
        self, x_bootstrap: np.ndarray, y_bootstrap: np.ndarray
    ) -> Optional[Union[RandomForestClassifier, RandomForestRegressor]]:
        try:
            x_transpose = x_bootstrap.T
            if self.categorical:
                model = RandomForestClassifier(
                    n_estimators=500,
                    max_features=int(math.sqrt(x_bootstrap.shape[0])),
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=500,
                    max_features=int(math.sqrt(x_bootstrap.shape[0])),
                    random_state=42,
                    n_jobs=-1,
                )
            model.fit(x_transpose, y_bootstrap)
            return model
        except Exception as e:
            log.warn(f"Random forest model training failed. Reason: {e}")
            return None

    def load_data(self) -> bool:
        try:
            if self.count:
                log.info(f"Opening HDF5 file: {self.input_file}")
                self.h5_file = h5py.File(self.input_file, "r")
                h5_utils = CachedH5Utils(self.h5_file)
                log.info("Retrieving chromosome list.")
                chromosome_list = h5_utils.get_chromosomes()
                if not chromosome_list:
                    raise ValueError(
                        "Failed to retrieve chromosome list from HDF5 file"
                    )
                print("Reading data...")
                data_list = []
                for chromosome in tqdm(chromosome_list, desc="Loading chromosomes"):
                    chromosome_data = h5_utils.read_chromosome(
                        chromosome, data_type="Methylation"
                    )
                    if chromosome_data is not None and not chromosome_data.empty:
                        CGID = chromosome_data.iloc[:, 0]
                        numeric_data = chromosome_data.select_dtypes(
                            include=[np.number]
                        )
                        numeric_data.index = CGID
                        data_list.append(numeric_data)
                self.data = pd.concat(data_list, axis=0)
                self.feature_names = self.data.index.tolist()
                self.data = self.data.values.T
            else:
                print("Reading data...")
                self.data = pd.read_csv(self.input_file, header=0)
                self.feature_names = self.data.columns.tolist()
                self.data = self.data.values.T
            missing_values = np.isnan(self.data).sum()
            if missing_values > 0:
                log.warn(f"Missing values detected in the data: {missing_values}")
                log.warn("Imputing missing values...")
                self.data = self.handle_missing_values(self.data)
            else:
                log.info("No missing values detected in the data.")
            log.info(f"Loading target variable from: {self.target_file}")
            target_df = pd.read_csv(self.target_file, header=None)
            if self.categorical:
                self.target = target_df.iloc[:, 0].astype("category").values
            else:
                self.target = target_df.iloc[:, 0].astype(float).values
            log.info("Checking data validity...")
            original_shape = self.data.shape
            self.data = self.check_data_validity(self.data)
            if self.data.shape[0] != original_shape[0]:
                log.warn(
                    f"Removed {original_shape[0] - self.data.shape[0]} invalid features"
                )
            return True
        except Exception as e:
            log.error(f"Error loading data: {e}")
            return False

    def run_bootstrap(self) -> Optional[pd.DataFrame]:
        try:
            n_features = self.data.shape[0]
            importance_matrix = np.zeros((n_features, self.iterations))
            sample_size = round(self.fraction * self.data.shape[1])
            print("Running bootstrap iterations...")
            for i in tqdm(range(self.iterations), desc="Bootstrap progress"):
                if sample_size <= 0:
                    importance_matrix[:, i] = 0.0
                    continue
                sample_indices = np.random.choice(
                    self.data.shape[1], size=sample_size, replace=True
                )
                x_bootstrap = self.data[:, sample_indices]
                y_bootstrap = self.target[sample_indices]
                x_bootstrap = self.check_data_validity(x_bootstrap, warn=False)
                rf_model = self.train_rf_model(x_bootstrap, y_bootstrap)
                if rf_model is not None:
                    importance_scores = rf_model.feature_importances_
                    if len(importance_scores) == n_features:
                        importance_matrix[:, i] = importance_scores
            mean_importance = np.nanmean(importance_matrix, axis=1)
            result_df = pd.DataFrame(
                {"FEATURE": self.feature_names, "IMPORTANCE": mean_importance}
            )
            result_df = result_df.sort_values("IMPORTANCE", ascending=False)
            return result_df
        except Exception as e:
            log.error(f"Error in bootstrap process: {e}")
            return None

    def run(self) -> Optional[pd.DataFrame]:
        try:
            log.info("Starting importance ranking.")
            if not self.load_data():
                log.error("Failed to load data")
                return None
            results = self.run_bootstrap()
            if results is None:
                log.error("Bootstrap process failed")
                return None
            if self.output_file:
                log.info(f"Writing results to: {self.output_file}")
                results.to_csv(self.output_file, index=False)
                log.success(
                    f"Importance ranking completed successfully. Results written to {self.output_file}"
                )
            return results
        except Exception as e:
            log.error(f"Error in ImportanceRank: {e}")
            return None
        finally:
            if self.h5_file is not None:
                self.h5_file.close()


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(flags=["-t", "--target"], type=str, required=True),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
    OptionConfig(flags=["-e", "--iterations"], type=int, default=100, required=False),
    OptionConfig(flags=["-r", "--fraction"], type=float, default=0.7, required=False),
    OptionConfig(flags=["-c", "--count"], type=bool, default=True, required=False),
    OptionConfig(
        flags=["-a", "--categorical"], type=bool, default=True, required=False
    ),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="ImportanceRank")
    opt = framework.run()
    ranker = ImportanceRank(
        input_file=opt.input,
        target_file=opt.target,
        output_file=opt.output,
        iterations=opt.iterations,
        fraction=opt.fraction,
        count=opt.count,
        categorical=opt.categorical,
    )
    ranker.run()
