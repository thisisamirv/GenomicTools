#!/usr/bin/env python
# Import required modules
import concurrent.futures
import datetime
import glob
import h5py
import multiprocessing as mp
import numpy as np
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.H5Utils import CachedH5Utils
from utils.LoggingUtils import log
from utils.SystemUtils import SystemUtils, monitor_resources

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("forkserver")


class DataProcessor:
    @staticmethod
    def optimize_parameters_advanced(threads: int, window_size: int) -> Dict[str, Any]:
        system_info = SystemUtils.get_system_info()
        memory_info = SystemUtils.get_memory_info()

        available_memory_gb = memory_info.get("available_gb", 8.0)
        total_memory_gb = memory_info.get("total_gb", 16.0)
        memory_source = memory_info.get("source", "System")

        log.info(f"System: {system_info['cpu_name']}")
        log.info(f"Environment: {system_info['environment']}")
        log.info(f"Memory: {available_memory_gb:.1f}GB available ({memory_source})")

        if system_info["environment"] in ["LSF", "SLURM"]:
            memory_per_chr_gb = 2.0
            reserved_memory_gb = 1.0
        else:
            memory_per_chr_gb = 3.0
            reserved_memory_gb = 2.0

        if total_memory_gb > 64:
            memory_per_chr_gb *= 0.8
        elif total_memory_gb < 16:
            memory_per_chr_gb *= 1.5

        usable_memory_gb = max(1.0, available_memory_gb - reserved_memory_gb)
        max_parallel_by_memory = max(1, int(usable_memory_gb / memory_per_chr_gb))
        max_parallel_chr = min(threads, max_parallel_by_memory)

        log.info(f"Optimized for {system_info['environment']} environment")
        log.info(f"Memory allocation: {memory_per_chr_gb}GB per chromosome")

        return {
            "max_parallel_chr": max_parallel_chr,
            "window_size": window_size,
            "process_large_chr_last": available_memory_gb < 16.0,
            "system_info": system_info,
            "memory_per_process_gb": memory_per_chr_gb,
        }

    @staticmethod
    def find_impute2() -> str:
        impute2_bin = os.environ.get("IMPUTE2_BIN")
        if impute2_bin and os.path.exists(impute2_bin):
            log.info(f"IMPUTE2 found at environment variable location: {impute2_bin}")
            return impute2_bin
        binary_names = ["impute2", "impute2_v2.3.2_x86_64_static", "impute2_v2.3.2"]
        for name in binary_names:
            try:
                path = shutil.which(name)
                if path:
                    log.info(f"IMPUTE2 found in PATH: {path}")
                    return path
            except Exception:
                continue
        common_dirs = [
            "/usr/bin",
            "/usr/local/bin",
            "/opt/impute2",
            str(Path.home() / "bin"),
            str(Path.home() / "impute2"),
            str(Path.home() / "software" / "impute2"),
        ]
        for directory in common_dirs:
            if os.path.exists(directory):
                for name in binary_names:
                    path = os.path.join(directory, name)
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        log.info(f"IMPUTE2 found at: {path}")
                        return path
        log.warn("IMPUTE2 not found, assuming 'impute2' is available in PATH")
        return "impute2"


class FastKNNImputer:
    def __init__(
        self,
        n_neighbors: int = 5,
        max_samples_for_knn: int = 10000,
        random_state: int = 42,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.max_samples_for_knn = max_samples_for_knn
        self.random_state = random_state

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        log.debug(f"FastKNNImputer processing data shape: {X.shape}")

        missing_mask = np.isnan(X)
        if not missing_mask.any():
            log.debug("No missing values found, returning original data")
            return X.copy()

        missing_count = missing_mask.sum()
        total_values = X.size
        missing_pct = (missing_count / total_values) * 100
        log.debug(
            f"Missing values: {missing_count}/{total_values} ({missing_pct:.2f}%)"
        )

        if n_samples <= self.max_samples_for_knn:
            log.debug(f"Using regular KNNImputer for {n_samples} samples")
            try:
                imputer = KNNImputer(n_neighbors=min(self.n_neighbors, n_samples - 1))
                return imputer.fit_transform(X)
            except Exception as e:
                log.warn(f"Regular KNN failed, falling back to mean imputation: {e}")
                return self._mean_imputation(X, missing_mask)

        log.debug(f"Using sampling-based KNN for {n_samples} samples")
        return self._sampling_based_imputation(X, missing_mask)

    def _mean_imputation(self, X: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        X_imputed = X.copy()
        col_means = np.nanmean(X, axis=0)

        all_nan_cols = np.isnan(col_means)
        col_means[all_nan_cols] = 0.0

        for j in range(X.shape[1]):
            if missing_mask[:, j].any():
                X_imputed[missing_mask[:, j], j] = col_means[j]

        return X_imputed

    def _sampling_based_imputation(
        self, X: np.ndarray, missing_mask: np.ndarray
    ) -> np.ndarray:
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        X_imputed = X.copy()

        missing_per_sample = missing_mask.sum(axis=1)
        max_missing_allowed = int(0.5 * n_features)

        complete_enough_samples = missing_per_sample <= max_missing_allowed
        candidate_indices = np.where(complete_enough_samples)[0]

        if len(candidate_indices) < self.n_neighbors:
            log.warn("Too few complete samples for KNN, using mean imputation")
            return self._mean_imputation(X, missing_mask)

        n_sample_for_knn = min(self.max_samples_for_knn, len(candidate_indices))
        sample_indices = np.random.choice(
            candidate_indices, size=n_sample_for_knn, replace=False
        )
        X_sample = X[sample_indices]

        log.debug(f"Using {len(sample_indices)} samples for KNN model building")

        sample_missing = np.isnan(X_sample)
        features_missing_rate = sample_missing.sum(axis=0) / len(X_sample)
        usable_features = features_missing_rate < 0.3

        if usable_features.sum() < 2:
            log.warn("Too few usable features for KNN, using mean imputation")
            return self._mean_imputation(X, missing_mask)

        usable_feature_indices = np.where(usable_features)[0]
        log.debug(f"Using {len(usable_feature_indices)} features for KNN")

        X_sample_usable = X_sample[:, usable_features]

        complete_sample_mask = ~np.isnan(X_sample_usable).any(axis=1)
        X_sample_complete = X_sample_usable[complete_sample_mask]
        complete_sample_indices = sample_indices[complete_sample_mask]

        if len(X_sample_complete) < self.n_neighbors:
            log.warn("Too few complete samples after filtering, using mean imputation")
            return self._mean_imputation(X, missing_mask)

        try:
            actual_k = min(self.n_neighbors, len(X_sample_complete))
            if actual_k <= 0:
                log.warn("No valid neighbors available, using mean imputation")
                return self._mean_imputation(X, missing_mask)
            nn = NearestNeighbors(n_neighbors=actual_k)
            nn.fit(X_sample_complete)
            log.debug(
                f"KNN model built with {len(X_sample_complete)} reference samples"
            )
        except Exception as e:
            log.warn(f"KNN model building failed: {e}, using mean imputation")
            return self._mean_imputation(X, missing_mask)

        samples_imputed = 0
        for i in range(n_samples):
            if not missing_mask[i].any():
                continue

            sample_usable = X[i, usable_features]

            if np.isnan(sample_usable).any():
                for j in np.where(missing_mask[i])[0]:
                    col_mean = np.nanmean(X[:, j])
                    X_imputed[i, j] = col_mean if not np.isnan(col_mean) else 0.0
                continue

            try:
                distances, indices = nn.kneighbors([sample_usable])
                neighbor_indices = complete_sample_indices[indices[0]]
                neighbors = X[neighbor_indices]

                for j in np.where(missing_mask[i])[0]:
                    neighbor_vals = neighbors[:, j]
                    valid_vals = neighbor_vals[~np.isnan(neighbor_vals)]

                    if len(valid_vals) > 0:
                        X_imputed[i, j] = np.mean(valid_vals)
                    else:
                        col_mean = np.nanmean(X[:, j])
                        X_imputed[i, j] = col_mean if not np.isnan(col_mean) else 0.0

                samples_imputed += 1

            except Exception as e:
                log.debug(f"KNN failed for sample {i}: {e}, using mean imputation")
                for j in np.where(missing_mask[i])[0]:
                    col_mean = np.nanmean(X[:, j])
                    X_imputed[i, j] = col_mean if not np.isnan(col_mean) else 0.0

        log.debug(f"Successfully imputed {samples_imputed} samples using KNN")

        return X_imputed


class ChunkedMethylationImputer:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        k: int = 5,
        chunk_size: int = 5000,
        max_samples_for_full_knn: int = 5000,
        compression_level: int = 6,
    ) -> None:
        self.input_file = input_file
        self.output_file = output_file
        self.k = k
        self.chunk_size = chunk_size
        self.max_samples_for_full_knn = max_samples_for_full_knn
        self.compression_level = compression_level

    def process_chromosome_chunked(self, grp_name: str, temp_dir: str) -> Optional[str]:
        log.info(f"Processing chromosome {grp_name} with chunking")

        try:
            with h5py.File(self.input_file, "r") as infile:
                grp = infile[grp_name]
                methylation_key = AliasUtils.find_keys(grp, "Methylation")

                if not methylation_key:
                    log.warn(f"No methylation data found for {grp_name}")
                    return None

                dataset = grp[methylation_key]
                n_probes, n_samples = dataset.shape

                log.info(
                    f"Chromosome {grp_name}: {n_probes} probes, {n_samples} samples"
                )

                chr_temp_file = os.path.join(temp_dir, f"{grp_name}_imputed.h5")

                with h5py.File(chr_temp_file, "w") as temp_outfile:
                    out_grp = temp_outfile.create_group(grp_name)
                    out_dataset = out_grp.create_dataset(
                        "Methylation",
                        shape=(n_probes, n_samples),
                        dtype=np.float32,
                        chunks=True,
                        compression="gzip",
                        compression_opts=self.compression_level,
                    )

                    n_chunks = (n_probes + self.chunk_size - 1) // self.chunk_size
                    log.info(
                        f"Processing {grp_name} in {n_chunks} chunks of size {self.chunk_size}"
                    )

                    for chunk_idx in tqdm(
                        range(n_chunks), desc=f"Imputing {grp_name} chunks"
                    ):
                        start_idx = chunk_idx * self.chunk_size
                        end_idx = min(start_idx + self.chunk_size, n_probes)

                        if n_chunks > 100 and chunk_idx % 50 == 0:
                            log.info(
                                f"Processing chunk {chunk_idx}/{n_chunks} for {grp_name}"
                            )

                        chunk = dataset[start_idx:end_idx, :].astype(np.float32)

                        if chunk.size == 0:
                            continue

                        missing_count = np.isnan(chunk).sum()
                        if missing_count == 0:
                            log.debug(
                                f"Chunk {chunk_idx} has no missing values, skipping imputation"
                            )
                            out_dataset[start_idx:end_idx, :] = chunk
                            continue

                        log.debug(
                            f"Chunk {chunk_idx}: {missing_count}/{chunk.size} missing values"
                        )

                        chunk_T = chunk.T

                        if n_samples <= self.max_samples_for_full_knn:
                            try:
                                imputer = KNNImputer(
                                    n_neighbors=min(self.k, n_samples - 1)
                                )
                                imputed_chunk_T = imputer.fit_transform(chunk_T)
                                log.debug(f"Used regular KNN for chunk {chunk_idx}")
                            except Exception as e:
                                log.warn(
                                    f"Regular KNN failed for chunk {chunk_idx}: {e}"
                                )
                                fast_imputer = FastKNNImputer(
                                    n_neighbors=self.k,
                                    max_samples_for_knn=self.max_samples_for_full_knn,
                                )
                                imputed_chunk_T = fast_imputer.fit_transform(chunk_T)
                        else:
                            fast_imputer = FastKNNImputer(
                                n_neighbors=self.k,
                                max_samples_for_knn=self.max_samples_for_full_knn,
                            )
                            imputed_chunk_T = fast_imputer.fit_transform(chunk_T)
                            log.debug(f"Used fast KNN for chunk {chunk_idx}")

                        imputed_chunk = imputed_chunk_T.T
                        out_dataset[start_idx:end_idx, :] = imputed_chunk

                    probe_key = AliasUtils.find_keys(grp, "ProbeList")
                    if probe_key:
                        probe_data = grp[probe_key][:]
                        out_grp.create_dataset("probeList", data=probe_data)
                        log.debug(f"Copied probe list for {grp_name}")

                log.info(f"Completed imputation for chromosome {grp_name}")
                return chr_temp_file

        except Exception as e:
            log.error(f"Error processing chromosome {grp_name}: {e}")
            return None


class ParallelMethylationImputer:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        k: int = 5,
        chunk_size: int = 5000,
        max_samples_for_full_knn: int = 5000,
        n_processes: Optional[int] = None,
        compression_level: int = 6,
    ) -> None:
        self.input_file = input_file
        self.output_file = output_file
        self.k = k
        self.chunk_size = chunk_size
        self.max_samples_for_full_knn = max_samples_for_full_knn
        self.compression_level = compression_level

        if n_processes is None:
            total_cores = mp.cpu_count()
            self.n_processes = min(4, max(1, total_cores // 2))
        else:
            self.n_processes = max(1, n_processes)

        log.info(f"Using {self.n_processes} processes for parallel imputation")

    def process_single_chromosome(
        self, args: Tuple[str, str, int, int, int, str, int]
    ) -> Optional[str]:
        (
            grp_name,
            input_file,
            k,
            chunk_size,
            max_samples_for_full_knn,
            temp_dir,
            compression_level,
        ) = args

        try:
            if hasattr(log, "child_init") and hasattr(log, "mp_queue"):
                try:
                    log.child_init(log.mp_queue)
                except Exception as e:
                    print(f"Warning: Could not initialize logging for {grp_name}: {e}")

            log.info(f"Worker process starting chromosome {grp_name}")

            with h5py.File(input_file, "r") as infile:
                if grp_name not in infile:
                    log.warn(f"Chromosome {grp_name} not found in input file")
                    return None

                grp = infile[grp_name]
                methylation_key = AliasUtils.find_keys(grp, "Methylation")

                if not methylation_key:
                    log.warn(f"No methylation data found for {grp_name}")
                    return None

                dataset = grp[methylation_key]
                n_probes, n_samples = dataset.shape

                log.info(
                    f"Processing {grp_name}: {n_probes} probes, {n_samples} samples"
                )

                chr_temp_file = os.path.join(temp_dir, f"{grp_name}_imputed.h5")

                with h5py.File(chr_temp_file, "w") as temp_outfile:
                    out_grp = temp_outfile.create_group(grp_name)
                    out_dataset = out_grp.create_dataset(
                        "Methylation",
                        shape=(n_probes, n_samples),
                        dtype=np.float32,
                        chunks=True,
                        compression="gzip",
                        compression_opts=compression_level,
                    )

                    n_chunks = (n_probes + chunk_size - 1) // chunk_size
                    log.info(f"Processing {grp_name} in {n_chunks} chunks")

                    for chunk_idx in range(n_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, n_probes)

                        if n_chunks > 100 and chunk_idx % 50 == 0:
                            log.info(
                                f"Processing chunk {chunk_idx}/{n_chunks} for {grp_name}"
                            )

                        chunk = dataset[start_idx:end_idx, :].astype(np.float32)

                        if chunk.size == 0:
                            continue

                        missing_count = np.isnan(chunk).sum()
                        if missing_count == 0:
                            out_dataset[start_idx:end_idx, :] = chunk
                            continue

                        chunk_T = chunk.T

                        if n_samples <= max_samples_for_full_knn:
                            try:
                                imputer = KNNImputer(n_neighbors=min(k, n_samples - 1))
                                imputed_chunk_T = imputer.fit_transform(chunk_T)
                                log.debug(
                                    f"Used regular KNN for {grp_name} chunk {chunk_idx}"
                                )
                            except Exception as e:
                                log.debug(
                                    f"KNN failed for {grp_name} chunk {chunk_idx}: {e}"
                                )
                                fast_imputer = FastKNNImputer(
                                    n_neighbors=k,
                                    max_samples_for_knn=max_samples_for_full_knn,
                                )
                                imputed_chunk_T = fast_imputer.fit_transform(chunk_T)
                        else:
                            fast_imputer = FastKNNImputer(
                                n_neighbors=k,
                                max_samples_for_knn=max_samples_for_full_knn,
                            )
                            imputed_chunk_T = fast_imputer.fit_transform(chunk_T)
                            log.debug(f"Used fast KNN for {grp_name} chunk {chunk_idx}")

                        imputed_chunk = imputed_chunk_T.T
                        out_dataset[start_idx:end_idx, :] = imputed_chunk

                    probe_key = AliasUtils.find_keys(grp, "ProbeList")
                    if probe_key:
                        probe_data = grp[probe_key][:]
                        out_grp.create_dataset("probeList", data=probe_data)

                log.info(f"Completed chromosome {grp_name}")
                return chr_temp_file

        except Exception as e:
            error_msg = f"Critical error processing chromosome {grp_name}: {e}"
            log.error(error_msg) if hasattr(log, "error") else print(error_msg)
            return None
        except KeyboardInterrupt:
            log.info(f"Process for {grp_name} interrupted by user")
            return None

    def run(self) -> Optional[str]:
        log.info("Starting parallel KNN imputation for methylation data")

        SystemUtils.print_system_info()
        success, message = SystemUtils.disable_core_dumps()
        log.info(f"Core dump prevention: {message}")

        with monitor_resources(interval=2.0) as stats:
            temp_dir = None
            try:
                output_path = os.path.dirname(self.output_file)
                temp_dir = SystemUtils.get_safe_tempdir(
                    output_dir=output_path, required_gb=5.0, prefix="methylation_impute"
                )
                log.debug(f"Using temporary directory: {temp_dir}")

                with h5py.File(self.input_file, "r") as infile:
                    chr_keys = [
                        grp
                        for grp in infile.keys()
                        if AliasUtils.strip_numeric_suffix(grp)
                        in AliasUtils.get_aliases("CHR")
                    ]

                log.info(f"Found {len(chr_keys)} chromosome groups: {chr_keys}")

                if not chr_keys:
                    log.error("No chromosome groups found")
                    return None

                process_args = [
                    (
                        grp_name,
                        self.input_file,
                        self.k,
                        self.chunk_size,
                        self.max_samples_for_full_knn,
                        temp_dir,
                        self.compression_level,
                    )
                    for grp_name in chr_keys
                ]

                temp_files: List[Tuple[str, str]] = []
                chromosomes_processed = 0

                if hasattr(log, "start_multiprocessing_logging"):
                    log.start_multiprocessing_logging()

                with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                    future_to_chr = {
                        executor.submit(self.process_single_chromosome, args): args[0]
                        for args in process_args
                    }

                    for future in tqdm(
                        concurrent.futures.as_completed(future_to_chr),
                        total=len(future_to_chr),
                        desc="Processing Chromosomes",
                    ):
                        if stats["samples"] > 0 and stats["samples"] % 10 == 0:
                            current_memory = SystemUtils.get_memory_info()
                            log.debug(
                                f"Current memory usage: {current_memory['percent_used']:.1f}%"
                            )

                            if current_memory["percent_used"] > 90:
                                log.warn(
                                    "High memory usage detected - consider reducing parallelism"
                                )

                        chr_name = future_to_chr[future]
                        try:
                            result = future.result()
                            if result is not None:
                                temp_files.append((chr_name, result))
                                chromosomes_processed += 1
                                log.info(f"Completed chromosome {chr_name}")
                            else:
                                log.warn(f"Failed to process chromosome {chr_name}")
                        except Exception as e:
                            log.error(
                                f"Exception processing chromosome {chr_name}: {e}"
                            )

                log.info("Merging chromosome results into final output file")
                self.merge_chromosome_files(temp_files, temp_dir)

                if chromosomes_processed == 0:
                    log.error("No chromosomes were processed successfully")
                    return None

                log.info(
                    f"Successfully imputed methylation data for {chromosomes_processed} chromosomes"
                )
                log.success(f"Saved imputed methylation data to {self.output_file}")
                return self.output_file

            except KeyboardInterrupt:
                log.info("Process interrupted by user")
                raise
            except Exception as e:
                log.error(f"Error during parallel KNN imputation: {e}")
                raise
            finally:
                if temp_dir and os.path.exists(temp_dir):
                    SystemUtils.cleanup_tempdir(temp_dir)

    def merge_chromosome_files(
        self, temp_files: List[Tuple[str, str]], temp_dir: str
    ) -> None:
        log.info(f"Merging {len(temp_files)} chromosome files")

        with h5py.File(self.output_file, "w") as outfile:
            for chr_name, temp_file in temp_files:
                if not os.path.exists(temp_file):
                    log.warn(f"Temporary file not found: {temp_file}")
                    continue

                try:
                    with h5py.File(temp_file, "r") as temp_h5:
                        if chr_name in temp_h5:
                            temp_h5.copy(chr_name, outfile)
                            log.debug(f"Merged chromosome {chr_name}")
                        else:
                            log.warn(f"Chromosome {chr_name} not found in temp file")
                except Exception as e:
                    log.error(f"Error merging chromosome {chr_name}: {e}")

            try:
                with h5py.File(self.input_file, "r") as infile:
                    metadata_key = AliasUtils.find_keys(infile, "Metadata")
                    if metadata_key:
                        infile.copy(metadata_key, outfile)
                        log.info("Metadata copied to output file")
            except Exception as e:
                log.warn(f"Could not copy metadata: {e}")


class MethylationImputer:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        k: int = 5,
        chunk_size: int = 5000,
        max_samples_for_full_knn: int = 5000,
        n_processes: Optional[int] = None,
        use_parallel: bool = True,
        compression_level: int = 6,
    ) -> None:
        self.input_file = input_file
        self.output_file = output_file
        self.k = k
        self.chunk_size = chunk_size
        self.max_samples_for_full_knn = max_samples_for_full_knn
        self.n_processes = n_processes
        self.use_parallel = use_parallel
        self.compression_level = compression_level

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_samples_for_full_knn <= 0:
            raise ValueError("max_samples_for_full_knn must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        if not 1 <= compression_level <= 9:
            raise ValueError("compression_level must be between 1 and 9")

        self._validate_system_resources()

    def _validate_system_resources(self) -> None:
        memory_info = SystemUtils.get_memory_info()
        available_memory_gb = memory_info.get("available_gb", 4.0)

        with h5py.File(self.input_file, "r") as f:
            chr_groups = [
                k
                for k in f.keys()
                if AliasUtils.strip_numeric_suffix(k) in AliasUtils.get_aliases("CHR")
            ]
            if chr_groups:
                first_chr = f[chr_groups[0]]
                methylation_data = AliasUtils.find_keys(first_chr, "Methylation")
                if methylation_data:
                    shape = first_chr[methylation_data].shape
                    data_size_gb = (shape[0] * shape[1] * 4) / (1024**3)
                    knn_overhead = data_size_gb * 3
                    process_multiplier = getattr(self, "n_processes", 1) or 1
                    safety_buffer = 2.0
                    knn_addition = data_size_gb + knn_overhead

                    total_estimated_memory = knn_addition * process_multiplier
                    total_estimated_memory = total_estimated_memory * safety_buffer

                    if total_estimated_memory > available_memory_gb * 0.8:
                        log.warn(
                            f"Estimated memory usage ({total_estimated_memory:.1f} GB)"
                        )
                        log.warn(f"Exceeds available ({available_memory_gb:.1f} GB)")

                        self.n_processes = 1
                        self.chunk_size = min(1000, getattr(self, "chunk_size", 1000))
                        self.max_samples_for_full_knn = min(
                            1000, getattr(self, "max_samples_for_full_knn", 1000)
                        )

                        log.info(
                            f"Reduced to: processes={self.n_processes}, chunk_size={self.chunk_size}"
                        )

    def run(self) -> Optional[str]:
        try:
            with h5py.File(self.input_file, "r") as infile:
                chr_keys = [
                    grp
                    for grp in infile.keys()
                    if AliasUtils.strip_numeric_suffix(grp)
                    in AliasUtils.get_aliases("CHR")
                ]

                if not chr_keys:
                    raise ValueError("No chromosome groups found in input file")

                first_chr = chr_keys[0]
                methylation_key = AliasUtils.find_keys(infile[first_chr], "Methylation")
                if methylation_key:
                    sample_shape = infile[first_chr][methylation_key].shape
                    estimated_total_size = (
                        len(chr_keys) * sample_shape[0] * sample_shape[1]
                    )

                    log.info(f"Estimated data size: {estimated_total_size:,} values")
                    log.info(f"Sample chromosome shape: {sample_shape}")

                    if self.use_parallel and (
                        len(chr_keys) > 1 or estimated_total_size > 10_000_000
                    ):
                        log.info("Using parallel processing strategy")
                        imputer = ParallelMethylationImputer(
                            input_file=self.input_file,
                            output_file=self.output_file,
                            k=self.k,
                            chunk_size=self.chunk_size,
                            max_samples_for_full_knn=self.max_samples_for_full_knn,
                            n_processes=self.n_processes,
                            compression_level=self.compression_level,
                        )
                        return imputer.run()
                    else:
                        log.info("Using single-process chunked strategy")
                        return self._run_chunked_single_process()
                else:
                    raise ValueError("No methylation data found in first chromosome")

        except Exception as e:
            log.error(f"Error during methylation imputation: {e}")
            raise

    def _run_chunked_single_process(self) -> Optional[str]:
        chunked_imputer = ChunkedMethylationImputer(
            input_file=self.input_file,
            output_file=self.output_file,
            k=self.k,
            chunk_size=self.chunk_size,
            max_samples_for_full_knn=self.max_samples_for_full_knn,
            compression_level=self.compression_level,
        )

        output_path = os.path.dirname(self.output_file)
        temp_dir = SystemUtils.get_safe_tempdir(
            output_dir=output_path, required_gb=2.0, prefix="methylation_impute_single"
        )

        try:
            with h5py.File(self.input_file, "r") as infile:
                chr_keys = [
                    grp
                    for grp in infile.keys()
                    if AliasUtils.strip_numeric_suffix(grp)
                    in AliasUtils.get_aliases("CHR")
                ]

                temp_files: List[Tuple[str, str]] = []
                for grp_name in tqdm(chr_keys, desc="Processing Chromosomes"):
                    result = chunked_imputer.process_chromosome_chunked(
                        grp_name, temp_dir
                    )
                    if result:
                        temp_files.append((grp_name, result))

                with h5py.File(self.output_file, "w") as outfile:
                    for chr_name, temp_file in temp_files:
                        with h5py.File(temp_file, "r") as temp_h5:
                            temp_h5.copy(chr_name, outfile)

                    metadata_key = AliasUtils.find_keys(infile, "Metadata")
                    if metadata_key:
                        infile.copy(metadata_key, outfile)

                return self.output_file

        finally:
            SystemUtils.cleanup_tempdir(temp_dir, silent=True)


class GenotypeImputer:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        reference_dir: str,
        window_size: int = 5000000,
        buffer_size: int = 250000,
        ne: int = 20000,
        sample_list: Optional[List[str]] = None,
    ) -> None:
        self.input_file = input_file
        self.output_file = output_file
        self.reference_dir = reference_dir
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.ne = ne
        self.threads = SystemUtils.get_optimal_cores(reserve_cores=1)
        log.info(f"Auto-detected optimal threads: {self.threads}")
        self.sample_list = sample_list

        impute_counts = ImputeCounts(
            input_file=self.input_file,
            data_type="Genotype",
            reference_dir=self.reference_dir,
        )
        required_gb = impute_counts._estimate_disk_requirement()
        output_path = os.path.dirname(self.output_file)
        self.temp_dir = SystemUtils.get_safe_tempdir(
            output_dir=output_path, required_gb=required_gb, prefix="impute2"
        )
        log.debug(f"Using temporary directory: {self.temp_dir}")

        self.impute2_bin = DataProcessor.find_impute2()
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        params = DataProcessor.optimize_parameters_advanced(
            self.threads, self.window_size
        )
        self.max_parallel_chr = params["max_parallel_chr"]
        self.window_size = params["window_size"]
        self.process_large_chr_last = params["process_large_chr_last"]
        self._check_reference_files()
        with h5py.File(self.input_file, "r") as h5f:
            chromosomes = [
                grp
                for grp in h5f.keys()
                if AliasUtils.strip_numeric_suffix(grp) in AliasUtils.get_aliases("CHR")
            ]
            if not chromosomes:
                raise ValueError("No chromosomes found in input file")
            first_chr = chromosomes[0]
            genotype_key = AliasUtils.find_keys(h5f[first_chr], "Genotype")
            if genotype_key is None:
                raise ValueError("No genotype data found in input file")
            self.input_dtype = h5f[first_chr][genotype_key].dtype
            data_shape = h5f[first_chr][genotype_key].shape
            metadata_key = AliasUtils.find_keys(h5f, "Metadata")
            if metadata_key:
                iid_key = AliasUtils.find_keys(h5f[metadata_key], "IID")
                if iid_key:
                    num_samples = len(h5f[f"{metadata_key}/{iid_key}"][:])
                else:
                    num_samples = None
            else:
                num_samples = None
            bp_key = AliasUtils.find_keys(h5f[first_chr], "BP")
            if bp_key:
                num_RSIDs = len(h5f[first_chr][bp_key][:])
            else:
                num_RSIDs = None
            if num_samples is None or num_RSIDs is None:
                raise ValueError("Could not determine number of samples or RSIDs")
            if data_shape[0] == num_samples and data_shape[1] == num_RSIDs:
                self.sample_dim = 0
                self.RSID_dim = 1
            elif data_shape[0] == num_RSIDs and data_shape[1] == num_samples:
                self.sample_dim = 1
                self.RSID_dim = 0
            else:
                raise ValueError(
                    f"Data shape {data_shape} does not match samples ({num_samples}) or RSIDs ({num_RSIDs})"
                )
            log.info(f"Input data dtype: {self.input_dtype}")
            log.info(
                f"Input data orientation: samples on dim {self.sample_dim}, RSIDs on dim {self.RSID_dim}"
            )

    def _check_reference_files(self) -> None:
        self.sample_file = os.path.join(self.reference_dir, "1000GP_Phase3.sample")
        if not os.path.exists(self.sample_file):
            raise FileNotFoundError(
                f"Reference sample file not found: {self.sample_file}"
            )
        chromosome = "1"
        hap_file = os.path.join(
            self.reference_dir, f"1000GP_Phase3_chr{chromosome}.hap.gz"
        )
        legend_file = os.path.join(
            self.reference_dir, f"1000GP_Phase3_chr{chromosome}.legend.gz"
        )
        map_file = os.path.join(
            self.reference_dir, f"genetic_map_chr{chromosome}_combined_b37.txt"
        )
        if not os.path.exists(hap_file):
            raise FileNotFoundError(f"Reference haplotype file not found: {hap_file}")
        if not os.path.exists(legend_file):
            raise FileNotFoundError(f"Reference legend file not found: {legend_file}")
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"Genetic map file not found: {map_file}")
        log.info(f"Reference files found in: {self.reference_dir}")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                result = subprocess.run(
                    [self.impute2_bin],
                    cwd=tmp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                combined_output = result.stdout.decode("utf-8") + result.stderr.decode(
                    "utf-8"
                )
                lines = combined_output.splitlines()
                version_line = next(
                    (line.strip() for line in lines if "IMPUTE version" in line), None
                )
                if version_line:
                    log.info(f"{version_line}")
                else:
                    raise RuntimeError(
                        "IMPUTE2 verification failed: Version information not found"
                    )
            log.info(f"IMPUTE2 executable verified at: {self.impute2_bin}")
        except Exception as e:
            log.error(f"Error checking IMPUTE2: {e}")
            raise

    def extract_chromosome_to_gen(self, chromosome: str) -> Optional[Dict[str, Any]]:
        log.info(f"Extracting {chromosome} from HDF5 and converting to GEN format")
        try:
            chr_temp_dir = os.path.join(self.temp_dir, chromosome)
            os.makedirs(chr_temp_dir, exist_ok=True)
            gen_file = os.path.join(chr_temp_dir, f"{chromosome}.gen")
            sample_file = os.path.join(chr_temp_dir, f"{chromosome}.sample")
            with h5py.File(self.input_file, "r") as h5f:
                if chromosome not in h5f:
                    log.warn(f"Chromosome {chromosome} not found in input file")
                    return None
                h5_utils = CachedH5Utils(h5f)
                sample_indices = None
                if self.sample_list:
                    sample_indices = h5_utils.get_sample_indices(self.sample_list)
                    if not sample_indices:
                        log.warn(f"No matching samples found for {chromosome}")
                        return None
                df = h5_utils.read_chromosome(chromosome, data_type="Genotype")
                if df is None:
                    log.warn(f"Failed to read data for {chromosome}")
                    return None
                if sample_indices:
                    sample_cols = ["RSID"] + [df.columns[i + 1] for i in sample_indices]
                    df = df[sample_cols]
                bp_key = AliasUtils.find_keys(h5f[chromosome], "BP")
                if bp_key is None:
                    log.warn(f"No position data found for {chromosome}")
                    return None
                positions = h5f[chromosome][bp_key][:]
                a1_key = AliasUtils.find_keys(h5f[chromosome], "A1")
                a2_key = AliasUtils.find_keys(h5f[chromosome], "A2")
                if a1_key is None or a2_key is None:
                    log.warn(f"Missing allele information for {chromosome}")
                    return None
                allele1 = h5f[chromosome][a1_key][:]
                allele2 = h5f[chromosome][a2_key][:]
                if isinstance(allele1[0], bytes):
                    allele1 = [a.decode("utf-8") for a in allele1]
                if isinstance(allele2[0], bytes):
                    allele2 = [a.decode("utf-8") for a in allele2]
                sort_idx = np.argsort(positions)
                positions = positions[sort_idx]
                allele1 = [allele1[j] for j in sort_idx]
                allele2 = [allele2[j] for j in sort_idx]
                df = df.iloc[sort_idx].reset_index(drop=True)
                sample_ids = df.columns[1:]
                with open(gen_file, "w") as gen:
                    chrom_num = chromosome.replace("CHR", "")
                    for i, (_, row) in enumerate(df.iterrows()):
                        RSID = row["RSID"]
                        gen.write(
                            f"{chrom_num} {RSID} {positions[i]} {allele1[i]} {allele2[i]}"
                        )
                        for sample_id in sample_ids:
                            gt = row[sample_id]
                            if gt == 0:
                                gen.write(" 1 0 0")
                            elif gt == 1:
                                gen.write(" 0 1 0")
                            elif gt == 2:
                                gen.write(" 0 0 1")
                            else:
                                gen.write(" 0 0 0")
                        gen.write("\n")
                sex = None
                if "metadata/sex" in h5f:
                    sex_all = h5f["metadata/sex"][:]
                    if sample_indices:
                        sex = [sex_all[i] for i in sample_indices]
                    else:
                        sex = sex_all
                with open(sample_file, "w") as sample:
                    sample.write("ID_1 ID_2 missing sex\n")
                    sample.write("0 0 0 D\n")
                    for i, sample_id in enumerate(sample_ids):
                        sex_val = sex[i] if sex is not None else 0
                        sample.write(f"{sample_id} {sample_id} 0 {sex_val}\n")
                n_markers = len(positions)
                n_samples = len(sample_ids)
            return {
                "chromosome": chromosome,
                "gen": gen_file,
                "sample": sample_file,
                "temp_dir": chr_temp_dir,
                "size": n_markers * n_samples,
            }
        except Exception as e:
            log.error(f"Error extracting {chromosome} to GEN format: {e}")
            return None

    def _run_impute2_window(
        self, args: Tuple[str, int, int, int, str, str, str, str, str, str]
    ) -> Optional[str]:
        (
            chromosome,
            start_pos,
            end_pos,
            i,
            gen_file,
            sample_file,
            chr_temp_dir,
            map_file,
            hap_file,
            legend_file,
        ) = args
        output_file = os.path.join(chr_temp_dir, f"{chromosome}_chunk{i}.imputed")
        cmd = [
            self.impute2_bin,
            "-m",
            map_file,
            "-h",
            hap_file,
            "-l",
            legend_file,
            "-g",
            gen_file,
            "-sample_g",
            sample_file,
            "-int",
            str(start_pos),
            str(end_pos),
            "-Ne",
            str(self.ne),
            "-o",
            output_file,
            "-verbose",
        ]
        log.info(f"Running IMPUTE2 for {chromosome} window {start_pos}-{end_pos}")
        try:
            result = subprocess.run(
                cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            log.debug(f"IMPUTE2 command: {' '.join(cmd)}")
            log.debug(f"IMPUTE2 stdout:\n{result.stdout.decode('utf-8')}")
            log.debug(f"IMPUTE2 stderr:\n{result.stderr.decode('utf-8')}")
            if result.returncode != 0:
                log.error(
                    f"IMPUTE2 failed with code {result.returncode} for {chromosome} window {start_pos}-{end_pos}"
                )
                log.error(f"IMPUTE2 stdout:\n{result.stdout.decode('utf-8')}")
                log.error(f"IMPUTE2 stderr:\n{result.stderr.decode('utf-8')}")
                return None
            if not os.path.exists(output_file):
                stdout_str = result.stdout.decode("utf-8")
                if "There are no RSIDs in the imputation interval" in stdout_str:
                    log.info(
                        f"No RSIDs in interval for {chromosome} window {start_pos}-{end_pos}, skipping"
                    )
                    for suffix in ["_summary", "_warnings", "_info", "_info_by_sample"]:
                        f = output_file + suffix
                        if os.path.exists(f):
                            os.remove(f)
                    return "empty"
                else:
                    log.error(
                        f"IMPUTE2 ran with exit 0 but output file not created: {output_file}"
                    )
                    log.error(f"IMPUTE2 stdout:\n{result.stdout.decode('utf-8')}")
                    log.error(f"IMPUTE2 stderr:\n{result.stderr.decode('utf-8')}")
                    return None
            summary_file = f"{output_file}_summary"
            if os.path.exists(summary_file):
                with open(summary_file, "r") as f:
                    summary_content = f.read()
                log.info(
                    f"IMPUTE2 Summary for {chromosome} window {start_pos}-{end_pos}:\n{summary_content}"
                )
                os.remove(summary_file)
            warnings_file = f"{output_file}_warnings"
            if os.path.exists(warnings_file):
                with open(warnings_file, "r") as f:
                    warnings_content = f.read().strip()
                if warnings_content:
                    log.warn(
                        f"IMPUTE2 Warnings for {chromosome} window {start_pos}-{end_pos}:\n{warnings_content}"
                    )
                os.remove(warnings_file)
            info_by_sample_file = f"{output_file}_info_by_sample"
            if os.path.exists(info_by_sample_file):
                os.remove(info_by_sample_file)
            return output_file
        except Exception as e:
            log.error(f"Error running IMPUTE2 window: {e}")
            return None

    def run_impute2(self, gen_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        log.info(f"Imputing genotypes for {gen_data['chromosome']} with IMPUTE2")
        try:
            chromosome = gen_data["chromosome"]
            gen_file = gen_data["gen"]
            sample_file = gen_data["sample"]
            chr_temp_dir = gen_data["temp_dir"]
            chrom_num = chromosome.replace("CHR", "")
            hap_file = os.path.join(
                self.reference_dir, f"1000GP_Phase3_chr{chrom_num}.hap.gz"
            )
            legend_file = os.path.join(
                self.reference_dir, f"1000GP_Phase3_chr{chrom_num}.legend.gz"
            )
            map_file = os.path.join(
                self.reference_dir, f"genetic_map_chr{chrom_num}_combined_b37.txt"
            )
            with open(map_file, "r") as mapf:
                lines = mapf.readlines()
                first_line = lines[0].strip().split()
                try:
                    int(first_line[0])
                    start_idx = 0
                except ValueError:
                    start_idx = 1
                    log.info(f"Detected header in genetic map file: {first_line}")
                if start_idx >= len(lines):
                    log.error(f"Genetic map file contains only headers: {map_file}")
                    return None
                positions = []
                for i in range(start_idx, len(lines)):
                    line_parts = lines[i].strip().split()
                    if line_parts:
                        try:
                            positions.append(int(line_parts[0]))
                        except (ValueError, IndexError):
                            log.warn(
                                f"Skipping invalid line in genetic map: {line_parts}"
                            )
                            continue
                if not positions:
                    log.error(
                        f"No valid positions found in genetic map file: {map_file}"
                    )
                    return None
                min_map_pos = min(positions)
                max_map_pos = max(positions)
                log.info(f"Genetic map position range: {min_map_pos} - {max_map_pos}")
            min_pos = max(1, min(positions) - self.buffer_size)
            max_pos = max(positions) + self.buffer_size
            min_pos = max(min_pos, min_map_pos)
            max_pos = min(max_pos, max_map_pos)
            if min_pos > max_pos:
                log.warn(f"Skipping {chromosome} as no overlap with genetic map")
                return None
            memory_info = SystemUtils.get_memory_info()
            available_memory = memory_info.get("available_gb", 8.0)
            window_size = self.window_size
            num_variants = len(positions)
            if available_memory > 100:
                window_size = 10000000
            elif available_memory > 50:
                window_size = 7000000
            if chrom_num in ["1", "2", "3", "4", "5", "6", "7"]:
                window_size = min(window_size, 7000000)
            if chrom_num in ["1", "2"]:
                window_size = min(window_size, 5000000)
            if available_memory < 8:
                window_size = min(window_size, 1000000)
                log.warn(
                    f"Low memory ({available_memory:.1f}GB) - reducing window size to {window_size}"
                )
            elif available_memory < 4:
                window_size = min(window_size, 500000)
                log.warn(
                    f"Very low memory ({available_memory:.1f}GB) - reducing window size to {window_size}"
                )
            log.info(
                f"Using window size of {window_size} for {chromosome} ({num_variants} variants)"
            )
            pos_array = np.array(positions)
            windows: List[Tuple[int, int]] = []
            for start_pos in range(min_pos, max_pos + 1, window_size):
                end_pos = min(start_pos + window_size, max_pos)
                left = np.searchsorted(pos_array, start_pos)
                right = np.searchsorted(pos_array, end_pos)
                if right > left:
                    windows.append((start_pos, end_pos))
            if not windows:
                log.warn(f"No valid windows with RSIDs for {chromosome}")
                return None
            num_potential_windows = (max_pos - min_pos) // window_size + 1
            log.info(
                f"Created {len(windows)} windows with RSIDs (out of potential ~{num_potential_windows})"
            )
            imputed_files: List[str] = []
            log.start_multiprocessing_logging()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.threads,
                initializer=log.child_init,
                initargs=(log.mp_queue,),
            ) as executor:
                args_list = [
                    (
                        chromosome,
                        start_pos,
                        end_pos,
                        i,
                        gen_file,
                        sample_file,
                        chr_temp_dir,
                        map_file,
                        hap_file,
                        legend_file,
                    )
                    for i, (start_pos, end_pos) in enumerate(windows)
                ]
                for result in tqdm(
                    executor.map(self._run_impute2_window, args_list),
                    total=len(windows),
                    desc=f"Imputing windows for {chromosome}",
                ):
                    if result is not None and result != "empty":
                        imputed_files.append(result)
            if len(imputed_files) < len(windows):
                log.warn(
                    f"Some windows failed for {chromosome}: {len(imputed_files)}/{len(windows)} succeeded"
                )
            if len(imputed_files) == 0:
                log.error(f"No windows succeeded for {chromosome}")
                return None
            combined_file = os.path.join(chr_temp_dir, f"{chromosome}_imputed.gen")
            info_file = os.path.join(chr_temp_dir, f"{chromosome}_imputed.info")
            with open(combined_file, "w") as outf:
                for imputed_file in imputed_files:
                    with open(imputed_file, "r") as inf:
                        for line in inf:
                            outf.write(line)
            with open(info_file, "w") as outf:
                header_written = False
                for i, imputed_file in enumerate(imputed_files):
                    info_chunk = f"{imputed_file}_info"
                    if os.path.exists(info_chunk):
                        with open(info_chunk, "r") as inf:
                            lines = inf.readlines()
                            if not header_written and i == 0:
                                outf.write(lines[0])
                                header_written = True
                            for line in lines[1:]:
                                outf.write(line)
            return {
                "chromosome": chromosome,
                "imputed_gen": combined_file,
                "info_file": info_file,
                "sample": sample_file,
                "temp_dir": chr_temp_dir,
            }
        except Exception as e:
            log.error(f"Error running IMPUTE2 for {gen_data['chromosome']}: {e}")
            return None

    def convert_imputed_to_hdf5(
        self, chrom_data: Union[str, Dict[str, Any]]
    ) -> Optional[Dict[str, str]]:
        try:
            chrom = (
                chrom_data["chromosome"] if isinstance(chrom_data, dict) else chrom_data
            )
            imputed_chunks = glob.glob(
                os.path.join(self.temp_dir, chrom, f"{chrom}_chunk*.imputed")
            )
            info_files = glob.glob(
                os.path.join(self.temp_dir, chrom, f"{chrom}_chunk*.imputed_info")
            )
            if not imputed_chunks or not info_files:
                raise FileNotFoundError(f"No imputed files found for {chrom}")
            output_file = os.path.join(self.temp_dir, f"{chrom}_imputed.h5")
            with h5py.File(self.input_file, "r") as src, h5py.File(
                output_file, "w"
            ) as dst:
                metadata_key = AliasUtils.find_keys(src, "Metadata")
                if metadata_key and metadata_key in src:
                    src.copy(metadata_key, dst)
                chr_group = dst.create_group(chrom)
                all_genotypes = []
                all_info = []
                all_positions = []
                all_rsids = []
                all_a1 = []
                all_a2 = []
                for imputed_file, info_file in zip(
                    sorted(imputed_chunks), sorted(info_files)
                ):
                    with open(imputed_file, "r") as f:
                        gen_lines = f.readlines()
                    with open(info_file, "r") as f:
                        info_header = f.readline().strip().split()
                        info_lines = f.readlines()
                    try:
                        rsid_idx = info_header.index("rs_id")
                    except ValueError:
                        rsid_idx = 0
                        log.debug(
                            "INFO file doesn't have rs_id column, using column 0 instead"
                        )
                    try:
                        position_idx = info_header.index("position")
                    except ValueError:
                        position_idx = 1
                        log.debug(
                            "INFO file doesn't have position column, using column 1 instead"
                        )
                    try:
                        a1_idx = info_header.index("a0")
                    except ValueError:
                        try:
                            a1_idx = info_header.index("a_0")
                        except ValueError:
                            a1_idx = 2
                            log.debug(
                                "INFO file doesn't have a0 column, using column 2 instead"
                            )
                    try:
                        a2_idx = info_header.index("a1")
                    except ValueError:
                        try:
                            a2_idx = info_header.index("a_1")
                        except ValueError:
                            a2_idx = 3
                            log.debug(
                                "INFO file doesn't have a1 column, using column 3 instead"
                            )
                    try:
                        info_score_idx = info_header.index("info")
                    except ValueError:
                        info_score_idx = -1
                        log.debug(
                            "INFO file doesn't have info column, using last column instead"
                        )
                    for i, (gen_line, info_line) in enumerate(
                        zip(gen_lines, info_lines)
                    ):
                        info_parts = info_line.strip().split()
                        rsid = info_parts[rsid_idx]
                        position = int(info_parts[position_idx])
                        a1 = info_parts[a1_idx]
                        a2 = info_parts[a2_idx]
                        info_score = float(info_parts[info_score_idx])
                        gen_parts = gen_line.strip().split()
                        gen_probs = gen_parts[5:]
                        n_samples = len(gen_probs) // 3
                        genotypes = np.zeros(n_samples, dtype=np.float32)
                        for j in range(n_samples):
                            p1 = float(gen_probs[j * 3 + 1])
                            p2 = float(gen_probs[j * 3 + 2])
                            genotypes[j] = p1 + 2 * p2
                        all_genotypes.append(genotypes)
                        all_info.append(info_score)
                        all_positions.append(position)
                        all_rsids.append(rsid)
                        all_a1.append(a1)
                        all_a2.append(a2)
                if all_genotypes:
                    genotypes = np.array(all_genotypes, dtype=np.float32)
                    info_scores = np.array(all_info, dtype=np.float32)
                    positions = np.array(all_positions, dtype=np.int32)
                    chr_group.create_dataset("Genotype", data=genotypes)
                    chr_group.create_dataset("INFO", data=info_scores)
                    chr_group.create_dataset("BP", data=positions)
                    chr_group.create_dataset(
                        "RSID", data=np.array(all_rsids, dtype="S10")
                    )
                    chr_group.create_dataset("A1", data=np.array(all_a1, dtype="S1"))
                    chr_group.create_dataset("A2", data=np.array(all_a2, dtype="S1"))
                    log.info(f"Successfully converted imputed GEN for {chrom} to HDF5")
                    return {"chromosome": chrom, "hdf5": output_file}
                else:
                    raise ValueError(f"No data found in imputed files for {chrom}")
        except Exception as e:
            log.error(f"Error converting imputed GEN to HDF5 for {chrom}: {e}")
            return None

    def _sort_chromosomes_by_size(
        self, gen_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return sorted(gen_data, key=lambda x: x["size"] if "size" in x else 0)

    def merge_hdf5_files(self, chromosome_files: List[Dict[str, str]]) -> Optional[str]:
        if not chromosome_files:
            log.error("No chromosomes were imputed successfully.")
            return None
        log.info("Merging imputed chromosome files into final HDF5")
        try:
            with h5py.File(self.output_file, "w") as out_h5:
                metadata_copied = False
                for chr_file in chromosome_files:
                    if chr_file is None:
                        continue
                    chromosome = chr_file["chromosome"]
                    with h5py.File(chr_file["hdf5"], "r") as in_h5:
                        in_h5.copy(chromosome, out_h5)
                        metadata_key = AliasUtils.find_keys(in_h5, "Metadata")
                        if not metadata_copied and metadata_key is not None:
                            in_h5.copy(metadata_key, out_h5)
                            metadata_copied = True
            log.info(f"Successfully created imputed genotype file: {self.output_file}")
            return self.output_file
        except Exception as e:
            log.error(f"Error merging HDF5 files: {e}")
            return None

    def run(self) -> Optional[str]:
        try:
            log.info("Starting IMPUTE2 imputation")
            log.info(f"Input file: {self.input_file}")
            log.info(f"Output file: {self.output_file}")

            temp_dir_space = SystemUtils.get_disk_info(self.temp_dir)
            log.info(f"Temp directory space: {temp_dir_space['free_gb']:.1f}GB free")

            SystemUtils.disable_core_dumps()

            with h5py.File(self.input_file, "r") as h5f:
                h5_utils = CachedH5Utils(h5f)
                chromosomes = h5_utils.get_chromosomes()
                log.info(f"Found {len(chromosomes)} chromosomes to process")
            log.info(
                f"Extracting chromosomes to GEN format using {self.threads} threads"
            )
            gen_data: List[Dict[str, Any]] = []
            has_mp_queue = hasattr(log, "mp_queue") and log.mp_queue is not None
            log.debug(f"Multiprocessing queue available: {has_mp_queue}")
            if has_mp_queue:
                log.start_multiprocessing_logging()
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.threads,
                    initializer=log.child_init,
                    initargs=(log.mp_queue,),
                ) as executor:
                    gen_results = list(
                        tqdm(
                            executor.map(self.extract_chromosome_to_gen, chromosomes),
                            total=len(chromosomes),
                            desc="Converting to GEN format",
                        )
                    )
                    gen_data = [result for result in gen_results if result is not None]
            else:
                if len(chromosomes) > 1 and self.threads > 1:
                    log.warn("Running in serial mode instead of parallel")
                gen_data = [self.extract_chromosome_to_gen(chr) for chr in chromosomes]
                gen_data = [r for r in gen_data if r is not None]
            if self.process_large_chr_last:
                log.info("Sorting chromosomes by size for memory efficiency")
                gen_data = self._sort_chromosomes_by_size(gen_data)
            log.info(
                f"Running IMPUTE2 imputation in parallel (max: {self.max_parallel_chr} processes)"
            )
            has_mp_queue = hasattr(log, "mp_queue") and log.mp_queue is not None
            log.debug(f"Multiprocessing queue available: {has_mp_queue}")
            if has_mp_queue:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.max_parallel_chr,
                    initializer=log.child_init,
                    initargs=(log.mp_queue,),
                ) as executor:
                    imputation_results = list(
                        tqdm(
                            executor.map(self.run_impute2, gen_data),
                            total=len(gen_data),
                            desc="Imputing with IMPUTE2",
                        )
                    )
                    imputed_data = [
                        result for result in imputation_results if result is not None
                    ]
            else:
                if len(gen_data) > 1 and self.threads > 1:
                    log.warn("Running IMPUTE2 in serial mode instead of parallel")
                imputation_results = []
                for data in gen_data:
                    result = self.run_impute2(data)
                    imputation_results.append(result)
                imputed_data = [
                    result for result in imputation_results if result is not None
                ]
            log.info("Converting imputed GEN to HDF5")
            hdf5_data: List[Dict[str, str]] = []
            if has_mp_queue:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.threads,
                    initializer=log.child_init,
                    initargs=(log.mp_queue,),
                ) as executor:
                    hdf5_results = list(
                        tqdm(
                            executor.map(self.convert_imputed_to_hdf5, imputed_data),
                            total=len(imputed_data),
                            desc="Converting to HDF5",
                        )
                    )
                    hdf5_data = [
                        result for result in hdf5_results if result is not None
                    ]
            else:
                if len(imputed_data) > 1 and self.threads > 1:
                    log.warn(
                        "Running conversion to HDF5 in serial mode instead of parallel"
                    )
                hdf5_results = []
                for data in imputed_data:
                    result = self.convert_imputed_to_hdf5(data)
                    hdf5_results.append(result)
                hdf5_data = [result for result in hdf5_results if result is not None]
            log.info("Checking disk space before merging results")
            output_dir = os.path.dirname(self.output_file)
            has_space, message = SystemUtils.check_disk_space(output_dir, 10.0)
            if not has_space:
                log.warn(f"{message} - merging may fail")

            log.info("Merging HDF5 files")
            result = self.merge_hdf5_files(hdf5_data)
            if result and os.path.exists(self.temp_dir):
                log.info(f"Cleaning up temporary files in {self.temp_dir}")
                SystemUtils.cleanup_tempdir(self.temp_dir)
            if has_mp_queue:
                log.mp_queue.put_nowait(None)
                log.listener.join()
            return result
        except Exception as e:
            log.error(f"Error during IMPUTE2 imputation: {e}")
            raise


class QualityFilter:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        threshold: float = 0.3,
        threads: Optional[int] = None,
    ) -> None:
        self.input_file = input_file
        self.output_file = output_file
        self.threshold = threshold
        if threads is None:
            self.threads = SystemUtils.get_optimal_cores(reserve_cores=1)
            log.info(f"Auto-detected optimal threads for filtering: {self.threads}")
        else:
            self.threads = threads
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        with h5py.File(self.input_file, "r") as h5f:
            has_info_scores = False
            h5_utils = CachedH5Utils(h5f)
            for CHR in h5_utils.get_chromosomes():
                info_key = AliasUtils.find_keys(h5f[CHR], "INFO")
                if info_key is not None:
                    has_info_scores = True
                    break
            if not has_info_scores:
                raise ValueError(
                    "No imputation info scores found in the input file. Is this an imputed dataset?"
                )

    def filter_chromosome(self, chromosome: str) -> Optional[Dict[str, Any]]:
        log.info(f"Filtering {chromosome} based on INFO score ≥ {self.threshold}")
        try:
            with h5py.File(self.input_file, "r") as h5f:
                if chromosome not in h5f:
                    log.warn(f"Chromosome {chromosome} not found in input file")
                    return None
                info_key = AliasUtils.find_keys(h5f[chromosome], "INFO")
                if info_key is None:
                    log.warn(f"No info score found for {chromosome}")
                    return None
                info_scores = h5f[chromosome][info_key][:]
                quality_mask = info_scores >= self.threshold
                n_variants_original = len(info_scores)
                n_variants_retained = np.sum(quality_mask)
                log.info(
                    f"{chromosome}: Retained {n_variants_retained}/{n_variants_original} variants "
                    f"({n_variants_retained / n_variants_original * 100:.2f}%)"
                )
                if n_variants_retained == 0:
                    log.warn(
                        f"No variants passed the INFO score filter for {chromosome}"
                    )
                    return None
                filtered_data = {
                    "chromosome": chromosome,
                    "mask": quality_mask,
                    "retained": n_variants_retained,
                    "total": n_variants_original,
                }
                return filtered_data
        except Exception as e:
            log.error(f"Error filtering {chromosome}: {e}")
            return None

    def write_filtered_data(
        self, filtered_data_list: List[Optional[Dict[str, Any]]]
    ) -> Optional[str]:
        log.info(f"Writing filtered data to {self.output_file}")
        try:
            with h5py.File(self.output_file, "w") as out_h5:
                with h5py.File(self.input_file, "r") as in_h5:
                    metadata_key = AliasUtils.find_keys(in_h5, "Metadata")
                    if metadata_key is not None:
                        in_h5.copy(metadata_key, out_h5)
                    for filtered_data in filtered_data_list:
                        if filtered_data is None:
                            continue
                        CHR = filtered_data["chromosome"]
                        mask = filtered_data["mask"]
                        grp = out_h5.create_group(CHR)
                        for dataset_name in in_h5[CHR]:
                            data = in_h5[CHR][dataset_name][:]
                            if data.ndim == 1 and len(data) == len(mask):
                                filtered = data[mask]
                                grp.create_dataset(dataset_name, data=filtered)
                            elif data.ndim == 2:
                                if data.shape[0] == len(mask):
                                    filtered = data[mask, :]
                                    grp.create_dataset(dataset_name, data=filtered)
                                elif data.shape[1] == len(mask):
                                    filtered = data[:, mask]
                                    grp.create_dataset(dataset_name, data=filtered)
                                else:
                                    grp.create_dataset(dataset_name, data=data)
                            else:
                                grp.create_dataset(dataset_name, data=data)
                filter_grp = (
                    out_h5.create_group("filter_info")
                    if "filter_info" not in out_h5
                    else out_h5["filter_info"]
                )
                filter_grp.attrs["threshold_score"] = self.threshold
                filter_grp.attrs["filter_date"] = np.bytes_(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                chrom_list = []
                original_counts = []
                retained_counts = []
                for data in filtered_data_list:
                    if data is not None:
                        chrom_list.append(np.bytes_(data["chromosome"]))
                        original_counts.append(data["total"])
                        retained_counts.append(data["retained"])
                if chrom_list:
                    filter_grp.create_dataset("chromosomes", data=chrom_list)
                    filter_grp.create_dataset(
                        "original_counts", data=np.array(original_counts)
                    )
                    filter_grp.create_dataset(
                        "retained_counts", data=np.array(retained_counts)
                    )
            total_original = sum(
                data["total"] for data in filtered_data_list if data is not None
            )
            total_retained = sum(
                data["retained"] for data in filtered_data_list if data is not None
            )
            retention_rate = (
                total_retained / total_original * 100 if total_original > 0 else 0
            )
            log.success(
                f"Successfully created filtered genotype file: {self.output_file}"
            )
            log.success(
                f"Overall: Retained {total_retained}/{total_original} variants ({retention_rate:.2f}%)"
            )
            return self.output_file
        except Exception as e:
            log.error(f"Error writing filtered data: {e}")
            raise

    def run(self) -> Optional[str]:
        try:
            log.info("Starting imputation quality filtering")
            log.info(f"Input file: {self.input_file}")
            log.info(f"Output file: {self.output_file}")
            log.info(f"Minimum INFO score: {self.threshold}")
            with h5py.File(self.input_file, "r") as h5f:
                h5_utils = CachedH5Utils(h5f)
                chromosomes = h5_utils.get_chromosomes()
                log.info(f"Found {len(chromosomes)} chromosomes to process")
            log.info(f"Filtering chromosomes using {self.threads} threads")
            filtered_data_list: List[Optional[Dict[str, Any]]] = []
            has_mp_queue = hasattr(log, "mp_queue") and log.mp_queue is not None
            log.debug(f"Multiprocessing queue available: {has_mp_queue}")
            if has_mp_queue:
                log.start_multiprocessing_logging()
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.threads,
                    initializer=log.child_init,
                    initargs=(log.mp_queue,),
                ) as executor:
                    filter_results = list(
                        tqdm(
                            executor.map(self.filter_chromosome, chromosomes),
                            total=len(chromosomes),
                            desc="Filtering chromosomes",
                        )
                    )
                    filtered_data_list = [
                        result for result in filter_results if result is not None
                    ]
            else:
                log.warn(
                    "Running chromosome filtering in serial mode instead of parallel"
                )
                filtered_data_list = [
                    self.filter_chromosome(chr) for chr in chromosomes
                ]
                filtered_data_list = [r for r in filtered_data_list if r is not None]
            if not filtered_data_list:
                log.error("No variants passed the INFO score filter for any chromosome")
                with h5py.File(self.output_file, "w") as out_h5:
                    with h5py.File(self.input_file, "r") as in_h5:
                        metadata_key = AliasUtils.find_keys(in_h5, "Metadata")
                        if metadata_key is not None:
                            in_h5.copy(metadata_key, out_h5)
                    filter_grp = out_h5.create_group("filter_info")
                    filter_grp.attrs["threshold_score"] = self.threshold
                    filter_grp.attrs["filter_date"] = np.bytes_(
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    filter_grp.attrs["total_variants_retained"] = 0
                    filter_grp.attrs["total_variants_original"] = 0
                log.warn(
                    f"Created empty output file with filter metadata: {self.output_file}"
                )
                return self.output_file
            result = self.write_filtered_data(filtered_data_list)
            return result
        except Exception as e:
            log.error(f"Error during imputation quality filtering: {e}")
            raise


class ImputeCounts:
    def __init__(
        self,
        output_file: Optional[str] = None,
        input_file: Optional[str] = None,
        data_type: Optional[str] = "Genotype",
        reference_dir: Optional[str] = None,
        k: int = 5,
        threshold: Optional[float] = None,
        window_size: int = 5000000,
        buffer_size: int = 250000,
        ne: int = 20000,
        threads: Optional[int] = None,
        samples: Optional[str] = None,
    ) -> None:
        self.input_file = input_file
        self.output_file = output_file
        self.data_type = self._resolve_or_detect_data_type(data_type)

        self.reference_dir = reference_dir
        self.k = k
        self.threshold = threshold
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.ne = ne
        if threads is None:
            self.threads = SystemUtils.get_optimal_cores(reserve_cores=1)
            log.info(f"Auto-detected optimal threads: {self.threads}")
        else:
            self.threads = threads
            log.info(f"Using specified threads: {self.threads}")
        self.samples = samples

        if self.data_type not in ["Genotype", "Methylation"]:
            raise ValueError("data_type must be either 'genotype' or 'methylation'")
        condition1 = self.data_type == "Genotype"
        condition2 = self.threshold is not None
        if condition1 and condition2:
            output_base = os.path.splitext(self.output_file)[0]
            self.intermediate_file = f"{output_base}_imputed.h5"
        else:
            self.intermediate_file = None

    def _estimate_disk_requirement(self) -> float:
        try:
            input_size_gb = os.path.getsize(self.input_file) / (1024**3)
            log.debug(f"Input file size: {input_size_gb:.2f}GB")

            if self.data_type == "Genotype":
                multiplier = 5.0

                ref_size_gb = 0
                if self.reference_dir and os.path.exists(self.reference_dir):
                    for root, _, files in os.walk(self.reference_dir):
                        for file in files:
                            if file.endswith(
                                (".hap", ".legend", ".sample", ".hap.gz", ".legend.gz")
                            ):
                                ref_path = os.path.join(root, file)
                                if os.path.isfile(ref_path):
                                    ref_size_gb += os.path.getsize(ref_path) / (1024**3)

                    log.debug(f"Reference data size: {ref_size_gb:.2f}GB")
                    ref_factor = min(ref_size_gb * 0.5, 20)
                else:
                    ref_factor = 5.0

                estimate = (input_size_gb * multiplier) + ref_factor

            elif self.data_type == "Methylation":
                multiplier = 3.0

                num_samples = 0
                num_features = 0
                try:
                    with h5py.File(self.input_file, "r") as h5f:
                        if "sample" in h5f:
                            num_samples = len(h5f["sample"])

                        for dataset_name in ["beta", "data", "methylation"]:
                            if dataset_name in h5f:
                                shape = h5f[dataset_name].shape
                                if len(shape) == 2:
                                    num_features = shape[1]
                                    break

                    log.debug(
                        f"Detected {num_samples} samples and {num_features} features"
                    )

                    if num_samples > 0 and num_features > 0:
                        sample_factor = min(num_samples / 100, 10)
                        feature_factor = min(num_features / 1e6, 5)
                        multiplier += (sample_factor * 0.2) + (feature_factor * 0.5)
                except Exception as e:
                    log.debug(f"Could not extract detailed info from input file: {e}")

                estimate = input_size_gb * multiplier
            else:
                estimate = input_size_gb * 4.0 + 10.0

            output_size_gb = input_size_gb * 1.2

            total_estimate = (estimate + output_size_gb) * 1.3

            return max(total_estimate, 10.0)
        except Exception as e:
            log.warning(f"Error estimating disk requirements: {e}")
            return 20.0

    def _resolve_or_detect_data_type(self, data_type: Optional[str]) -> str:
        error_message = "data_type must be either 'genotype' or 'methylation'"
        if data_type is None:
            if not self.input_file:
                raise ValueError(error_message)
            log.info(
                "No data type specified, attempting auto-detection from file structure"
            )
            try:
                return self._detect_from_file_structure()
            except Exception as exc:
                raise ValueError(error_message) from exc

        resolved_field = AliasUtils.get_field(data_type)

        if resolved_field == "Methylation":
            log.info(
                f"Data type '{data_type}' resolved to 'Methylation' via AliasUtils"
            )
            return "Methylation"
        if resolved_field == "Genotype":
            log.info(f"Data type '{data_type}' resolved to 'Genotype' via AliasUtils")
            return "Genotype"

        if self.input_file:
            try:
                detected_type = self._detect_from_file_structure()
                log.info(
                    f"Auto-detection successful: resolved '{data_type}' to '{detected_type}' based on file structure"
                )
                return detected_type
            except Exception:
                pass
        raise ValueError(error_message)

    def _detect_from_file_structure(self) -> str:
        if not self.input_file or not os.path.exists(self.input_file):
            raise ValueError(
                "Input file not specified or does not exist - cannot auto-detect data type"
            )

        with h5py.File(self.input_file, "r") as f:
            chr_groups = [
                k
                for k in f.keys()
                if AliasUtils.strip_numeric_suffix(k) in AliasUtils.get_aliases("CHR")
            ]
            if not chr_groups:
                raise ValueError("No chromosome groups found in input file")

            first_chr = chr_groups[0]
            chr_group = f[first_chr]

            condition1 = AliasUtils.find_keys(chr_group, "Genotype") is not None
            condition2 = AliasUtils.find_keys(chr_group, "A1") is not None
            condition3 = AliasUtils.find_keys(chr_group, "A2") is not None
            condition4 = AliasUtils.find_keys(chr_group, "Methylation") is not None
            condition5 = AliasUtils.find_keys(chr_group, "ProbeList") is not None

            if condition1 and condition2 and condition3:
                detected_type = "Genotype"
            elif condition4 and condition5:
                detected_type = "Methylation"
            else:
                raise ValueError(
                    "Could not detect data type from input file structure - missing expected fields"
                )

        log.info(f"Auto-detected data type: {detected_type}")
        return detected_type

    def run(self) -> Optional[str]:
        try:
            log.info(f"Starting {self.data_type} imputation pipeline")

            self._log_system_context()

            success, message = SystemUtils.disable_core_dumps()
            log.info(f"Core dump prevention: {message}")

            output_path = os.path.dirname(self.output_file)
            required_space_gb = self._estimate_disk_requirement()
            log.info(f"Estimated disk space required: {required_space_gb:.1f}GB")
            has_space, space_message = SystemUtils.check_disk_space(
                output_path, required_space_gb
            )
            if not has_space:
                log.warn(space_message)
                log.warn("Proceeding with limited disk space may cause failures")
            else:
                log.info(space_message)

            safety = SystemUtils.configure_safe_environment()
            log.debug(f"Safety measures applied: {safety}")

            memory_info = SystemUtils.get_memory_info()
            available_memory = memory_info.get("available_gb", 8.0)
            total_memory = memory_info.get("total_gb", 16.0)
            log.info(f"Available memory: {available_memory:.1f}GB")
            log.info(f"Total memory: {total_memory:.1f}GB")
            log.info(f"Using {self.threads} threads")
            if self.data_type == "Methylation":
                log.info("=" * 60)
                log.info("METHYLATION IMPUTATION (Optimized KNN)")
                log.info("=" * 60)

                memory_info = SystemUtils.get_memory_info()
                available_memory = memory_info.get("available_gb", 8.0)

                if available_memory >= 32:
                    chunk_size = 10000
                    max_samples_for_full_knn = 10000
                    n_processes = 4
                elif available_memory >= 16:
                    chunk_size = 5000
                    max_samples_for_full_knn = 5000
                    n_processes = 3
                elif available_memory >= 8:
                    chunk_size = 2000
                    max_samples_for_full_knn = 2000
                    n_processes = 2
                else:
                    chunk_size = 1000
                    max_samples_for_full_knn = 1000
                    n_processes = 1

                log.info(f"Memory-optimized parameters: chunk_size={chunk_size}")
                log.info(f"Using {n_processes} processes for parallel processing")

                imputer = MethylationImputer(
                    input_file=self.input_file,
                    output_file=self.output_file,
                    k=self.k,
                    chunk_size=chunk_size,
                    max_samples_for_full_knn=max_samples_for_full_knn,
                    n_processes=n_processes,
                    use_parallel=True,
                    compression_level=6,
                )

                result = imputer.run()
                log.success("Optimized methylation imputation completed successfully")
                return result
            elif self.data_type == "Genotype":
                if not self.reference_dir:
                    raise ValueError(
                        "reference_dir is required for genotype imputation"
                    )
                log.info("=" * 60)
                log.info("GENOTYPE IMPUTATION (IMPUTE2)")
                log.info("=" * 60)
                sample_list: Optional[List[str]] = None
                if self.samples:
                    sample_list = self.samples.split(",")
                if self.threshold is None:
                    impute_output = self.output_file
                    log.info(
                        "No filtering requested - imputation output will be final result"
                    )
                else:
                    impute_output = self.intermediate_file
                    log.info(
                        f"Filtering requested - imputation output will be saved to intermediate file: {impute_output}"
                    )
                imputer = GenotypeImputer(
                    input_file=self.input_file,
                    output_file=impute_output,
                    reference_dir=self.reference_dir,
                    window_size=self.window_size,
                    buffer_size=self.buffer_size,
                    ne=self.ne,
                    sample_list=sample_list,
                )
                impute_result = imputer.run()
                if not impute_result:
                    log.error("Genotype imputation failed")
                    return None
                log.success("Genotype imputation completed successfully")
                if self.threshold is not None:
                    log.info("=" * 60)
                    log.info("IMPUTATION QUALITY FILTERING")
                    log.info("=" * 60)
                    filter_tool = QualityFilter(
                        input_file=impute_result,
                        output_file=self.output_file,
                        threshold=self.threshold,
                        threads=self.threads,
                    )
                    filter_result = filter_tool.run()
                    if not filter_result:
                        log.error("Filtering failed")
                        return None
                    log.success("Quality filtering completed successfully")
                    if os.path.exists(self.intermediate_file):
                        log.info(
                            f"Cleaning up intermediate file: {self.intermediate_file}"
                        )
                        os.remove(self.intermediate_file)
                else:
                    log.info("No filtering requested - pipeline completed")
                return self.output_file
        except Exception as e:
            log.error(f"Pipeline failed: {e}")
            if self.intermediate_file and os.path.exists(self.intermediate_file):
                log.info(
                    f"Cleaning up intermediate file due to error: {self.intermediate_file}"
                )
                os.remove(self.intermediate_file)
            raise

    def _log_system_context(self) -> None:
        system_info = SystemUtils.get_system_info()
        memory_info = SystemUtils.get_memory_info()

        log.info("=== SYSTEM CONTEXT ===")
        log.info(f"CPU: {system_info['cpu_name']}")
        log.info(
            f"Architecture: {system_info['platform']} {system_info['architecture']}"
        )
        log.info(f"Environment: {system_info['environment']}")

        if system_info["allocated_cores"]:
            log.info(
                f"Allocated cores: {system_info['allocated_cores']} ({system_info['environment']})"
            )
        log.info(f"Physical cores: {system_info['physical_cores']}")
        log.info(f"Logical cores: {system_info['logical_cores']}")
        log.info(f"Effective cores: {system_info['effective_cores']}")
        log.info(f"Using threads: {self.threads}")

        log.info(f"Memory total: {memory_info['total_gb']:.1f}GB")
        log.info(f"Memory available: {memory_info['available_gb']:.1f}GB")
        log.info(f"Memory source: {memory_info['source']}")
        log.info("=" * 23)


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
    OptionConfig(
        flags=["-dt", "--data_type"],
        type=str,
        default=None,
        required=False,
        choices=["Genotype", "Methylation"],
    ),
    OptionConfig(
        flags=["-r", "--reference_dir"], type=str, default=None, required=False
    ),
    OptionConfig(flags=["-w", "--window"], type=int, default=5000000, required=False),
    OptionConfig(flags=["-bf", "--buffer"], type=int, default=250000, required=False),
    OptionConfig(
        flags=["-ne", "--effective_size"], type=int, default=20000, required=False
    ),
    OptionConfig(
        flags=["-th", "--threshold"], type=float, default=None, required=False
    ),
    OptionConfig(flags=["-s", "--samples"], type=str, default=None, required=False),
    OptionConfig(flags=["-k", "--k"], type=int, default=5, required=False),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="ImputeCounts")
    opt = framework.run()
    pipeline = ImputeCounts(
        input_file=opt.input,
        output_file=opt.output,
        data_type=opt.data_type,
        reference_dir=opt.reference_dir,
        k=opt.k,
        threshold=opt.threshold,
        window_size=opt.window,
        buffer_size=opt.buffer,
        ne=opt.effective_size,
        samples=opt.samples,
    )
    pipeline.run()
