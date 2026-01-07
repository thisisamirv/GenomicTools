#!/usr/bin/env python
# Import required modules
import concurrent.futures
import gc
import h5py
import multiprocessing
import numpy as np
import os
import pandas as pd
import shutil
import tempfile
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.H5Utils import CachedH5Utils
from utils.LoggingUtils import log
from utils.SystemUtils import SystemUtils, monitor_resources


class CountsQC:
    # Class constant for metadata columns to avoid repeated list creation
    METADATA_COLS = frozenset(["CHR", "BP", "A1", "A2", "INFO", "DataType"])
    
    def __init__(
        self,
        input_file: str,
        output_file: str,
        metric: str = "marker_call_rate",
        threshold: Optional[float] = None,
        data_type: Optional[str] = None,
    ) -> None:
        try:
            if multiprocessing.get_start_method(allow_none=True) is None:
                multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

        self.input_file = input_file
        self.metric = metric.lower()
        self.output_file = output_file
        self.data_type = data_type

        self.max_workers = SystemUtils.get_optimal_cores(reserve_cores=1)
        memory_info = SystemUtils.get_memory_info()
        self.available_memory = memory_info.get("available_gb", 8.0)
        self.total_memory = memory_info.get("total_gb", 16.0)
        safe_config = SystemUtils.configure_safe_environment()
        if safe_config.get("memory_limit_set", False):
            log.debug("Memory limits configured to prevent OOM errors")
        if safe_config.get("core_dumps_disabled", False):
            log.debug("Core dumps disabled for stability")

        log.info(
            f"System resources: {self.max_workers} cores, {self.available_memory:.1f}GB available memory"
        )

        valid_metrics = ["marker_call_rate", "sample_call_rate", "probe_variance"]
        if self.metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of: {valid_metrics}"
            )

        if threshold is None:
            default_thresholds = {
                "marker_call_rate": 0.98,
                "sample_call_rate": 0.90,
                "probe_variance": 0.05,
            }
            self.threshold = default_thresholds[self.metric]
        else:
            self.threshold = threshold

        self.temp_dirs = []
        condition1 = self.metric == "probe_variance"
        condition2 = self.data_type is not None and self.data_type != "Methylation"
        if condition1 and condition2:
            raise ValueError(
                "Probe variance calculation is only applicable to methylation data"
            )

        self._adjust_workers()

        if self.data_type is None:
            self._detect_data_type()

        if self.metric == "probe_variance" and self.data_type != "Methylation":
            raise ValueError(
                "Probe variance calculation is only applicable to methylation data"
            )

        log.start_multiprocessing_logging()

    def _cleanup(self) -> None:
        temp_dirs = getattr(self, "temp_dirs", None)
        if temp_dirs:
            for temp_dir in temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        log.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    log.warn(f"Error cleaning up temp directory {temp_dir}: {e}")

        try:
            cleanup_result = SystemUtils.cleanup_stale_temp_files(
                prefix=f"countsqc_{self.metric}",
                max_age_hours=24,
                dry_run=False,
            )

            condition1 = isinstance(cleanup_result, dict)
            condition2 = cleanup_result.get("dirs_deleted", 0) > 0
            condition3 = cleanup_result.get("files_deleted", 0) > 0
            condition2_3 = condition2 or condition3
            if condition1 and condition2_3:
                log.debug(
                    f"Cleaned up {cleanup_result.get('dirs_deleted', 0)} stale directories and "
                    f"{cleanup_result.get('files_deleted', 0)} files"
                )
        except Exception as e:
            log.debug(f"Error during stale file cleanup: {e}")

        gc.collect()

    def __del__(self) -> None:
        try:
            self._cleanup()
            log.stop_multiprocessing_logging()
        except Exception:
            pass

    def _adjust_workers(self) -> None:
        try:
            file_size = os.path.getsize(self.input_file) / (1024**3)

            memory_info = SystemUtils.get_memory_info()
            self.available_memory = memory_info.get("available_gb", 8.0)
            self.total_memory = memory_info.get("total_gb", 16.0)

            self.max_workers = SystemUtils.get_optimal_cores(reserve_cores=1)

            memory_per_worker = {
                "marker_call_rate": 2.0,
                "sample_call_rate": 3.0,
                "probe_variance": 4.0,
            }.get(self.metric, 3.0)

            usable_memory = max(0.0, self.available_memory - 2.0)
            memory_based_workers = max(1, int(usable_memory / memory_per_worker))

            if file_size > 50:
                size_based_limit = min(4, self.max_workers)
                log.info(f"Large file detected ({file_size:.1f} GB). Limiting workers.")
            elif file_size > 20:
                size_based_limit = min(6, self.max_workers)
                log.info(
                    f"Large file detected ({file_size:.1f} GB). Adjusting workers."
                )
            elif file_size > 10:
                size_based_limit = min(8, self.max_workers)
            else:
                size_based_limit = self.max_workers

            if self.metric == "probe_variance":
                size_based_limit = max(1, size_based_limit - 1)

            self.max_workers = min(
                self.max_workers, memory_based_workers, size_based_limit
            )

            self.max_workers = max(1, self.max_workers)

            log.info(
                f"Resource allocation: {self.max_workers} workers (from CPU: {SystemUtils.get_optimal_cores()}, "
                f"memory: {memory_based_workers}, file size: {size_based_limit})"
            )
            log.info(
                f"Available memory: {self.available_memory:.1f} GB, File size: {file_size:.1f} GB"
            )

        except FileNotFoundError:
            log.warn(f"Could not determine file size for {self.input_file}")
            self.max_workers = max(
                1, min(2, SystemUtils.get_optimal_cores(reserve_cores=1))
            )
            log.info(f"Defaulting to {self.max_workers} workers")

    def _calculate_chunk_size(self, total_items: int) -> int:
        memory_per_worker = self.available_memory / max(1, self.max_workers)

        if total_items <= 10000:
            return max(1000, min(5000, total_items))

        if self.metric == "sample_call_rate":
            base_chunk = int(memory_per_worker * 2000)
        else:
            base_chunk = int(memory_per_worker * 10000)

        min_chunk = 1000
        max_chunk = min(100000, max(1000, total_items // 10))

        chunk_size = max(min_chunk, min(base_chunk, max_chunk))
        log.info(
            f"Calculated chunk size: {chunk_size} (from {total_items} total items)"
        )

        return chunk_size

    def _setup_temp_directory(self) -> str:
        output_dir = os.path.dirname(os.path.abspath(self.output_file))

        try:
            temp_dir, temp_info = SystemUtils.create_safe_tempdir(
                default_path=output_dir,
                required_gb=0.5,
                prefix=f"countsqc_{self.metric}",
                buffer_percent=10.0,
            )
            log.info(f"Created temporary directory: {temp_dir}")
            try:
                free_gb = temp_info.get("disk_info", {}).get("free_gb")
                if free_gb is not None:
                    log.debug(f"Temp directory has {free_gb:.1f}GB free")
            except Exception:
                pass
            return temp_dir
        except Exception as e:
            log.warn(f"Failed to create safe temp directory: {e}")
            log.warn("Using system temp directory as fallback")
            return tempfile.mkdtemp(prefix=f"countsqc_{self.metric}")

    def _check_system_health(self) -> bool:
        log.info("Performing system health check...")

        required_output_gb = self._estimate_output_size()

        health = SystemUtils.check_system_health(
            min_free_disk_gb=required_output_gb,
            max_cpu_percent=95.0,
            max_memory_percent=90.0,
        )

        if health["status"] == "critical":
            log.error("CRITICAL SYSTEM ISSUES:")
            for issue in health["critical"]:
                log.error(f"  - {issue}")
            return False

        if health["status"] == "warning":
            log.warn("SYSTEM WARNINGS:")
            for issue in health["warnings"]:
                log.warn(f"  - {issue}")

        output_dir = os.path.dirname(os.path.abspath(self.output_file))
        disk_ok, disk_message = SystemUtils.check_disk_space(
            path=output_dir, required_gb=required_output_gb, buffer_percent=15.0
        )

        if not disk_ok:
            log.error(f"Disk space issue: {disk_message}")
            return False

        log.info("System health check passed")
        return True

    def _estimate_output_size(self) -> float:
        try:
            input_size_gb = os.path.getsize(self.input_file) / (1024**3)

            if self.metric == "marker_call_rate":
                return max(0.05 * input_size_gb, 0.01)

            elif self.metric == "sample_call_rate":
                return 0.01

            elif self.metric == "probe_variance":
                return max(self.threshold * input_size_gb * 0.1, 0.01)

            return 0.1

        except Exception as e:
            log.debug(f"Error estimating output size: {e}")
            return 0.1

    def _detect_data_type(self) -> None:
        try:
            with h5py.File(self.input_file, "r") as h5_file:
                metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
                if metadata_key is None:
                    log.warn("No metadata group found in HDF5 file")
                    chr_keys = [
                        key for key in h5_file.keys() if key.startswith(("chr", "CHR"))
                    ]
                    if not chr_keys:
                        raise ValueError(
                            "Cannot determine data type: no chromosome groups found"
                        )
                    first_chr = chr_keys[0]
                    chr_group = h5_file[first_chr]
                    meth_indicator = AliasUtils.find_keys(chr_group, "Methylation")
                    probe_indicator = AliasUtils.find_keys(chr_group, "CGID")
                    geno_indicator = AliasUtils.find_keys(chr_group, "Genotype")
                    snp_indicator = AliasUtils.find_keys(chr_group, "RSID")
                    if meth_indicator or probe_indicator:
                        self.data_type = "Methylation"
                    elif geno_indicator or snp_indicator:
                        self.data_type = "Genotype"
                    else:
                        raise ValueError(
                            "Cannot determine data type from chromosome data"
                        )
                else:
                    metadata_group = h5_file[metadata_key]
                    sample_list_key = AliasUtils.find_keys(metadata_group, "SampleList")
                    iid_key = AliasUtils.find_keys(metadata_group, "IID")
                    if sample_list_key and not iid_key:
                        self.data_type = "Methylation"
                    elif iid_key and not sample_list_key:
                        self.data_type = "Genotype"
                    else:
                        chr_keys = [
                            key
                            for key in h5_file.keys()
                            if key.startswith(("chr", "CHR")) and key != metadata_key
                        ]
                        if chr_keys:
                            first_chr = chr_keys[0]
                            chr_group = h5_file[first_chr]
                            meth_indicator = AliasUtils.find_keys(
                                chr_group, "Methylation"
                            )
                            probe_indicator = AliasUtils.find_keys(chr_group, "CGID")
                            geno_indicator = AliasUtils.find_keys(chr_group, "Genotype")
                            snp_indicator = AliasUtils.find_keys(chr_group, "RSID")
                            if meth_indicator or probe_indicator:
                                self.data_type = "Methylation"
                            elif geno_indicator or snp_indicator:
                                self.data_type = "Genotype"
                if self.data_type is None:
                    raise ValueError("Cannot determine data type from file structure")
                log.info(f"Detected {self.data_type} data format")
        except Exception as e:
            log.error(f"Error detecting data type: {e}")
            raise

    def _get_missing_value_condition(self, data_values: Any) -> np.ndarray:
        """Detect missing values based on data type and dtype."""
        dtype = getattr(data_values, "dtype", None)
        is_float = dtype is not None and np.issubdtype(dtype, np.floating)
        is_int = dtype is not None and np.issubdtype(dtype, np.integer)
        
        try:
            if self.data_type == "Methylation":
                # Methylation: NaN for floats, sentinel values for ints
                if is_float:
                    return np.isnan(data_values)
                if is_int:
                    return (data_values == -1) | (data_values == -999) | (data_values == 0)
                # Unknown type: try float conversion
                try:
                    return np.isnan(data_values.astype(float))
                except (ValueError, TypeError):
                    return data_values == -1
            else:
                # Genotype: -1 is missing, NaN for floats
                if is_int:
                    return data_values == -1
                if is_float:
                    return np.isnan(data_values) | (data_values == -1)
                # Unknown type: try numeric conversion
                try:
                    numeric_data = pd.to_numeric(data_values.flatten(), errors="coerce")
                    numeric_data = numeric_data.reshape(data_values.shape)
                    return np.isnan(numeric_data) | (numeric_data == -1)
                except (ValueError, TypeError, AttributeError):
                    # String-based fallback
                    return (
                        (data_values == "-1") | (data_values == "NA") |
                        (data_values == "nan") | (data_values == "")
                    )
        except Exception as e:
            log.error(f"Error in missing value detection: {e}")
            # Fallback chain
            for check in [lambda d: d == -1, lambda d: np.isnan(d)]:
                try:
                    return check(data_values)
                except Exception:
                    continue
            log.warn("Could not detect missing values, assuming no missing data")
            return np.zeros(getattr(data_values, "shape", (0,)), dtype=bool)

    def _read_chromosome_data_direct(
        self, h5_file: h5py.File, chromosome: Union[str, int]
    ) -> Optional[pd.DataFrame]:
        try:
            chrom_key = str(chromosome)
            if chrom_key not in h5_file:
                return None
            chr_group = h5_file[chrom_key]
            if self.data_type == "Methylation":
                data_key = AliasUtils.find_keys(chr_group, "Methylation")
                id_key = AliasUtils.find_keys(chr_group, "CGID")
            else:
                data_key = AliasUtils.find_keys(chr_group, "Genotype")
                id_key = AliasUtils.find_keys(chr_group, "RSID")
            if data_key is None or id_key is None:
                log.warn(f"Required datasets not found in chromosome {chromosome}")
                return None
            data_values = chr_group[data_key][:]
            id_values = chr_group[id_key][:]
            if isinstance(id_values[0], bytes):
                id_values = [s.decode("utf-8") for s in id_values]
            num_samples = data_values.shape[1]
            sample_cols = [f"sample_{i}" for i in range(num_samples)]
            df_data: Dict[str, Any] = {id_key: id_values}
            for i, col in enumerate(sample_cols):
                df_data[col] = data_values[:, i]
            return pd.DataFrame(df_data)
        except Exception as e:
            log.error(f"Error reading chromosome {chromosome} directly: {e}")
            return None

    def process_chromosome_marker_call_rate(
        self, chromosome: Union[str, int]
    ) -> Optional[pd.DataFrame]:
        try:
            log.debug(f"Processing chromosome: {chromosome}")
            with h5py.File(self.input_file, "r") as h5_file:
                try:
                    with CachedH5Utils(h5_file) as h5_utils:
                        chr_data = h5_utils.read_chromosome(
                            chromosome, data_type=self.data_type
                        )
                except Exception as e:
                    log.debug(f"CachedH5Utils failed: {e}, trying direct read")
                    chr_data = self._read_chromosome_data_direct(h5_file, chromosome)
                if chr_data is None or chr_data.empty:
                    log.warn(f"No data found for chromosome: {chromosome}")
                    return None
                id_field = "CGID" if self.data_type == "Methylation" else "RSID"
                id_col_name = AliasUtils.find_keys(chr_data, id_field)
                if id_col_name is None:
                    id_col = chr_data.iloc[:, 0]
                    data_cols = chr_data.columns[1:]
                else:
                    id_col = chr_data[id_col_name]
                    data_cols = [col for col in chr_data.columns if col != id_col_name]
                data_cols = [col for col in data_cols if col not in self.METADATA_COLS]
                if not data_cols:
                    log.warn(f"No data columns found for chromosome {chromosome}")
                    return None
                data_values = chr_data[data_cols].values
                log.debug(
                    f"Data shape: {data_values.shape}, dtype: {data_values.dtype}"
                )
                try:
                    missing_condition = self._get_missing_value_condition(data_values)
                    call_rate_values = (~missing_condition).sum(
                        axis=1
                    ) / data_values.shape[1]
                    call_rate_values = np.round(call_rate_values, 3)
                except Exception as e:
                    log.error(f"Error calculating missing values for {chromosome}: {e}")
                    return None
                below_threshold_indices = call_rate_values <= self.threshold
                below_threshold_count = below_threshold_indices.sum()
                if below_threshold_count == 0:
                    entity_type = (
                        "probes" if self.data_type == "Methylation" else "SNPs"
                    )
                    log.debug(
                        f"No {entity_type} below the threshold for chromosome: {chromosome}"
                    )
                    return None
                filtered_results = pd.DataFrame(id_col[below_threshold_indices])
                entity_type = "probes" if self.data_type == "Methylation" else "SNPs"
                log.debug(
                    f"Number of {entity_type} below the threshold for chromosome {chromosome}: {len(filtered_results)}"
                )
                return filtered_results
        except Exception as e:
            log.error(f"Error processing chromosome {chromosome}: {e}")
            return None

    def process_chromosome_sample_call_rate(
        self, chromosome: Union[str, int]
    ) -> Optional[Dict[str, Any]]:
        try:
            log.debug(f"Worker process: Starting for chromosome {chromosome}")
            with h5py.File(self.input_file, "r") as h5_file:
                log.debug(f"Worker: Opened file for chromosome {chromosome}")
                try:
                    with CachedH5Utils(h5_file) as h5_utils:
                        log.debug(
                            f"Worker: Using CachedH5Utils for chromosome {chromosome}"
                        )
                        chr_data = h5_utils.read_chromosome(
                            chromosome, data_type=self.data_type
                        )
                except Exception as e:
                    log.debug(
                        f"Worker: CachedH5Utils failed for {chromosome}, using direct read: {str(e)}"
                    )
                    chr_data = self._read_chromosome_data_direct(h5_file, chromosome)
                if chr_data is None or chr_data.empty:
                    log.warn(f"Worker: No data for chromosome {chromosome}")
                    return None
                log.debug(f"Worker: Got data for chromosome {chromosome}")
                id_field = "CGID" if self.data_type == "Methylation" else "RSID"
                id_col_name = AliasUtils.find_keys(chr_data, id_field)
                if id_col_name is None:
                    data_cols = chr_data.columns[1:]
                    sample_names = list(data_cols)
                else:
                    data_cols = [col for col in chr_data.columns if col != id_col_name]
                    sample_names = data_cols
                data_cols = [col for col in data_cols if col not in self.METADATA_COLS]
                sample_names = [col for col in sample_names if col not in self.METADATA_COLS]
                if hasattr(chr_data, "select_dtypes"):
                    numeric_data = chr_data[data_cols].select_dtypes(
                        include=[np.number]
                    )
                    if len(numeric_data.columns) != len(data_cols):
                        log.debug("Filtered out non-numeric columns.")
                        log.debug(
                            f"Original: {len(data_cols)}, Numeric: {len(numeric_data.columns)}"
                        )
                        data_cols = list(numeric_data.columns)
                        sample_names = [
                            name for name in sample_names if name in data_cols
                        ]
                if not data_cols:
                    log.warn(
                        f"No numeric data columns found for chromosome {chromosome}"
                    )
                    return None
                data_values = chr_data[data_cols].values
                log.debug(
                    f"Worker: Final data shape for {chromosome}: {data_values.shape}, dtype: {data_values.dtype}"
                )
                try:
                    missing_condition = self._get_missing_value_condition(data_values)
                    non_missing_counts = np.sum(~missing_condition, axis=0)
                    num_markers = data_values.shape[0]
                except Exception as e:
                    log.error(f"Error calculating missing values for {chromosome}: {e}")
                    return None
                log.debug(
                    f"Worker: Completed processing for {chromosome}, markers: {num_markers}"
                )
                return {
                    "non_missing": non_missing_counts,
                    "num_markers": np.full(len(non_missing_counts), num_markers),
                    "sample_names": sample_names,
                }
        except Exception as e:
            log.error(f"Worker error processing chromosome {chromosome}: {e}")
            return None

    def process_chromosome_probe_variance(
        self, chromosome: Union[str, int]
    ) -> Optional[pd.DataFrame]:
        try:
            with h5py.File(self.input_file, "r") as h5_file:
                log.debug(f"Processing chromosome: {chromosome}")

                # Use ChunkedH5Utils for memory-efficient processing
                from utils.H5Utils import ChunkedH5Utils
                
                # Container for results from chunks
                chunk_variances = []
                chunk_probe_ids = []

                def process_chunk(chunk_data: np.ndarray):
                    """Callback to process each chunk of data."""
                    # Calculate variance for the chunk
                    # axis=1 means variance across samples for each probe
                    variances = np.nanvar(chunk_data, axis=1)
                    chunk_variances.append(variances)
                
                try:
                    with ChunkedH5Utils(h5_file) as h5_utils:
                        # We use read_chromosome_chunked with a callback
                        # This avoids loading the whole data matrix into memory
                        h5_utils.read_chromosome_chunked(
                            chromosome, 
                            data_type="Methylation",
                            chunk_callback=process_chunk
                        )
                        
                        # Now retrieve the probe IDs
                        # We need to map the chromosome name first to find the group
                        mapper = h5_utils.chromosome_mapper
                        actual_chrom = mapper.map_chromosome_name(chromosome)
                        if actual_chrom:
                            chr_group = h5_file[actual_chrom]
                            probe_id_field = AliasUtils.find_keys(chr_group, "CGID")
                            if probe_id_field is None:
                                # Fallback if no specific ID field
                                total_probes = sum(len(v) for v in chunk_variances)
                                probe_ids = [f"probe_{i}" for i in range(total_probes)]
                            else:
                                probe_ids = chr_group[probe_id_field][:]
                                if isinstance(probe_ids[0], bytes):
                                    probe_ids = [s.decode("utf-8").rstrip('\x00').strip() for s in probe_ids]
                        else:
                            log.warn(f"Could not map chromosome {chromosome} for ID retrieval")
                            return None

                except Exception as e:
                    log.error(f"Error during chunked processing of {chromosome}: {e}")
                    return None

                if not chunk_variances:
                    log.warn(f"No data processed for chromosome {chromosome}")
                    return None

                # Combine results
                all_variances = np.concatenate(chunk_variances)
                
                # Verify lengths match
                if len(all_variances) != len(probe_ids):
                    log.warn(
                        f"Mismatch in probe counts for {chromosome}: "
                        f"Variances={len(all_variances)}, IDs={len(probe_ids)}"
                    )
                    # Truncate to shorter length to be safe (though this indicates an issue)
                    min_len = min(len(all_variances), len(probe_ids))
                    all_variances = all_variances[:min_len]
                    probe_ids = probe_ids[:min_len]

                variance_df = pd.DataFrame(
                    {"probe_id": probe_ids, "variance": all_variances}
                )
                
                log.debug(f"Completed variance calculation for {chromosome}")
                return variance_df

        except Exception as e:
            log.error(f"Error processing chromosome {chromosome}: {e}")
            return None

    def run_marker_call_rate(self) -> str:
        try:
            entity_type = "probe" if self.data_type == "Methylation" else "SNP"
            log.info(
                f"Starting {entity_type} call rate calculation. Data type: {self.data_type}"
            )
            with h5py.File(self.input_file, "r") as h5_file:
                try:
                    with CachedH5Utils(h5_file) as h5_utils:
                        chromosome_list = h5_utils.get_chromosomes()
                except Exception:
                    chromosome_list = None
                if not chromosome_list:
                    metadata_aliases = AliasUtils.get_aliases("Metadata")
                    chromosome_list = [
                        key for key in h5_file.keys() if key not in metadata_aliases
                    ]
                if not chromosome_list:
                    raise ValueError(
                        "Failed to retrieve chromosome list from HDF5 file"
                    )
                log.info(
                    f"Found {len(chromosome_list)} chromosomes using direct detection"
                )
            log.info(f"Using {self.max_workers} cores for parallel processing.")
            entity_desc = "probe" if self.data_type == "Methylation" else "SNP"
            print(f"Calculating {entity_desc} call rates...")
            results: List[pd.DataFrame] = []
            log_queue = log.get_queue()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=log.child_init if log_queue else None,
                initargs=(log_queue,) if log_queue else (),
            ) as executor:
                future_to_chr = {
                    executor.submit(
                        self.process_chromosome_marker_call_rate, chromosome
                    ): chromosome
                    for chromosome in chromosome_list
                }
                for future in concurrent.futures.as_completed(future_to_chr):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        log.error(f"Error in chromosome processing: {e}")
            results = [r for r in results if r is not None]
            if not results:
                entity_desc = "CpGs" if self.data_type == "Methylation" else "SNPs"
                log.info(f"No {entity_desc} below the threshold.")
                open(self.output_file, "w").close()
                log.success(
                    f"{entity_desc} call rates calculated successfully and written to: {self.output_file}"
                )
                log.success(f"The number of {entity_desc} below the threshold: 0")
                return self.output_file

            combined_results = pd.concat(results, ignore_index=True)
            dropped_count = len(combined_results)
            entity_desc = "CpGs" if self.data_type == "Methylation" else "SNPs"
            combined_results.to_csv(self.output_file, header=False, index=False)
            log.success(
                f"{entity_desc} call rates calculated successfully and written to: {self.output_file}"
            )
            log.success(
                f"The number of {entity_desc} below the threshold: {dropped_count}"
            )
            return self.output_file
        except Exception as e:
            log.error(f"Error in marker call rate calculation: {e}")
            raise RuntimeError(f"Marker call rate calculation failed: {e}") from e

    def run_sample_call_rate(self) -> str:
        try:
            log.info(
                f"Starting sample call rate calculation. Data type: {self.data_type}"
            )
            chromosome_list: List[str] = []
            sample_list: List[str] = []
            with h5py.File(self.input_file, "r") as h5_file:
                try:
                    chromosome_list = [
                        k
                        for k in h5_file.keys()
                        if k != AliasUtils.find_keys(h5_file, "Metadata")
                    ]
                except Exception as e:
                    log.error(f"Error getting chromosome list: {e}")
                    chromosome_list = []
                if not chromosome_list:
                    raise ValueError("No chromosomes found in input file.")
                metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
                if metadata_key is None:
                    raise ValueError("Could not find Metadata group in input file.")
                sample_key = None
                if self.data_type == "Methylation":
                    sample_key = AliasUtils.find_keys(
                        h5_file[metadata_key], "SampleList"
                    )
                else:
                    sample_key = AliasUtils.find_keys(h5_file[metadata_key], "IID")
                if sample_key is None:
                    raise ValueError("Could not find sample list in Metadata group.")
                log.debug(f"Reading samples from /{metadata_key}/{sample_key}")
                sample_list = h5_file[metadata_key][sample_key][:]
                if isinstance(sample_list[0], bytes):
                    sample_list = [s.decode("utf-8") for s in sample_list]
                log.debug(f"Got {len(sample_list)} samples")
            result = pd.DataFrame(
                {
                    "Sample": sample_list,
                    "non_missing": np.zeros(len(sample_list), dtype=np.int64),
                    "num_markers": np.zeros(len(sample_list), dtype=np.int64),
                }
            )
            log.info(f"Total samples to process: {len(sample_list)}")
            log.info(f"Using {self.max_workers} cores for parallel processing.")
            print("Calculating call rates...")
            chr_results: List[Dict[str, Any]] = []
            log_queue = log.get_queue()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=log.child_init if log_queue else None,
                initargs=(log_queue,) if log_queue else (),
            ) as executor:
                future_to_chr = {
                    executor.submit(
                        self.process_chromosome_sample_call_rate, chromosome
                    ): chromosome
                    for chromosome in chromosome_list
                }
                for future in concurrent.futures.as_completed(future_to_chr):
                    res = future.result()
                    if res is not None:
                        chr_results.append(res)
            log.info("Aggregating results from all chromosomes.")
            sample_to_idx = {s: i for i, s in enumerate(sample_list)}
            for res in chr_results:
                if "sample_names" in res:
                    for i, sample in enumerate(res["sample_names"]):
                        if sample in sample_to_idx:
                            idx = sample_to_idx[sample]
                            result.at[idx, "non_missing"] += res["non_missing"][i]
                            result.at[idx, "num_markers"] += res["num_markers"][i]
                else:
                    if len(res["non_missing"]) == len(result):
                        result["non_missing"] += res["non_missing"]
                        result["num_markers"] += res["num_markers"]
                    else:
                        raise ValueError(
                            "Sample count mismatch and no sample_names in result."
                        )
            log.info("Calculating call rates for each sample.")
            result["call_rate"] = result["non_missing"] / result["num_markers"]
            log.info(
                f"Filtering samples with call rates below the threshold: {self.threshold}"
            )
            filtered_result = result[result["call_rate"] < self.threshold][["Sample"]]
            total_samples = len(sample_list)
            filtered_count = len(filtered_result)
            remaining_count = total_samples - filtered_count
            log.info("Sample call rate summary:")
            log.info(f"  Total samples processed: {total_samples}")
            log.info(f"  Samples below threshold ({self.threshold}): {filtered_count}")
            log.info(f"  Samples passing QC: {remaining_count}")

            filtered_result.to_csv(self.output_file, header=False, index=False)
            log.success(
                f"Sample call rates calculated successfully and written to: {self.output_file}"
            )
            log.success(f"The number of samples below the threshold: {filtered_count}")
            return self.output_file
        except Exception as e:
            log.error(f"Error in sample call rate calculation: {e}")
            raise RuntimeError(f"Sample call rate calculation failed: {e}") from e

    def run_probe_variance(self) -> str:
        try:
            log.info("Starting probe variance calculation.")
            with h5py.File(self.input_file, "r") as h5_file:
                try:
                    with CachedH5Utils(h5_file) as h5_utils:
                        chromosome_list = h5_utils.get_chromosomes()
                except Exception:
                    chromosome_list = None
                if chromosome_list is None:
                    metadata_aliases = AliasUtils.get_aliases("Metadata")
                    chromosome_list = [
                        key for key in h5_file.keys() if key not in metadata_aliases
                    ]
                    if not chromosome_list:
                        raise ValueError("Failed to get chromosome list")
                    log.info(
                        f"Found {len(chromosome_list)} chromosomes using direct detection"
                    )
            log.info(f"Using {self.max_workers} cores for parallel processing.")
            log.info("Calculating probe variances...")
            variances_list: List[pd.DataFrame] = []
            log_queue = log.get_queue()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=log.child_init if log_queue else None,
                initargs=(log_queue,) if log_queue else (),
            ) as executor:
                futures = [
                    executor.submit(self.process_chromosome_probe_variance, chromosome)
                    for chromosome in chromosome_list
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Processing chromosomes",
                ):
                    result = future.result()
                    if result is not None:
                        variances_list.append(result)
            log.info("Variance calculation completed for all chromosomes.")
            if not variances_list:
                log.warn("No valid variance data found")
                open(self.output_file, "w").close()
                log.success("Empty output file created")
                return self.output_file
            combined_results = pd.concat(variances_list, ignore_index=True)
            del variances_list
            gc.collect()
            combined_results = combined_results.dropna(subset=["variance"])
            log.info(f"Combined results contain {len(combined_results)} probes.")
            variance_threshold = combined_results["variance"].quantile(self.threshold)
            log.info(f"Variance threshold calculated: {variance_threshold}")
            filtered_results = combined_results[
                combined_results["variance"] < variance_threshold
            ]
            log.info(
                f"Found {len(filtered_results)} probes below the variance threshold."
            )
            filtered_results = filtered_results[["probe_id"]]
            del combined_results
            gc.collect()
            if len(filtered_results) > 0:
                log.info(
                    f"Writing {len(filtered_results)} probes to {self.output_file}"
                )
                filtered_results.to_csv(self.output_file, header=False, index=False)
            else:
                log.info(
                    "No probes below the threshold. Creating an empty output file."
                )
                open(self.output_file, "w").close()
            log.success(f"CpG variances calculated and saved to {self.output_file}")

            log.success(
                f"The number of CpGs below the threshold: {len(filtered_results)}"
            )
            return self.output_file
        except Exception as e:
            log.error(f"Error in probe variance calculation: {e}")
            raise RuntimeError(f"Probe variance calculation failed: {e}") from e

    def run(self) -> str:
        if not self._check_system_health():
            log.error("System health check failed - QC may be unstable")
            user_input = input("Continue anyway? (y/N): ")
            if user_input.lower() != "y":
                log.info("QC cancelled by user")
                raise RuntimeError("QC cancelled due to system health check failure")

        with monitor_resources(interval=2.0) as stats:
            try:
                if self.metric == "marker_call_rate":
                    result = self.run_marker_call_rate()
                elif self.metric == "sample_call_rate":
                    result = self.run_sample_call_rate()
                elif self.metric == "probe_variance":
                    result = self.run_probe_variance()
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")

                log.info(
                    f"QC completed. Peak resource usage - CPU: {stats['max_cpu']:.1f}%, "
                    f"Memory: {stats['max_memory']:.1f}%"
                )
                return result

            except Exception as e:
                log.error("QC failed. Peak resource usage before failure:")
                log.error(
                    f"CPU: {stats['max_cpu']:.1f}%, Memory: {stats['max_memory']:.1f}%"
                )
                raise e


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(
        flags=["-m", "--metric"],
        type=str,
        default="marker_call_rate",
        required=True,
        choices=["marker_call_rate", "sample_call_rate", "probe_variance"],
    ),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
    OptionConfig(flags=["-t", "--threshold"], type=float, default=None, required=True),
    OptionConfig(
        flags=["-d", "--data_type"],
        type=str,
        default=None,
        required=False,
        choices=["Methylation", "Genotype"],
    ),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="CountsQC")
    opt = framework.run()
    calculator = CountsQC(
        input_file=opt.input,
        metric=opt.metric,
        output_file=opt.output,
        threshold=opt.threshold,
        data_type=opt.data_type,
    )
    calculator.run()
