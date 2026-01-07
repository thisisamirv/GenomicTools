#!/usr/bin/env python
# Import required modules
import concurrent.futures
import gc
import h5py
import numpy as np
import os
import pandas as pd
import psutil
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.H5Utils import CachedH5Utils
from utils.LoggingUtils import log
from utils.SystemUtils import SystemUtils, monitor_resources


class ProcessHDF5:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        operation: str = "subset",
        samples: Optional[Union[str, List[str]]] = None,
        markers: Optional[Union[str, List[str]]] = None,
        chromosomes: Optional[Union[str, List[str]]] = None,
        data_type: Optional[str] = None,
        chunk_size: int = 30000,
        names: Optional[Union[str, List[str]]] = None,
        metadata: Optional[str] = None,
    ) -> None:
        self.input_file = input_file
        if not os.path.isfile(self.input_file):
            raise FileNotFoundError(f"Input file '{self.input_file}' not found")
        if not output_file:
            raise ValueError("output_file is required")
        self.output_file = output_file
        self.operation = (operation or "subset").lower()
        self.data_type = data_type
        self.names_type = names
        self.metadata_file = metadata
        self.temp_dirs = []
        self.system_config = SystemUtils.load_config()
        safe_config = SystemUtils.configure_safe_environment()
        if safe_config.get("core_dumps_disabled", False):
            log.debug("Core dumps disabled for stability")
        if safe_config.get("memory_limit_set", False):
            log.debug("Memory limits configured to prevent OOM errors")

        valid_operations = [
            "subset",
            "remove",
            "read",
            "names",
            "add_metadata",
            "extract_metadata",
        ]
        if self.operation not in valid_operations:
            raise ValueError(
                f"Invalid operation '{operation}'. Must be one of: {valid_operations}"
            )

        if self.data_type:
            self.data_type = AliasUtils.get_field(self.data_type)

        SystemUtils.print_system_info()

        self.max_workers = SystemUtils.get_optimal_cores(reserve_cores=1)
        self.chunk_size = chunk_size
        self._adjust_resources()

        dynamic_chunk_size = self._calculate_optimal_chunk_size()
        if dynamic_chunk_size != self.chunk_size:
            log.info(
                f"Using dynamically calculated chunk size {dynamic_chunk_size} instead of {self.chunk_size}"
            )
            self.chunk_size = dynamic_chunk_size

        self._check_memory_usage("Initialization")

        if self.operation not in ["names", "add_metadata", "extract_metadata"]:
            self.sample_ids, self.sample_file = (
                self._process_input(samples, "sample") if samples else (None, None)
            )
            self.marker_ids, self.marker_file = (
                self._process_input(markers, "marker") if markers else (None, None)
            )
            self.chromosomes = (
                self._process_chromosomes(chromosomes) if chromosomes else None
            )
        else:
            self.sample_ids = self.marker_ids = self.chromosomes = None
            self.sample_file = self.marker_file = None

        if self.operation == "add_metadata" and self.names_type:
            if isinstance(self.names_type, str):
                self.columns_to_add = [
                    col.strip() for col in self.names_type.split(",")
                ]
            else:
                self.columns_to_add = self.names_type
        else:
            self.columns_to_add = None

        if self.operation == "extract_metadata":
            if self.names_type:
                if isinstance(self.names_type, str):
                    self.columns_to_extract = [
                        col.strip() for col in self.names_type.split(",")
                    ]
                else:
                    self.columns_to_extract = self.names_type
            else:
                raise ValueError("names required for extract_metadata")

        if self.operation in ["subset", "remove", "add_metadata"] and self.output_file:
            self.temp_dir = self._setup_temp_directory()
        else:
            self.temp_dir = None

        self.sample_indices: Optional[List[int]] = None
        self.markers_dict: Dict[str, List[str]] = {}

        if self.data_type is None:
            self._detect_data_type()

    def _check_system_health(self) -> bool:
        log.info("Checking system health before processing...")

        required_gb = self._estimate_output_size()

        check_paths = []
        if self.output_file and self.operation in ["subset", "remove", "add_metadata"]:
            output_dir = os.path.dirname(os.path.abspath(self.output_file))
            check_paths.append(output_dir)

        try:
            health = SystemUtils.check_system_health(
                min_free_disk_gb=required_gb,
                max_cpu_percent=95.0,
                max_memory_percent=90.0,
                check_paths=check_paths if check_paths else None,
            )
        except Exception as e:
            log.warn(f"Failed to perform system health check: {e}")
            return True

        if health.get("status") == "critical":
            log.warn("CRITICAL SYSTEM ISSUES:")
            for issue in health.get("critical", []):
                log.warn(f"  - {issue}")
            return False

        if health.get("status") == "warning":
            log.warn("SYSTEM WARNINGS:")
            for issue in health.get("warnings", []):
                log.warn(f"  - {issue}")

        log.info("System health check passed")
        return True

    def _estimate_output_size(self) -> float:
        try:
            input_size_gb = os.path.getsize(self.input_file) / (1024**3)

            if self.operation == "subset":
                size_multiplier = 1.0
                if self.sample_ids and self.marker_ids:
                    with h5py.File(self.input_file, "r") as h5_file:
                        metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
                        if metadata_key:
                            sample_key = (
                                AliasUtils.find_keys(
                                    h5_file[metadata_key], "SampleList"
                                )
                                if self.data_type == "Methylation"
                                else AliasUtils.find_keys(h5_file[metadata_key], "IID")
                            )
                            if sample_key:
                                sample_path = f"/{metadata_key}/{sample_key}"
                                total_samples = len(h5_file[sample_path])
                                if total_samples > 0:
                                    sample_ratio = len(self.sample_ids) / total_samples
                                    size_multiplier = sample_ratio
                elif self.sample_ids:
                    with h5py.File(self.input_file, "r") as h5_file:
                        metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
                        if metadata_key:
                            sample_key = (
                                AliasUtils.find_keys(
                                    h5_file[metadata_key], "SampleList"
                                )
                                if self.data_type == "Methylation"
                                else AliasUtils.find_keys(h5_file[metadata_key], "IID")
                            )
                            if sample_key:
                                sample_path = f"/{metadata_key}/{sample_key}"
                                total_samples = len(h5_file[sample_path])
                                if total_samples > 0:
                                    size_multiplier = (
                                        len(self.sample_ids) / total_samples
                                    )

                elif self.chromosomes:
                    all_chroms = self._get_all_chromosomes()
                    if all_chroms:
                        size_multiplier = len(self.chromosomes) / len(all_chroms)

                return max(
                    0.1, min(input_size_gb * size_multiplier * 1.2, input_size_gb)
                )

            elif self.operation == "remove":
                size_multiplier = 0.9
                if self.sample_ids or self.marker_ids or self.chromosomes:
                    size_multiplier = 0.8
                return input_size_gb * size_multiplier

            elif self.operation == "read":
                return input_size_gb * 1.2

            elif self.operation == "names":
                return 0.1

            elif self.operation in ["add_metadata", "extract_metadata"]:
                return input_size_gb * 1.05

            return input_size_gb

        except Exception as e:
            log.debug(f"Error estimating output size: {e}")
            return max(1.0, input_size_gb if "input_size_gb" in locals() else 1.0)

    def _setup_temp_directory(self) -> str:
        if not self.output_file:
            default_path = os.getcwd()
        else:
            default_path = os.path.dirname(os.path.abspath(self.output_file))

        required_gb = self._estimate_output_size() * 0.5

        try:
            temp_dir, temp_info = SystemUtils.create_safe_tempdir(
                default_path=default_path,
                required_gb=required_gb,
                prefix=f"process_hdf5_{self.operation}",
                buffer_percent=10.0,
            )
            log.info(f"Created temporary directory: {temp_dir}")
            try:
                log.debug(
                    f"Temp directory has {temp_info['disk_info']['free_gb']:.1f}GB free"
                )
            except Exception:
                log.debug("Temp directory info not available")
            return temp_dir
        except Exception as e:
            log.warn(f"Failed to create safe temp directory: {e}")
            log.warn("Using system temp directory as fallback")

            temp_dir = tempfile.mkdtemp(prefix=f"process_hdf5_{self.operation}")
            log.info(f"Created fallback temporary directory: {temp_dir}")
            return temp_dir

    def _calculate_optimal_chunk_size(self) -> int:
        try:
            memory_info = SystemUtils.get_memory_info()
            available_memory_gb = memory_info.get("available_gb", 8.0)
            max_memory_gb = self.system_config.get("max_memory_gb", 1024)
            available_memory_gb = min(available_memory_gb, max_memory_gb)

            memory_based_chunk = int((available_memory_gb * (1024**3) * 0.1) / 8)

            file_size_gb = 1.0
            try:
                file_size_gb = os.path.getsize(self.input_file) / (1024**3)
            except (OSError, AttributeError):
                file_size_gb = 1.0

            if self.operation in ("subset", "remove"):
                if file_size_gb > 50:
                    memory_based_chunk = int(memory_based_chunk * 0.7)
            elif self.operation == "read":
                memory_based_chunk = int(memory_based_chunk * 0.8)

            min_chunk = 5000 if available_memory_gb > 16 else 2000
            max_chunk = 100000 if available_memory_gb > 64 else 50000

            optimal_chunk = max(min_chunk, min(memory_based_chunk, max_chunk))
            log.info(
                f"Calculated optimal chunk size: {optimal_chunk} based on {available_memory_gb:.1f}GB available memory"
            )

            return optimal_chunk

        except Exception as e:
            log.warning(f"Error calculating optimal chunk size: {e}")
            return self.chunk_size

    def _cleanup(self) -> None:
        if hasattr(self, "temp_dir") and self.temp_dir is not None:
            if os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                    log.debug(f"Cleaned up temporary directory: {self.temp_dir}")
                except Exception as e:
                    log.warn(f"Error cleaning up temp directory {self.temp_dir}: {e}")

        if hasattr(self, "temp_dirs") and self.temp_dirs:
            for temp_dir in self.temp_dirs:
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        log.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    log.warn(f"Error cleaning up temp directory {temp_dir}: {e}")

        try:
            cleanup_result = SystemUtils.cleanup_stale_temp_files(
                prefix=f"process_hdf5_{self.operation}",
                max_age_hours=24,
                dry_run=False,
            )

            dirs_deleted = cleanup_result.get("dirs_deleted", 0)
            files_deleted = cleanup_result.get("files_deleted", 0)
            if dirs_deleted > 0 or files_deleted > 0:
                log.debug(
                    f"Cleaned up {dirs_deleted} stale directories and "
                    f"{files_deleted} files"
                )
        except Exception as e:
            log.debug(f"Error during stale file cleanup: {e}")

        gc.collect()

    def _adjust_resources(self) -> None:
        try:
            memory_info = SystemUtils.get_memory_info()
            available_memory_gb = memory_info.get("available_gb", 8.0)
            file_size_gb = os.path.getsize(self.input_file) / (1024**3)
            if available_memory_gb < 8:
                self.chunk_size = min(self.chunk_size, 5000)
            elif available_memory_gb > 64:
                self.chunk_size = max(self.chunk_size, 50000)
            with h5py.File(self.input_file, "r") as h5_file:
                chrom_keys = []
                for key in h5_file.keys():
                    base_key = AliasUtils.strip_numeric_suffix(key)
                    if AliasUtils.get_field(base_key) == "CHR":
                        chrom_keys.append(key)
                if chrom_keys:
                    first_chr = chrom_keys[0]
                    data_key = AliasUtils.find_keys(
                        h5_file[first_chr], "Methylation"
                    ) or AliasUtils.find_keys(h5_file[first_chr], "Genotype")
                    if data_key:
                        data_path = f"{first_chr}/{data_key}"
                        if data_path in h5_file:
                            dataset = h5_file[data_path]
                            estimated_memory_per_chr_gb = (
                                (dataset.size * dataset.dtype.itemsize) / (1024**3) * 3
                            )
                            if estimated_memory_per_chr_gb > 0:
                                mem_to_use = available_memory_gb * 0.7
                                mem_max = int(mem_to_use / estimated_memory_per_chr_gb)
                                memory_based_workers = max(1, mem_max)
                                self.max_workers = min(
                                    self.max_workers, memory_based_workers
                                )
                    if file_size_gb > 50:
                        self.max_workers = min(self.max_workers, 6)
            valid_resources, message = SystemUtils.validate_resources(
                cores=self.max_workers, memory_gb=available_memory_gb * 0.8
            )

            if not valid_resources:
                log.warn(f"Resource validation issue: {message}")
                log.warn("Processing performance may be affected")

            log.info(
                f"Resources adjusted: {self.max_workers} workers, chunk size {self.chunk_size}"
            )
        except Exception as e:
            log.warn(f"Could not adjust resources: {e}")

    def _check_memory_usage(self, operation_name: str = "") -> float:
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024**3)
            available_gb = psutil.virtual_memory().available / (1024**3)
            log.debug(
                f"{operation_name} - Memory: {memory_gb:.1f}GB used, {available_gb:.1f}GB available"
            )
            if memory_gb > available_gb * 0.5:
                log.warn(f"High memory usage: {memory_gb:.1f}GB")
            return memory_gb
        except Exception as e:
            log.debug(f"Could not check memory: {e}")
            return 0.0

    def _decode_array(self, arr: Union[np.ndarray, List[Any]]) -> List[str]:
        if len(arr) == 0:
            return []
        first = arr[0]
        # Handle bytes - detect encoding once using first element
        if isinstance(first, bytes):
            # Try encodings on first element only
            detected_encoding = None
            for encoding in ("utf-8", "latin-1", "ascii"):
                try:
                    first.decode(encoding)
                    detected_encoding = encoding
                    break
                except (UnicodeDecodeError, AttributeError):
                    continue
            if detected_encoding:
                try:
                    # Strip null bytes and whitespace (common in fixed-width HDF5 strings)
                    return [x.decode(detected_encoding).rstrip('\x00').strip() for x in arr]
                except (UnicodeDecodeError, AttributeError):
                    pass
            # Fallback to str() if nothing works
            return [str(x).rstrip('\x00').strip() for x in arr]
        # Handle numpy bytes/string types (numpy.bytes_)
        if hasattr(first, "decode"):
            try:
                return [x.decode("utf-8").rstrip('\x00').strip() for x in arr]
            except (UnicodeDecodeError, AttributeError):
                return [str(x).rstrip('\x00').strip() for x in arr]
        # Handle regular strings or other types
        return [str(x).strip() for x in arr if not pd.isna(x)]

    def _normalize_sample_ids(self, sample_list: List[Any]) -> List[str]:
        normalized: List[str] = []
        for item in sample_list:
            try:
                float_val = float(item)
                if float_val.is_integer():
                    normalized.append(str(int(float_val)))
                else:
                    normalized.append(str(item))
            except (ValueError, TypeError):
                normalized.append(str(item))
        return normalized

    def _process_data_in_chunks(
        self,
        h5_file: h5py.File,
        chromosome: str,
        sample_indices: Optional[List[int]] = None,
        sample_names: Optional[List[str]] = None,
    ) -> Optional[List[pd.DataFrame]]:
        if chromosome not in h5_file:
            return None
        data_key = (
            AliasUtils.find_keys(h5_file[chromosome], "Methylation")
            if self.data_type == "Methylation"
            else AliasUtils.find_keys(h5_file[chromosome], "Genotype")
        )
        marker_key = (
            AliasUtils.find_keys(h5_file[chromosome], "ProbeList")
            if self.data_type == "Methylation"
            else AliasUtils.find_keys(h5_file[chromosome], "RSID")
        )
        if not data_key or not marker_key:
            return None
        data_path = f"/{chromosome}/{data_key}"
        marker_path = f"/{chromosome}/{marker_key}"
        if data_path not in h5_file or marker_path not in h5_file:
            return None
        dataset = h5_file[data_path]
        markers_dataset = h5_file[marker_path]
        total_markers = len(markers_dataset)
        processed_chunks: List[pd.DataFrame] = []
        sample_indices = sample_indices or list(
            range(dataset.shape[1 - (self.data_type == "Methylation")])
        )
        sample_names = sample_names or [
            f"sample_{i}" for i in range(len(sample_indices))
        ]
        for start_idx in range(0, total_markers, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_markers)
            chunk_markers = self._decode_array(markers_dataset[start_idx:end_idx])
            if self.marker_ids:
                marker_mask = [m in self.marker_ids for m in chunk_markers]
                if not any(marker_mask):
                    continue
                local_marker_indices = [
                    i for i, match in enumerate(marker_mask) if match
                ]
                filtered_markers = [chunk_markers[i] for i in local_marker_indices]
            else:
                local_marker_indices = list(range(len(chunk_markers)))
                filtered_markers = chunk_markers
            if not filtered_markers:
                continue
            if self.data_type == "Methylation":
                data_chunk = dataset[sample_indices, start_idx:end_idx][
                    :, local_marker_indices
                ]
            else:
                data_chunk = dataset[start_idx:end_idx, sample_indices][
                    local_marker_indices, :
                ]
            filtered_chunk = self._filter_chunk_data(
                data_chunk, filtered_markers, chromosome, sample_names
            )
            if filtered_chunk is not None and len(filtered_chunk) > 0:
                processed_chunks.append(filtered_chunk)
            del data_chunk
            gc.collect()
            self._check_memory_usage(f"Chunk {start_idx}-{end_idx}")
        return processed_chunks

    def _filter_chunk_data(
        self,
        chunk_data: Union[np.ndarray, List[List[Any]]],
        chunk_markers: List[str],
        chromosome: str,
        sample_names: List[str],
    ) -> Optional[pd.DataFrame]:
        try:
            if self.data_type == "Methylation":
                df = pd.DataFrame(
                    chunk_data.T, index=chunk_markers, columns=sample_names
                )
                df.reset_index(inplace=True)
                df.rename(columns={"index": "CGID"}, inplace=True)
            else:
                df = pd.DataFrame(chunk_data, index=chunk_markers, columns=sample_names)
                df.reset_index(inplace=True)
                df.rename(columns={"index": "RSID"}, inplace=True)
            chrom_num = re.sub(r"^(chr|CHR|Chr)", "", chromosome)
            df["CHR"] = chrom_num
            return df
        except Exception as e:
            log.error(f"Error filtering chunk: {e}")
            return None

    def _process_input(
        self, input_data: Optional[Union[str, List[Any]]], data_type: str
    ) -> Tuple[List[str], Optional[str]]:
        if isinstance(input_data, str) and os.path.isfile(input_data):
            with open(input_data, "r") as f:
                ids_list = [line.strip() for line in f if line.strip()]
        else:
            if isinstance(input_data, str):
                ids_list = (
                    [s.strip() for s in input_data.split(",")]
                    if "," in input_data
                    else [input_data]
                )
            else:
                ids_list = list(input_data) if input_data is not None else []
        if data_type == "sample":
            ids_list = self._normalize_sample_ids(ids_list)
        return ids_list, (
            input_data
            if isinstance(input_data, str) and os.path.isfile(input_data)
            else None
        )

    def _process_chromosomes(self, chromosomes: Union[str, List[Any]]) -> List[str]:
        if isinstance(chromosomes, str):
            chr_list = (
                [s.strip() for s in chromosomes.split(",")]
                if "," in chromosomes
                else [chromosomes.strip()]
            )
        else:
            chr_list = [str(c).strip() for c in chromosomes]

        return chr_list

    def _map_requested_chromosomes(
        self, requested: Optional[List[Any]], available: List[str]
    ) -> Tuple[List[str], List[str]]:
        if not requested:
            return [], []
        normalized_map: Dict[str, str] = {}
        for chrom in available:
            lower = chrom.lower()
            stripped = re.sub(r"^(chr|CHR|Chr)", "", chrom).lower()
            normalized_map.setdefault(lower, chrom)
            normalized_map.setdefault(stripped, chrom)
            normalized_map.setdefault(f"chr{stripped}", chrom)
        mapped: List[str] = []
        missing: List[str] = []
        for item in requested:
            if item is None:
                continue
            item_str = str(item)
            candidates = [
                item_str,
                item_str.lower(),
                re.sub(r"^(chr|CHR|Chr)", "", item_str).lower(),
            ]
            candidates.append(f"chr{candidates[2]}")
            match = None
            for candidate in candidates:
                if candidate in normalized_map:
                    match = normalized_map[candidate]
                    break
            if match:
                mapped.append(match)
            else:
                missing.append(item_str)
        return mapped, missing

    def _detect_data_type(self) -> None:
        with h5py.File(self.input_file, "r") as h5_file:
            chromosomes: List[str] = []
            for key in h5_file.keys():
                base_key = AliasUtils.strip_numeric_suffix(key)
                if AliasUtils.get_field(base_key) == "CHR":
                    chromosomes.append(key)
            if not chromosomes:
                raise ValueError("No chromosome groups found")
            first_chr = chromosomes[0]
            if AliasUtils.find_keys(
                h5_file[first_chr], "ProbeList"
            ) or AliasUtils.find_keys(h5_file[first_chr], "Methylation"):
                self.data_type = "Methylation"
            elif AliasUtils.find_keys(
                h5_file[first_chr], "RSID"
            ) or AliasUtils.find_keys(h5_file[first_chr], "Genotype"):
                self.data_type = "Genotype"
            else:
                raise ValueError("Unknown HDF5 format")
            log.info(f"Detected {self.data_type} data")

    def _get_data_paths(
        self, chromosome: str, h5_file: h5py.File
    ) -> Dict[str, Union[str, List[str]]]:
        if self.data_type == "Methylation":
            methylation_key = AliasUtils.find_keys(h5_file[chromosome], "Methylation")
            cgid_key = AliasUtils.find_keys(h5_file[chromosome], "ProbeList")
            metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
            sample_key = AliasUtils.find_keys(h5_file[metadata_key], "SampleList")
            return {
                "Methylation": f"/{chromosome}/{methylation_key}",
                "CGID": f"/{chromosome}/{cgid_key}",
                "SampleList": f"/{metadata_key}/{sample_key}",
            }
        else:
            genotype_key = AliasUtils.find_keys(h5_file[chromosome], "Genotype")
            rsid_key = AliasUtils.find_keys(h5_file[chromosome], "RSID")
            metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
            sample_key = AliasUtils.find_keys(h5_file[metadata_key], "IID")
            additional_keys = ["A1", "A2", "BP"]
            return {
                "Genotype": f"/{chromosome}/{genotype_key}",
                "RSID": f"/{chromosome}/{rsid_key}",
                "SampleList": f"/{metadata_key}/{sample_key}",
                "Additional": additional_keys,
            }

    def _get_sample_info(self, h5_file: h5py.File) -> Tuple[List[str], List[str]]:
        metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
        if not metadata_key:
            raise ValueError("No metadata group found")
        if self.data_type == "Methylation":
            sample_key = AliasUtils.find_keys(h5_file[metadata_key], "SampleList")
        else:
            sample_key = AliasUtils.find_keys(h5_file[metadata_key], "IID")
        if not sample_key:
            raise ValueError(
                f"Could not find sample list key in metadata for {self.data_type}"
            )
        sample_path = f"/{metadata_key}/{sample_key}"
        if sample_path not in h5_file:
            raise ValueError(f"Sample list not found at {sample_path}")
        full_sample_list = self._decode_array(h5_file[sample_path][:])
        full_sample_list = self._normalize_sample_ids(full_sample_list)
        if not self.sample_ids:
            self.sample_indices = None
            log.info(f"Using all {len(full_sample_list)} samples")
            return full_sample_list, full_sample_list
        requested = set(str(s) for s in self.sample_ids)
        sample_indices: List[int] = []
        found_samples: List[str] = []
        for idx, sample in enumerate(full_sample_list):
            if str(sample) in requested:
                sample_indices.append(idx)
                found_samples.append(sample)
        if not sample_indices:
            log.debug("Sample matching failed. Details:")
            log.debug(f"Total samples in HDF5: {len(full_sample_list)}")
            log.debug(f"Total requested samples: {len(self.sample_ids)}")
            sample_preview = (
                full_sample_list[:10]
                if len(full_sample_list) > 10
                else full_sample_list
            )
            log.debug(f"First {len(sample_preview)} samples in HDF5: {sample_preview}")
            requested_preview = (
                list(requested)[:10] if len(requested) > 10 else list(requested)
            )
            log.debug(
                f"First {len(requested_preview)} requested samples: {requested_preview}"
            )
            if full_sample_list:
                h5_sample_types = set(type(s).__name__ for s in full_sample_list[:5])
                requested_sample_types = set(
                    type(s).__name__ for s in list(requested)[:5]
                )
                log.debug(f"HDF5 sample types: {h5_sample_types}")
                log.debug(f"Requested sample types: {requested_sample_types}")
            raise ValueError("No requested samples found in dataset")
        self.sample_indices = sample_indices
        log.info(
            f"Found {len(found_samples)} of {len(self.sample_ids)} requested samples"
        )
        return full_sample_list, found_samples

    def _get_marker_info(self, h5_file: h5py.File) -> Dict[str, List[str]]:
        if not self.marker_ids:
            return {}
        
        entity_type = "probes" if self.data_type == "Methylation" else "SNPs"
        h5_utils = CachedH5Utils(h5_file)
        all_chroms = h5_utils.get_chromosomes()
        
        mapped_chroms, missing_req = self._map_requested_chromosomes(
            self.chromosomes, all_chroms
        )
        
        # If user didn't specify chromosomes, scan all available ones
        chr_list = mapped_chroms if (self.chromosomes and mapped_chroms) else all_chroms
        
        if not chr_list:
            log.warn("No valid chromosomes found for marker scanning")
            return {}

        # Use the first available chromosome to find the marker dataset key
        sample_chr = chr_list[0]
        marker_alias = "ProbeList" if self.data_type == "Methylation" else "RSID"
        marker_key = AliasUtils.find_keys(h5_file[sample_chr], marker_alias)
        
        if not marker_key:
            log.warn(f"Could not find marker key '{marker_alias}' in {sample_chr}")
            return {}
            
        marker_suffix = f"/{marker_key}"
        markers_dict: Dict[str, List[str]] = {}
        
        # Normalize requested marker IDs: strip whitespace and convert to string
        requested_markers = set(str(m).strip() for m in self.marker_ids)
        remaining_markers = requested_markers.copy()
        
        log.info(f"Scanning {len(chr_list)} chromosomes for {len(requested_markers)} {entity_type}")
        
        for chr in tqdm(chr_list, desc="Scanning chromosomes"):
            if not remaining_markers: # Early exit if all markers found
                break
                
            marker_path = f"/{chr}{marker_suffix}"
            if marker_path not in h5_file:
                continue
                
            # Decode array and ensure whitespace/null bytes are stripped
            chr_markers = self._decode_array(h5_file[marker_path][:])
            chr_set = set(m.strip() for m in chr_markers)
            
            # Use set intersection for O(min(len(remaining_markers), len(chr_set)))
            found_in_chr = remaining_markers.intersection(chr_set)
            if found_in_chr:
                markers_dict[chr] = list(found_in_chr)
                remaining_markers.difference_update(found_in_chr)
                
        if remaining_markers:
            log.warn(f"Missing {len(remaining_markers)} {entity_type}")
            
            # DIAGNOSTICS: Help identify why markers are missing
            missing_sample = list(remaining_markers)[:5]
            log.warn(f"Sample missing {entity_type}: {missing_sample}")
            
            # Check a sample chromosome to see what the markers actually look like
            sample_chr_path = f"/{sample_chr}{marker_suffix}"
            if sample_chr_path in h5_file:
                actual_sample = self._decode_array(h5_file[sample_chr_path][:5])
                log.warn(f"Markers in {sample_chr} look like: {actual_sample}")
                
                # Check for case-insensitive matches
                lower_remaining = set(m.lower() for m in remaining_markers)
                chr_markers_full = self._decode_array(h5_file[sample_chr_path][:])
                chr_lower = set(m.lower().strip() for m in chr_markers_full)
                
                case_matches = lower_remaining.intersection(chr_lower)
                if case_matches:
                    log.warn(f"Found {len(case_matches)} case-insensitive matches in {sample_chr}. Data may be case-sensitive.")
                    log.warn(f"Sample case matches: {list(case_matches)[:3]}")

        return markers_dict

    def _get_indices_for_chromosome(
        self, h5_file: h5py.File, chromosome: str
    ) -> Tuple[List[int], Optional[List[int]], List[str]]:
        marker_key = (
            AliasUtils.find_keys(h5_file[chromosome], "ProbeList")
            if self.data_type == "Methylation"
            else AliasUtils.find_keys(h5_file[chromosome], "RSID")
        )
        if self.data_type == "Methylation":
            marker_path = f"/{chromosome}/{marker_key}"
        else:
            marker_path = f"/{chromosome}/{marker_key}"
        if marker_path not in h5_file:
            raise ValueError(
                f"Marker path {marker_path} not found in chromosome {chromosome}"
            )
        marker_list = self._decode_array(h5_file[marker_path][:])
        if self.marker_ids:
            chr_markers = self.markers_dict.get(chromosome, [])
            if not chr_markers:
                return [], self.sample_indices, marker_list
            # Build index lookup dict once: O(n), then lookups are O(1) each
            marker_to_idx = {m: i for i, m in enumerate(marker_list)}
            marker_indices = [
                marker_to_idx[str(marker)]
                for marker in chr_markers
                if str(marker) in marker_to_idx
            ]
        else:
            if self.operation == "remove":
                marker_indices = []
            else:
                marker_indices = list(range(len(marker_list)))
        return marker_indices, self.sample_indices, marker_list

    def process_chromosome_hdf5(self, chromosome: str) -> Tuple[str, Optional[str]]:
        temp_filename = f"{chromosome}_{int(time.time() * 1000)}.h5"
        temp_path = os.path.join(self.temp_dir, temp_filename)

        with monitor_resources(interval=2.0) as stats:
            try:
                with h5py.File(self.input_file, "r") as h5_file, h5py.File(
                    temp_path, "w"
                ) as temp_file:
                    temp_file.create_group(chromosome)
                    affected_marker_idx, affected_sample_idx, marker_list = (
                        self._get_indices_for_chromosome(h5_file, chromosome)
                    )
                    affected_sample_idx = (
                        affected_sample_idx if affected_sample_idx is not None else []
                    )
                    paths = self._get_data_paths(chromosome, h5_file)
                    data_key = (
                        AliasUtils.find_keys(h5_file[chromosome], "Methylation")
                        if self.data_type == "Methylation"
                        else AliasUtils.find_keys(h5_file[chromosome], "Genotype")
                    )
                    data_dataset = h5_file[
                        paths[
                            (
                                data_key
                                if data_key in paths
                                else (
                                    "Methylation"
                                    if self.data_type == "Methylation"
                                    else "Genotype"
                                )
                            )
                        ]
                    ]
                    data_shape = data_dataset.shape
                    marker_path_key = (
                        "CGID" if self.data_type == "Methylation" else "RSID"
                    )
                    # Deduplicated list of markers for faster membership tests
                    all_marker_idx = range(
                        data_shape[0]
                        if self.data_type != "Methylation"
                        else data_shape[1]
                    )
                    all_sample_idx = range(
                        data_shape[1]
                        if self.data_type != "Methylation"
                        else data_shape[0]
                    )
                    
                    if self.operation == "subset":
                        if self.marker_ids and not affected_marker_idx:
                            return chromosome, None
                        keep_marker = (
                            affected_marker_idx if self.marker_ids else list(all_marker_idx)
                        )
                        keep_sample = (
                            affected_sample_idx if self.sample_ids else list(all_sample_idx)
                        )
                    else:
                        affected_marker_set = set(affected_marker_idx)
                        affected_sample_set = set(affected_sample_idx)
                        
                        keep_marker = [
                            i
                            for i in all_marker_idx
                            if i not in affected_marker_set
                        ]
                        keep_sample = [
                            i
                            for i in all_sample_idx
                            if i not in affected_sample_set
                        ]
                    
                    filtered_markers = [marker_list[i] for i in keep_marker]
                    if len(keep_marker) == 0 or len(keep_sample) == 0:
                        return chromosome, None
                    data_path_key = (
                        "Methylation" if self.data_type == "Methylation" else "Genotype"
                    )
                    
                    # Get the source dataset without loading it fully
                    source_dataset = h5_file[paths[data_path_key]]
                    source_dtype = source_dataset.dtype
                    
                    row_idx = (
                        keep_marker if self.data_type != "Methylation" else keep_sample
                    )
                    col_idx = (
                        keep_sample if self.data_type != "Methylation" else keep_marker
                    )
                    
                    # Output shape after filtering
                    out_rows = len(row_idx)
                    out_cols = len(col_idx)
                    
                    if out_rows == 0 or out_cols == 0:
                        return chromosome, None
                    
                    # Calculate chunk sizes for the output dataset
                    row_chunk = min(out_rows, max(32, int(out_rows / 10)))
                    base_bytes = 32 * 1024 * 1024
                    dtype_itemsize = source_dtype.itemsize
                    row_divisor = max(1, row_chunk)
                    computed = int(base_bytes / dtype_itemsize / row_divisor)
                    col_chunk = min(out_cols, max(1, computed))
                    chunk_size = (row_chunk, col_chunk)
                    
                    log.debug(
                        f"Creating output dataset with shape ({out_rows}, {out_cols}), chunks {chunk_size}"
                    )
                    
                    # Create the output dataset with the final shape
                    out_dataset = temp_file.create_dataset(
                        paths[data_path_key],
                        shape=(out_rows, out_cols),
                        dtype=source_dtype,
                        chunks=chunk_size,
                        compression="gzip",
                        compression_opts=4,
                    )
                    
                    # Sort indices for efficient HDF5 access
                    # HDF5 requires sorted indices for efficient fancy indexing
                    row_idx_arr = np.array(row_idx)
                    col_idx_arr = np.array(col_idx)
                    sorted_row_idx = np.sort(row_idx_arr)
                    sorted_col_idx = np.sort(col_idx_arr)
                    
                    # Process in chunks to avoid OOM
                    available_mem = psutil.virtual_memory().available
                    target_mem = available_mem * 0.1
                    dtype_itemsize = source_dtype.itemsize
                    row_size = out_cols * dtype_itemsize
                    
                    # Ideal number of rows to process at once
                    processing_chunk_size = int(target_mem / row_size)
                    # Stay between 100 and 10,000 to keep progress bar moving but avoid excessive overhead
                    processing_chunk_size = min(10000, max(100, processing_chunk_size))
                    # Don't exceed total rows
                    processing_chunk_size = min(out_rows, processing_chunk_size)
                    
                    log.debug(f"Processing in chunks of {processing_chunk_size} rows (based on {available_mem/1e9:.1f}GB available RAM)")
                    
                    # Pre-compute column reorder once using np.argsort
                    # This maps sorted indices back to original requested order
                    col_reorder_indices = np.argsort(np.argsort(col_idx))
                    
                    # Process rows in sorted order for efficient reads, write to contiguous output
                    out_row_write_pos = 0
                    for chunk_start in range(0, len(sorted_row_idx), processing_chunk_size):
                        chunk_end = min(chunk_start + processing_chunk_size, len(sorted_row_idx))
                        chunk_row_indices = sorted_row_idx[chunk_start:chunk_end]
                        
                        # Read chunk from source (sorted indices for efficient read)
                        chunk_data = source_dataset[chunk_row_indices, :][:, sorted_col_idx]
                        
                        # Reorder columns to match original col_idx order using pre-computed indices
                        chunk_data = chunk_data[:, col_reorder_indices]
                        
                        # Write entire chunk as contiguous block
                        chunk_rows = chunk_end - chunk_start
                        out_dataset[out_row_write_pos:out_row_write_pos + chunk_rows, :] = chunk_data
                        out_row_write_pos += chunk_rows
                        
                        del chunk_data
                        gc.collect()
                    
                    temp_file.create_dataset(
                        paths[marker_path_key],
                        data=np.array(filtered_markers, dtype=h5py.string_dtype()),
                        compression="gzip",
                        compression_opts=4,
                    )
                    if "Additional" in paths:
                        for ds_name in paths["Additional"]:
                            ds_path = f"/{chromosome}/{ds_name}"
                            if ds_path in h5_file:
                                original_ds = h5_file[ds_path]
                                # Use fancy indexing directly on HDF5 dataset to avoid loading full array
                                # only if it matches marker count
                                if len(original_ds) == len(marker_list):
                                    idx_arr = np.array(keep_marker)
                                    sorted_idx = np.sort(idx_arr)
                                    # HDF5 requires sorted indices for fancy indexing
                                    filtered_ds = original_ds[sorted_idx]
                                    # Reorder back to preserved order if it was a subset operation with specific order
                                    reorder = np.argsort(np.argsort(idx_arr))
                                    filtered_ds = filtered_ds[reorder]
                                else:
                                    filtered_ds = original_ds[:]
                                
                                # Optimization: Handle decoding without full list creation if possible
                                if len(filtered_ds) > 0 and isinstance(filtered_ds[0], (bytes, np.bytes_)):
                                    filtered_ds = self._decode_array(filtered_ds)
                                    temp_file.create_dataset(
                                        ds_path,
                                        data=np.array(filtered_ds, dtype=h5py.string_dtype()),
                                        compression="gzip",
                                        compression_opts=4,
                                    )
                                else:
                                    temp_file.create_dataset(
                                        ds_path,
                                        data=filtered_ds,
                                        compression="gzip",
                                        compression_opts=4,
                                    )
                    gc.collect()

                    memory_used = stats["max_memory"]
                    if memory_used > 80:
                        log.warn(
                            f"High memory usage ({memory_used:.1f}%) during processing of chromosome {chromosome}"
                        )

                    return chromosome, temp_path

            except Exception as e:
                log.error(f"Error processing chromosome {chromosome}: {e}")
                log.info(
                    f"Peak resources during failure: CPU {stats['max_cpu']:.1f}%, Memory: {stats['max_memory']:.1f}%"
                )
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                return chromosome, None

    def process_chromosome_read(
        self,
        chromosome: str,
        sample_indices: Optional[List[int]] = None,
        sample_names: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        try:
            with h5py.File(self.input_file, "r") as h5_file:
                h5_utils = CachedH5Utils(h5_file)
                if self.marker_ids:
                    marker_dict = self._get_marker_info(h5_file)
                    if chromosome in marker_dict:
                        markers_in_chr = marker_dict[chromosome]
                        if markers_in_chr:
                            pass
                chr_data = h5_utils.read_chromosome(chromosome, self.data_type)
                if chr_data is None:
                    log.warn(f"No data found for chromosome {chromosome}")
                    return None
                if "CHR" not in chr_data.columns:
                    chrom_num = re.sub(r"^(chr|CHR|Chr)", "", chromosome)
                    try:
                        chr_data["CHR"] = int(chrom_num)
                    except ValueError:
                        chr_data["CHR"] = chrom_num
                else:
                    if chr_data["CHR"].dtype == object:
                        chr_data["CHR"] = pd.to_numeric(
                            chr_data["CHR"], errors="ignore"
                        )
                marker_col = "CGID" if self.data_type == "Methylation" else "RSID"
                if self.marker_ids and marker_col in chr_data.columns:
                    chr_data = chr_data[
                        chr_data[marker_col]
                        .astype(str)
                        .isin([str(m) for m in self.marker_ids])
                    ]
                if sample_names and len(sample_names) > 0:
                    cols_to_keep = [
                        col
                        for col in chr_data.columns
                        if col in sample_names or col in [marker_col, "CHR"]
                    ]
                    chr_data = chr_data[cols_to_keep]
                return chr_data
        except Exception as e:
            log.error(f"Error reading chromosome {chromosome}: {e}")
            return None

    def copy_datasets(self, temp_file_path: str, chromosome: str) -> bool:
        with h5py.File(temp_file_path, "r") as temp_file, h5py.File(
            self.output_file, "r+"
        ) as out_file:
            if chromosome not in out_file:
                out_file.create_group(chromosome)
            for dataset_name in temp_file[chromosome]:
                temp_file.copy(f"/{chromosome}/{dataset_name}", out_file[chromosome])
        os.remove(temp_file_path)
        return True

    def _create_metadata(
        self,
        input_h5_file: h5py.File,
        output_h5_file: h5py.File,
        kept_sample_names: List[str],
    ) -> bool:
        metadata_key = AliasUtils.find_keys(input_h5_file, "Metadata")
        if not metadata_key:
            metadata_key = "Metadata"
        metadata_group = output_h5_file.create_group(metadata_key)
        if self.data_type == "Genotype":
            sample_key = AliasUtils.find_keys(input_h5_file[metadata_key], "IID")
            if sample_key:
                samples_array = np.array(
                    [str(s) for s in kept_sample_names], dtype=h5py.string_dtype()
                )
                metadata_group.create_dataset(
                    sample_key, data=samples_array, compression="gzip"
                )
            metadata_fields = ["FID", "Father", "Mother", "Sex", "Phenotype"]
            for field in metadata_fields:
                field_key = AliasUtils.find_keys(input_h5_file[metadata_key], field)
                if field_key and field_key in input_h5_file[metadata_key]:
                    original_data = input_h5_file[metadata_key][field_key][:]
                    if len(original_data) == len(kept_sample_names):
                        metadata_group.create_dataset(
                            field_key, data=original_data, compression="gzip"
                        )
        else:
            sample_key = AliasUtils.find_keys(input_h5_file[metadata_key], "SampleList")
            if sample_key:
                samples_array = np.array(
                    [str(s) for s in kept_sample_names], dtype=h5py.string_dtype()
                )
                metadata_group.create_dataset(
                    sample_key, data=samples_array, compression="gzip"
                )
        return True

    def extract_marker_names(self, h5_file: h5py.File) -> pd.DataFrame:
        marker_type = "probe" if self.data_type == "Methylation" else "SNP"
        h5_utils = CachedH5Utils(h5_file)
        chromosome_list = h5_utils.get_chromosomes()
        all_markers: List[str] = []
        for chromosome in tqdm(chromosome_list, desc=f"Reading {marker_type} lists"):
            marker_key = AliasUtils.find_keys(
                h5_file[chromosome],
                "ProbeList" if self.data_type == "Methylation" else "RSID",
            )
            marker_path = f"/{chromosome}/{marker_key or ('probeList' if self.data_type == 'Methylation' else 'snp')}"
            if marker_path in h5_file:
                marker_list = self._decode_array(h5_file[marker_path][:])
                all_markers.extend(marker_list)
        column_name = "probe_id" if self.data_type == "Methylation" else "snp_id"
        return pd.DataFrame({column_name: all_markers})

    def extract_sample_names(self, h5_file: h5py.File) -> pd.DataFrame:
        metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
        sample_key = (
            AliasUtils.find_keys(h5_file[metadata_key], "SampleList")
            if self.data_type == "Methylation"
            else AliasUtils.find_keys(h5_file[metadata_key], "IID")
        )
        sample_path = f"/{metadata_key}/{sample_key}"
        if sample_path not in h5_file:
            raise ValueError(f"Sample list not found at {sample_path}")
        sample_list = self._decode_array(h5_file[sample_path][:])
        return pd.DataFrame({"sample_id": sample_list})

    def run_hdf5_operation(self) -> bool:
        os.makedirs(self.temp_dir, exist_ok=True)
        with h5py.File(self.input_file, "r") as h5_file:
            h5_utils = CachedH5Utils(h5_file)
            all_chromosomes = h5_utils.get_chromosomes()

            mapped_chromosomes, missing = self._map_requested_chromosomes(
                self.chromosomes, all_chromosomes
            )
            for missing_chr in missing:
                log.warn(f"Chromosome {missing_chr} not found in HDF5 file")

            mapped_chromosomes_set = set(mapped_chromosomes)
            if self.operation == "subset":
                chromosome_list = (
                    mapped_chromosomes if mapped_chromosomes else all_chromosomes
                )
            else:
                chromosome_list = [
                    chr for chr in all_chromosomes if chr not in mapped_chromosomes_set
                ]
                if not chromosome_list and mapped_chromosomes:
                    log.info("All requested chromosomes will be removed from output")

            full_sample_list, selected_samples = self._get_sample_info(h5_file)
            self.markers_dict = self._get_marker_info(h5_file)

            entity_type = "probes" if self.data_type == "Methylation" else "SNPs"
            # SAFETY CHECK: If markers were requested but none found, stop early
            if self.marker_ids and not self.markers_dict:
                log.error(f"None of the {len(self.marker_ids)} requested {entity_type} were found.")
                return False

            # SAFETY CHECK: If samples were requested but none found, stop early
            if self.sample_ids and not selected_samples:
                log.error("None of the requested samples were found.")
                return False

            if self.operation == "subset":
                chromosomes_to_process = [
                    chr
                    for chr in chromosome_list
                    if chr in self.markers_dict or not self.marker_ids
                ]
                kept_sample_names = selected_samples
            else:
                chromosomes_to_process = chromosome_list
                remove_indices = set(self.sample_indices) if self.sample_ids else set()
                keep_idx = [
                    i for i in range(len(full_sample_list)) if i not in remove_indices
                ]
                kept_sample_names = [full_sample_list[i] for i in keep_idx]
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        chromosome_files: List[Tuple[str, str]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(self.process_chromosome_hdf5, chr)
                for chr in chromosomes_to_process
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing chromosomes",
            ):
                chr, path = future.result()
                if path:
                    chromosome_files.append((chr, path))
        with h5py.File(self.output_file, "w") as out_file, h5py.File(
            self.input_file, "r"
        ) as input_file:
            if not self._create_metadata(input_file, out_file, kept_sample_names):
                return False
        for chr, temp_path in tqdm(chromosome_files, desc="Copying to output"):
            self.copy_datasets(temp_path, chr)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        return True

    def run_read_operation(self) -> Union[bool, pd.DataFrame]:
        file_size_gb = os.path.getsize(self.input_file) / (1024**3)
        use_chunked = file_size_gb > 10
        with h5py.File(self.input_file, "r") as h5_file:
            full_sample_list, selected_samples = self._get_sample_info(h5_file)
            sample_names = selected_samples
            sample_indices = self.sample_indices
            self.markers_dict = self._get_marker_info(h5_file)
            all_chromosomes = self._get_all_chromosomes()
            mapped_chromosomes, missing = self._map_requested_chromosomes(
                self.chromosomes, all_chromosomes
            )
            for missing_chr in missing:
                log.warn(f"Chromosome {missing_chr} not found in HDF5 file")
            if mapped_chromosomes:
                chromosomes_to_process = mapped_chromosomes
            else:
                chromosomes_to_process = all_chromosomes if not self.chromosomes else []
        all_results: List[pd.DataFrame] = []
        for chromosome in tqdm(chromosomes_to_process, desc="Processing chromosomes"):
            chr_data: Optional[pd.DataFrame] = None
            with h5py.File(self.input_file, "r") as h5_file:
                if use_chunked:
                    chunk_results = self._process_data_in_chunks(
                        h5_file, chromosome, sample_indices, sample_names
                    )
                    if chunk_results:
                        chr_data = pd.concat(chunk_results, ignore_index=True)
                else:
                    chr_data = self.process_chromosome_read(
                        chromosome, sample_indices, sample_names
                    )
            if chr_data is not None and not chr_data.empty:
                all_results.append(chr_data)
            gc.collect()
            self._check_memory_usage(f"After {chromosome}")
        if not all_results:
            log.warn("No data found for any chromosome")
            return False
        combined_data = pd.concat(all_results, ignore_index=True)
        if "CHR" in combined_data.columns:
            combined_data["CHR"] = pd.to_numeric(combined_data["CHR"], errors="coerce")
            sort_cols = ["CHR"]
            if "BP" in combined_data.columns:
                sort_cols.append("BP")
            combined_data = combined_data.sort_values(sort_cols)
        if self.output_file:
            combined_data.to_csv(self.output_file, index=False)
            return True
        return combined_data

    def _get_all_chromosomes(self) -> List[str]:
        with h5py.File(self.input_file, "r") as h5_file:
            chromosomes: List[str] = []
            for key in h5_file.keys():
                base_key = AliasUtils.strip_numeric_suffix(key)
                if AliasUtils.get_field(base_key) == "CHR":
                    chromosomes.append(key)
            return chromosomes

    def run_names_operation(self) -> Union[bool, pd.DataFrame]:
        with h5py.File(self.input_file, "r") as h5_file:
            if self.names_type.lower() in ["markers", "probes", "snps"]:
                results = self.extract_marker_names(h5_file)
            elif self.names_type.lower() == "samples":
                results = self.extract_sample_names(h5_file)
            else:
                raise ValueError("Invalid names_type")
        if self.output_file:
            results.to_csv(self.output_file, header=False, index=False)
            return True
        return results

    def _find_sample_id_column(
        self, metadata_df: pd.DataFrame, h5_sample_list: List[Any]
    ) -> str:
        test_samples_set = set(h5_sample_list[:10])
        best_col: Optional[str] = None
        best_count = 0
        for col in metadata_df.columns:
            col_values = [str(val) for val in metadata_df[col].dropna()]
            col_normalized = set(self._normalize_sample_ids(col_values))
            matches = len(test_samples_set & col_normalized)
            if matches > best_count:
                best_count = matches
                best_col = col
        if best_col:
            return best_col
        raise ValueError("No matching sample ID column")

    def _load_and_process_metadata(self) -> Tuple[pd.DataFrame, str]:
        file_ext = os.path.splitext(self.metadata_file)[1].lower()
        if file_ext in [".csv", ".txt", ".tsv"]:
            sep = "\t" if file_ext in [".txt", ".tsv"] else ","
            metadata_df = pd.read_csv(self.metadata_file, sep=sep)
        else:
            metadata_df = pd.read_csv(self.metadata_file)
        with h5py.File(self.input_file, "r") as h5_file:
            metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
            sample_key = (
                AliasUtils.find_keys(h5_file[metadata_key], "SampleList")
                if self.data_type == "Methylation"
                else AliasUtils.find_keys(h5_file[metadata_key], "IID")
            )
            sample_path = f"/{metadata_key}/{sample_key}"
            h5_sample_list = self._decode_array(h5_file[sample_path][:])
        sample_id_col = self._find_sample_id_column(metadata_df, h5_sample_list)
        metadata_df[sample_id_col] = metadata_df[sample_id_col].astype(str)
        h5_sample_str = [str(s) for s in h5_sample_list]
        metadata_dict = {
            str(row[sample_id_col]): row for _, row in metadata_df.iterrows()
        }
        ordered_rows: List[pd.Series] = []
        for s in h5_sample_str:
            if s in metadata_dict:
                ordered_rows.append(metadata_dict[s])
            else:
                nan_row = pd.Series(
                    [np.nan] * len(metadata_df.columns), index=metadata_df.columns
                )
                nan_row[sample_id_col] = s
                ordered_rows.append(nan_row)
        ordered_df = pd.DataFrame(ordered_rows)
        if self.columns_to_add:
            missing_cols = [
                col for col in self.columns_to_add if col not in ordered_df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Requested columns not found in metadata: {missing_cols}"
                )
            cols = [sample_id_col] + [
                col for col in self.columns_to_add if col != sample_id_col
            ]
            ordered_df = ordered_df[cols]
        metadata_for_hdf5 = ordered_df[
            [col for col in ordered_df.columns if col != sample_id_col]
        ]
        return metadata_for_hdf5, sample_id_col

    def run_add_metadata_operation(self) -> bool:
        metadata_df, _ = self._load_and_process_metadata()
        shutil.copy2(self.input_file, self.output_file)
        with h5py.File(self.output_file, "r+") as h5_file:
            metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
            if not metadata_key:
                raise ValueError("No metadata group found in HDF5 file")
            metadata_group = h5_file[metadata_key]
            for col in metadata_df.columns:
                col_data = metadata_df[col].values
                if col_data.dtype == "object":
                    col_clean = [
                        str(val) if not pd.isna(val) else "missing" for val in col_data
                    ]
                    dtype = h5py.string_dtype()
                    data_array = np.array(col_clean, dtype=dtype)
                else:
                    fill = np.nan if np.issubdtype(col_data.dtype, np.floating) else -9
                    col_clean = np.where(pd.isna(col_data), fill, col_data)
                    dtype = col_clean.dtype
                    data_array = col_clean
                if col in metadata_group:
                    del metadata_group[col]
                metadata_group.create_dataset(
                    col, data=data_array, compression="gzip", compression_opts=4
                )
        return True

    def run_extract_metadata_operation(self) -> Union[bool, pd.DataFrame]:
        with h5py.File(self.input_file, "r") as h5_file:
            metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
            if not metadata_key:
                raise ValueError("No metadata group found")
            sample_key = (
                AliasUtils.find_keys(h5_file[metadata_key], "SampleList")
                if self.data_type == "Methylation"
                else AliasUtils.find_keys(h5_file[metadata_key], "IID")
            )
            sample_path = f"/{metadata_key}/{sample_key}"
            if sample_path not in h5_file:
                raise ValueError(f"Sample list not found at {sample_path}")
            full_sample_list = self._decode_array(h5_file[sample_path][:])
            if self.sample_ids:
                requested = set(str(s) for s in self.sample_ids)
                sample_indices = [
                    i for i, s in enumerate(full_sample_list) if str(s) in requested
                ]
                sample_names = [full_sample_list[i] for i in sample_indices]
                if not sample_indices:
                    raise ValueError("No requested samples found")
            else:
                sample_indices = list(range(len(full_sample_list)))
                sample_names = full_sample_list
            metadata_group = h5_file[metadata_key]
            data_dict: Dict[str, Any] = {"sample_id": sample_names}
            for col in self.columns_to_extract:
                if col in metadata_group:
                    col_data = metadata_group[col][:]
                    if len(col_data) != len(full_sample_list):
                        raise ValueError(f"Metadata column {col} length mismatch")
                    filtered_data = col_data[sample_indices]
                    if col_data.dtype.kind in {"O", "S", "U"}:
                        filtered_data = self._decode_array(filtered_data)
                    data_dict[col] = filtered_data
                else:
                    log.warn(f"Column {col} not found in metadata")
            df = pd.DataFrame(data_dict)
        if self.output_file:
            df.to_csv(self.output_file, index=False)
            return True
        return df

    def run(self) -> bool:
        operations_require_output = {"subset", "remove", "add_metadata"}

        try:
            if not self._check_system_health():
                log.warn("System health check failed - processing may be unstable")

            if self.operation in operations_require_output and not self.output_file:
                raise ValueError(
                    f"output_file required for '{self.operation}' operation"
                )

            with monitor_resources(interval=5.0) as stats:
                try:
                    result = False

                    if self.operation in ["subset", "remove"]:
                        has_filters = bool(
                            self.sample_ids or self.marker_ids or self.chromosomes
                        )
                        if not has_filters:
                            if self.operation == "remove":
                                shutil.copy(self.input_file, self.output_file)
                                result = True
                            else:
                                raise ValueError(
                                    "At least one of samples, markers, or chromosomes required"
                                )
                        else:
                            result = self.run_hdf5_operation()

                    elif self.operation == "read":
                        result = self.run_read_operation()

                    elif self.operation == "names":
                        if not self.names_type:
                            raise ValueError("names_type required")
                        valid_names = ["markers", "probes", "snps", "samples"]
                        if self.names_type.lower() not in valid_names:
                            raise ValueError(
                                f"Invalid names_type '{self.names_type}'. Must be one of: {valid_names}"
                            )
                        result = self.run_names_operation()

                    elif self.operation == "add_metadata":
                        if not self.metadata_file:
                            raise ValueError(
                                "metadata file required for add_metadata operation"
                            )
                        result = self.run_add_metadata_operation()

                    elif self.operation == "extract_metadata":
                        result = self.run_extract_metadata_operation()

                    log.info(f"Operation '{self.operation}' completed")
                    log.info(
                        f"Peak resource usage - CPU: {stats['max_cpu']:.1f}%, Memory: {stats['max_memory']:.1f}%"
                    )

                    return result

                except Exception as e:
                    log.error(f"Error during {self.operation} operation: {e}")
                    log.info("Peak resource usage before failure:")
                    log.info(
                        f"CPU: {stats['max_cpu']:.1f}%, Memory: {stats['max_memory']:.1f}%"
                    )
                    raise

        except Exception as e:
            log.error(f"Error in ProcessHDF5: {e}")
            return False
        finally:
            self._cleanup()

        return False


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
    OptionConfig(
        flags=["-op", "--operation"],
        type=str,
        default="subset",
        required=True,
        choices=[
            "subset",
            "remove",
            "read",
            "names",
            "add_metadata",
            "extract_metadata",
        ],
    ),
    OptionConfig(flags=["-s", "--samples"], type=str, default=None, required=False),
    OptionConfig(flags=["-m", "--markers"], type=str, default=None, required=False),
    OptionConfig(flags=["-c", "--chromosomes"], type=str, default=None, required=False),
    OptionConfig(
        flags=["-t", "--type"],
        type=str,
        default=None,
        required=False,
        choices=["Methylation", "Genotype"],
    ),
    OptionConfig(
        flags=["-cs", "--chunk_size"], type=int, default=30000, required=False
    ),
    OptionConfig(flags=["-n", "--names"], type=str, default=None, required=False),
    OptionConfig(flags=["-md", "--metadata"], type=str, default=None, required=False),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="ProcessHDF5")
    opt = framework.run()
    processor = ProcessHDF5(
        input_file=opt.input,
        output_file=opt.output,
        operation=opt.operation,
        samples=opt.samples,
        markers=opt.markers,
        chromosomes=opt.chromosomes,
        data_type=opt.type,
        chunk_size=opt.chunk_size,
        names=opt.names,
        metadata=opt.metadata,
    )
    result = processor.run()
    if result is False:
        sys.exit(1)
