#!/usr/bin/env python
# Import required modules
import concurrent.futures
import gc
import h5py
import numba
import numpy as np
import os
import pandas as pd
import re
import shutil
import tempfile
import time
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.DownloadAndExtract import DownloadAndExtract
from utils.LoggingUtils import log
from utils.SystemUtils import SystemUtils, monitor_resources


@numba.jit(nopython=True)
def _pack_genotypes_to_bed(genotypes: np.ndarray, n_samples: int) -> np.ndarray:
    bytes_per_variant = (n_samples + 3) // 4
    n_variants = genotypes.shape[0]
    packed_bytes = np.zeros((n_variants, bytes_per_variant), dtype=np.uint8)
    for variant_idx in range(n_variants):
        for sample_idx in range(n_samples):
            byte_idx = sample_idx // 4
            bit_pair = sample_idx % 4
            geno = genotypes[variant_idx, sample_idx]
            if geno == 0:
                code = 0
            elif geno == 1:
                code = 2
            elif geno == 2:
                code = 3
            else:
                code = 1
            packed_bytes[variant_idx, byte_idx] |= code << (bit_pair * 2)
    return packed_bytes


class ConvertCounts:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        chip: Optional[str] = None,
        hg: str = "hg38",
        transpose: bool = False,
    ) -> None:
        self.input_file: str = input_file
        self.output_file: str = output_file
        self.chip: Optional[str] = chip
        self.hg: str = hg
        self.transpose: bool = transpose
        self.data: Optional[pd.DataFrame] = None
        self.h5_file: Optional[h5py.File] = None
        self.input_format: str = self._detect_format(input_file, is_input=True)
        self.output_format: str = self._detect_format(output_file, is_input=False)
        self.operation: str = f"{self.input_format}_to_{self.output_format}"
        self.bim: Optional[pd.DataFrame] = None
        self.fam: Optional[pd.DataFrame] = None
        self.temp_dirs: List[str] = []
        self.compression_params: Dict[str, Any] = {
            "compression": "gzip",
            "chunks": True,
            "compression_opts": 4,
        }
        log.info(
            f"Detected conversion: {self.input_format.upper()} → {self.output_format.upper()}"
        )

        SystemUtils.print_system_info()

    def _check_system_health(self) -> bool:
        log.info("Checking system health before conversion...")

        required_gb = self._estimate_output_size()

        health = SystemUtils.check_system_health(
            min_free_disk_gb=required_gb,
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
            path=output_dir,
            required_gb=required_gb,
            buffer_percent=20.0,
        )

        if not disk_ok:
            log.error(f"Disk space issue: {disk_message}")
            return False

        log.info("System health check passed")
        return True

    def _estimate_output_size(self) -> float:
        try:
            if self.operation == "csv_to_hdf5":
                if os.path.exists(self.input_file):
                    input_size_gb = os.path.getsize(self.input_file) / (1024**3)
                    return max(input_size_gb, 0.1)
                return 1.0

            elif self.operation == "hdf5_to_csv":
                if os.path.exists(self.input_file):
                    input_size_gb = os.path.getsize(self.input_file) / (1024**3)
                    return input_size_gb * 1.5
                return 1.0

            elif self.operation == "plink_to_hdf5":
                total_size_gb = 0
                for ext in [".bed", ".bim", ".fam"]:
                    file_path = f"{self.input_file}{ext}"
                    if os.path.exists(file_path):
                        total_size_gb += os.path.getsize(file_path) / (1024**3)
                return max(total_size_gb * 0.9, 0.1)

            elif self.operation == "hdf5_to_plink":
                if os.path.exists(self.input_file):
                    input_size_gb = os.path.getsize(self.input_file) / (1024**3)
                    return input_size_gb * 1.2
                return 1.0

            return 1.0

        except Exception as e:
            log.debug(f"Error estimating output size: {e}")
            return 1.0

    def _setup_temp_directory(self) -> str:
        output_dir = os.path.dirname(os.path.abspath(self.output_file))

        if self.operation in ("csv_to_hdf5", "hdf5_to_csv"):
            required_gb = max(0.5, self._estimate_output_size() * 0.25)
        else:
            required_gb = max(0.5, self._estimate_output_size() * 0.5)

        try:
            temp_dir, temp_info = SystemUtils.create_safe_tempdir(
                default_path=output_dir,
                required_gb=required_gb,
                prefix=f"convert_{self.input_format}_to_{self.output_format}",
                buffer_percent=10.0,
            )
            log.info(f"Created temporary directory: {temp_dir}")
            condition1 = isinstance(temp_info, dict)
            condition2 = "disk_info" in temp_info
            condition3 = "free_gb" in temp_info["disk_info"]
            if condition1 and condition2 and condition3:
                log.debug(
                    f"Temp directory has {temp_info['disk_info']['free_gb']:.1f}GB free space"
                )
            return temp_dir
        except Exception as e:
            log.warn(f"Failed to create safe temp directory: {e}")
            log.warn("Using system temp directory as fallback")
            return tempfile.mkdtemp(
                prefix=f"convert_{self.input_format}_to_{self.output_format}"
            )

    def _detect_format(self, file_path: str, is_input: bool = True) -> str:
        _, ext = os.path.splitext(file_path.lower())
        if ext in [".csv", ".tsv", ".txt"]:
            return "csv"
        elif ext in [".h5", ".hdf5"]:
            return "hdf5"
        elif ext == "":
            if is_input:
                plink_files = [
                    f"{file_path}.bed",
                    f"{file_path}.bim",
                    f"{file_path}.fam",
                ]
                if all(os.path.exists(f) for f in plink_files):
                    return "plink"
                else:
                    if any(
                        os.path.exists(f"{file_path}.{ext}")
                        for ext in ["bed", "bim", "fam"]
                    ):
                        return "plink"
                    else:
                        log.error(
                            f"No extension provided and PLINK files not found for: {file_path}"
                        )
                        log.error(
                            "Expected files: {}.bed, {}.bim, {}.fam".format(
                                file_path, file_path, file_path
                            )
                        )
                        raise ValueError("Cannot determine input format")
            else:
                return "plink"
        else:
            log.error(f"Unsupported file extension: {ext}")
            raise ValueError(f"Unsupported file format: {ext}")

    def convert(self) -> bool:
        try:
            safe_config = SystemUtils.configure_safe_environment()
            if safe_config.get("memory_limit_set", False):
                log.debug("Memory limits configured to prevent OOM errors")
            if safe_config.get("core_dumps_disabled", False):
                log.debug("Core dumps disabled for stability")

            health = SystemUtils.check_system_health()
            if health["status"] != "healthy":
                for warning in health.get("warnings", []):
                    log.warn(f"System warning: {warning}")
                for critical in health.get("critical", []):
                    log.warn(f"Critical system issue: {critical}")

            if self.input_format == self.output_format:
                log.error(
                    f"Input and output formats are the same ({self.input_format})"
                )
                return False

            valid_conversions = {
                "csv_to_hdf5",
                "hdf5_to_csv",
                "plink_to_hdf5",
                "hdf5_to_plink",
            }

            if self.operation not in valid_conversions:
                log.error(f"Conversion {self.operation} is not supported")
                log.info(
                    f"Supported conversions: {', '.join(sorted(valid_conversions))}"
                )
                return False

            if self.output_format == "hdf5":
                output_dir = os.path.dirname(os.path.abspath(self.output_file))
                if not os.path.exists(output_dir):
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                        log.info(f"Created output directory: {output_dir}")
                    except PermissionError:
                        log.error(
                            f"No permission to create output directory: {output_dir}"
                        )
                        return False

                if os.path.exists(self.output_file):
                    if not os.access(self.output_file, os.W_OK):
                        log.error(
                            f"No write permission for existing output file: {self.output_file}"
                        )
                        return False
                else:
                    if not os.access(output_dir, os.W_OK):
                        log.error(
                            f"No write permission for output directory: {output_dir}"
                        )
                        return False

            if self.operation == "hdf5_to_csv":
                with h5py.File(self.input_file, "r") as h5_file:
                    has_methylation = False
                    chromosome_keys: List[str] = []
                    for key in h5_file.keys():
                        metadata_aliases = AliasUtils.get_aliases("Metadata")
                        if key in metadata_aliases:
                            continue
                        base_key = AliasUtils.strip_numeric_suffix(key)
                        chr_aliases = AliasUtils.get_aliases("CHR")
                        if base_key in chr_aliases or base_key.upper() in [
                            alias.upper() for alias in chr_aliases
                        ]:
                            chromosome_keys.append(key)
                    for chr_key in chromosome_keys:
                        probe_key = AliasUtils.find_keys(h5_file[chr_key], "ProbeList")
                        if probe_key:
                            has_methylation = True
                            break
                    if not has_methylation:
                        log.error("hdf5_to_csv is only supported for methylation data")
                        raise ValueError("HDF5 file does not contain methylation data")

            if not self._check_system_health():
                log.error("System health check failed - conversion may be unstable")
                user_input = input("Continue anyway? (y/N): ")
                if user_input.lower() != "y":
                    log.info("Conversion cancelled by user")
                    return False

            log.info(f"Starting {self.operation} conversion")
            log.info(f"Input: {self.input_file} ({self.input_format.upper()})")
            log.info(f"Output: {self.output_file} ({self.output_format.upper()})")
            if self.chip:
                log.info(f"Chip: {self.chip}, Genome: {self.hg}")
            if self.transpose:
                log.info("Data will be transposed")

            start_time = time.time()

            with monitor_resources(interval=5.0) as stats:
                result = False

                if self.operation == "csv_to_hdf5":
                    result = self._csv_to_hdf5()
                elif self.operation == "hdf5_to_csv":
                    result = self._hdf5_to_csv()
                elif self.operation == "plink_to_hdf5":
                    result = self._plink_to_hdf5()
                elif self.operation == "hdf5_to_plink":
                    result = self._hdf5_to_plink()

                execution_time = time.time() - start_time
                resource_str = (
                    f"CPU: {stats['max_cpu']:.1f}%, Memory: {stats['max_memory']:.1f}%"
                )

                if result:
                    log.success(f"Conversion completed in {execution_time:.1f}s")
                    log.info(f"Peak resource usage: {resource_str}")

                    if os.path.exists(self.output_file):
                        output_size = os.path.getsize(self.output_file) / (1024**2)
                        log.info(f"Output file size: {output_size:.1f} MB")
                    elif self.output_format == "plink":
                        total_size = 0
                        for ext in [".bed", ".bim", ".fam"]:
                            file_path = f"{self.output_file}{ext}"
                            if os.path.exists(file_path):
                                total_size += os.path.getsize(file_path)
                        log.info(
                            f"Output files total size: {total_size / (1024**2):.1f} MB"
                        )
                else:
                    log.error(f"Conversion failed after {execution_time:.1f}s")
                    log.info(f"Peak resource usage before failure: {resource_str}")

                return result

        except Exception as e:
            log.error(f"Error in data conversion: {e}")
            return False
        finally:
            self._cleanup()

    def _csv_to_hdf5(self) -> bool:
        steps: List[Tuple[str, Callable[[], bool]]] = [
            ("read CSV data", self._read_csv_data),
            ("ensure chromosome column", self._ensure_chromosome_column),
            ("remove existing output", self._remove_existing_output),
            ("save to HDF5", self._save_csv_to_hdf5),
        ]
        return self._execute_steps(steps, "CSV to HDF5")

    def _hdf5_to_csv(self) -> bool:
        steps: List[Tuple[str, Callable[[], bool]]] = [
            ("read HDF5 data", self._read_hdf5_methylation_data),
            ("process transpose if needed", self._process_transpose),
            ("save to CSV", self._save_to_csv),
        ]
        return self._execute_steps(steps, "HDF5 to CSV")

    def _plink_to_hdf5(self) -> bool:
        steps: List[Tuple[str, Callable[[], bool]]] = [
            ("read PLINK data", self._read_plink_metadata),
            ("remove existing output", self._remove_existing_output),
            ("save to HDF5", self._save_plink_to_hdf5),
        ]
        return self._execute_steps(steps, "PLINK to HDF5")

    def _hdf5_to_plink(self) -> bool:
        steps: List[Tuple[str, Callable[[], bool]]] = [
            ("read HDF5 genotype data", self._read_hdf5_genotype_data),
            ("remove existing output", self._remove_existing_output),
            ("save to PLINK", self._save_to_plink),
        ]
        return self._execute_steps(steps, "HDF5 to PLINK")

    def _execute_steps(
        self, steps: List[Tuple[str, Callable[[], bool]]], operation_name: str
    ) -> bool:
        for step_name, step_func in steps:
            if not step_func():
                log.error(f"Failed to {step_name}")
                return False
        log.success(f"{operation_name} conversion completed: {self.output_file}")
        return True

    def _read_csv_data(self) -> bool:
        try:
            memory_info = SystemUtils.get_memory_info()
            available_memory = memory_info.get("available_gb", 8.0)

            chunk_size = int((available_memory * 1024**3 * 0.25) / 8)

            chunk_size = max(10000, min(chunk_size, 1000000))

            log.info(
                f"Using chunk size {chunk_size} based on {available_memory:.1f}GB available memory"
            )

            if not os.path.exists(self.input_file):
                log.error(f"Input file not found: {self.input_file}")
                return False

            if not os.access(self.input_file, os.R_OK):
                log.error(f"No read permission for input file: {self.input_file}")
                return False

            with open(self.input_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()

            delimiter = "," if "," in first_line else "\t"
            log.debug(f"Detected delimiter: {'comma' if delimiter == ',' else 'tab'}")

            chunks: List[pd.DataFrame] = []
            total_rows = 0

            with monitor_resources(interval=1.0) as stats:
                reader = pd.read_csv(
                    self.input_file, sep=delimiter, chunksize=chunk_size, iterator=True
                )
                i = 0
                while True:
                    try:
                        chunk = next(reader)
                    except StopIteration:
                        break
                    except pd.errors.EmptyDataError:
                        break

                    chunks.append(chunk)
                    total_rows += len(chunk)
                    i += 1

                    if i > 0 and i % 10 == 0:
                        log.info(
                            f"Read {i} chunks, {total_rows} rows so far. Memory: {stats['max_memory']:.1f}%"
                        )

                    if stats.get("max_memory", 0) > 85.0:
                        log.warn(
                            f"High memory usage detected ({stats['max_memory']:.1f}%). Reducing chunk size."
                        )
                        chunk_size = max(1000, chunk_size // 2)
                        log.info(f"Adjusted chunk size to {chunk_size}")
                        reader = pd.read_csv(
                            self.input_file,
                            sep=delimiter,
                            chunksize=chunk_size,
                            iterator=True,
                        )
                        rows_to_skip = total_rows
                        if rows_to_skip > 0:
                            skipped = 0
                            while skipped < rows_to_skip:
                                try:
                                    _ = next(reader)
                                    skipped += len(_)
                                except StopIteration:
                                    break

            if not chunks:
                log.error("No data read from CSV file")
                return False

            self.data = pd.concat(chunks, ignore_index=True)
            log.info(f"Combined CSV data: {self.data.shape} rows×columns")
            return True

        except pd.errors.EmptyDataError:
            log.error("The CSV file is empty")
            return False
        except pd.errors.ParserError as e:
            log.error(f"CSV parsing error: {e}")
            log.info("Check if the file format is correct")
            return False
        except Exception as e:
            log.error(f"Error reading CSV file: {e}")
            return False

    def _ensure_chromosome_column(self) -> bool:
        if self.data is None:
            log.error("No data loaded to ensure chromosome column")
            return False
        if "chromosome" in self.data.columns:
            return True
        if "CGID" in self.data.columns or any(
            "cg" in str(col) for col in self.data.columns
        ):
            if self.chip is None:
                log.error("Methylation data detected but no chip parameter provided")
                log.info("Use --chip 450k or --chip EPIC for methylation data")
                return False
            return self._add_methylation_chromosome_column()
        log.error("Cannot determine data type.")
        log.error(
            "Please ensure data has 'chromosome' column or specify --chip for methylation data"
        )
        return False

    def _add_methylation_chromosome_column(self) -> bool:
        if self.data is None:
            log.error("No data loaded to add chromosome column")
            return False
        try:
            log.info(f"Adding chromosome column for {self.chip}")
            if self.chip not in ["450k", "EPIC"]:
                log.error(f"Invalid chip: {self.chip}. Must be '450k' or 'EPIC'")
                return False
            chr_data = self._get_manifest_chromosome_data(self.data["CGID"])
            self.data = self.data.merge(chr_data, on="CGID", how="left")
            condition1 = "chromosome" not in self.data.columns
            condition2 = self.data["chromosome"].isna().all()
            if condition1 or condition2:
                log.warn(
                    "No chromosome mappings found; assigning to chr1 as placeholder"
                )
                self.data["chromosome"] = 1
            else:
                self.data = self.data.dropna(subset=["chromosome"])
                self.data = self.data[
                    ~self.data["chromosome"].astype(str).isin(["nan", "", "0"])
                ]
                self.data = self.data[self.data["chromosome"] != 0]
                self.data["chromosome"] = self.data["chromosome"].astype(int)
            cols = list(self.data.columns)
            cols.insert(1, cols.pop(cols.index("chromosome")))
            self.data = self.data[cols]
            log.info(f"Added chromosome column, final shape: {self.data.shape}")
            return True
        except Exception as e:
            log.error(f"Error adding chromosome column: {e}")
            return False

    def _get_manifest_chromosome_data(
        self, probes: Union[pd.Series, List[str]]
    ) -> pd.DataFrame:
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                manifest = self._download_and_process_manifest(tmpdirname)
                probes_df = (
                    probes.to_frame(name="CGID")
                    if isinstance(probes, pd.Series)
                    else pd.DataFrame({"CGID": probes})
                )
                return pd.merge(probes_df, manifest, on="CGID", how="left")
        except Exception as e:
            log.error(f"Error in manifest processing: {e}")
            raise

    def _download_and_process_manifest(self, tmpdirname: str) -> pd.DataFrame:
        log.debug(f"Downloading manifest for {self.chip}")
        if self.chip == "450k":
            file_name = "humanmethylation450_15017482_v1-2.csv"
            file_url = "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/"
            complete_url = f"{file_url}{file_name}"
            download_config = (
                complete_url,
                os.path.join(tmpdirname, "450k_manifest_v1.2.csv"),
                "Downloading 450k manifest",
            )
            csv_file = os.path.join(tmpdirname, "450k_manifest_v1.2.csv")
            cols = ["IlmnID", "CHR"]
            rename_map = {"IlmnID": "CGID", "CHR": "chromosome"}
        elif self.chip == "EPIC":
            file_name = "infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip"
            file_url = (
                "https://webdata.illumina.com/downloads/productfiles/methylationEPIC/"
            )
            complete_url = f"{file_url}{file_name}"
            download_config = (
                complete_url,
                os.path.join(tmpdirname, "EPIC_array_v1.B5.csv.zip"),
                "Downloading EPIC manifest",
            )
            csv_file = os.path.join(tmpdirname, "EPIC_array_v1.B5.csv")
            if self.hg == "hg38":
                cols = ["IlmnID", "CHR_hg38"]
                rename_map = {"IlmnID": "CGID", "CHR_hg38": "chromosome"}
            elif self.hg == "hg19":
                cols = ["IlmnID", "CHR"]
                rename_map = {"IlmnID": "CGID", "CHR": "chromosome"}
            else:
                raise ValueError(f"Invalid genome version: {self.hg}")
        DownloadAndExtract([download_config])
        manifest = pd.read_csv(
            csv_file, sep=",", comment="#", skiprows=7, low_memory=False
        )
        manifest = manifest.loc[:, cols].rename(columns=rename_map)
        manifest.dropna(subset=["chromosome"], inplace=True)
        manifest["chromosome"] = manifest["chromosome"].apply(
            lambda x: re.sub(r"^chr", "", str(x)) if isinstance(x, str) else x
        )
        autosomal_chromosomes = [str(i) for i in range(1, 23)]
        manifest = manifest[manifest["chromosome"].isin(autosomal_chromosomes)]
        manifest = manifest[manifest["chromosome"].apply(lambda x: str(x).isdigit())]
        manifest["chromosome"] = manifest["chromosome"].astype(int)
        manifest = manifest[manifest["chromosome"] > 0]
        return manifest.drop_duplicates().reset_index(drop=True)

    def _save_csv_to_hdf5(self) -> bool:
        try:
            with h5py.File(self.output_file, "w", libver="latest") as hdf5_file:
                hdf5_file.swmr_mode = True
                log.info(f"Created HDF5 file: {self.output_file}")
                if self.data is None:
                    log.error("No data available to save to HDF5")
                    return False
                id_column = "CGID" if "CGID" in self.data.columns else "RSID"
                chromosomes: List[int] = []
                autosomal_chromosomes = [str(i) for i in range(1, 23)]
                for chrom in self.data["chromosome"].unique():
                    try:
                        chrom_str = str(chrom)
                        if chrom_str in autosomal_chromosomes:
                            chromosomes.append(int(chrom))
                        else:
                            log.warn(f"Skipping non-autosomal chromosome: {chrom}")
                    except (ValueError, TypeError):
                        log.warn(f"Non-integer chromosome: {chrom}")
                if not chromosomes:
                    log.error("No valid autosomal chromosomes found in data")
                    return False
                memory_info = SystemUtils.get_memory_info()
                available_memory_gb = memory_info.get("available_gb", 8.0)

                cores_by_memory = max(1, int(available_memory_gb / 2))
                cores_by_cpu = SystemUtils.get_optimal_cores(reserve_cores=1)

                max_workers = min(cores_by_memory, cores_by_cpu)

                condition1 = self.data is not None
                condition2 = self.data.shape[0] * self.data.shape[1] > 1e8
                if condition1 and condition2:
                    max_workers = min(max_workers, 4)
                    log.info(
                        f"Large dataset detected - limiting to {max_workers} workers"
                    )
                else:
                    log.info(f"Using {max_workers} workers for parallel processing")

                def process_chromosome(chromosome: int) -> None:
                    chrom_group = hdf5_file.create_group(f"CHR{chromosome}")
                    chrom_data = self.data[self.data["chromosome"] == chromosome]
                    data_values = chrom_data.drop(
                        columns=["chromosome", id_column]
                    ).values
                    ids = chrom_data[id_column].values
                    if id_column == "CGID":
                        data_values = data_values.astype(np.float32).T
                        chrom_group.create_dataset(
                            AliasUtils.get_aliases("Methylation")[0],
                            data=data_values,
                            dtype="float32",
                            **self.compression_params,
                        )
                        chrom_group.create_dataset(
                            AliasUtils.get_aliases("ProbeList")[0],
                            data=ids,
                            dtype=h5py.string_dtype(),
                            **self.compression_params,
                        )
                    else:
                        data_values = data_values.astype(np.int8)
                        chrom_group.create_dataset(
                            AliasUtils.get_aliases("Genotype")[0],
                            data=data_values,
                            dtype=np.int8,
                            **self.compression_params,
                        )
                        chrom_group.create_dataset(
                            AliasUtils.get_aliases("RSID")[0],
                            data=ids,
                            dtype=h5py.string_dtype(),
                            **self.compression_params,
                        )
                        chrom_group.create_dataset(
                            AliasUtils.get_aliases("BP")[0],
                            data=np.arange(len(ids)),
                            dtype=np.int32,
                            **self.compression_params,
                        )
                        chrom_group.create_dataset(
                            AliasUtils.get_aliases("A1")[0],
                            data=np.array(["A"] * len(ids), dtype="S1"),
                            **self.compression_params,
                        )
                        chrom_group.create_dataset(
                            AliasUtils.get_aliases("A2")[0],
                            data=np.array(["G"] * len(ids), dtype="S1"),
                            **self.compression_params,
                        )
                    gc.collect()

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    list(
                        tqdm(
                            executor.map(process_chromosome, chromosomes),
                            total=len(chromosomes),
                            desc="Saving chromosomes",
                        )
                    )
                metadata_group = hdf5_file.create_group("/metadata")
                sample_list = self.data.drop(
                    columns=[id_column, "chromosome"]
                ).columns.values
                if id_column == "CGID":
                    metadata_group.create_dataset(
                        AliasUtils.get_aliases("SampleList")[0],
                        data=sample_list,
                        dtype=h5py.string_dtype(),
                        **self.compression_params,
                    )
                else:
                    metadata_group.create_dataset(
                        AliasUtils.get_aliases("IID")[0],
                        data=sample_list,
                        dtype=h5py.string_dtype(),
                        **self.compression_params,
                    )
                    metadata_group.create_dataset(
                        AliasUtils.get_aliases("FID")[0],
                        data=sample_list,
                        dtype=h5py.string_dtype(),
                        **self.compression_params,
                    )
                    metadata_group.create_dataset(
                        AliasUtils.get_aliases("Father")[0],
                        data=np.array(["0"] * len(sample_list), dtype="S50"),
                        **self.compression_params,
                    )
                    metadata_group.create_dataset(
                        AliasUtils.get_aliases("Mother")[0],
                        data=np.array(["0"] * len(sample_list), dtype="S50"),
                        **self.compression_params,
                    )
                    metadata_group.create_dataset(
                        AliasUtils.get_aliases("Sex")[0],
                        data=np.zeros(len(sample_list), dtype=np.int8),
                        **self.compression_params,
                    )
                    metadata_group.create_dataset(
                        AliasUtils.get_aliases("Phenotype")[0],
                        data=np.zeros(len(sample_list), dtype=np.int8),
                        **self.compression_params,
                    )
            log.info("HDF5 file saved successfully")
            return True
        except Exception as e:
            log.error(f"Error saving to HDF5: {e}")
            return False

    def _read_hdf5_methylation_data(self) -> bool:
        try:
            log.info(f"Opening HDF5 file: {self.input_file}")
            self.h5_file = h5py.File(self.input_file, "r")
            metadata_key = AliasUtils.find_keys(self.h5_file, "Metadata")
            if not metadata_key:
                log.error("HDF5 file does not contain required metadata")
                return False
            sample_key = AliasUtils.find_keys(self.h5_file[metadata_key], "SampleList")
            if not sample_key:
                log.error(f"No sample list found in /{metadata_key}")
                return False
            sample_list = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in self.h5_file[metadata_key][sample_key][:]
            ]
            n_samples = len(sample_list)
            log.info(f"Found {n_samples} samples in metadata")
            chr_list: List[str] = []
            for key in self.h5_file.keys():
                metadata_aliases = AliasUtils.get_aliases("Metadata")
                if key.lower() in [alias.lower() for alias in metadata_aliases]:
                    continue
                base_key = AliasUtils.strip_numeric_suffix(key)
                chr_aliases = AliasUtils.get_aliases("CHR")
                if base_key.lower() in [alias.lower() for alias in chr_aliases]:
                    chr_list.append(key)
            if not chr_list:
                log.error("No chromosomes found in HDF5 file")
                return False
            log.info("Processing chromosomes from HDF5")
            all_data: List[pd.DataFrame] = []
            for chr_name in tqdm(sorted(chr_list), desc="Processing chromosomes"):
                chr_group = self.h5_file[chr_name]
                methylation_key = AliasUtils.find_keys(chr_group, "Methylation")
                probelist_key = AliasUtils.find_keys(chr_group, "ProbeList")
                if not methylation_key or not probelist_key:
                    log.warn(
                        f"Skipping {chr_name}: missing methylation data or probe list"
                    )
                    continue
                betas = chr_group[methylation_key][:]
                probe_list = [
                    p.decode() if isinstance(p, bytes) else str(p)
                    for p in chr_group[probelist_key][:]
                ]
                log.debug(
                    f"{chr_name} methylation shape: {betas.shape}, expected samples: {n_samples}"
                )
                if betas.shape[0] == n_samples:
                    betas = betas.T
                elif betas.shape[1] == n_samples:
                    pass
                else:
                    log.error(f"Dimension mismatch in {chr_name}:")
                    log.error(
                        f"methylation data shape {betas.shape} doesn't match {n_samples} samples"
                    )
                    continue
                df = pd.DataFrame(betas, columns=sample_list, index=probe_list)
                df = df.reset_index().rename(columns={"index": "CGID"})
                all_data.append(df)
            if not all_data:
                log.error("No data processed from any chromosome")
                return False
            self.data = pd.concat(all_data, ignore_index=True)
            log.info(f"Combined HDF5 data: {self.data.shape}")
            return True
        except Exception as e:
            log.error(f"Error reading HDF5 file: {e}")
            return False

    def _read_hdf5_genotype_data(self) -> bool:
        try:
            log.info(f"Opening HDF5 file for genotype data: {self.input_file}")
            self.h5_file = h5py.File(self.input_file, "r")
            metadata_key = AliasUtils.find_keys(self.h5_file, "Metadata")
            if not metadata_key:
                log.error("HDF5 file does not contain required metadata")
                return False
            metadata = self.h5_file[metadata_key]
            iid_key = AliasUtils.find_keys(metadata, "IID")
            if not iid_key:
                log.error(f"No IID information found in /{metadata_key}")
                return False
            log.debug(f"Metadata keys: {list(self.h5_file[metadata_key].keys())}")
            sample_ids = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in metadata[iid_key][:]
            ]
            n_samples = len(sample_ids)
            log.debug(f"Found {n_samples} samples in metadata")
            self.fam = pd.DataFrame(
                {
                    "fid": sample_ids,
                    "iid": sample_ids,
                    "father": ["0"] * n_samples,
                    "mother": ["0"] * n_samples,
                    "sex": (
                        [0] * n_samples if "sex" not in metadata else metadata["sex"][:]
                    ),
                    "phenotype": (
                        [0] * n_samples
                        if "phenotype" not in metadata
                        else metadata["phenotype"][:]
                    ),
                }
            )
            bim_data: List[Dict[str, Any]] = []
            chromosomes: List[str] = []
            for key in self.h5_file.keys():
                metadata_aliases = AliasUtils.get_aliases("Metadata")
                if key.lower() in [alias.lower() for alias in metadata_aliases]:
                    continue
                base_key = AliasUtils.strip_numeric_suffix(key)
                chr_aliases = AliasUtils.get_aliases("CHR")
                if base_key.lower() in [alias.lower() for alias in chr_aliases]:
                    chromosomes.append(key)
            log.debug(f"Found chromosomes: {chromosomes}")
            for chr_name in sorted(chromosomes):
                chr_group = self.h5_file[chr_name]
                match = re.search(r"(\d+)", chr_name)
                chrom_num = match.group(1) if match else "1"
                log.debug(f"Processing chromosome {chr_name}")
                snp_key = AliasUtils.find_keys(chr_group, "RSID")
                pos_key = AliasUtils.find_keys(chr_group, "BP")
                a1_key = AliasUtils.find_keys(chr_group, "A1")
                a2_key = AliasUtils.find_keys(chr_group, "A2")
                genotype_key = AliasUtils.find_keys(chr_group, "Genotype")
                if not all([snp_key, pos_key, a1_key, a2_key]):
                    log.warn(
                        f"Skipping chr{chr_name}: missing required SNP metadata (keys: {list(chr_group.keys())})"
                    )
                    continue
                snp_names = [
                    s.decode() if isinstance(s, bytes) else str(s)
                    for s in chr_group[snp_key][:]
                ]
                positions = chr_group[pos_key][:]
                a1_alleles = [
                    s.decode() if isinstance(s, bytes) else str(s)
                    for s in chr_group[a1_key][:]
                ]
                a2_alleles = [
                    s.decode() if isinstance(s, bytes) else str(s)
                    for s in chr_group[a2_key][:]
                ]
                log.debug(
                    f"chr{chr_name}: {len(snp_names)} SNPs, {len(positions)} positions"
                )
                if genotype_key:
                    genotypes = chr_group[genotype_key][:]
                    log.debug(f"chr{chr_name} genotype shape: {genotypes.shape}")
                    if genotypes.shape[1] != n_samples:
                        log.warn(
                            f"Dimension mismatch in chr{chr_name}: expected {n_samples} samples"
                        )
                        log.warn(f"Got {genotypes.shape[1]} samples in genotype data")
                        continue
                for i, snp in enumerate(snp_names):
                    bim_data.append(
                        {
                            "CHR": chrom_num,
                            "RSID": snp,
                            "CM": 0,
                            "BP": positions[i],
                            "A1": a1_alleles[i],
                            "A2": a2_alleles[i],
                        }
                    )
            if not bim_data:
                log.error("No valid genotype data found in any chromosome")
                return False
            self.bim = pd.DataFrame(bim_data)
            log.info(
                f"Read genotype data: {len(self.fam)} samples, {len(self.bim)} variants"
            )
            return True
        except Exception as e:
            log.error(f"Error reading HDF5 genotype data: {e}")
            return False

    def _read_plink_metadata(self) -> bool:
        try:
            log.info(f"Reading PLINK files with prefix: {self.input_file}")
            bim_file = f"{self.input_file}.bim"
            fam_file = f"{self.input_file}.fam"
            for file_path in [bim_file, fam_file]:
                if not os.path.exists(file_path):
                    log.error(f"PLINK file not found: {file_path}")
                    return False
            self.bim = pd.read_csv(
                bim_file,
                sep="\t",
                header=None,
                names=["CHR", "RSID", "CM", "BP", "A1", "A2"],
            )
            with open(fam_file, "r") as f:
                first_line = f.readline().strip()
                separator = "\t" if "\t" in first_line else r"\s+"
            self.fam = pd.read_csv(
                fam_file,
                sep=separator,
                header=None,
                names=["fid", "iid", "father", "mother", "sex", "phenotype"],
                engine="python",
            )
            log.info(f"Read {len(self.bim)} variants and {len(self.fam)} samples")
            return True
        except Exception as e:
            log.error(f"Error reading PLINK files: {e}")
            return False

    def _read_chromosome_genotypes(
        self, chromosome: Union[int, str]
    ) -> Optional[np.ndarray]:
        try:
            bed_file = f"{self.input_file}.bed"
            if not os.path.exists(bed_file):
                log.error(f"BED file not found: {bed_file}")
                return None
            chrom_indices = self.bim["CHR"] == chromosome
            variant_indices = np.where(chrom_indices)[0]
            n_variants = len(variant_indices)
            n_samples = len(self.fam)
            if n_variants == 0:
                return np.array([]).reshape(0, n_samples)
            bytes_per_variant = int(np.ceil(n_samples / 4))
            with open(bed_file, "rb") as f:
                header = f.read(3)
                if header != b"\x6c\x1b\x01":
                    log.error("Invalid BED file format")
                    return None
            genotypes = np.zeros((n_variants, n_samples), dtype=np.int8)
            lookup = np.array([0, -1, 1, 2], dtype=np.int8)
            with open(bed_file, "rb") as f:
                f.seek(3)
                for i, global_variant_idx in enumerate(variant_indices):
                    f.seek(3 + global_variant_idx * bytes_per_variant)
                    variant_bytes = f.read(bytes_per_variant)
                    for byte_idx, byte_val in enumerate(variant_bytes):
                        for bit_pair in range(4):
                            sample_idx = byte_idx * 4 + bit_pair
                            if sample_idx >= n_samples:
                                break
                            genotype_code = (byte_val >> (bit_pair * 2)) & 0x03
                            genotypes[i, sample_idx] = lookup[genotype_code]
            return genotypes
        except Exception as e:
            log.error(f"Error reading chromosome genotypes: {e}")
            return None

    def _save_plink_to_hdf5(self) -> bool:
        try:
            with h5py.File(self.output_file, "w") as hf:
                log.info("Creating HDF5 file structure for genotype data")
                metadata_group = hf.create_group("Metadata")

                def safe_string_convert(series: pd.Series) -> List[str]:
                    result: List[str] = []
                    for val in series:
                        if pd.isna(val):
                            result.append("missing")
                        else:
                            result.append(str(val))
                    return result

                def safe_numeric_convert(
                    series: pd.Series, dtype: Any = np.int8
                ) -> np.ndarray:
                    result: List[int] = []
                    for val in series:
                        if pd.isna(val):
                            result.append(-9)
                        else:
                            try:
                                result.append(dtype(val))
                            except (ValueError, TypeError):
                                result.append(-9)
                    return np.array(result, dtype=dtype)

                metadata_datasets = {
                    AliasUtils.get_aliases("FID")[0]: {
                        "data": safe_string_convert(self.fam["fid"]),
                        "dtype": "S50",
                    },
                    AliasUtils.get_aliases("IID")[0]: {
                        "data": safe_string_convert(self.fam["iid"]),
                        "dtype": "S50",
                    },
                    AliasUtils.get_aliases("Father")[0]: {
                        "data": safe_string_convert(self.fam["father"]),
                        "dtype": "S50",
                    },
                    AliasUtils.get_aliases("Mother")[0]: {
                        "data": safe_string_convert(self.fam["mother"]),
                        "dtype": "S50",
                    },
                    AliasUtils.get_aliases("Sex")[0]: {
                        "data": safe_numeric_convert(self.fam["sex"], np.int8),
                        "dtype": np.int8,
                    },
                    AliasUtils.get_aliases("Phenotype")[0]: {
                        "data": safe_numeric_convert(self.fam["phenotype"], np.int8),
                        "dtype": np.int8,
                    },
                }
                for name, params in metadata_datasets.items():
                    metadata_group.create_dataset(
                        name,
                        data=params["data"],
                        dtype=params["dtype"],
                        **self.compression_params,
                    )
                chromosomes = sorted(self.bim["CHR"].unique())
                log.info(f"Processing {len(chromosomes)} chromosomes")
                valid_chromosomes = 0
                for chrom in tqdm(chromosomes, desc="Processing chromosomes"):
                    chrom_group = hf.create_group(f"CHR{chrom}")
                    chrom_variants = self.bim[self.bim["CHR"] == chrom]
                    n_variants = len(chrom_variants)
                    if n_variants == 0:
                        continue
                    snp_names = [str(name) for name in chrom_variants["RSID"].values]
                    chrom_group.create_dataset(
                        AliasUtils.get_aliases("RSID")[0],
                        data=snp_names,
                        dtype="S50",
                        **self.compression_params,
                    )
                    positions = chrom_variants["BP"].values.astype(np.int32)
                    chrom_group.create_dataset(
                        AliasUtils.get_aliases("BP")[0],
                        data=positions,
                        dtype=np.int32,
                        **self.compression_params,
                    )
                    a1_alleles = [str(allele) for allele in chrom_variants["A1"].values]
                    chrom_group.create_dataset(
                        AliasUtils.get_aliases("A1")[0],
                        data=a1_alleles,
                        dtype="S10",
                        **self.compression_params,
                    )
                    a2_alleles = [str(allele) for allele in chrom_variants["A2"].values]
                    chrom_group.create_dataset(
                        AliasUtils.get_aliases("A2")[0],
                        data=a2_alleles,
                        dtype="S10",
                        **self.compression_params,
                    )
                    genotypes = self._read_chromosome_genotypes(chrom)
                    if genotypes is not None and genotypes.size > 0:
                        chrom_group.create_dataset(
                            AliasUtils.get_aliases("Genotype")[0],
                            data=genotypes,
                            dtype=np.int8,
                            **self.compression_params,
                        )
                        valid_chromosomes += 1
                if valid_chromosomes == 0:
                    log.error("No valid genotype data processed for any chromosome")
                    return False
            log.info("PLINK to HDF5 conversion completed")
            return True
        except Exception as e:
            log.error(f"Error saving PLINK to HDF5: {e}")
            return False

    def _save_to_plink(self) -> bool:
        try:
            log.info("Converting HDF5 to PLINK format")
            base_name = self.output_file
            bim_file = f"{base_name}.bim"
            fam_file = f"{base_name}.fam"
            bed_file = f"{base_name}.bed"
            log.info("Writing FAM file")
            self.fam.to_csv(fam_file, sep="\t", header=False, index=False)
            log.info("Writing BIM file")
            self.bim.to_csv(bim_file, sep="\t", header=False, index=False)
            log.info("Writing BED file")
            n_samples = len(self.fam)
            bytes_per_variant = int(np.ceil(n_samples / 4))
            bim_sorted = self.bim.sort_values(by=["CHR", "BP"]).reset_index(drop=True)
            chromosomes = sorted(
                bim_sorted["CHR"].unique(),
                key=lambda x: int(x) if str(x).isdigit() else float("inf"),
            )
            memory_info = SystemUtils.get_memory_info()
            available_memory_gb = memory_info.get("available_gb", 8.0)
            max_workers = min(
                SystemUtils.get_optimal_cores(reserve_cores=1),
                int(available_memory_gb),
            )
            log.info(f"Using {max_workers} workers for BED writing")
            total_variants = len(bim_sorted)
            bed_size = 3 + total_variants * bytes_per_variant
            bed_memmap = np.memmap(
                bed_file, dtype=np.uint8, mode="w+", shape=(bed_size,)
            )
            bed_memmap[:3] = np.frombuffer(b"\x6c\x1b\x01", dtype=np.uint8)
            offset = 3

            def write_bed_chunk(
                chr_name: Union[int, str],
            ) -> Tuple[Optional[np.ndarray], int]:
                chr_key = None
                possible_keys = [
                    f"CHR{chr_name}",
                    f"chr{chr_name}",
                    f"chromosome{chr_name}",
                    str(chr_name),
                ]
                for possible_key in possible_keys:
                    if possible_key in self.h5_file:
                        chr_key = possible_key
                        break
                if not chr_key:
                    log.warn(f"Skipping chromosome {chr_name}: not found in HDF5 file")
                    log.debug(f"Available keys: {list(self.h5_file.keys())}")
                    return None, 0
                chr_group = self.h5_file[chr_key]
                genotype_key = AliasUtils.find_keys(chr_group, "Genotype")
                if not genotype_key:
                    log.warn(f"Skipping chr{chr_name}: no genotype data")
                    return None, 0
                genotypes = chr_group[genotype_key][:]
                chr_variants = bim_sorted[bim_sorted["CHR"] == chr_name]
                if len(chr_variants) != genotypes.shape[0]:
                    log.error(f"Mismatch in variant count for chr{chr_name}")
                    log.error(
                        f"BIM has {len(chr_variants)}, genotypes has {genotypes.shape[0]}"
                    )
                    return None, 0
                if genotypes.shape[1] != n_samples:
                    log.error(
                        f"Dimension mismatch in chr{chr_name}: expected {n_samples} samples, got {genotypes.shape[1]}"
                    )
                    return None, 0
                chunk_bytes = _pack_genotypes_to_bed(genotypes, n_samples)
                return chunk_bytes.ravel(), chunk_bytes.size

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                for chunk_bytes, chunk_size in tqdm(
                    executor.map(write_bed_chunk, chromosomes),
                    total=len(chromosomes),
                    desc="Writing genotypes",
                ):
                    if chunk_bytes is None:
                        return False
                    offset_increment = offset + chunk_size
                    bed_memmap[offset:offset_increment] = chunk_bytes
                    offset += chunk_size
            log.info("HDF5 to PLINK conversion completed")
            return True
        except Exception as e:
            log.error(f"Error saving to PLINK: {e}")
            return False

    def _process_transpose(self) -> bool:
        if not self.transpose:
            return True
        try:
            log.info("Transposing data")
            if self.data is None:
                log.error("No data available to transpose")
                return False
            id_column: Optional[str] = None
            probe_ids: Optional[List[str]] = None
            probe_id_aliases = AliasUtils.get_aliases("CGID")
            for alias in probe_id_aliases:
                if alias in self.data.columns:
                    id_column = "CGID"
                    probe_ids = self.data[alias].tolist()
                    break
            if id_column is None:
                snp_id_aliases = AliasUtils.get_aliases("RSID")
                for alias in snp_id_aliases:
                    if alias in self.data.columns:
                        id_column = "RSID"
                        probe_ids = self.data[alias].tolist()
                        break
            if id_column is None:
                log.error("No probe ID or SNP ID column found for transposition")
                log.debug(f"Available columns: {list(self.data.columns)}")
                log.debug(f"Expected probe ID aliases: {probe_id_aliases}")
                log.debug(f"Expected SNP ID aliases: {snp_id_aliases}")
                return False
            sample_ids = [col for col in self.data.columns if col != id_column]
            data_matrix = self.data[sample_ids].values
            transposed_matrix = data_matrix.T
            transposed_data = pd.DataFrame(
                transposed_matrix, index=sample_ids, columns=probe_ids
            )
            self.data = transposed_data.reset_index().rename(
                columns={"index": "sample_id"}
            )
            log.info(f"Data transposed, new shape: {self.data.shape}")
            return True
        except Exception as e:
            log.error(f"Error transposing data: {e}")
            return False

    def _save_to_csv(self) -> bool:
        try:
            log.info(f"Writing data to CSV: {self.output_file}")
            if self.data is None:
                log.error("No data available to save to CSV")
                return False
            self.data.to_csv(self.output_file, index=False)
            log.info("CSV file saved successfully")
            return True
        except Exception as e:
            log.error(f"Error saving to CSV: {e}")
            return False

    def _remove_existing_output(self) -> bool:
        try:
            if self.output_format == "plink":
                for ext in [".bed", ".bim", ".fam"]:
                    file_path = f"{self.output_file}{ext}"
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        log.info(f"Removed existing file: {file_path}")
            else:
                if os.path.exists(self.output_file):
                    os.remove(self.output_file)
                    log.info(f"Removed existing output file: {self.output_file}")
            return True
        except Exception as e:
            log.error(f"Error removing existing files: {e}")
            return False

    def _cleanup(self) -> None:
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception as e:
                log.warn(f"Error closing HDF5 file: {e}")

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
                prefix=f"convert_{self.input_format}_to_{self.output_format}",
                max_age_hours=24,
                dry_run=False,
            )

            condition1 = cleanup_result["dirs_deleted"] > 0
            condition2 = cleanup_result["files_deleted"] > 0

            if condition1 or condition2:
                log.debug(
                    f"Cleaned up {cleanup_result['dirs_deleted']} stale directories and "
                    f"{cleanup_result['files_deleted']} files"
                )
        except Exception as e:
            log.debug(f"Error during stale file cleanup: {e}")

        gc.collect()


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
    OptionConfig(
        flags=["-c", "--chip"],
        type=str,
        default=None,
        required=False,
        choices=["450K", "EPIC"],
    ),
    OptionConfig(
        flags=["-g", "--hg"],
        type=str,
        default="hg38",
        required=False,
        choices=["hg19", "hg38"],
    ),
    OptionConfig(flags=["-t", "--transpose"], type=bool, default=False, required=False),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="ConvertCounts")
    opt = framework.run()
    converter = ConvertCounts(
        input_file=opt.input,
        output_file=opt.output,
        chip=opt.chip,
        hg=opt.hg,
        transpose=opt.transpose,
    )
    converter.convert()
