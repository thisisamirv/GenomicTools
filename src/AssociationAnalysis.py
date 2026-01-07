#!/usr/bin/env python
# Import required modules
import h5py
import numpy as np
import os
import pandas as pd
import re
import subprocess
import sys
import threading
from typing import Optional, Dict, Any, List
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.LoggingUtils import log
from utils.ParsingUtils import ParseFormula, ParseToList
from utils.SystemUtils import SystemUtils, monitor_resources


class AssociationAnalysisLauncher:
    def __init__(
        self,
        h5_file: str,
        metadata: Optional[str],
        output: str,
        formula: str,
        **kwargs,
    ) -> None:
        self.h5_file = h5_file
        self.metadata = metadata
        self.output = output
        self.sample_id = kwargs.get("sample_id")

        if self.metadata and not self.sample_id:
            self.sample_id = self._detect_sample_id_column()

        self.formula_components, self.formula = ParseFormula(
            formula, build_formula=True
        )

        requested_type = kwargs.get("analysis_type", "Auto")
        if requested_type.lower() == "auto":
            self.analysis_type = (
                "EWAS"
                if self.formula_components["data_variable"] == "Methylation"
                else "GWAS"
            )
        else:
            requested_type = requested_type.upper()
            if requested_type not in ["EWAS", "GWAS"]:
                raise ValueError(
                    "Invalid analysis_type. Must be 'Auto', 'EWAS', or 'GWAS'."
                )
            self.analysis_type = requested_type

        expected_data_var = (
            "Methylation" if self.analysis_type == "EWAS" else "Genotype"
        )
        data_variable = self.formula_components.get("data_variable")
        dependent_var = self.formula_components.get("dependent_var")

        if not data_variable:
            raise ValueError(
                "Formula must reference either 'Methylation' or 'Genotype' to indicate the data variable."
            )

        if data_variable != expected_data_var:
            raise ValueError(
                f"Data variable must be '{expected_data_var}' for {self.analysis_type}. "
                f"Found '{data_variable}' instead. Ensure your formula references the correct data stream."
            )

        if not dependent_var:
            raise ValueError("Formula did not specify a dependent variable.")

        if self.analysis_type == "EWAS":
            if dependent_var != expected_data_var:
                raise ValueError(
                    "For EWAS analyses the dependent variable must be the methylation data. "
                    f"Found '{dependent_var}'. Update your formula to use '{expected_data_var}' as the outcome."
                )
        else:
            if dependent_var == expected_data_var:
                raise ValueError(
                    "For GWAS analyses the dependent variable must be the phenotype, not the genotype data. "
                    "Update your formula so the outcome is the phenotype you want to model."
                )

        self.data_variable = data_variable
        self.dependent_var = dependent_var

        raw_stat_var = kwargs.get("var")
        requested_vars = ParseToList(raw_stat_var)

        data_variables = {"Methylation", "Genotype"}
        invalid_data_vars = [v for v in requested_vars if v in data_variables]
        if invalid_data_vars:
            raise ValueError(
                f"Cannot use data variables for statistics extraction: {', '.join(invalid_data_vars)}. "
                f"Use covariates or other formula variables instead."
            )

        covs = self.formula_components.get("covariates") or []

        if requested_vars:
            valid_vars = [v for v in requested_vars if v in covs]
            if not valid_vars and requested_vars:
                log.warn(
                    f"Requested stat var(s) not found in covariates: {', '.join(requested_vars)}"
                )
                valid_vars = covs[:1] if covs else []
        else:
            valid_vars = covs[:1] if covs else []

        self.stat_vars = valid_vars
        self.stat_var = ",".join(valid_vars)

        if self.stat_vars:
            log.info(f"Statistics will be extracted for: {', '.join(self.stat_vars)}")
        else:
            log.warn("No covariates found for statistics extraction")

        self.model = kwargs.get("model", "linear")
        self.threads = kwargs.get(
            "threads", SystemUtils.get_optimal_cores(reserve_cores=1)
        )
        self.chunk_size = kwargs.get("chunk_size", None)
        self.log_level = kwargs.get("log_level", "INFO")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.r_script_path = os.path.join(
            script_dir, "utils", f"{self.analysis_type}.R"
        )

        self.system_config = SystemUtils.load_config()
        self._configure_resources()
        self._validate_inputs()

        safe_config = SystemUtils.configure_safe_environment()
        if safe_config.get("core_dumps_disabled", False):
            log.debug("Core dumps disabled for stability")
        if safe_config.get("memory_limit_set", False):
            log.debug("Memory limits configured to prevent OOM errors")

        log.info(f"Initialized {self.analysis_type} analysis launcher")

    def _configure_resources(self) -> None:
        SystemUtils.print_system_info()

        reserve_cores = self.system_config.get("reserve_cores", 1)
        max_memory_gb = self.system_config.get("max_memory_gb", 1024)

        if not self.threads:
            self.threads = SystemUtils.get_optimal_cores(reserve_cores=reserve_cores)

        memory_info = SystemUtils.get_memory_info()
        self.available_memory_gb = min(memory_info["available_gb"], max_memory_gb)

        self.memory_per_core = min(4.0, self.available_memory_gb / max(1, self.threads))
        log.info(f"Memory allocation: {self.memory_per_core:.1f}GB per core")

        valid_resources, message = SystemUtils.validate_resources(
            cores=self.threads,
            memory_gb=self.memory_per_core * self.threads,
        )

        if not valid_resources:
            log.warn(f"Resource validation issue: {message}")
            log.warn("Analysis performance may be affected")

        if not self.chunk_size:
            try:
                num_samples = self._get_num_samples()
            except Exception as e:
                log.warn(f"Unable to determine number of samples for chunk sizing: {e}")
                num_samples = 1000

            bytes_per_entry = 4 if self.analysis_type == "EWAS" else 1
            target_memory_gb = self.available_memory_gb * 0.5 / max(1, self.threads)
            max_chunk_size = int(
                (target_memory_gb * 1024**3) / (num_samples * bytes_per_entry)
            )
            min_chunk_size = 500 if self.analysis_type == "EWAS" else 1000
            self.chunk_size = max(min_chunk_size, min(max_chunk_size, 5000))
            log.info(f"Auto-configured chunk size: {self.chunk_size}")

    def _validate_inputs(self) -> None:
        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_file}")
        if self.metadata and not os.path.exists(self.metadata):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata}")
        if self.model not in ["linear", "logistic"]:
            raise ValueError(
                f"Invalid model type '{self.model}'. Use 'linear' or 'logistic'."
            )
        if not os.path.exists(self.r_script_path):
            raise FileNotFoundError(f"R script not found: {self.r_script_path}")
        if self.metadata and not self.sample_id:
            log.warn(
                "Metadata file provided but sample_id not specified or auto-detection failed."
            )
            log.warn("This may cause issues with sample matching between data sources.")

        self._validate_formula_variables()

    def _check_system_health(self) -> bool:
        log.info("Performing system health check...")
        health = SystemUtils.check_system_health(
            min_free_disk_gb=1.0, max_cpu_percent=95.0, max_memory_percent=90.0
        )

        if health.get("status") == "critical":
            log.error("CRITICAL SYSTEM ISSUES:")
            for issue in health.get("critical", []):
                log.error(f"  - {issue}")
            return False

        if health.get("status") == "warning":
            log.warn("SYSTEM WARNINGS:")
            for issue in health.get("warnings", []):
                log.warn(f"  - {issue}")

        estimated_size_gb = self._estimate_output_size()
        log.info(f"Estimated output size: {estimated_size_gb:.2f} GB")

        output_dir = os.path.dirname(os.path.abspath(self.output))
        disk_ok, disk_message = SystemUtils.check_disk_space(
            path=output_dir, required_gb=estimated_size_gb, buffer_percent=20.0
        )

        if not disk_ok:
            log.error(f"Disk space issue: {disk_message}")
            return False

        log.info("System health check completed successfully")
        return True

    def _estimate_output_size(self) -> float:
        try:
            n_features = 0
            with h5py.File(self.h5_file, "r") as h5f:
                metadata_key = AliasUtils.find_keys(h5f, "Metadata")
                chr_groups = [key for key in h5f.keys() if key != metadata_key]

                for group in chr_groups:
                    try:
                        for key in h5f[group].keys():
                            feature_key = AliasUtils.find_keys(
                                h5f[group], "ProbeList"
                            ) or AliasUtils.find_keys(h5f[group], "CGID")
                        if feature_key:
                            n_features += len(h5f[group][feature_key])
                    except Exception as e:
                        log.debug(f"Error counting features in {group}: {e}")

            if n_features == 0:
                log.warn("Could not count features, using fallback estimates")
                n_features = 500000 if self.analysis_type == "GWAS" else 800000

            bytes_per_feature = 0

            base_cols = 4 if self.analysis_type == "EWAS" else 8
            n_covariates = len(self.formula_components.get("covariates") or [])
            interaction_term = self.formula_components.get("interaction_term")
            if interaction_term:
                n_covariates += 2

            correction_cols = 4
            bytes_per_value = 12

            total_columns = base_cols + (correction_cols * n_covariates)
            bytes_per_feature = total_columns * bytes_per_value
            total_bytes = n_features * bytes_per_feature

            size_gb = (total_bytes / (1024**3)) * 1.5

            min_size = 0.1
            return max(size_gb, min_size)

        except Exception as e:
            log.warn(f"Error estimating output size: {e}")
            if self.analysis_type == "GWAS":
                return 1.0
            else:
                return 0.5

    def _setup_temp_directory(self) -> str:
        output_dir = os.path.dirname(os.path.abspath(self.output))
        try:
            temp_dir, temp_info = SystemUtils.create_safe_tempdir(
                default_path=output_dir,
                required_gb=1.0,
                prefix=f"{self.analysis_type.lower()}_temp",
                buffer_percent=10.0,
            )
            log.info(f"Created temporary directory: {temp_dir}")

            if temp_info.get("checked_paths") and len(temp_info["checked_paths"]) > 0:
                chosen_path_info = None
                for path_info in temp_info["checked_paths"]:
                    if path_info.get("has_space", False) and "error" not in path_info:
                        chosen_path_info = path_info
                        break

                if chosen_path_info:
                    log.debug(f"Temp directory created at: {chosen_path_info['path']}")
                else:
                    log.debug(f"Temp directory created successfully at: {temp_dir}")
            else:
                log.debug(f"Temp directory created successfully at: {temp_dir}")

            return temp_dir
        except Exception as e:
            log.warn(f"Failed to create safe temp directory: {e}")
            log.warn("Using system temp directory as fallback")
            return SystemUtils.get_safe_tempdir(
                prefix=f"{self.analysis_type.lower()}_temp"
            )

    def _cleanup_temp_files(self, temp_dir: Optional[str] = None) -> None:
        if temp_dir and os.path.exists(temp_dir):
            try:
                if SystemUtils.cleanup_tempdir(temp_dir):
                    log.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                log.debug(f"Error cleaning up temp directory: {e}")

        try:
            cleanup_result = SystemUtils.cleanup_stale_temp_files(
                directories=None,
                prefix=f"{self.analysis_type.lower()}_temp",
                max_age_hours=48,
                dry_run=False,
            )

            condition1 = cleanup_result.get("dirs_deleted", 0) > 0
            condition2 = cleanup_result.get("files_deleted", 0) > 0
            if condition1 or condition2:
                log.info(
                    f"Cleaned up {cleanup_result.get('dirs_deleted', 0)} directories and "
                    f"{cleanup_result.get('files_deleted', 0)} files "
                    f"({cleanup_result.get('space_reclaimed_gb', 0.0):.2f} GB reclaimed)"
                )
        except Exception as e:
            log.debug(f"Error during temp file cleanup: {e}")

    def _validate_formula_variables(self) -> None:
        log.debug("Validating formula variables against available metadata...")

        dependent_var = self.dependent_var
        data_variable = self.data_variable

        if self.analysis_type == "EWAS":
            if dependent_var != data_variable:
                raise ValueError(
                    "For EWAS analyses the dependent variable must match the methylation data variable."
                )
        else:
            if dependent_var == data_variable:
                raise ValueError(
                    "For GWAS analyses the dependent variable cannot be the genotype data variable."
                )

        covariates = self.formula_components.get("covariates", [])
        random_effects = self.formula_components.get("random_effects")

        required_vars = set()

        if covariates:
            required_vars.update(covariates)

        if random_effects:
            if isinstance(random_effects, list):
                required_vars.update(random_effects)
            else:
                required_vars.add(random_effects)

        if not required_vars:
            log.debug("No variables to validate in formula")
            return

        metadata_columns = set()
        if self.metadata:
            try:
                meta_df = pd.read_csv(self.metadata)
                metadata_columns = set(meta_df.columns)
                log.debug(f"Metadata file columns: {metadata_columns}")
            except Exception as e:
                log.error(f"Failed to read metadata file: {e}")
                raise ValueError(
                    f"Cannot read metadata file to validate variables: {e}"
                )

        h5_metadata_vars = set()
        try:
            with h5py.File(self.h5_file, "r") as h5f:
                metadata_key = AliasUtils.find_keys(h5f, "Metadata")
                if metadata_key:
                    metadata_group = h5f[metadata_key]
                    h5_metadata_vars = set(metadata_group.keys())
                else:
                    log.debug("No metadata group found in HDF5 file")
        except Exception as e:
            log.error(f"Failed to read HDF5 metadata: {e}")
            raise ValueError(f"Cannot read HDF5 file to validate variables: {e}")

        available_vars = metadata_columns | h5_metadata_vars

        if required_vars and not self.metadata and not h5_metadata_vars:
            raise ValueError(
                f"Formula requires variables {required_vars} but no metadata file was provided "
                f"and no metadata group found in HDF5 file"
            )

        missing_vars = required_vars - available_vars
        if missing_vars:
            log.debug(f"Missing variables: {missing_vars}")
            raise ValueError(
                f"The following variables are missing: {', '.join(missing_vars)}. "
            )

        found_in = []
        if metadata_columns & required_vars:
            found_in.append(
                f"metadata file ({len(metadata_columns & required_vars)} variables)"
            )
        if h5_metadata_vars & required_vars:
            found_in.append(
                f"HDF5 metadata ({len(h5_metadata_vars & required_vars)} variables)"
            )

        log.info(f"✓ All {len(required_vars)} formula variables validated successfully")
        log.debug(f"  Found in: {', '.join(found_in)}")

        for var in sorted(required_vars):
            locations = []
            if var in metadata_columns:
                locations.append("metadata file")
            if var in h5_metadata_vars:
                locations.append("HDF5")

    def _get_num_samples(self) -> int:
        with h5py.File(self.h5_file, "r") as h5f:
            metadata_key = AliasUtils.find_keys(h5f, "Metadata")
            if metadata_key is None:
                raise ValueError("No metadata group in HDF5 file")

            sample_key_name = "SampleList" if self.analysis_type == "EWAS" else "IID"
            sample_key = AliasUtils.find_keys(h5f[metadata_key], sample_key_name)
            if sample_key is None:
                raise ValueError(f"No {sample_key_name} found in metadata group")

            return len(h5f[metadata_key][sample_key])

    def _detect_sample_id_column(self) -> Optional[str]:
        log.warn("Sample ID column not specified. Attempting automatic detection...")

        try:
            with h5py.File(self.h5_file, "r") as h5f:
                metadata_key = AliasUtils.find_keys(h5f, "Metadata")
                if metadata_key is None:
                    log.warn(
                        "No metadata group in HDF5 file, cannot auto-detect sample ID"
                    )
                    return None

                sample_key_candidates = ["SampleList", "IID"]
                sample_key = None
                for candidate in sample_key_candidates:
                    sample_key = AliasUtils.find_keys(h5f[metadata_key], candidate)
                    if sample_key:
                        break

                if not sample_key:
                    log.warn("Could not find sample list in HDF5 file")
                    return None

                sample_list = self._decode_string_array(
                    h5f[metadata_key][sample_key][:5]
                )
                log.debug(f"Sample IDs from HDF5: {sample_list}")

                if not sample_list:
                    log.warn("Empty sample list in HDF5 file")
                    return None

            try:
                meta_df = pd.read_csv(self.metadata)

                if meta_df.empty:
                    log.warn("Metadata file appears to be empty")
                    return None

                matching_cols = []
                for col in meta_df.columns:
                    col_vals = meta_df[col].astype(str).values

                    if all(sample_id in col_vals for sample_id in sample_list):
                        matching_cols.append(col)

                if matching_cols:
                    sample_id_col = matching_cols[0]
                    log.info(f"Auto-detected sample ID column: '{sample_id_col}'")
                    return sample_id_col
                else:
                    log.warn(
                        "Could not find a column containing all sample IDs in metadata file"
                    )
                    return None

            except Exception as e:
                log.error(f"Error reading metadata file: {e}")
                return None

        except Exception as e:
            log.error(f"Error auto-detecting sample ID column: {e}")
            return None

    def _decode_string_array(self, array: Any) -> List[str]:
        """Decode bytes array, detecting encoding once and stripping null padding."""
        if len(array) == 0:
            return []
        
        first = array[0]
        # Detect encoding from first element
        if isinstance(first, (bytes, np.bytes_)):
            detected_encoding = None
            for enc in ("utf-8", "latin-1", "ascii"):
                try:
                    first.decode(enc)
                    detected_encoding = enc
                    break
                except Exception:
                    continue
            
            if detected_encoding:
                try:
                    return [item.decode(detected_encoding).rstrip('\x00').strip() for item in array]
                except Exception:
                    pass
            # Fallback
            return [str(item).rstrip('\x00').strip() for item in array]
        
        # Already strings
        return [str(item).strip() for item in array]

    def _extract_h5_paths(self) -> Dict[str, Any]:
        log.info(f"Extracting HDF5 paths from {self.h5_file}")

        paths: Dict[str, Any] = {
            "metadata_group": None,
            "sample_list_name": None,
            "chrom_groups": [],
        }

        if self.analysis_type == "EWAS":
            paths.update({"probe_list_name": None, "betas_name": None})
        else:
            paths.update(
                {
                    "variant_list_name": None,
                    "geno_name": None,
                    "a1_path": None,
                    "a2_path": None,
                    "bp_path": None,
                }
            )

        with h5py.File(self.h5_file, "r") as h5f:
            metadata_aliases = AliasUtils.get_aliases("Metadata")
            for key in h5f.keys():
                if key in metadata_aliases or any(
                    alias in key.lower()
                    for alias in [m.lower() for m in metadata_aliases]
                ):
                    paths["metadata_group"] = key
                    break

            if paths["metadata_group"] is None:
                raise ValueError("No metadata group found in HDF5 file")

            meta_grp = h5f[paths["metadata_group"]]
            sample_key_name = "SampleList" if self.analysis_type == "EWAS" else "IID"
            sample_key = AliasUtils.find_keys(meta_grp, sample_key_name)
            if sample_key:
                paths["sample_list_name"] = sample_key

            metadata_group = paths["metadata_group"]
            chr_groups = [key for key in h5f.keys() if key != metadata_group]

            if self.analysis_type == "EWAS":
                valid_chr_groups = []
                # Once we find the keys in one group, reuse them for others
                probe_key_cached = None
                beta_key_cached = None
                for chr_group in chr_groups:
                    try:
                        grp = h5f[chr_group]
                        # Use cached keys if available, otherwise search
                        probe_key = probe_key_cached or AliasUtils.find_keys(
                            grp, "ProbeList"
                        ) or AliasUtils.find_keys(grp, "CGID")
                        beta_key = beta_key_cached or AliasUtils.find_keys(
                            grp, "betas"
                        ) or AliasUtils.find_keys(grp, "Methylation")
                        
                        # Verify keys exist in this group
                        if probe_key in grp and beta_key in grp:
                            valid_chr_groups.append(chr_group)
                            # Cache for subsequent groups
                            if probe_key_cached is None:
                                probe_key_cached = probe_key
                                paths["probe_list_name"] = probe_key
                            if beta_key_cached is None:
                                beta_key_cached = beta_key
                                paths["betas_name"] = beta_key
                    except Exception as e:
                        log.debug(f"Error checking group {chr_group}: {e}")
                paths["chrom_groups"] = valid_chr_groups
            else:
                valid_chr_groups = []
                # Cache keys once found
                rsid_key_cached = None
                geno_key_cached = None
                a1_key_cached = None
                a2_key_cached = None
                bp_key_cached = None
                
                for chr_group in chr_groups:
                    try:
                        grp = h5f[chr_group]
                        rsid_key = rsid_key_cached or AliasUtils.find_keys(
                            grp, "RSID"
                        ) or AliasUtils.find_keys(grp, "SNP")
                        geno_key = geno_key_cached or AliasUtils.find_keys(
                            grp, "Genotype"
                        ) or AliasUtils.find_keys(grp, "genotypes")
                        
                        if rsid_key in grp and geno_key in grp:
                            valid_chr_groups.append(chr_group)
                            # Cache keys on first success
                            if rsid_key_cached is None:
                                rsid_key_cached = rsid_key
                                paths["variant_list_name"] = rsid_key
                            if geno_key_cached is None:
                                geno_key_cached = geno_key
                                paths["geno_name"] = geno_key
                            if a1_key_cached is None:
                                a1_key = AliasUtils.find_keys(grp, "A1") or AliasUtils.find_keys(grp, "REF")
                                if a1_key and a1_key in grp:
                                    a1_key_cached = a1_key
                                    paths["a1_path"] = a1_key
                            if a2_key_cached is None:
                                a2_key = AliasUtils.find_keys(grp, "A2") or AliasUtils.find_keys(grp, "ALT")
                                if a2_key and a2_key in grp:
                                    a2_key_cached = a2_key
                                    paths["a2_path"] = a2_key
                            if bp_key_cached is None:
                                bp_key = AliasUtils.find_keys(grp, "BP") or AliasUtils.find_keys(grp, "POS")
                                if bp_key and bp_key in grp:
                                    bp_key_cached = bp_key
                                    paths["bp_path"] = bp_key
                    except Exception as e:
                        log.debug(f"Error checking group {chr_group}: {e}")
                paths["chrom_groups"] = valid_chr_groups

        log.info(f"Found {len(paths['chrom_groups'])} valid chromosome groups")
        return paths

    def _detect_data_orientation(self) -> str:
        log.info("Detecting data orientation in HDF5 file...")

        try:
            with h5py.File(self.h5_file, "r") as h5f:
                metadata_key = AliasUtils.find_keys(h5f, "Metadata")
                if not metadata_key:
                    raise ValueError("Metadata group not found")

                sample_key_name = (
                    "SampleList" if self.analysis_type == "EWAS" else "IID"
                )
                sample_key = AliasUtils.find_keys(h5f[metadata_key], sample_key_name)
                if not sample_key:
                    raise ValueError(f"{sample_key_name} not found")
                sample_count = len(h5f[metadata_key][sample_key])

                chr_groups = [key for key in h5f.keys() if key != metadata_key]
                if not chr_groups:
                    raise ValueError("No data groups found")

                first_group = chr_groups[0]
                if self.analysis_type == "EWAS":
                    data_key = AliasUtils.find_keys(
                        h5f[first_group], "betas"
                    ) or AliasUtils.find_keys(h5f[first_group], "Methylation")
                else:
                    data_key = AliasUtils.find_keys(h5f[first_group], "Genotype")

                if not data_key:
                    raise ValueError(f"No data matrix found in {first_group}")

                data_shape = h5f[first_group][data_key].shape
                log.debug(f"Data shape: {data_shape}, Sample count: {sample_count}")

                if len(data_shape) < 2:
                    return "unknown"

                if data_shape[0] == sample_count:
                    return "samples_as_rows"
                elif data_shape[1] == sample_count:
                    return "markers_as_rows"
                else:
                    if abs(data_shape[0] - sample_count) < abs(
                        data_shape[1] - sample_count
                    ):
                        return "samples_as_rows"
                    else:
                        return "markers_as_rows"

        except Exception as e:
            log.warn(f"Failed to detect data orientation: {e}")
            return "unknown"

    def run(self) -> str:
        if not self._check_system_health():
            log.error("System health check failed - analysis may be unstable")

        log.info(f"Starting {self.analysis_type} analysis using R")

        temp_dir = self._setup_temp_directory()

        h5_paths = self._extract_h5_paths()
        dependent_var = self.dependent_var

        data_orientation = self._detect_data_orientation()

        if data_orientation == "samples_as_rows":
            data_orientation = "markers_as_rows"
        elif data_orientation == "markers_as_rows":
            data_orientation = "samples_as_rows"

        log.debug(f"Data orientation for R: {data_orientation}")
        log.debug(f"Dependent variable: {dependent_var}")
        log.debug(f"Data variable: {self.data_variable}")
        log.debug(f"Covariates: {self.formula_components['covariates']}")
        log.debug(f"Interaction: {self.formula_components['interaction_term']}")
        log.debug(f"Random effects: {self.formula_components['random_effects']}")

        r_script_path = os.path.abspath(self.r_script_path)
        log.debug(f"Metadata group: {h5_paths['metadata_group']}")

        log.info(f"Running {self.analysis_type} with the following parameters:")
        log.info(f"  Data file: {self.h5_file}")
        log.info(f"  Output file: {self.output}")
        log.info(f"  Metadata file: {self.metadata if self.metadata else 'NONE'}")
        log.info(f"  Sample ID column: {self.sample_id if self.sample_id else 'NONE'}")
        log.info(f"  Dependent variable: {dependent_var}")
        log.info(f"  Data variable: {self.data_variable}")
        log.info(f"  Threads: {self.threads}")
        log.info(f"  Chunk size: {self.chunk_size}")
        log.info(f"  Chromosome groups: {', '.join(h5_paths.get('chrom_groups', []))}")
        log.info(f"  Sample dataset: {h5_paths.get('sample_list_name')}")
        log.info(f"  Metadata dataset: {h5_paths.get('metadata_group')}")
        log.info(f"  Data orientation: {data_orientation}")

        if self.analysis_type == "EWAS":
            log.info(f"  Probe dataset: {h5_paths.get('probe_list_name')}")
            log.info(f"  Betas dataset: {h5_paths.get('betas_name')}")
            log.info("  M-value analysis: True")
        else:
            log.info(f"  Variant dataset: {h5_paths.get('variant_list_name')}")
            log.info(f"  Genotype dataset: {h5_paths.get('geno_name')}")
            log.info(f"  Model type: {self.model}")
            log.info(f"  A1 path: {h5_paths.get('a1_path', 'None')}")
            log.info(f"  A2 path: {h5_paths.get('a2_path', 'None')}")
            log.info(f"  BP path: {h5_paths.get('bp_path', 'None')}")

        cmd: List[str] = [
            "Rscript",
            r_script_path,
            "--output",
            os.path.abspath(self.output),
            "--metadata",
            os.path.abspath(self.metadata) if self.metadata else "NONE",
            "--sample_id",
            self.sample_id if self.sample_id else "NONE",
            "--stat_var",
            self.stat_var,
            "--processes",
            str(self.threads),
            "--chunk_size",
            str(self.chunk_size),
            "--chrom_groups",
            ",".join(h5_paths.get("chrom_groups", [])),
            "--sample_list_name",
            h5_paths.get("sample_list_name", "NONE"),
            "--metadata_group",
            h5_paths.get("metadata_group", "NONE"),
            "--data_orientation",
            data_orientation,
            "--log_level",
            self.log_level,
        ]

        if self.analysis_type == "EWAS":
            cmd[2:2] = ["--methylation_betas_file", os.path.abspath(self.h5_file)]
            cmd.extend(
                [
                    "--marker_list_name",
                    h5_paths.get("probe_list_name", "NONE"),
                    "--betas_name",
                    h5_paths.get("betas_name", "NONE"),
                ]
            )
        else:
            cmd[2:2] = ["--genotype_file", os.path.abspath(self.h5_file)]
            cmd.extend(
                [
                    "--dependent_var",
                    dependent_var,
                    "--marker_list_name",
                    h5_paths.get("variant_list_name", "NONE"),
                    "--geno_name",
                    h5_paths.get("geno_name", "NONE"),
                    "--test_type",
                    "logistic" if self.model == "logistic" else "linear",
                ]
            )

            for param in ["a1_path", "a2_path", "bp_path"]:
                value = h5_paths.get(param)
                if value and value != "NONE":
                    cmd.extend([f"--{param}", value])
                else:
                    cmd.extend([f"--{param}", "None"])

        covariates = self.formula_components.get("covariates", [])
        if covariates:
            cmd.extend(["--covariate_names", ",".join(covariates)])

        interaction_term = self.formula_components.get("interaction_term")
        random_effects = self.formula_components.get("random_effects")

        if interaction_term:
            cmd.extend(["--interaction_term", interaction_term])
        if random_effects:
            if isinstance(random_effects, list):
                cmd.extend(["--random_effects", ",".join(random_effects)])
            else:
                cmd.extend(["--random_effects", str(random_effects)])

        cmd.extend(["--temp_dir", temp_dir])
        cmd.extend(["--memory_per_core", str(self.memory_per_core)])

        try:
            with monitor_resources(interval=5.0) as stats:
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=1,
                        universal_newlines=True,
                    )

                    stdout_lines: List[str] = []
                    stderr_lines: List[str] = []

                    def _stream(pipe, collect, log_fn):
                        try:
                            startup_re = re.compile(
                                r"^\s*starting worker pid=\d+ on", re.IGNORECASE
                            )
                            for ln in iter(pipe.readline, ""):
                                if startup_re.search(ln):
                                    continue
                                collect.append(ln)
                                sys.stdout.write(ln)
                                sys.stdout.flush()
                        finally:
                            try:
                                pipe.close()
                            except Exception:
                                pass

                    t_out = threading.Thread(
                        target=_stream,
                        args=(process.stdout, stdout_lines, log.info),
                        daemon=True,
                    )
                    t_err = threading.Thread(
                        target=_stream,
                        args=(process.stderr, stderr_lines, log.error),
                        daemon=True,
                    )
                    t_out.start()
                    t_err.start()
                    process.wait()
                    t_out.join(timeout=1.0)
                    t_err.join(timeout=1.0)
                    stdout = "".join(stdout_lines)
                    stderr = "".join(stderr_lines)

                    if process.returncode != 0:
                        log.error(
                            f"R script failed with return code {process.returncode}"
                        )
                        if stderr:
                            log.error(f"Error output: {stderr}")
                        raise RuntimeError(
                            f"R script failed with return code {process.returncode}"
                        )

                    log.info(
                        f"Peak resource usage - CPU: {stats['max_cpu']:.1f}%, Memory: {stats['max_memory']:.1f}%"
                    )

                    if self.analysis_type in ["EWAS", "GWAS"]:
                        standardized_df = self._standardize_result_headers(self.output)
                        if standardized_df is not None:
                            log.info("Header standardization completed.")
                        else:
                            log.warn(
                                "Header standardization skipped (read/write issue)."
                            )

                    log.success(f"Analysis completed. Results written to {self.output}")
                    return self.output

                except Exception as e:
                    log.error(f"Error during R script execution: {e}")
                    log.info("Peak resource usage before failure:")
                    log.info(
                        f"CPU: {stats['max_cpu']:.1f}%, Memory: {stats['max_memory']:.1f}%"
                    )
                    if "stderr" in locals() and stderr:
                        log.error(f"R stderr: {stderr[:500]}...")
                    raise RuntimeError(f"Error running R script: {e}") from e
        finally:
            self._cleanup_temp_files(temp_dir)

        return self.output

    def _standardize_result_headers(self, output_path: str) -> Optional[pd.DataFrame]:
        try:
            if output_path is None or not os.path.exists(output_path):
                log.warn(
                    f"Result file not found for header standardization: {output_path}"
                )
                return None

            df = pd.read_csv(output_path)
            if df is None or df.empty:
                log.warn("Received an empty DataFrame for header standardization.")
                return df

            special_columns = {"U_SE", "COEF_SE", "OR_SE", "ELOG2FC"}

            stat_vars = self.stat_vars if hasattr(self, "stat_vars") else []
            if not stat_vars and hasattr(self, "stat_var") and self.stat_var:
                stat_vars = self.stat_var.split(",")

            log.debug(f"Using stat variables for header standardization: {stat_vars}")

            standardized_mapping = {}
            renamed_count = 0

            for col in df.columns:
                original_col = str(col)
                prefix = original_col.split("_")[0]
                if prefix.lower() == "elog2fc":
                    standardized_mapping[original_col] = original_col
                    continue

                if original_col.upper() in special_columns:
                    standardized_mapping[original_col] = original_col
                    continue

                parts = original_col.split("_")
                if len(parts) >= 2 and parts[-1] in stat_vars:
                    stat_var = parts[-1]
                    prefix = "_".join(parts[:-1])

                    std_prefix = AliasUtils.get_field(prefix)
                    if std_prefix is None:
                        std_prefix = prefix.upper()

                    new_name = f"{std_prefix}_{stat_var}"
                    standardized_mapping[original_col] = new_name
                    if new_name != original_col:
                        renamed_count += 1

                elif any(original_col.startswith(f"{var}_") for var in stat_vars):
                    for var in stat_vars:
                        if original_col.startswith(f"{var}_"):
                            var_len = len(var) + 1
                            remainder = original_col[var_len:]
                            std_remainder = AliasUtils.get_field(remainder)
                            if std_remainder is None:
                                std_remainder = remainder.upper()

                            new_name = f"{var}_{std_remainder}"
                            standardized_mapping[original_col] = new_name
                            if new_name != original_col:
                                renamed_count += 1
                            break
                else:
                    std_name = AliasUtils.get_field(original_col)
                    if std_name is None:
                        std_name = original_col.upper()

                    standardized_mapping[original_col] = std_name
                    if std_name != original_col:
                        renamed_count += 1

            df = df.rename(columns=standardized_mapping)

            if renamed_count > 0:
                log.info(f"Standardized {renamed_count} column names")
                for old, new in standardized_mapping.items():
                    if old != new:
                        log.debug(f"  {old} → {new}")
                df.to_csv(output_path, index=False)
                log.info(f"Updated result file: {output_path}")
            else:
                log.info("All column names were already standardized.")

            return df

        except Exception as e:
            log.error(f"Failed to standardize result headers: {e}")
            return None

    def print_info(self) -> None:
        print("=" * 50)
        print(f"Analysis Type: {self.analysis_type}")
        print(f"Model Type: {self.model}")
        print(f"Formula: {self.formula}")
        print(f"R Script: {self.r_script_path}")
        print(f"Dependent Variable: {self.dependent_var}")
        print(f"Data Variable: {self.data_variable}")
        system_info = SystemUtils.get_system_info()
        print(f"Environment: {system_info['environment']}")
        print(f"CPU: {system_info['cpu_name']}")
        print(f"Effective cores: {system_info['effective_cores']}")
        print(f"Using threads: {self.threads}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Available memory: {system_info['ram_available_gb']:.1f} GB")
        print("=" * 50)


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(flags=["-m", "--metadata"], type=str, default=None, required=False),
    OptionConfig(flags=["-s", "--sample_id"], type=str, default=None, required=False),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
    OptionConfig(flags=["-f", "--formula"], type=str, required=True),
    OptionConfig(
        flags=["-d", "--model"],
        type=str,
        default="linear",
        required=False,
        choices=["linear", "logistic"],
    ),
    OptionConfig(flags=["-p", "--threads"], type=int, default=None, required=False),
    OptionConfig(flags=["-c", "--chunk_size"], type=int, default=None, required=False),
    OptionConfig(
        flags=["-a", "--var"],
        type=str,
        default=None,
        required=False,
    ),
    OptionConfig(
        flags=["-t", "--analysis_type"],
        type=str,
        default="Auto",
        required=False,
        choices=["Auto", "EWAS", "GWAS"],
    ),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="AssociationAnalysis")
    opt = framework.run()

    analysis = AssociationAnalysisLauncher(
        h5_file=opt.input,
        metadata=opt.metadata,
        sample_id=opt.sample_id,
        output=opt.output,
        formula=opt.formula,
        model=opt.model,
        threads=opt.threads,
        chunk_size=opt.chunk_size,
        log_level=opt.verbose,
        analysis_type=opt.analysis_type,
        var=opt.var,
    )

    analysis.print_info()
    analysis.run()
