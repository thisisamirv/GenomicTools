#!/usr/bin/env python
# Import required modules
import gc
import math
import numba
import numpy as np
import os
import pandas as pd
import psutil
from decimal import Decimal, getcontext
from joblib import Parallel, delayed
from mpmath import mp, erfc, sqrt
from scipy.stats import chi2, norm
from threading import Lock
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.LoggingUtils import log
from utils.ParsingUtils import ParseToList
from utils.SystemUtils import SystemUtils


@numba.jit(nopython=True)
def _fast_fixed_effects_meta(
    coefs: np.ndarray, ses: np.ndarray
) -> Tuple[float, float, float, float, float]:
    ses_safe = np.maximum(ses, 1e-10)
    weights = 1.0 / (ses_safe**2)
    pooled_coef = np.sum(weights * coefs) / np.sum(weights)
    pooled_se = np.sqrt(1.0 / np.sum(weights))
    pooled_z = pooled_coef / pooled_se
    q_stat = np.sum(weights * (coefs - pooled_coef) ** 2)
    return pooled_coef, pooled_se, pooled_z, q_stat, np.sum(weights)


@numba.jit(nopython=True)
def _fast_random_effects_meta(
    coefs: np.ndarray, ses: np.ndarray
) -> Tuple[float, float, float, float]:
    ses_safe = np.maximum(ses, 1e-10)
    weights = 1.0 / (ses_safe**2)
    pooled_coef_fe = np.sum(weights * coefs) / np.sum(weights)
    q_stat = np.sum(weights * (coefs - pooled_coef_fe) ** 2)
    df_q = len(coefs) - 1
    if df_q > 0 and q_stat > df_q:
        sum_weights = np.sum(weights)
        sum_weights_sq = np.sum(weights**2)
        tau_squared = (q_stat - df_q) / (sum_weights - sum_weights_sq / sum_weights)
    else:
        tau_squared = 0.0
    re_weights = 1.0 / (ses_safe**2 + tau_squared)
    pooled_coef_re = np.sum(re_weights * coefs) / np.sum(re_weights)
    pooled_se_re = np.sqrt(1.0 / np.sum(re_weights))
    pooled_z_re = pooled_coef_re / pooled_se_re
    return pooled_coef_re, pooled_se_re, pooled_z_re, tau_squared


@numba.jit(nopython=True)
def _fast_heterogeneity_stats(q_stat: float, df_q: int) -> float:
    i_squared = max(0.0, (q_stat - df_q) / q_stat) if q_stat > 0 else 0.0
    return i_squared


@numba.jit(nopython=True)
def _validate_study_data(
    coefs: np.ndarray, ses: np.ndarray, pvals: np.ndarray
) -> np.ndarray:
    n_studies = len(coefs)
    valid_mask = np.empty(n_studies, dtype=numba.boolean)

    for i in range(n_studies):
        valid = True

        if np.isnan(coefs[i]) or np.isnan(ses[i]) or np.isnan(pvals[i]):
            valid = False
        elif ses[i] <= 0:
            valid = False
        elif pvals[i] < 0 or pvals[i] > 1:
            valid = False

        valid_mask[i] = valid

    return valid_mask


getcontext().prec = 50


@numba.jit(nopython=True)
def _fast_log_p_value_calculation(z_stat: float) -> float:
    abs_z = abs(z_stat)
    if abs_z > 8.0:
        term1 = -(abs_z * abs_z) / (2.0 * math.log(10))
        term2 = -0.5 * math.log10(2 * math.pi)
        term3 = -math.log10(abs_z)
        log10_p = term1 + term2 + term3
        return log10_p
    else:
        x = abs_z / math.sqrt(2.0)
        p_value = math.erfc(x)
        if p_value <= 0:
            return -300.0
        return math.log10(p_value)


@numba.jit(nopython=True)
def _fallback_precise_p_value(z_stat: float) -> float:
    abs_z = abs(z_stat)
    if abs_z > 8.0:
        term1 = -(abs_z * abs_z) / (2.0 * math.log(10))
        term2 = -0.5 * math.log10(2 * math.pi)
        term3 = -math.log10(abs_z)
        log10_p = term1 + term2 + term3
        if log10_p > -300:
            return 10.0**log10_p
        else:
            return 0.0
    else:
        x = abs_z / math.sqrt(2.0)
        return math.erfc(x)


def precise_p_value_from_z(z_stat: float) -> float:
    abs_z = abs(z_stat)
    if abs_z > 8.0:
        log10_p = _fast_log_p_value_calculation(z_stat)
        if log10_p < -300:
            return 0.0
        else:
            try:
                decimal_log10_p = Decimal(str(log10_p))
                decimal_p = Decimal(10) ** decimal_log10_p
                return float(decimal_p)
            except Exception:
                return 0.0
    else:
        return 2 * (1 - norm.cdf(abs_z))


def ultra_precise_p_value(z_stat: float) -> float:
    abs_z = abs(z_stat)
    if abs_z <= 8.0:
        return 2 * (1 - norm.cdf(abs_z))
    try:
        original_dps = mp.dps
        try:
            mp.dps = 100
            x = abs_z / sqrt(2)
            p_value_mp = erfc(x)
            p_value_float = float(p_value_mp)
            if p_value_float == 0.0 and p_value_mp > 0:
                return 0.0
            return p_value_float
        except OverflowError:
            return 0.0
        finally:
            mp.dps = original_dps
    except Exception:
        return _fallback_precise_p_value(z_stat)


def calculate_precise_p_value(z_stat: float, precision_mode: str = "auto") -> float:
    abs_z = abs(z_stat)
    if precision_mode == "auto":
        if abs_z <= 6.0:
            precision_mode = "standard"
        elif abs_z <= 15.0:
            precision_mode = "high"
        else:
            precision_mode = "ultra"

    if precision_mode == "standard":
        return 2 * (1 - norm.cdf(abs_z))
    elif precision_mode == "high":
        try:
            log_p = norm.logsf(abs_z) + math.log(2)
            return math.exp(log_p) if log_p > -700 else 0.0
        except OverflowError:
            x = abs_z / math.sqrt(2.0)
            return math.erfc(x)
    elif precision_mode == "ultra":
        return ultra_precise_p_value(z_stat)
    else:
        raise ValueError(f"Unknown precision_mode: {precision_mode}")


def calculate_precise_q_pvalue(q_stat: float, df_q: int) -> float:
    if df_q <= 0:
        return 1.0
    try:
        return chi2.sf(q_stat, df_q)
    except Exception:
        return 1e-300


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR. Preserves NaNs in input positions."""
    p = np.asarray(pvals, dtype=float)
    res = np.full(p.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    if not finite_mask.any():
        return res
    pv = p[finite_mask]
    n = pv.size
    order = np.argsort(pv)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adjusted = pv * n / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted[order[::-1]])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
    res_vals = np.empty(n, dtype=float)
    res_vals[order] = adjusted_sorted
    res[finite_mask] = res_vals
    return res


def _holm_correction(pvals: np.ndarray) -> np.ndarray:
    """Holm (step-down) correction. Preserves NaNs."""
    p = np.asarray(pvals, dtype=float)
    res = np.full(p.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    if not finite_mask.any():
        return res
    pv = p[finite_mask]
    n = pv.size
    order = np.argsort(pv)
    sorted_p = pv[order]
    adjusted = np.empty(n, dtype=float)
    for i in range(n):
        adjusted[i] = sorted_p[i] * (n - i)
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.clip(adjusted, 0, 1)
    inv = np.empty(n, dtype=float)
    inv[order] = adjusted
    res[finite_mask] = inv
    return res


def _bonferroni_correction(pvals: np.ndarray) -> np.ndarray:
    """Simple Bonferroni correction. Preserves NaNs."""
    p = np.asarray(pvals, dtype=float)
    res = np.full(p.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    if not finite_mask.any():
        return res
    pv = p[finite_mask]
    n = pv.size
    corrected = np.minimum(pv * n, 1.0)
    res[finite_mask] = corrected
    return res


class MetaAssociation:
    def __init__(
        self,
        output: str,
        input: Optional[Union[str, List[str]]] = None,
        names: Optional[Union[str, List[str]]] = None,
        sample_sizes: Optional[Union[str, List[int]]] = None,
        populations: Optional[Union[str, List[str]]] = None,
        method: str = "fixed",
        data_type: str = "auto",
        var: Optional[str] = None,
        precision_mode: str = "auto",
    ) -> None:
        valid_methods = ["fixed", "random", "both"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {valid_methods}"
            )

        valid_precision_modes = ["auto", "standard", "high", "ultra"]
        if precision_mode not in valid_precision_modes:
            raise ValueError(
                f"Invalid precision_mode '{precision_mode}'. "
                f"Must be one of: {valid_precision_modes}"
            )
        self.output = output
        self.method = method
        self.studies: List[Dict[str, Any]] = []
        self.results: Optional[pd.DataFrame] = None
        self.data_type = data_type
        self.id_col: Optional[str] = None
        self.target_variable = var
        self.precision_mode = precision_mode
        self.threads = SystemUtils.get_optimal_cores(reserve_cores=1)
        memory_info = SystemUtils.get_memory_info()
        self.available_memory = memory_info["available_gb"]
        self.total_memory = memory_info["total_gb"]
        log.info(
            f"System resources: {self.threads} cores, {self.available_memory:.1f}GB available memory"
        )
        self.parallel_params = self._configure_parallel_settings()
        self.chunk_size = self._calculate_optimal_chunk_size()
        if self.precision_mode == "ultra" or self.precision_mode == "auto":
            self._verify_extreme_precision()
        if input and names:
            self.validate_and_setup_studies(input, names, sample_sizes, populations)

    def validate_input_files(self, input_files: List[str]) -> bool:
        missing_files: List[str] = []
        invalid_files: List[str] = []

        for file_path in input_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                continue

            try:
                test_df = pd.read_csv(file_path, nrows=5)
                if len(test_df.columns) < 3:
                    invalid_files.append(file_path)
            except Exception as e:
                invalid_files.append(f"{file_path} (Error: {e})")

        if missing_files:
            log.error(f"Input files not found: {missing_files}")
            return False

        if invalid_files:
            log.error(f"Invalid input files: {invalid_files}")
            return False

        return True

    def validate_required_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        validation_results: Dict[str, Any] = {
            "missing_columns": [],
            "calculated_columns": {},
            "valid": True,
        }
        required_columns = [self.id_col, "COEF", "SE", "P"]
        for col in required_columns:
            if col not in df.columns:
                validation_results["missing_columns"].append(col)
        if "SE" in validation_results["missing_columns"]:
            calculated_se, method = AliasUtils.calculate_se_from_other_stats(df, {})
            if calculated_se is not None:
                df["SE"] = calculated_se
                validation_results["calculated_columns"]["SE"] = method
                validation_results["missing_columns"].remove("SE")
        if "P" in validation_results["missing_columns"]:
            calculated_p, method = AliasUtils.calculate_p_from_other_stats(df, {})
            if calculated_p is not None:
                df["P"] = calculated_p
                validation_results["calculated_columns"]["P"] = method
                validation_results["missing_columns"].remove("P")
        critical_missing = [
            col for col in validation_results["missing_columns"] if col in ["COEF", "P"]
        ]
        validation_results["valid"] = len(critical_missing) == 0
        return validation_results

    def _verify_extreme_precision(self) -> None:
        try:
            test_p = ultra_precise_p_value(25.0)
            if test_p is not None and test_p > 0:
                log.info(f"Extreme precision verified: Z=25 gives P={test_p:.2e}")
            else:
                log.warn("Extreme precision may have precision limits")
        except Exception as e:
            log.warn(f"Extreme precision test failed: {e}")

    def calculate_fdr_and_holm(
        self, p_column: str, use_original_names: bool = False
    ) -> None:
        if p_column not in self.results.columns:
            return

        numeric_p = pd.to_numeric(self.results[p_column], errors="coerce")
        if numeric_p.isna().all():
            log.warn(f"No valid p-values found for {p_column}; skipping correction")
            return

        pvalues_array = numeric_p.to_numpy(dtype=float, copy=True)
        pvalues_array[(pvalues_array <= 0) & ~np.isnan(pvalues_array)] = 1e-300
        pvalues_array[(pvalues_array > 1) & ~np.isnan(pvalues_array)] = np.nan

        fdr_pvalues = _fdr_bh(pvalues_array.copy())
        holm_pvalues = _holm_correction(pvalues_array.copy())
        bonf_pvalues = _bonferroni_correction(pvalues_array.copy())

        base_col = p_column
        if base_col.startswith("P_"):
            fdr_col = base_col.replace("P_", "FDR_")
            holm_col = base_col.replace("P_", "HOLM_")
            bonf_col = base_col.replace("P_", "BONFERRONI_")
        else:
            fdr_col = f"FDR_{base_col}"
            holm_col = f"HOLM_{base_col}"
            bonf_col = f"BONFERRONI_{base_col}"

        self.results[fdr_col] = fdr_pvalues
        self.results[holm_col] = holm_pvalues
        self.results[bonf_col] = bonf_pvalues

        log.info(
            f"Applied FDR, Holm and Bonferroni corrections to {p_column} -> {fdr_col}, {holm_col}, {bonf_col}"
        )

    def apply_genomic_control(
        self, p_column: str
    ) -> Tuple[Optional[str], Optional[float]]:
        if p_column not in self.results.columns:
            return None, None

        valid_pvals = pd.to_numeric(self.results[p_column], errors="coerce").dropna()

        if len(valid_pvals) < 100:
            log.warn(
                f"Insufficient data for genomic control correction: {len(valid_pvals)}"
            )
            return None, None

        try:
            chi2_obs = -2 * np.log(valid_pvals.clip(lower=1e-300))
            lambda_gc = np.median(chi2_obs) / chi2.ppf(0.5, df=1)

            if lambda_gc <= 1.0:
                log.info(
                    f"No inflation detected (λ = {lambda_gc:.3f}), skipping genomic control"
                )
                return None, lambda_gc

            corrected_chi2 = chi2_obs / lambda_gc
            corrected_pvals = 1 - chi2.cdf(corrected_chi2, df=1)
            corrected_pvals = np.clip(corrected_pvals, 1e-300, 1.0)

            gc_col = f"{p_column}_GC"
            self.results[gc_col] = np.nan
            self.results.loc[valid_pvals.index, gc_col] = corrected_pvals

            new_lambda = self.calculate_genomic_lambda(corrected_pvals)

            log.info(
                f"Applied genomic control correction: λ {lambda_gc:.3f} → {new_lambda:.3f}"
            )
            return gc_col, new_lambda

        except Exception as e:
            log.error(f"Genomic control correction failed: {e}")
            return None, None

    def calculate_genomic_lambda(self, pvalues: Union[np.ndarray, pd.Series]) -> float:
        try:
            valid_pvals = pvalues[~np.isnan(pvalues)]
            valid_pvals = valid_pvals[valid_pvals > 0]

            if len(valid_pvals) == 0:
                return 1.0

            chi2_obs = -2 * np.log(valid_pvals.clip(lower=1e-300))
            lambda_gc = np.median(chi2_obs) / chi2.ppf(0.5, df=1)
            return lambda_gc

        except Exception:
            return 1.0

    def _configure_parallel_settings(self) -> Dict[str, Any]:
        n_jobs = self.threads
        if self.available_memory < 8:
            batch_size = max(1, n_jobs // 2)
        elif self.available_memory < 16:
            batch_size = n_jobs
        else:
            batch_size = n_jobs * 2
        parallel_params: Dict[str, Any] = {
            "n_jobs": n_jobs,
            "verbose": 0,
            "backend": "threading",
            "batch_size": batch_size,
            "pre_dispatch": f"{min(batch_size * 2, n_jobs * 3)}",
        }
        log.info(
            f"Configured parallel processing: {n_jobs} jobs, batch_size={batch_size}"
        )
        return parallel_params

    def _calculate_optimal_chunk_size(self) -> int:
        max_studies = 50
        bytes_per_marker = max_studies * 24 * 4
        target_memory_gb = self.available_memory * 0.25
        target_memory_bytes = target_memory_gb * 1024**3
        chunk_size = max(1000, int(target_memory_bytes / bytes_per_marker))
        chunk_size = min(chunk_size, 50000)
        log.info(f"Calculated optimal chunk size: {chunk_size} markers")
        return chunk_size

    def _monitor_memory_usage(self, stage: str = "") -> float:
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024**3)
            system_memory = psutil.virtual_memory()
            available_gb = system_memory.available / (1024**3)
            total_gb = system_memory.total / (1024**3)

            log.debug(
                f"{stage} - Process memory: {memory_gb:.1f}GB, "
                f"Available: {available_gb:.1f}GB, "
                f"Total: {total_gb:.1f}GB"
            )

            if memory_gb > total_gb * 0.7:
                log.warn(
                    f"High memory usage detected at {stage}: {memory_gb:.1f}GB "
                    f"({memory_gb / total_gb * 100:.1f}% of total)"
                )
            return memory_gb
        except Exception as e:
            log.debug(f"Could not check memory usage: {e}")
            return 0.0

    def determine_target_variable(self, df: pd.DataFrame) -> Union[str, None, bool]:
        if self.target_variable:
            var_coef_col = f"{self.target_variable}_COEF"
            if var_coef_col in df.columns:
                log.info(f"Using specified variable: {self.target_variable}")
                return self.target_variable
            elif all(col in df.columns for col in ["COEF", "SE", "P"]):
                log.info(f"Using generic columns for variable: {self.target_variable}")
                return None
            else:
                raise ValueError(
                    f"Specified variable '{self.target_variable}' not found in data. "
                    f"Available columns: {list(df.columns)}"
                )

        detected_var = AliasUtils.auto_detect_variable(df)
        if detected_var:
            var_columns = AliasUtils.get_all_variable_columns(df, detected_var)
            if len(var_columns) >= 2:
                log.info(
                    f"Auto-detected variable with sufficient columns: {detected_var}"
                )
                return detected_var
        if self.data_type.upper() == "EWAS":
            methylation_aliases = AliasUtils.get_aliases("Methylation")
            for alias in methylation_aliases:
                if f"{alias}_COEF" in df.columns or f"{alias}_P" in df.columns:
                    log.info(f"Found EWAS variable: {alias}")
                    return alias
        elif self.data_type.upper() == "GWAS":
            genotype_aliases = AliasUtils.get_aliases("Genotype")
            for alias in genotype_aliases:
                if f"{alias}_COEF" in df.columns or f"{alias}_P" in df.columns:
                    log.info(f"Found GWAS variable: {alias}")
                    return alias
        if all(col in df.columns for col in ["COEF", "SE", "P"]):
            log.info("Using generic columns (COEF, SE, P) - no variable prefix")
            return None
        log.error(f"No suitable variable found. Available columns: {list(df.columns)}")
        return False

    def validate_and_setup_studies(
        self,
        input_files: Union[str, List[str]],
        names: Union[str, List[str]],
        sample_sizes: Optional[Union[str, List[int]]],
        populations: Optional[Union[str, List[str]]] = None,
    ) -> bool:
        input_files = ParseToList(input_files) if input_files else []
        names = ParseToList(names) if names else []
        if not input_files:
            log.error("No input files provided")
            return False
        if not names:
            log.error("No study names provided")
            return False
        if not self.validate_input_files(input_files):
            return False
        if sample_sizes:
            sample_sizes = (
                [int(x) for x in ParseToList(str(sample_sizes))] if sample_sizes else []
            )
        else:
            sample_sizes = None
        if populations:
            populations = ParseToList(populations)
        if len(input_files) != len(names):
            log.error(
                f"Number of input files ({len(input_files)}) and names ({len(names)}) must match"
            )
            return False
        if sample_sizes and len(input_files) != len(sample_sizes):
            log.error(
                f"Number of input files ({len(input_files)}) and sample sizes ({len(sample_sizes)}) must match"
            )
            return False
        if populations and len(populations) != len(input_files):
            log.error(
                f"Number of populations ({len(populations)}) must match number of input files ({len(input_files)})"
            )
            return False
        valid_methods = ["fixed", "random", "both"]
        if self.method not in valid_methods:
            log.error(
                f"Invalid method '{self.method}'. Must be one of: {valid_methods}"
            )
            return False
        populations = populations or [None] * len(input_files)
        successful_studies = 0
        for i, (file_path, name) in enumerate(zip(input_files, names)):
            if sample_sizes:
                sample_size = sample_sizes[i]
            else:
                sample_size = self.detect_sample_size(file_path, name)
                if sample_size is None:
                    log.error(f"Could not determine sample size for {name}")
                    continue
            population = populations[i] if populations else None
            self.add_study(file_path, name, sample_size, population)
            successful_studies += 1
        if successful_studies < 2:
            log.error(
                f"Need at least 2 valid studies for meta-analysis, got {successful_studies}"
            )
            return False
        log.info(
            f"Successfully validated and added {successful_studies} studies for meta-analysis"
        )
        return True

    def add_study(
        self,
        file_path: str,
        study_name: str,
        sample_size: int,
        population: Optional[str] = None,
    ) -> None:
        study_info: Dict[str, Any] = {
            "file_path": file_path,
            "study_name": study_name,
            "sample_size": sample_size,
            "population": population or "Unknown",
        }
        self.studies.append(study_info)
        log.info(
            f"Added study: {study_name} (N={sample_size}, Pop={study_info['population']})"
        )

    def detect_sample_size(
        self, file_path: str, study_name: str, data_type: Optional[str] = None
    ) -> Optional[int]:
        try:
            df_sample = pd.read_csv(file_path, nrows=100)
            df_sample = self.standardize_columns(df_sample)
            if "N" in df_sample.columns:
                sample_sizes = df_sample["N"].dropna()
                if len(sample_sizes) > 0:
                    detected_size = int(sample_sizes.mode().iloc[0])
                    log.info(
                        f"Auto-detected sample size for {study_name}: {detected_size}"
                    )
                    return detected_size
            df_full = pd.read_csv(file_path)
            df_full = self.standardize_columns(df_full)
            if self.data_type == "EWAS":
                id_col = "CGID"
            else:
                id_col = "RSID"
            if all(col in df_full.columns for col in [id_col, "COEF", "SE", "P"]):
                valid_records = df_full.dropna(subset=[id_col, "COEF", "SE", "P"])
                estimated_size = len(valid_records)
                log.warn(
                    f"No sample size column found for {study_name}. Estimated from valid records: {estimated_size}"
                )
                return estimated_size
            else:
                log.warn(
                    f"Could not auto-detect sample size for {study_name}. Using default: 1000"
                )
                return 1000
        except Exception as e:
            log.error(f"Error detecting sample size for {study_name}: {e}")
            return None

    def standardize_columns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        column_mappings: Dict[str, str] = {}
        if self.data_type == "auto":
            id_col = AliasUtils.find_keys(
                dict.fromkeys(df.columns), "CGID"
            ) or AliasUtils.find_keys(dict.fromkeys(df.columns), "RSID")
            if id_col:
                sample_ids = df[id_col].dropna().head(10).astype(str)
                if sum(id.startswith("cg") for id in sample_ids) > len(sample_ids) / 2:
                    self.data_type = "EWAS"
                    log.info("Auto-detected data type: EWAS (CpG identifiers)")
                elif (
                    sum(id.startswith("rs") for id in sample_ids) > len(sample_ids) / 2
                ):
                    self.data_type = "GWAS"
                    log.info("Auto-detected data type: GWAS (SNP identifiers)")
                else:
                    self.data_type = "EWAS"
                    log.info(
                        "Could not confidently detect data type, defaulting to EWAS"
                    )
        self.id_col = "CGID" if self.data_type == "EWAS" else "RSID"
        standard_fields = {
            "EWAS": ["CGID", "CHR", "BP", "N"],
            "GWAS": ["RSID", "CHR", "BP", "A1", "A2", "INFO", "EAF", "N"],
        }
        for field in standard_fields.get(self.data_type, []):
            found_col = AliasUtils.find_keys(dict.fromkeys(df.columns), field)
            if found_col and found_col not in column_mappings:
                column_mappings[found_col] = field
        stat_fields = ["COEF", "T-STAT", "P", "OR", "Z"]
        for field in stat_fields:
            found_col = AliasUtils.find_keys(dict.fromkeys(df.columns), field)
            if found_col and found_col not in column_mappings:
                column_mappings[found_col] = field
        target_var = AliasUtils.auto_detect_variable(df)
        if target_var:
            log.info(f"Auto-detected target variable: {target_var}")
            self.determined_variable = target_var
        else:
            if not hasattr(self, "determined_variable"):
                self.determined_variable = None
        se_col = AliasUtils.find_se_column_comprehensive(
            df, column_mappings, target_variable=target_var
        )
        if se_col and se_col not in column_mappings:
            column_mappings[se_col] = "SE"
        else:
            calculated_se, method = AliasUtils.calculate_se_from_other_stats(
                df, column_mappings
            )
            if calculated_se is not None:
                df["SE"] = calculated_se
                log.success(f"Successfully calculated SE using method: {method}")
            else:
                log.error("Could not find or calculate SE column")
        if column_mappings:
            df = df.rename(columns=column_mappings)
            log.debug(f"Applied column mappings: {column_mappings}")
        if "P" not in df.columns:
            p_col = AliasUtils.find_p_column_comprehensive(
                df, column_mappings, target_variable=target_var
            )
            if p_col:
                df = df.rename(columns={p_col: "P"})
            else:
                calculated_p, method = AliasUtils.calculate_p_from_other_stats(
                    df, column_mappings
                )
                if calculated_p is not None:
                    df["P"] = calculated_p
                    log.success(
                        f"Successfully calculated P-values using method: {method}"
                    )
        if "COEF" not in df.columns:
            coef_col = AliasUtils.find_coef_column_comprehensive(
                df, column_mappings, target_variable=target_var
            )
            if coef_col:
                df = df.rename(columns={coef_col: "COEF"})
        final_target_var = self.determine_target_variable(df)
        if final_target_var is False:
            log.error("Failed to determine target variable for analysis")
            return None
        self.determined_variable = final_target_var
        return df

    def _early_variable_detection(self, df: pd.DataFrame) -> Optional[str]:
        if self.target_variable:
            return self.target_variable
        for col in df.columns:
            if "_COEF" in col or "_coef" in col:
                var_name = col.replace("_COEF", "").replace("_coef", "")
                if var_name in ["Methylation", "methylation", "Genotype", "genotype"]:
                    return var_name
        return None

    def load_and_validate_study(
        self, study_info: Dict[str, Any], data_type: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(study_info["file_path"])
            log.info(f"Loaded {study_info['study_name']}: {len(df)} sites")
            df = self.standardize_columns(df)
            if df is None:
                return None
            validation_results = self.validate_required_columns(df)
            for field, method in validation_results["calculated_columns"].items():
                log.info(f"Using calculated {field} column (method: {method})")
            if validation_results["missing_columns"]:
                critical_missing = [
                    col
                    for col in validation_results["missing_columns"]
                    if col in ["COEF", "P"]
                ]
                if critical_missing:
                    log.error(
                        f"Study {study_info['study_name']} missing critical columns: {critical_missing}"
                    )
                    return None
            condition1 = hasattr(self, "determined_variable")
            condition2 = self.determined_variable is not None
            if condition1 and condition2:
                var = self.determined_variable
                var_mappings = {
                    f"{var}_COEF": "COEF",
                    f"{var}_SE": "SE",
                    f"{var}_P": "P",
                }
                for var_col, generic_col in var_mappings.items():
                    if var_col in df.columns and generic_col not in df.columns:
                        df[generic_col] = df[var_col]
                        log.debug(f"Mapped {var_col} to {generic_col}")
            df["STUDY"] = study_info["study_name"]
            df["SAMPLE_SIZE"] = study_info["sample_size"]
            df["POPULATION"] = study_info["population"]
            if "P" in df.columns:
                zero_p_count = (df["P"] == 0).sum()
                if zero_p_count > 0:
                    log.info(
                        f"Found {zero_p_count} p-values set to 0 in {study_info['study_name']}"
                    )
                    log.info("Treating as very small p-values (1e-300)")
                    df.loc[df["P"] == 0, "P"] = 1e-300
            p_value_cols = [
                col
                for col in df.columns
                if col.startswith("P_") or col.startswith("p_") or col.startswith("P.")
            ]
            for p_col in p_value_cols:
                if p_col in df.columns:
                    zero_count = (df[p_col] == 0).sum()
                    if zero_count > 0:
                        log.debug(
                            f"Found {zero_count} zero values in {p_col} column, replacing with 1e-300"
                        )
                        df.loc[df[p_col] == 0, p_col] = 1e-300
            condition1 = self.data_type == "GWAS"
            condition2 = "OR" in df.columns
            condition3 = "COEF" not in df.columns
            if condition1 and condition2 and condition3:
                df["COEF"] = np.log(pd.to_numeric(df["OR"], errors="coerce"))
                log.info(f"Converted OR to log(OR) for {study_info['study_name']}")
            if "N" in df.columns:
                file_sample_sizes = df["N"].dropna()
                if len(file_sample_sizes) > 0:
                    file_sample_size = int(file_sample_sizes.mode().iloc[0])
                    if file_sample_size != study_info["sample_size"]:
                        log.info(
                            f"Using sample size from file for {study_info['study_name']}: {file_sample_size}"
                        )
                        log.info(
                            f"Overriding provided sample size: {study_info['sample_size']}"
                        )
                        df["SAMPLE_SIZE"] = file_sample_size
            initial_count = len(df)
            df = df.dropna(subset=[self.id_col, "COEF", "SE", "P"])
            df = df[df["SE"] > 0]
            df = df[df["P"] >= 0]
            df = df[df["P"] <= 1]
            final_count = len(df)
            if initial_count != final_count:
                log.warn(
                    f"Removed {initial_count - final_count} invalid records from {study_info['study_name']}"
                )
            return df
        except Exception as e:
            log.error(f"Error loading study {study_info['study_name']}: {e}")
            return None

    def process_marker_chunk_with_progress(
        self,
        marker_chunk_data: List[Tuple[str, pd.DataFrame]],
        progress_callback: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for marker, group_data in marker_chunk_data:
            if len(group_data) < 2:
                if progress_callback:
                    progress_callback()
                continue
            result = self.process_single_marker(marker, group_data)
            if result:
                results.append(result)
            if progress_callback:
                progress_callback()
        return results

    def process_marker_chunk(
        self, marker_chunk_data: List[Tuple[str, pd.DataFrame]]
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for marker, group_data in marker_chunk_data:
            if len(group_data) < 2:
                continue
            result = self.process_single_marker(marker, group_data)
            if result:
                results.append(result)
        return results

    def process_single_marker(
        self, marker: str, group_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        try:
            result: Dict[str, Any] = {self.id_col: marker}
            for col in ["CHR", "BP"]:
                if col in group_data.columns:
                    non_null_values = group_data[col].dropna()
                    if len(non_null_values) > 0:
                        result[col] = non_null_values.iloc[0]
            result["STUDIES"] = ";".join(group_data["STUDY"].tolist())
            result["POPULATIONS"] = ";".join(group_data["POPULATION"].unique().tolist())
            coefs = group_data["COEF"].values.astype(np.float64)
            ses = group_data["SE"].values.astype(np.float64)
            pvals = group_data["P"].values.astype(np.float64)
            sample_sizes = group_data["SAMPLE_SIZE"].values.astype(np.int64)
            valid_mask = _validate_study_data(coefs, ses, pvals)
            if np.sum(valid_mask) < 2:
                return None
            valid_coefs = coefs[valid_mask]
            valid_ses = ses[valid_mask]
            valid_sample_sizes = sample_sizes[valid_mask]
            if self.method in ["fixed", "both"]:
                pooled_coef, pooled_se, pooled_z, q_stat, sum_weights = (
                    _fast_fixed_effects_meta(valid_coefs, valid_ses)
                )
                df_q = len(valid_coefs) - 1
                i_squared = _fast_heterogeneity_stats(q_stat, df_q)
                p_fixed = calculate_precise_p_value(
                    pooled_z, precision_mode=self.precision_mode
                )
                q_p = calculate_precise_q_pvalue(q_stat, df_q)
                result.update(
                    {
                        "COEF_FIXED": pooled_coef,
                        "SE_FIXED": pooled_se,
                        "Z_FIXED": pooled_z,
                        "P_FIXED": p_fixed,
                        "Q_STAT": q_stat,
                        "Q_DF": df_q,
                        "Q_P": q_p,
                        "I_SQUARED": i_squared,
                        "N_STUDIES": len(valid_coefs),
                        "TOTAL_N": np.sum(valid_sample_sizes),
                    }
                )
            if self.method in ["random", "both"]:
                pooled_coef_re, pooled_se_re, pooled_z_re, tau_squared = (
                    _fast_random_effects_meta(valid_coefs, valid_ses)
                )
                p_random = calculate_precise_p_value(
                    pooled_z_re, precision_mode=self.precision_mode
                )
                result.update(
                    {
                        "COEF_RANDOM": pooled_coef_re,
                        "SE_RANDOM": pooled_se_re,
                        "Z_RANDOM": pooled_z_re,
                        "P_RANDOM": p_random,
                        "TAU_SQUARED": tau_squared,
                    }
                )
            for i, (_, study_row) in enumerate(group_data.iterrows()):
                result[f'COEF_{study_row["STUDY"]}'] = study_row["COEF"]
                result[f'SE_{study_row["STUDY"]}'] = study_row["SE"]
                result[f'P_{study_row["STUDY"]}'] = study_row["P"]
            return result
        except Exception as e:
            log.debug(f"Error processing marker {marker}: {e}")
            return None

    def perform_meta_analysis(self) -> Optional[pd.DataFrame]:
        if len(self.studies) < 2:
            log.error("Need at least 2 studies for meta-analysis")
            return None
        log.info(f"Starting parallel meta-analysis of {len(self.studies)} studies")
        self._monitor_memory_usage("Starting meta-analysis")
        study_data: List[pd.DataFrame] = []
        log.info("Loading and validating studies...")
        for study_info in tqdm(self.studies, desc="Loading studies", unit="study"):
            df = self.load_and_validate_study(study_info)
            if df is not None:
                study_data.append(df)
        if len(study_data) < 2:
            log.error("Failed to load sufficient valid studies")
            return None
        log.info("Combining study data...")
        combined_data = pd.concat(study_data, ignore_index=True)
        log.info(f"Combined data: {len(combined_data)} total records")

        del study_data
        gc.collect()
        self._monitor_memory_usage("After combining and cleaning data")
        log.info("Grouping markers...")
        marker_groups = combined_data.groupby(self.id_col)
        marker_chunks: List[List[Tuple[str, pd.DataFrame]]] = []
        current_chunk: List[Tuple[str, pd.DataFrame]] = []
        valid_markers_count = 0
        log.info("Creating processing chunks...")
        for marker, group_data in tqdm(
            marker_groups, desc="Creating chunks", unit="marker"
        ):
            if len(group_data) >= 2:
                current_chunk.append((marker, group_data))
                valid_markers_count += 1
            if len(current_chunk) >= self.chunk_size:
                marker_chunks.append(current_chunk)
                current_chunk = []
        if current_chunk:
            marker_chunks.append(current_chunk)
        log.info(
            f"Created {len(marker_chunks)} chunks for {valid_markers_count} markers"
        )
        progress_lock = Lock()

        def update_progress() -> None:
            with progress_lock:
                pbar.update(1)

        all_results: List[Dict[str, Any]] = []
        log.info("Starting parallel meta-analysis computation...")
        with tqdm(
            total=valid_markers_count, desc="Processing markers", unit="marker"
        ) as pbar:
            chunk_results = Parallel(**self.parallel_params)(
                delayed(self.process_marker_chunk_with_progress)(chunk, update_progress)
                for chunk in marker_chunks
            )
            for i, results in enumerate(chunk_results):
                all_results.extend(results)
                if i % 10 == 0:
                    gc.collect()
                    self._monitor_memory_usage(
                        f"After chunk {i + 1}/{len(marker_chunks)}"
                    )

            del chunk_results
            del marker_chunks
            gc.collect()
            if not all_results:
                log.error("No valid markers found for meta-analysis")
                return None
        self.results = pd.DataFrame(all_results)
        log.info(f"Meta-analysis completed: {len(self.results)} markers analyzed")
        self._monitor_memory_usage("Meta-analysis completed")
        return self.results

    def generate_summary_statistics(self) -> Dict[str, Any]:
        if self.results is None:
            log.error("No results available for summary")
            return {}
        summary: Dict[str, Any] = {
            "total_markers": len(self.results),
            "studies_included": len(self.studies),
            "total_sample_size": sum(study["sample_size"] for study in self.studies),
        }
        if "I_SQUARED" in self.results.columns:
            i_squared = self.results["I_SQUARED"].dropna()
            if len(i_squared) > 0:
                summary["mean_i_squared"] = i_squared.mean()
                summary["high_heterogeneity_sites"] = (i_squared > 0.75).sum()
        return summary

    def format_scientific_notation(self, value: Any, precision: int = 2) -> Any:
        if pd.isna(value) or value is None:
            return value
        try:
            value = float(value)
            if value == 0.0:
                return "<1.00e-300"
            elif abs(value) > 1e100:
                return "inf" if value > 0 else "-inf"
            elif value >= 0.01:
                return f"{value:.{precision + 2}f}"
            elif value >= 1e-300:
                return f"{value:.{precision}e}"
            else:
                return "<1.00e-300"
        except (ValueError, OverflowError, TypeError):
            return str(value)

    def save_results(self) -> None:
        if self.results is None:
            log.error("No results to save")
            return

        columns_to_remove: List[str] = []

        if columns_to_remove:
            self.results = self.results.drop(columns=columns_to_remove)
            for col in columns_to_remove:
                log.info(
                    f"Removed intermediate column {col} (GC correction was applied)"
                )

        processed_columns = set()

        base_p_columns = [
            col for col in self.results.columns if col in ["P_FIXED", "P_RANDOM"]
        ]
        for p_col in base_p_columns:
            if p_col not in processed_columns:
                self.calculate_fdr_and_holm(p_col, use_original_names=False)

        sort_column = "P_FIXED" if "P_FIXED" in self.results.columns else "P_RANDOM"
        if sort_column in self.results.columns:
            self.results = self.results.sort_values(sort_column)

        if "Q_DF" in self.results.columns:
            self.results = self.results.drop(columns=["Q_DF"])
            log.info("Removed Q_DF column from the output")

        if "STUDIES" in self.results.columns:
            self.results = self.results.drop(columns=["STUDIES"])
            log.info("Removed STUDIES column from the output")

        if "POPULATIONS" in self.results.columns:
            pop_col = self.results["POPULATIONS"].dropna()
            if len(pop_col) > 0:
                condition1 = pop_col.nunique() == 1
                condition2 = pop_col.iloc[0] == "Unknown"
                if condition1 and condition2:
                    self.results = self.results.drop(columns=["POPULATIONS"])
                    log.info(
                        "Removed POPULATIONS column from the output as all values are 'Unknown'"
                    )
            else:
                self.results = self.results.drop(columns=["POPULATIONS"])
                log.info(
                    "Removed POPULATIONS column from the output as all values are NaN"
                )

        z_score_columns = ["Z_FIXED", "Z_RANDOM"]
        for z_col in z_score_columns:
            if z_col in self.results.columns:
                self.results = self.results.drop(columns=[z_col])
                log.info(f"Removed {z_col} column from the output")

        if self.data_type == "GWAS":
            base_columns = ["RSID", "CHR", "BP", "POPULATIONS", "N_STUDIES", "TOTAL_N"]
        else:
            base_columns = ["CGID", "CHR", "BP", "POPULATIONS", "N_STUDIES", "TOTAL_N"]

        fixed_effects_columns = [
            "COEF_FIXED",
            "SE_FIXED",
            "P_FIXED",
            "FDR_FIXED",
            "HOLM_FIXED",
        ]

        random_effects_columns = [
            "COEF_RANDOM",
            "SE_RANDOM",
            "P_RANDOM",
            "TAU_SQUARED",
            "FDR_RANDOM",
            "HOLM_RANDOM",
        ]

        heterogeneity_columns = ["Q_STAT", "Q_P", "I_SQUARED"]

        drop_study_corrections = [
            col
            for col in self.results.columns
            if col.startswith("FDR_") or col.startswith("HOLM_")
        ]
        drop_study_corrections = [
            col
            for col in drop_study_corrections
            if not col.endswith(("_FIXED", "_RANDOM"))
        ]
        if drop_study_corrections:
            self.results = self.results.drop(columns=drop_study_corrections)

        study_coef_columns: List[str] = []
        for col in self.results.columns:
            condition1 = col.startswith("COEF_")
            condition2 = not col.endswith("_FIXED")
            condition3 = not col.endswith("_RANDOM")
            if condition1 and condition2 and condition3:
                study_coef_columns.append(col)

        study_se_columns: List[str] = []
        for col in self.results.columns:
            condition1 = col.startswith("SE_")
            condition2 = not col.endswith("_FIXED")
            condition3 = not col.endswith("_RANDOM")
            if condition1 and condition2 and condition3:
                study_se_columns.append(col)

        study_p_columns: List[str] = []
        for col in self.results.columns:
            condition1 = col.startswith("P_")
            condition2 = not col.endswith("_FIXED")
            condition3 = not col.endswith("_RANDOM")
            condition4 = col.count("_") >= 1
            if condition1 and condition2 and condition3 and condition4:
                study_p_columns.append(col)

        study_columns = study_coef_columns + study_se_columns + study_p_columns

        ordered_columns = base_columns
        if "COEF_FIXED" in self.results.columns:
            ordered_columns += fixed_effects_columns
        if "COEF_RANDOM" in self.results.columns:
            ordered_columns += random_effects_columns
        ordered_columns += heterogeneity_columns
        ordered_columns += study_columns

        ordered_columns = [
            col for col in ordered_columns if col in self.results.columns
        ]
        self.results = self.results[ordered_columns]

        correction_columns = [
            col
            for col in self.results.columns
            if col.startswith(("P_", "FDR_", "HOLM_"))
        ]
        for col in correction_columns:
            if col in self.results.columns:
                self.results[col] = pd.to_numeric(self.results[col], errors="coerce")
                self.results[col] = self.results[col].apply(
                    lambda x: self.format_scientific_notation(x, precision=2)
                )

        self.results.to_csv(self.output, index=False)
        log.success(f"Results saved to {self.output}")

    def run(self) -> Optional[pd.DataFrame]:
        if not self.studies:
            log.error(
                "No studies added for meta-analysis. Check validation errors above."
            )
            return None
        if len(self.studies) < 2:
            log.error(
                f"Need at least 2 studies for meta-analysis, have {len(self.studies)}"
            )
            return None
        log.info(f"Starting optimized meta-analysis with {self.threads} threads")
        log.info(
            f"Memory available: {self.available_memory:.1f}GB, chunk size: {self.chunk_size}"
        )
        results = self.perform_meta_analysis()
        if results is None:
            return None
        try:
            self.save_results()
        except Exception as e:
            log.error(f"Failed to save results: {e}")
        return results


options = [
    OptionConfig(flags=["-i", "--input"], type=str),
    OptionConfig(flags=["-n", "--names"], type=str),
    OptionConfig(flags=["-s", "--sample_sizes"], type=str),
    OptionConfig(flags=["-t", "--populations"], type=str, default=None),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
    OptionConfig(flags=["-m", "--method"], type=str, default="both"),
    OptionConfig(flags=["-d", "--data_type"], type=str, default="auto"),
    OptionConfig(flags=["-a", "--var"], type=str, default=None),
    OptionConfig(flags=["-p", "--precision"], type=str, default="auto"),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="MetaAssociation")
    opt = framework.run()
    analysis = MetaAssociation(
        input=opt.input,
        names=opt.names,
        sample_sizes=opt.sample_sizes,
        populations=opt.populations,
        output=opt.output,
        method=opt.method,
        data_type=opt.data_type,
        var=opt.var,
        precision_mode=opt.precision,
    )
    analysis.run()
