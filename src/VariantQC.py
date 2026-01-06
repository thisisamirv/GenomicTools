#!/usr/bin/env python
# Import required modules
import concurrent.futures
import gc
import h5py
import multiprocessing
import numba
import numpy as np
import os
import shutil
import sys
import tempfile
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.H5Utils import CachedH5Utils
from utils.LoggingUtils import log
from utils.SystemUtils import SystemUtils, monitor_resources

multiprocessing.set_start_method("spawn", force=True)


@numba.jit(nopython=True)
def _calculate_maf_jit(genotypes: np.ndarray) -> float:
    valid_mask = ~np.isnan(genotypes) & (genotypes != -1)
    if np.sum(valid_mask) == 0:
        return 0.0
    alt_count = np.sum(genotypes[valid_mask])
    total_alleles = 2 * np.sum(valid_mask)
    alt_freq = alt_count / total_alleles
    return min(alt_freq, 1 - alt_freq)


@numba.jit(nopython=True, parallel=True)
def _calculate_maf_batch_jit(genotypes: np.ndarray) -> np.ndarray:
    n_variants, n_samples = genotypes.shape
    mafs = np.zeros(n_variants, dtype=np.float32)
    for i in numba.prange(n_variants):
        valid_mask = ~np.isnan(genotypes[i]) & (genotypes[i] != -1)
        valid_count = np.sum(valid_mask)
        if valid_count == 0:
            mafs[i] = 0.0
            continue
        alt_count = np.sum(genotypes[i][valid_mask])
        total_alleles = 2 * valid_count
        alt_freq = alt_count / total_alleles
        mafs[i] = min(alt_freq, 1 - alt_freq)
    return mafs


@numba.jit(nopython=True)
def _hardy_weinberg_test_jit(n_hom_ref: int, n_het: int, n_hom_alt: int) -> float:
    n_samples = n_hom_ref + n_het + n_hom_alt
    if n_samples < 5:
        return 1.0
    n_alleles = 2 * n_samples
    n_ref = 2 * n_hom_ref + n_het
    n_alt = 2 * n_hom_alt + n_het
    p = n_ref / n_alleles
    q = n_alt / n_alleles
    expected_hom_ref = n_samples * (p * p)
    expected_het = n_samples * (2 * p * q)
    expected_hom_alt = n_samples * (q * q)
    if min(expected_hom_ref, expected_het, expected_hom_alt) < 5:
        return 1.0
    section1 = (n_hom_ref - expected_hom_ref) ** 2 / expected_hom_ref
    section2 = (n_het - expected_het) ** 2 / expected_het
    section3 = (n_hom_alt - expected_hom_alt) ** 2 / expected_hom_alt
    chisq = section1 + section2 + section3
    if chisq > 10.83:
        return 0.0
    elif chisq > 6.635:
        return 0.005
    elif chisq > 3.841:
        return 0.025
    else:
        return 0.5


@numba.jit(nopython=True, parallel=True)
def _process_hwe_batch_jit(genotypes: np.ndarray) -> np.ndarray:
    n_variants, n_samples = genotypes.shape
    hwe_pvalues = np.ones(n_variants, dtype=np.float32)
    for i in numba.prange(n_variants):
        counts = np.zeros(3, dtype=np.int32)
        for j in range(n_samples):
            geno = genotypes[i, j]
            if not np.isnan(geno) and geno != -1:
                ig = int(geno)
                if 0 <= ig <= 2:
                    counts[ig] += 1
        total = np.sum(counts)
        if total >= 10:
            hwe_pvalues[i] = _hardy_weinberg_test_jit(counts[0], counts[1], counts[2])
    return hwe_pvalues


@numba.jit(nopython=True, parallel=True)
def _calculate_r2_jit(geno1: np.ndarray, geno2: np.ndarray) -> float:
    n = len(geno1)
    valid_count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    for i in numba.prange(n):
        x = geno1[i]
        y = geno2[i]
        if not (np.isnan(x) or np.isnan(y) or x == -1 or y == -1):
            valid_count += 1
            sum_x += x
            sum_y += y
            sum_xx += x * x
            sum_yy += y * y
            sum_xy += x * y
    if valid_count < 10:
        return 0.0
    mean_x = sum_x / valid_count
    mean_y = sum_y / valid_count
    var_x = (sum_xx / valid_count) - (mean_x * mean_x)
    var_y = (sum_yy / valid_count) - (mean_y * mean_y)
    cov_xy = (sum_xy / valid_count) - (mean_x * mean_y)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    r = cov_xy / np.sqrt(var_x * var_y)
    return r * r


@numba.jit(nopython=True)
def _count_genotypes_jit(genotypes: np.ndarray) -> np.ndarray:
    counts = np.zeros(3, dtype=np.int32)
    for i in range(len(genotypes)):
        geno = genotypes[i]
        if not np.isnan(geno) and geno != -1:
            if geno == 0:
                counts[0] += 1
            elif geno == 1:
                counts[1] += 1
            elif geno == 2:
                counts[2] += 1
    return counts


@numba.jit(nopython=True, parallel=True)
def _ld_prune_full(
    genotypes: np.ndarray,
    r2_threshold: float,
    maf_threshold: float,
    window_size: int,
    step_size: int,
    mafs: Optional[np.ndarray],
) -> np.ndarray:
    n_variants, n_samples = genotypes.shape
    if mafs is None:
        mafs = _calculate_maf_batch_jit(genotypes)
    maf_mask = (
        mafs >= maf_threshold
        if maf_threshold > 0
        else np.ones(n_variants, dtype=np.bool_)
    )
    maf_indices = np.where(maf_mask)[0]
    keep = np.ones(n_variants, dtype=np.bool_)
    n = len(maf_indices)
    for start in range(0, n, step_size):
        end = min(start + window_size, n)
        window_indices = maf_indices[start:end]
        win_size = len(window_indices)
        if win_size <= 1:
            continue
        G_win = genotypes[window_indices]
        means = np.zeros((win_size, 1), dtype=np.float32)
        for i in numba.prange(win_size):
            valid = ~np.isnan(G_win[i]) & (G_win[i] != -1)
            if np.sum(valid) > 0:
                means[i] = np.sum(G_win[i][valid]) / np.sum(valid)
        for i in numba.prange(win_size):
            miss_mask = np.isnan(G_win[i]) | (G_win[i] == -1)
            G_win[i][miss_mask] = means[i]
        centered = G_win - means
        cov = np.zeros((win_size, win_size), dtype=np.float32)
        for i in numba.prange(win_size):
            for j in range(i, win_size):
                cov[i, j] = np.dot(centered[i], centered[j]) / (n_samples - 1)
                cov[j, i] = cov[i, j]
        vars_ = np.diag(cov).copy()
        r2_mat = np.zeros((win_size, win_size), dtype=np.float32)
        for i in numba.prange(win_size):
            for j in range(i + 1, win_size):
                if vars_[i] > 0 and vars_[j] > 0:
                    r = cov[i, j] / np.sqrt(vars_[i] * vars_[j])
                    r2_mat[i, j] = r * r
                    r2_mat[j, i] = r2_mat[i, j]
        win_mafs = mafs[window_indices]
        sort_idx = np.argsort(-win_mafs)
        kept = np.ones(win_size, dtype=np.bool_)
        for ii in range(win_size):
            i = sort_idx[ii]
            if not kept[i]:
                continue
            for jj in range(ii + 1, win_size):
                j = sort_idx[jj]
                if kept[j] and r2_mat[i, j] > r2_threshold:
                    kept[j] = False
        keep[window_indices[~kept]] = False
    return keep


class VariantQC:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        analysis_type: str = "maf",
        threshold: Optional[float] = None,
        pop_code: Optional[int] = None,
        window_size: int = 50,
        step_size: int = 5,
        r2_threshold: float = 0.2,
        maf_threshold: float = 0.01,
    ) -> None:
        self.safe_config = SystemUtils.configure_safe_environment()
        if self.safe_config.get("core_dumps_disabled", False):
            log.debug("Core dumps disabled for stability")
        if self.safe_config.get("memory_limit_set", False):
            log.debug("Memory limits configured to prevent OOM errors")

        SystemUtils.print_system_info()

        self.input_file = input_file
        self.analysis_type = analysis_type.lower()
        self.output_file = output_file
        self.pop_code = pop_code
        self.max_workers = SystemUtils.get_optimal_cores(reserve_cores=1)
        log.info(f"Auto-detected optimal threads: {self.max_workers}")
        self.window_size = window_size
        self.step_size = step_size
        self.r2_threshold = r2_threshold
        self.maf_threshold = maf_threshold
        self.temp_dirs = []

        valid_analyses = ["maf", "hwe", "ld_prune", "maf_filter"]
        if self.analysis_type not in valid_analyses:
            raise ValueError(
                f"Invalid analysis type '{analysis_type}'. Must be one of: {valid_analyses}"
            )
        if threshold is None:
            default_thresholds = {
                "maf": 0.01,
                "hwe": 1e-6,
                "ld_prune": 0.2,
                "maf_filter": 0.01,
            }
            self.threshold = default_thresholds[self.analysis_type]
        else:
            self.threshold = threshold
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        self.sample_mask: Optional[np.ndarray] = None

        memory_info = SystemUtils.get_memory_info()
        self.available_memory = memory_info.get("available_gb", 8.0)
        self.total_memory = memory_info.get("total_gb", 16.0)
        log.info(f"Available memory: {self.available_memory:.1f}GB")

        self._adjust_workers()
        self._setup_target_file()
        if self.analysis_type in ["ld_prune", "maf_filter"]:
            self._check_maf_availability()
        self._warmup_jit()

    def _warmup_jit(self) -> None:
        dummy_genotypes = np.array([[0, 1, 2, 0, 1]], dtype=np.float32)
        _calculate_maf_batch_jit(dummy_genotypes)
        _process_hwe_batch_jit(dummy_genotypes)

    def _check_system_health(self) -> bool:
        log.info("Checking system health before processing...")

        required_output_gb = self._estimate_output_size()

        health = SystemUtils.check_system_health(
            min_free_disk_gb=required_output_gb,
            max_cpu_percent=95.0,
            max_memory_percent=90.0,
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

        if self.output_file:
            output_dir = os.path.dirname(os.path.abspath(self.output_file))
            disk_ok, disk_message = SystemUtils.check_disk_space(
                path=output_dir,
                required_gb=required_output_gb,
                buffer_percent=15.0,
            )

            if not disk_ok:
                log.error(f"Disk space issue: {disk_message}")
                return False

        log.info("System health check passed")
        return True

    def _estimate_output_size(self) -> float:
        try:
            input_size_gb = os.path.getsize(self.input_file) / (1024**3)

            if self.analysis_type in ["maf", "hwe"]:
                return input_size_gb * 1.05

            elif self.analysis_type == "maf_filter":
                if self.threshold is None:
                    return input_size_gb * 0.95
                if self.threshold <= 0.01:
                    return input_size_gb * 0.95
                elif self.threshold <= 0.05:
                    return input_size_gb * 0.8
                else:
                    return input_size_gb * 0.6

            elif self.analysis_type == "ld_prune":
                return input_size_gb * 0.5

            return input_size_gb

        except Exception as e:
            log.debug(f"Error estimating output size: {e}")
            return 1.0

    def _cleanup(self) -> None:
        """Clean up resources after analysis"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    log.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                log.warn(f"Error cleaning up temp directory {temp_dir}: {e}")

        try:
            cleanup_result = SystemUtils.cleanup_stale_temp_files(
                prefix="variantqc_",
                max_age_hours=24,
                dry_run=False,
            )

            condition1 = cleanup_result.get("dirs_deleted", 0) > 0
            condition2 = cleanup_result.get("files_deleted", 0) > 0
            if condition1 or condition2:
                log.debug(
                    f"Cleaned up {cleanup_result.get('dirs_deleted', 0)} stale directories and "
                    f"{cleanup_result.get('files_deleted', 0)} files"
                )
        except Exception as e:
            log.debug(f"Error during stale file cleanup: {e}")

        gc.collect()

    def _setup_temp_directory(self) -> str:
        if not self.output_file:
            default_path = os.getcwd()
        else:
            default_path = os.path.dirname(os.path.abspath(self.output_file))

        required_gb = 0.5
        if self.analysis_type in ["ld_prune", "maf_filter"]:
            file_size_gb = os.path.getsize(self.input_file) / (1024**3)
            required_gb = max(0.5, file_size_gb * 0.2)

        try:
            temp_dir, temp_info = SystemUtils.create_safe_tempdir(
                default_path=default_path,
                required_gb=required_gb,
                prefix=f"variantqc_{self.analysis_type}",
                buffer_percent=10.0,
            )
            log.info(f"Created temporary directory: {temp_dir}")
            self.temp_dirs.append(temp_dir)
            return temp_dir
        except Exception as e:
            log.warn(f"Failed to create safe temp directory: {e}")
            log.warn("Using system temp directory as fallback")
            temp_dir = tempfile.mkdtemp(prefix=f"variantqc_{self.analysis_type}")
            self.temp_dirs.append(temp_dir)
            return temp_dir

    def _adjust_workers(self) -> None:
        try:
            file_size = os.path.getsize(self.input_file) / (1024**3)
            log.debug(f"Input file size: {file_size:.2f}GB")

            memory_per_worker = {
                "maf": 2.0,
                "hwe": 2.5,
                "ld_prune": 3.0,
                "maf_filter": 2.0,
            }

            required_memory = memory_per_worker.get(self.analysis_type, 2.5)
            usable_memory = max(0.0, self.available_memory - 2.0)
            memory_based_workers = max(1, int(usable_memory / required_memory))

            if file_size > 50:
                size_based_limit = min(4, self.max_workers)
                log.info(
                    f"Large file detected ({file_size:.1f}GB). Limiting workers to {size_based_limit}"
                )
            elif file_size > 20:
                size_based_limit = min(6, self.max_workers)
                log.info(
                    f"Large file detected ({file_size:.1f}GB). Adjusting workers to {size_based_limit}"
                )
            elif file_size > 10:
                size_based_limit = min(8, self.max_workers)
            else:
                size_based_limit = self.max_workers

            if self.analysis_type == "ld_prune":
                size_based_limit = max(1, size_based_limit - 1)

            self.max_workers = min(
                self.max_workers, memory_based_workers, size_based_limit
            )

            self.max_workers = max(1, self.max_workers)

            log.info(
                f"Adjusted worker count: {self.max_workers} based on available resources"
            )
            log.debug("Worker adjustment:")
            log.debug(f"CPU: {SystemUtils.get_optimal_cores()}")
            log.debug(f"memory: {memory_based_workers}, file size: {size_based_limit}")

        except Exception as e:
            log.warn(f"Error adjusting workers: {e}")
            self.max_workers = max(
                1, min(2, SystemUtils.get_optimal_cores(reserve_cores=1))
            )
            log.info(f"Using fallback worker count: {self.max_workers}")

    def _setup_target_file(self) -> None:
        if self.analysis_type in ["ld_prune", "maf_filter"]:
            if not self.output_file:
                raise ValueError(f"{self.analysis_type} requires an output file")
            self.target_file = self.output_file
        else:
            if self.output_file and self.output_file != self.input_file:
                if os.path.exists(self.output_file):
                    log.warn(
                        f"Output file {self.output_file} already exists. It will be overwritten."
                    )
                log.info(f"Creating a copy of the input file at: {self.output_file}")
                shutil.copy2(self.input_file, self.output_file)
                self.target_file = self.output_file
            else:
                self.target_file = self.input_file
                log.info(
                    f"Will add {self.analysis_type.upper()} values directly to: {self.target_file}"
                )

    def _check_maf_availability(self) -> None:
        self.has_maf_values: bool = False
        try:
            with h5py.File(self.input_file, "r") as h5f:
                h5_utils = CachedH5Utils(h5f)
                chromosomes = h5_utils.get_chromosomes()
                if chromosomes:
                    self.has_maf_values = (
                        AliasUtils.find_keys(h5f[chromosomes[0]], "MAF") is not None
                    )
            if self.has_maf_values:
                log.info(
                    f"Found pre-calculated MAF values - will use these for {self.analysis_type}"
                )
            else:
                log.info(
                    f"No pre-calculated MAF values found - will calculate as needed for {self.analysis_type}"
                )
        except Exception as e:
            log.debug(f"Error checking MAF availability: {e}")

    def create_population_mask(self) -> bool:
        try:
            if self.pop_code is None or self.pop_code == -9:
                log.info("Using all samples (no population filtering)")
                self.sample_mask = None
                return True
            log.info(f"Creating sample mask for population code: {self.pop_code}")
            with h5py.File(self.input_file, "r") as h5_file:
                metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
                if metadata_key is None:
                    log.warn("No metadata group found in HDF5 file. Using all samples.")
                    self.sample_mask = None
                    return True
                population_key = AliasUtils.find_keys(
                    h5_file[metadata_key], "Population"
                )
                if population_key is None:
                    log.warn("No population data found in metadata. Using all samples.")
                    self.sample_mask = None
                    return True
                populations = h5_file[f"{metadata_key}/{population_key}"][:]
                if isinstance(populations[0], bytes):
                    populations = [p.decode("utf-8") if p else "" for p in populations]
                try:
                    pop_codes = [int(p) if p else -9 for p in populations]
                    self.sample_mask = np.array(pop_codes) == self.pop_code
                except ValueError:
                    self.sample_mask = np.array(populations) == str(self.pop_code)
                if np.sum(self.sample_mask) == 0:
                    log.error(f"No samples found for population code: {self.pop_code}")
                    return False
                log.info(
                    f"Found {np.sum(self.sample_mask)} samples for population code: {self.pop_code}"
                )
                return True
        except Exception as e:
            log.error(f"Error creating population mask: {e}")
            self.sample_mask = None
            return False

    def calculate_maf(self, genotypes: np.ndarray) -> Union[float, np.ndarray]:
        if genotypes.ndim == 1:
            return _calculate_maf_jit(genotypes)
        else:
            return _calculate_maf_batch_jit(genotypes)

    def hardy_weinberg_test(
        self, observed_genotypes: Union[List[int], np.ndarray]
    ) -> float:
        n_hom_ref, n_het, n_hom_alt = observed_genotypes
        return _hardy_weinberg_test_jit(int(n_hom_ref), int(n_het), int(n_hom_alt))

    def calculate_r2(self, geno1: np.ndarray, geno2: np.ndarray) -> float:
        return _calculate_r2_jit(geno1, geno2)

    def process_chromosome_maf(self, chromosome: str) -> Optional[Dict[str, Any]]:
        try:
            log.debug(f"Processing MAF for chromosome: {chromosome}")
            with h5py.File(self.input_file, "r") as h5_file:
                if chromosome not in h5_file:
                    log.warn(f"Chromosome {chromosome} not found")
                    return None
                genotype_key = AliasUtils.find_keys(h5_file[chromosome], "Genotype")
                snp_key = AliasUtils.find_keys(h5_file[chromosome], "RSID")
                if genotype_key is None:
                    log.warn(f"No genotype data found in {chromosome}")
                    return None
                if snp_key is None:
                    log.warn(f"No SNP IDs found in {chromosome}")
                    return None
                genotypes = h5_file[f"{chromosome}/{genotype_key}"][:]
                snp_ids = h5_file[f"{chromosome}/{snp_key}"][:]
                if isinstance(snp_ids[0], bytes):
                    snp_ids = [s.decode("utf-8") for s in snp_ids]
                log.debug(
                    f"Calculating MAF for {len(snp_ids)} variants in {chromosome}"
                )
                variant_mafs = _calculate_maf_batch_jit(genotypes)
                below_threshold_count = np.sum(variant_mafs < self.threshold)
                result: Dict[str, Any] = {
                    "chromosome": chromosome,
                    "maf_values": np.array(variant_mafs, dtype=np.float32),
                    "below_threshold": int(below_threshold_count),
                    "total_variants": int(len(variant_mafs)),
                }
                log.debug(f"Calculated {len(variant_mafs)} MAF values for {chromosome}")
                return result
        except Exception as e:
            log.error(f"Error processing MAF for chromosome {chromosome}: {e}")
            return None

    def process_chromosome_hwe(self, chromosome: str) -> Optional[Dict[str, Any]]:
        try:
            log.debug(f"Processing HWE for chromosome: {chromosome}")
            with h5py.File(self.input_file, "r") as h5_file:
                if chromosome not in h5_file:
                    log.warn(f"Chromosome {chromosome} not found")
                    return None
                genotype_key = AliasUtils.find_keys(h5_file[chromosome], "Genotype")
                snp_key = AliasUtils.find_keys(h5_file[chromosome], "RSID")
                if genotype_key is None:
                    log.warn(f"No genotype data found in {chromosome}")
                    return None
                if snp_key is None:
                    log.warn(f"No SNP IDs found in {chromosome}")
                    return None
                genotypes = h5_file[f"{chromosome}/{genotype_key}"][:]
                if self.sample_mask is not None:
                    genotypes = genotypes[:, self.sample_mask]
                snp_ids = h5_file[f"{chromosome}/{snp_key}"][:]
                if isinstance(snp_ids[0], bytes):
                    snp_ids = [s.decode("utf-8") for s in snp_ids]
                log.debug(
                    f"Calculating HWE for {len(snp_ids)} variants in {chromosome}"
                )
                hwe_pvalues = _process_hwe_batch_jit(genotypes)
                deviation_count = int(np.sum(hwe_pvalues < self.threshold))
                result: Dict[str, Any] = {
                    "chromosome": chromosome,
                    "hwe_pvalues": hwe_pvalues,
                    "below_threshold": deviation_count,
                    "total_variants": int(len(hwe_pvalues)),
                }
                log.debug(
                    f"Calculated {len(hwe_pvalues)} HWE p-values for {chromosome}"
                )
                return result
        except Exception as e:
            log.error(f"Error processing HWE for chromosome {chromosome}: {e}")
            return None

    def process_chromosome_maf_filter(
        self, chromosome: str
    ) -> Optional[Dict[str, Any]]:
        try:
            log.info(f"Filtering {chromosome} with MAF threshold {self.threshold}")
            with h5py.File(self.input_file, "r") as h5f:
                if chromosome not in h5f:
                    log.warn(f"Chromosome {chromosome} not found in input file")
                    return None
                genotype_key = AliasUtils.find_keys(h5f[chromosome], "Genotype")
                if genotype_key is None:
                    log.warn(f"No genotype data found in {chromosome}")
                    return None
                genotypes = h5f[chromosome][genotype_key][:]
                n_variants, n_samples = genotypes.shape
                log.debug(f"Processing {n_variants} variants in {chromosome}")
                maf_key = AliasUtils.find_keys(h5f[chromosome], "MAF")
                if self.has_maf_values and maf_key is not None:
                    log.debug(f"Using pre-calculated MAF values for {chromosome}")
                    mafs = h5f[chromosome][maf_key][:]
                else:
                    log.debug(f"Calculating MAF values for {chromosome}")
                    mafs = self.calculate_maf(genotypes)
                keep_mask = mafs >= self.threshold
                n_kept = int(np.sum(keep_mask))
                if n_kept == 0:
                    log.warn(
                        f"No variants in {chromosome} pass MAF filter (threshold: {self.threshold})"
                    )
                    return None
                log.info(
                    f"Retained {n_kept}/{n_variants} variants in {chromosome} after MAF filtering"
                )
                result: Dict[str, Any] = {
                    "chromosome": chromosome,
                    "keep_mask": keep_mask,
                    "maf_values": mafs,
                    "n_original": int(n_variants),
                    "n_kept": n_kept,
                }
                return result
        except Exception as e:
            log.error(f"Error filtering {chromosome}: {e}")
            return None

    def process_chromosome_ld_prune(self, chromosome: str) -> Optional[Dict[str, Any]]:
        try:
            log.info(
                f"Pruning {chromosome} with window={self.window_size}, step={self.step_size}, r²={self.r2_threshold}"
            )
            with h5py.File(self.input_file, "r") as h5f:
                if chromosome not in h5f:
                    log.warn(f"Chromosome {chromosome} not found in input file")
                    return None
                genotype_key = AliasUtils.find_keys(h5f[chromosome], "Genotype")
                if genotype_key is None:
                    log.warn(f"No genotype data found in {chromosome}")
                    return None
                genotypes = h5f[chromosome][genotype_key][:]
                n_variants, n_samples = genotypes.shape
                log.debug(f"Processing {n_variants} variants in {chromosome}")
                maf_key = AliasUtils.find_keys(h5f[chromosome], "MAF")
                if self.maf_threshold > 0 or not self.has_maf_values or maf_key is None:
                    log.debug(f"Calculating MAF values for {chromosome}")
                    mafs = _calculate_maf_batch_jit(genotypes)
                else:
                    log.debug(f"Using pre-calculated MAF values for {chromosome}")
                    mafs = h5f[chromosome][maf_key][:]
                keep = _ld_prune_full(
                    genotypes,
                    self.r2_threshold,
                    self.maf_threshold,
                    self.window_size,
                    self.step_size,
                    mafs,
                )
                n_kept = int(np.sum(keep))
                if n_kept == 0:
                    log.warn(f"No variants retained for {chromosome} after pruning")
                    return None
                log.info(
                    f"Retained {n_kept}/{n_variants} variants in {chromosome} after pruning"
                )
                result: Dict[str, Any] = {
                    "chromosome": chromosome,
                    "keep_mask": keep,
                    "n_original": int(n_variants),
                    "n_kept": n_kept,
                }
                return result
        except Exception as e:
            log.error(f"Error pruning {chromosome}: {e}")
            return None

    def add_values_to_h5(
        self, results_dict: Dict[str, Any], analysis_type: str
    ) -> None:
        try:
            log.info(
                f"Adding {analysis_type.upper()} values to HDF5 file: {self.target_file}"
            )
            if analysis_type == "hwe":
                dataset_name = (
                    "hwe" if self.pop_code in [None, -9] else f"hwe_pop{self.pop_code}"
                )
            else:
                dataset_name = analysis_type
            with h5py.File(self.target_file, "r+") as h5_file:
                for chrom, result in results_dict.items():
                    if chrom not in h5_file:
                        log.warn(f"Chromosome {chrom} not found in HDF5 file")
                        continue
                    if dataset_name in h5_file[chrom]:
                        del h5_file[chrom][dataset_name]
                    if analysis_type == "maf":
                        h5_file[chrom].create_dataset(
                            dataset_name, data=result["maf_values"]
                        )
                    elif analysis_type == "hwe":
                        h5_file[chrom].create_dataset(
                            dataset_name, data=result["hwe_pvalues"]
                        )
                    log.debug(f"Added {analysis_type.upper()} values for {chrom}")
            log.success(
                f"Successfully added {analysis_type.upper()} values to HDF5 file"
            )
        except Exception as e:
            log.error(f"Error adding {analysis_type.upper()} values to HDF5 file: {e}")

    def write_filtered_data(
        self, filter_results: List[Optional[Dict[str, Any]]], filter_type: str
    ) -> str:
        log.info(f"Writing {filter_type} filtered data to {self.output_file}")
        try:
            with h5py.File(self.output_file, "w") as out_h5:
                with h5py.File(self.input_file, "r") as in_h5:
                    metadata_key = AliasUtils.find_keys(in_h5, "Metadata")
                    if metadata_key is not None:
                        in_h5.copy(metadata_key, out_h5)
                    if "filter_info" not in out_h5:
                        filter_grp = out_h5.create_group("filter_info")
                    else:
                        filter_grp = out_h5["filter_info"]
                    if filter_type == "maf_filter":
                        filter_grp.attrs["maf_filter_threshold"] = self.threshold
                        filter_grp.attrs["filter_type"] = "maf_filter"
                    elif filter_type == "ld_prune":
                        filter_grp.attrs["ld_prune_window"] = self.window_size
                        filter_grp.attrs["ld_prune_step"] = self.step_size
                        filter_grp.attrs["ld_prune_r2"] = self.r2_threshold
                        filter_grp.attrs["ld_prune_maf"] = self.maf_threshold
                        filter_grp.attrs["filter_type"] = "ld_prune"
                    total_variants = 0
                    total_kept = 0
                    for result in filter_results:
                        if result is None:
                            continue
                        chrom = result["chromosome"]
                        keep_mask = result["keep_mask"]
                        grp = out_h5.create_group(chrom)
                        for dataset_name in in_h5[chrom]:
                            data = in_h5[chrom][dataset_name][:]
                            if data.ndim == 1 and len(data) == len(keep_mask):
                                filtered = data[keep_mask]
                                grp.create_dataset(dataset_name, data=filtered)
                            elif data.ndim == 2 and data.shape[0] == len(keep_mask):
                                filtered = data[keep_mask]
                                grp.create_dataset(dataset_name, data=filtered)
                            elif data.ndim == 2 and data.shape[1] == len(keep_mask):
                                filtered = data[:, keep_mask]
                                grp.create_dataset(dataset_name, data=filtered)
                            else:
                                grp.create_dataset(dataset_name, data=data)
                        if filter_type == "maf_filter" and "maf_values" in result:
                            filtered_mafs = result["maf_values"][keep_mask]
                            grp.create_dataset("maf", data=filtered_mafs)
                        total_variants += int(result.get("n_original", 0))
                        total_kept += int(result.get("n_kept", 0))
                    if total_variants > 0:
                        filter_grp.attrs["original_variants"] = total_variants
                        filter_grp.attrs["filtered_variants"] = total_kept
                        filter_grp.attrs["retention_rate"] = total_kept / total_variants
            if total_variants > 0:
                retention_pct = (total_kept / total_variants) * 100
                filter_name = (
                    "MAF filtering" if filter_type == "maf_filter" else "LD pruning"
                )
                log.success(f"{filter_name} complete.")
                log.success(
                    f"Retained {total_kept:,}/{total_variants:,} variants ({retention_pct:.2f}%)"
                )
            return self.output_file
        except Exception as e:
            log.error(f"Error writing filtered data: {e}")
            sys.exit(1)

    def write_pruned_data(self, pruned_results: List[Optional[Dict[str, Any]]]) -> str:
        return self.write_filtered_data(pruned_results, "ld_prune")

    def run_maf_analysis(self) -> Optional[str]:
        try:
            log.info("Starting MAF calculation")
            log.info(f"Input file: {self.input_file}")
            log.info(f"Output file: {self.output_file}")
            with h5py.File(self.input_file, "r") as h5_file:
                h5_utils = CachedH5Utils(h5_file)
                chromosome_list = h5_utils.get_chromosomes()
            if not chromosome_list:
                raise ValueError("No chromosomes found in the HDF5 file")
            log.info(f"Found {len(chromosome_list)} chromosomes")
            log.info(f"Using {self.max_workers} cores for parallel processing")
            print("Calculating Minor Allele Frequencies...")
            results_dict: Dict[str, Any] = {}
            total_variants = 0
            total_below_threshold = 0
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                future_to_chr = {
                    executor.submit(self.process_chromosome_maf, chromosome): chromosome
                    for chromosome in chromosome_list
                }
                for future in concurrent.futures.as_completed(future_to_chr):
                    chromosome = future_to_chr[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results_dict[result["chromosome"]] = result
                            total_variants += int(result["total_variants"])
                            total_below_threshold += int(result["below_threshold"])
                    except Exception as e:
                        log.error(f"Error in chromosome {chromosome} processing: {e}")
            if results_dict:
                self.add_values_to_h5(results_dict, "maf")
                log.info(f"Total variants processed: {total_variants}")
                log.info(
                    f"Variants with MAF < {self.threshold}: {total_below_threshold}"
                )
                log.info(
                    f"Percentage below threshold: {total_below_threshold / total_variants * 100:.2f}%"
                )
                log.success("MAF calculation complete. Values added to HDF5 file.")
                return getattr(self, "target_file", None)
            else:
                log.error("No MAF values were calculated")
                return None
        except Exception as e:
            log.error(f"Error in MAF calculation: {e}")
            return None

    def run_hwe_analysis(self) -> Optional[str]:
        try:
            log.info("Starting HWE calculation")
            log.info(f"Input file: {self.input_file}")
            log.info(f"Output file: {self.output_file}")
            log.info(f"Significance threshold: {self.threshold}")
            if not self.create_population_mask():
                log.error("Failed to create population filter. Aborting.")
                return None
            with h5py.File(self.input_file, "r") as h5_file:
                h5_utils = CachedH5Utils(h5_file)
                chromosome_list = h5_utils.get_chromosomes()
            if not chromosome_list:
                raise ValueError("No chromosomes found in the HDF5 file")
            log.info(f"Found {len(chromosome_list)} chromosomes")
            log.info(f"Using {self.max_workers} cores for parallel processing")
            pop_desc = (
                f"population {self.pop_code}"
                if self.pop_code not in [None, -9]
                else "all samples"
            )
            print(f"Calculating Hardy-Weinberg Equilibrium p-values for {pop_desc}...")
            results_dict: Dict[str, Any] = {}
            total_variants = 0
            total_below_threshold = 0
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                future_to_chr = {
                    executor.submit(self.process_chromosome_hwe, chromosome): chromosome
                    for chromosome in chromosome_list
                }
                for future in concurrent.futures.as_completed(future_to_chr):
                    chromosome = future_to_chr[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results_dict[result["chromosome"]] = result
                            total_variants += int(result["total_variants"])
                            total_below_threshold += int(result["below_threshold"])
                    except Exception as e:
                        log.error(f"Error in chromosome {chromosome} processing: {e}")
            if results_dict:
                self.add_values_to_h5(results_dict, "hwe")
                log.info(f"Total variants processed: {total_variants}")
                log.info(
                    f"Variants deviating from HWE (p < {self.threshold}): {total_below_threshold}"
                )
                log.info(
                    f"Percentage deviating: {total_below_threshold / total_variants * 100:.2f}%"
                )
                log.success("HWE calculation complete. Values added to HDF5 file.")
                return getattr(self, "target_file", None)
            else:
                log.error("No HWE p-values were calculated")
                return None
        except Exception as e:
            log.error(f"Error in HWE calculation: {e}")
            return None

    def run_maf_filter_analysis(self) -> Optional[str]:
        try:
            log.info("Starting MAF filtering")
            log.info(f"Input file: {self.input_file}")
            log.info(f"Output file: {self.output_file}")
            log.info(
                f"MAF threshold: {self.threshold} (variants below this will be removed)"
            )
            with h5py.File(self.input_file, "r") as h5f:
                h5_utils = CachedH5Utils(h5f)
                chromosomes = h5_utils.get_chromosomes()
                log.info(f"Found {len(chromosomes)} chromosomes to process")
            if not chromosomes:
                log.error("No chromosomes found in input file")
                return None
            log.info(f"Filtering chromosomes using {self.max_workers} threads")
            results: List[Dict[str, Any]] = []
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                filtering_results = list(
                    tqdm(
                        executor.map(self.process_chromosome_maf_filter, chromosomes),
                        total=len(chromosomes),
                        desc="Filtering chromosomes by MAF",
                    )
                )
                results = [result for result in filtering_results if result is not None]
            if not results:
                log.error("No variants retained after MAF filtering for any chromosome")
                return None
            output_file = self.write_filtered_data(results, "maf_filter")
            return output_file
        except Exception as e:
            log.error(f"Error during MAF filtering: {e}")
            raise

    def run_ld_prune_analysis(self) -> Optional[str]:
        try:
            log.info("Starting LD pruning")
            log.info(f"Input file: {self.input_file}")
            log.info(f"Output file: {self.output_file}")
            log.info(
                f"Window parameters: size={self.window_size}, step={self.step_size}"
            )
            log.info(
                f"Threshold parameters: r²={self.r2_threshold}, MAF={self.maf_threshold}"
            )
            with h5py.File(self.input_file, "r") as h5f:
                h5_utils = CachedH5Utils(h5f)
                chromosomes = h5_utils.get_chromosomes()
                log.info(f"Found {len(chromosomes)} chromosomes to process")
            if not chromosomes:
                log.error("No chromosomes found in input file")
                return None
            log.info(f"Pruning chromosomes using {self.max_workers} threads")
            results: List[Dict[str, Any]] = []
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                pruning_results = list(
                    tqdm(
                        executor.map(self.process_chromosome_ld_prune, chromosomes),
                        total=len(chromosomes),
                        desc="Pruning chromosomes",
                    )
                )
                results = [result for result in pruning_results if result is not None]
            if not results:
                log.error("No variants retained after pruning for any chromosome")
                return None
            output_file = self.write_pruned_data(results)
            return output_file
        except Exception as e:
            log.error(f"Error during LD pruning: {e}")
            raise

    def run(self) -> Optional[str]:
        try:
            if not self._check_system_health():
                log.error("System health check failed - analysis may be unstable")
                user_input = input("Continue anyway? (y/N): ")
                if user_input.lower() != "y":
                    log.info("Analysis cancelled by user")
                    return None

            with monitor_resources(interval=5.0) as stats:
                try:
                    result = None

                    if self.analysis_type == "maf":
                        result = self.run_maf_analysis()
                    elif self.analysis_type == "hwe":
                        result = self.run_hwe_analysis()
                    elif self.analysis_type == "ld_prune":
                        result = self.run_ld_prune_analysis()
                    elif self.analysis_type == "maf_filter":
                        result = self.run_maf_filter_analysis()
                    else:
                        raise ValueError(f"Unknown analysis type: {self.analysis_type}")

                    log.info("Analysis completed successfully")
                    log.info(
                        f"Peak resource usage - CPU: {stats['max_cpu']:.1f}%, Memory: {stats['max_memory']:.1f}%"
                    )

                    return result

                except Exception as e:
                    log.error(f"Error during analysis: {e}")
                    log.info("Peak resource usage before failure:")
                    log.info(
                        f"CPU: {stats['max_cpu']:.1f}%, Memory: {stats['max_memory']:.1f}%"
                    )
                    raise
        except Exception as e:
            log.error(f"Error in VariantQC: {e}")
            return None
        finally:
            self._cleanup()


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(
        flags=["-a", "--analysis"],
        type=str,
        default="maf",
        required=False,
        choices=["maf", "hwe", "ld_prune", "maf_filter"],
    ),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
    OptionConfig(flags=["-t", "--threshold"], type=float, default=None, required=False),
    OptionConfig(flags=["-p", "--pop_code"], type=int, default=None, required=False),
    OptionConfig(flags=["-w", "--window"], type=int, default=50, required=False),
    OptionConfig(flags=["-s", "--step"], type=int, default=5, required=False),
    OptionConfig(flags=["-r", "--r2"], type=float, default=0.2, required=False),
    OptionConfig(flags=["-m", "--maf"], type=float, default=0.01, required=False),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="VariantQC")
    opt = framework.run()
    analyzer = VariantQC(
        input_file=opt.input,
        analysis_type=opt.analysis,
        output_file=opt.output,
        threshold=opt.threshold,
        pop_code=opt.pop_code,
        window_size=opt.window,
        step_size=opt.step,
        r2_threshold=opt.r2,
        maf_threshold=opt.maf,
    )
    analyzer.run()
