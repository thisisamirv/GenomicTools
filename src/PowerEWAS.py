#!/usr/bin/env python
# Import required modules
import ctypes
import datetime
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import subprocess
import tempfile
from dataclasses import dataclass
from scipy import stats
from scipy.stats import truncnorm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.ExperimentHub import ExperimentHub
from utils.LoggingUtils import log
from utils.SystemUtils import SystemUtils, monitor_resources


@dataclass
class ChunkConfiguration:
    generation_chunk_size: int
    testing_chunk_size: int
    n_cnt: int
    n_tx: int
    total_samples: int


@dataclass
class PowerAnalysisConfig:
    min_sample_size: int
    max_sample_size: int
    sample_size_steps: int
    control_proportion: float
    n_cpgs: int
    target_dm_cpgs: int
    tissue_type: str
    detection_limit: float
    dm_method: str
    fdr_threshold: float
    n_simulations: int
    output_file: Optional[str]
    seed: int
    target_delta: Optional[List[float]] = None
    delta_sd: Optional[List[float]] = None


class MemoryManager:
    def __init__(self) -> None:
        self.system_info = SystemUtils.get_system_info()
        self.available_memory_gb = self.system_info["ram_available_gb"]
        self._chunk_size_cache = {}

    def calculate_optimal_chunk_size(
        self,
        n_cpgs: int,
        n_samples_total: int,
        dtype_size: int = 8,
        safety_factor: float = 0.6,
    ) -> int:
        cache_key = (n_cpgs, n_samples_total, dtype_size, safety_factor)

        if cache_key in self._chunk_size_cache:
            return self._chunk_size_cache[cache_key]

        memory_per_cpg_bytes = n_samples_total * dtype_size * 6 + 2 * dtype_size
        usable_memory_bytes = self.available_memory_gb * (1024**3) * safety_factor
        optimal_chunk_size = int(usable_memory_bytes / memory_per_cpg_bytes)

        min_chunk = 1000
        optimal_chunk_size = max(min_chunk, min(optimal_chunk_size, n_cpgs))

        self._chunk_size_cache[cache_key] = optimal_chunk_size
        return optimal_chunk_size

    def get_cached_chunk_size(
        self,
        n_cpgs: int,
        n_samples_total: int,
        dtype_size: int = 8,
        safety_factor: float = 0.6,
    ) -> int:
        cache_key = (n_cpgs, n_samples_total, dtype_size, safety_factor)

        if cache_key not in self._chunk_size_cache:
            return self.calculate_optimal_chunk_size(
                n_cpgs, n_samples_total, dtype_size, safety_factor
            )

        return self._chunk_size_cache[cache_key]

    def cleanup_memory(self) -> None:
        gc.collect()
        if hasattr(ctypes, "windll"):
            try:
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            except Exception:
                pass

    def clear_cache(self) -> None:
        self._chunk_size_cache.clear()


class PowerEWASDataManager:
    _instance = None
    _data_cache = {}

    TISSUE_MAPPING = {
        "Saliva": "EH3068",
        "Lymphoma": "EH3069",
        "Placenta": "EH3070",
        "Liver": "EH3071",
        "Colon": "EH3072",
        "Blood adult": "EH3073",
        "Blood 5 year olds": "EH3074",
        "Blood newborns": "EH3075",
        "Cord-blood (whole blood)": "EH3076",
        "Cord-blood (PBMC)": "EH3077",
        "Adult (PBMC)": "EH3078",
        "Sperm": "EH3079",
    }

    def __new__(cls) -> "PowerEWASDataManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self._initialized = True

    @classmethod
    def get_instance(cls) -> "PowerEWASDataManager":
        return cls()

    def load_dataset(self, tissue_type: str) -> Dict[str, np.ndarray]:
        if tissue_type in self._data_cache:
            log.info(f"Using cached data for {tissue_type}")
            return self._data_cache[tissue_type]

        if tissue_type not in self.TISSUE_MAPPING:
            available_types = list(self.TISSUE_MAPPING.keys())
            raise ValueError(
                f"Tissue type not found: {tissue_type}. Available: {available_types}"
            )

        log.info(f"Loading dataset for tissue type: {tissue_type}")

        try:
            result = self._load_from_experiment_hub(tissue_type)
            self._validate_dataset(result, tissue_type)
            self._data_cache[tissue_type] = result

            log.info(f"Successfully loaded {len(result['mu'])} CpGs")
            log.debug(
                f"Mu range: {np.min(result['mu']):.6f} - {np.max(result['mu']):.6f}"
            )
            log.debug(
                f"Var range: {np.min(result['var']):.8f} - {np.max(result['var']):.6f}"
            )

            return result

        except Exception as e:
            log.error(f"Failed to load dataset from ExperimentHub: {e}")
            raise RuntimeError(f"Could not load {tissue_type} dataset. Error: {e}")

    def _load_from_experiment_hub(self, tissue_type: str) -> Dict[str, np.ndarray]:
        hub = ExperimentHub.get_cached_hub(local_hub=True, ask=False, auto_convert=True)
        eh_id = self.TISSUE_MAPPING[tissue_type]
        log.debug(f"Fetching ExperimentHub resource: {eh_id}")

        resource = hub[eh_id]

        if isinstance(resource, dict):
            if "data" in resource:
                return self._extract_from_enhanced_hub(resource["data"], tissue_type)
            elif "mu" in resource and "var" in resource:
                return {
                    "mu": np.array(resource["mu"]),
                    "var": np.array(resource["var"]),
                }
            else:
                raise ValueError(f"Unexpected resource format: {list(resource.keys())}")
        else:
            return self._extract_from_object(resource)

    def _extract_from_enhanced_hub(
        self, data: Dict[str, Any], tissue_type: str
    ) -> Dict[str, np.ndarray]:
        log.debug(f"Extracting from enhanced hub data: {list(data.keys())}")

        for key, value in data.items():
            if isinstance(value, dict) and "mu" in value and "var" in value:
                return {"mu": np.array(value["mu"]), "var": np.array(value["var"])}

            elif hasattr(value, "__array__") or isinstance(value, (list, tuple)):
                try:
                    arr = np.array(value)
                    if len(arr.shape) == 2 and arr.shape[1] == 2:
                        return {"mu": arr[:, 0], "var": arr[:, 1]}
                except Exception as e:
                    log.debug(f"Failed to convert {key} to array: {e}")

        numeric_data = ExperimentHub.get_numeric_arrays(
            self.TISSUE_MAPPING[tissue_type], local_hub=True
        )

        if len(numeric_data) >= 2:
            arrays = list(numeric_data.values())
            if len(arrays[0]) == len(arrays[1]):
                log.info("Using numeric data extraction")
                return {"mu": arrays[0], "var": arrays[1]}

        raise ValueError("Could not extract mu/var from enhanced hub data")

    def _extract_from_object(self, obj: Any) -> Dict[str, np.ndarray]:
        if hasattr(obj, "mu") and hasattr(obj, "var"):
            return {"mu": np.array(obj.mu), "var": np.array(obj.var)}

        if hasattr(obj, "__getitem__"):
            try:
                return {"mu": np.array(obj["mu"]), "var": np.array(obj["var"])}
            except (KeyError, TypeError):
                pass

        raise ValueError(f"Could not extract mu/var from object of type {type(obj)}")

    def _validate_dataset(self, result: Dict[str, np.ndarray], tissue_type: str):
        if "mu" not in result or "var" not in result:
            raise ValueError("Could not extract mu and var arrays")

        mu_data = result["mu"]
        var_data = result["var"]

        if len(mu_data) != len(var_data):
            raise ValueError(
                f"Mismatched array lengths: mu={len(mu_data)}, var={len(var_data)}"
            )


class StatisticalUtilities:
    @staticmethod
    def beta_to_m_value(beta_vals: np.ndarray) -> np.ndarray:
        beta_clipped = np.clip(beta_vals, 1e-15, 1 - 1e-15)
        return np.log2(beta_clipped / (1 - beta_clipped))

    @staticmethod
    def calculate_fdr(p_values: np.ndarray) -> np.ndarray:
        return multipletests(p_values, method="fdr_bh")[1]


class StatisticalTestEngine:
    def __init__(self, memory_manager: MemoryManager) -> None:
        self.memory_manager = memory_manager
        self.stats_utils = StatisticalUtilities()

    def perform_test(
        self,
        g1_beta: np.ndarray,
        g2_beta: np.ndarray,
        n_cnt: int,
        n_tx: int,
        method: str,
        chunk_size: int,
    ) -> Dict[str, np.ndarray]:
        n_cpgs = g1_beta.shape[0]
        n_chunks = (n_cpgs + chunk_size - 1) // chunk_size

        all_pvals = np.zeros(n_cpgs, dtype=np.float64)
        show_progress = n_chunks > 1

        for chunk_idx in tqdm(
            range(n_chunks),
            desc="Test chunks",
            unit="chunk",
            disable=not show_progress,
            leave=False,
        ):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_cpgs)

            g1_chunk = g1_beta[start_idx:end_idx].copy()
            g2_chunk = g2_beta[start_idx:end_idx].copy()

            chunk_result = self._run_test_method(
                g1_chunk, g2_chunk, n_cnt, n_tx, method
            )
            all_pvals[start_idx:end_idx] = chunk_result["pval"]

            del g1_chunk, g2_chunk, chunk_result

            if n_chunks > 1 and (chunk_idx + 1) % 5 == 0:
                self.memory_manager.cleanup_memory()

        final_fdr = self.stats_utils.calculate_fdr(all_pvals)
        self.memory_manager.cleanup_memory()

        return {"pval": all_pvals, "fdr": final_fdr}

    def _run_test_method(
        self,
        g1_beta: np.ndarray,
        g2_beta: np.ndarray,
        n_cnt: int,
        n_tx: int,
        method: str,
    ) -> Dict[str, np.ndarray]:
        test_methods = {
            "limma": self._limma_test,
            "t-test (equal var)": self._t_test_equal_var,
            "t-test (unequal var)": self._t_test_unequal_var,
            "Wilcox rank sum": self._wilcox_test,
            "CPGassoc": self._cpg_assoc,
        }

        if method not in test_methods:
            raise ValueError(
                f"Unknown method: {method}. Available: {list(test_methods.keys())}"
            )

        return test_methods[method](g1_beta, g2_beta, n_cnt, n_tx)

    def _prepare_mvals(
        self, g1_beta: np.ndarray, g2_beta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        g1_mvals = self.stats_utils.beta_to_m_value(g1_beta)
        g2_mvals = self.stats_utils.beta_to_m_value(g2_beta)
        return g1_mvals, g2_mvals

    def _limma_test(
        self, g1_beta: np.ndarray, g2_beta: np.ndarray, n_cnt: int, n_tx: int
    ) -> Dict[str, np.ndarray]:
        g1_mvals, g2_mvals = self._prepare_mvals(g1_beta, g2_beta)
        mvals = np.concatenate([g1_mvals, g2_mvals], axis=1)

        r_script_path = os.path.join(os.path.dirname(__file__), "utils", "PowerLimma.R")
        if not os.path.exists(r_script_path):
            raise RuntimeError("R limma helper not found: expected " + r_script_path)

        in_fh = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        out_fh = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        in_path, out_path = in_fh.name, out_fh.name
        in_fh.close()
        out_fh.close()

        try:
            np.savetxt(in_path, mvals, delimiter=",")
            cmd = ["Rscript", r_script_path, in_path, str(n_cnt), str(n_tx), out_path]
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"R limma helper failed (rc={proc.returncode}): {proc.stderr.strip()[:400]}"
                )

            df = pd.read_csv(out_path)
            if "pval" in df.columns:
                p_values = df["pval"].to_numpy(dtype=float)
            else:
                p_values = df.iloc[:, 0].to_numpy(dtype=float)

            if len(p_values) != mvals.shape[0]:
                raise RuntimeError("Returned p-value length mismatch")

            p_values = np.where(np.isnan(p_values), 1.0, p_values)
            return {"pval": p_values}
        finally:
            for p in (in_path, out_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    def _t_test_equal_var(
        self, g1_beta: np.ndarray, g2_beta: np.ndarray, n_cnt: int, n_tx: int
    ) -> Dict[str, np.ndarray]:
        g1_mvals, g2_mvals = self._prepare_mvals(g1_beta, g2_beta)

        mean1 = np.mean(g1_mvals, axis=1)
        mean2 = np.mean(g2_mvals, axis=1)
        var1 = np.var(g1_mvals, axis=1, ddof=1)
        var2 = np.var(g2_mvals, axis=1, ddof=1)

        pooled_var = ((n_cnt - 1) * var1 + (n_tx - 1) * var2) / (n_cnt + n_tx - 2)
        pooled_se = np.sqrt(pooled_var * (1 / n_cnt + 1 / n_tx))

        t_stats = (mean1 - mean2) / pooled_se
        df = n_cnt + n_tx - 2
        p_values = 2 * stats.t.sf(np.abs(t_stats), df)

        p_values = np.where(np.isnan(p_values), 1.0, p_values)
        return {"pval": p_values}

    def _t_test_unequal_var(
        self, g1_beta: np.ndarray, g2_beta: np.ndarray, n_cnt: int, n_tx: int
    ) -> Dict[str, np.ndarray]:
        g1_mvals, g2_mvals = self._prepare_mvals(g1_beta, g2_beta)

        mean1 = np.mean(g1_mvals, axis=1)
        mean2 = np.mean(g2_mvals, axis=1)
        var1 = np.var(g1_mvals, axis=1, ddof=1)
        var2 = np.var(g2_mvals, axis=1, ddof=1)

        se1 = var1 / n_cnt
        se2 = var2 / n_tx
        se_combined = np.sqrt(se1 + se2)

        t_stats = (mean1 - mean2) / se_combined
        df_welch = (se1 + se2) ** 2 / (se1**2 / (n_cnt - 1) + se2**2 / (n_tx - 1))

        p_values = 2 * stats.t.sf(np.abs(t_stats), df_welch)
        p_values = np.where(np.isnan(p_values), 1.0, p_values)

        return {"pval": p_values}

    def _wilcox_test(
        self, g1_beta: np.ndarray, g2_beta: np.ndarray, n_cnt: int, n_tx: int
    ) -> Dict[str, np.ndarray]:
        g1_mvals, g2_mvals = self._prepare_mvals(g1_beta, g2_beta)

        n_genes = g1_mvals.shape[0]
        p_values = np.ones(n_genes)

        for i in range(n_genes):
            try:
                combined = np.concatenate([g1_mvals[i, :], g2_mvals[i, :]])
                ranks = stats.rankdata(combined, method="average")
                r1 = np.sum(ranks[:n_cnt])
                u1 = r1 - n_cnt * (n_cnt + 1) / 2
                u2 = n_cnt * n_tx - u1
                u_stat = min(u1, u2)

                mu = n_cnt * n_tx / 2
                sigma = np.sqrt(n_cnt * n_tx * (n_cnt + n_tx + 1) / 12)

                if sigma > 0:
                    z = (u_stat - mu) / sigma
                    p_values[i] = 2 * stats.norm.sf(abs(z))
                else:
                    p_values[i] = 1.0

            except (ValueError, ZeroDivisionError):
                p_values[i] = 1.0

        p_values = np.where(np.isnan(p_values), 1.0, p_values)
        return {"pval": p_values}

    def _cpg_assoc(
        self, g1_beta: np.ndarray, g2_beta: np.ndarray, n_cnt: int, n_tx: int
    ) -> Dict[str, np.ndarray]:
        mean1 = np.mean(g1_beta, axis=1)
        mean2 = np.mean(g2_beta, axis=1)
        var1 = np.var(g1_beta, axis=1, ddof=1)
        var2 = np.var(g2_beta, axis=1, ddof=1)

        pooled_var = ((n_cnt - 1) * var1 + (n_tx - 1) * var2) / (n_cnt + n_tx - 2)
        pooled_se = np.sqrt(pooled_var * (1 / n_cnt + 1 / n_tx))

        t_stats = (mean1 - mean2) / pooled_se
        df = n_cnt + n_tx - 2
        p_values = 2 * stats.t.sf(np.abs(t_stats), df)

        p_values = np.where(np.isnan(p_values), 1.0, p_values)
        return {"pval": p_values}


class SimulationEngine:
    def __init__(
        self,
        config: PowerAnalysisConfig,
        memory_manager: MemoryManager,
        test_engine: StatisticalTestEngine,
        data_manager: PowerEWASDataManager,
    ) -> None:
        self.config = config
        self.memory_manager = memory_manager
        self.test_engine = test_engine
        self.data_manager = data_manager
        self.chunk_configurations = self._precalculate_chunk_sizes()

    def _precalculate_chunk_sizes(self) -> Dict[int, ChunkConfiguration]:
        log.info("Pre-calculating optimal chunk sizes for all sample configurations...")

        tot_sample_sizes = list(
            range(
                self.config.min_sample_size,
                self.config.max_sample_size + 1,
                self.config.sample_size_steps,
            )
        )

        chunk_info = {}

        for n_tot in tot_sample_sizes:
            n_cnt = round(n_tot * self.config.control_proportion)
            n_tx = n_tot - n_cnt
            total_samples = n_cnt + n_tx

            gen_chunk_size = self.memory_manager.calculate_optimal_chunk_size(
                self.config.n_cpgs, total_samples, safety_factor=0.5
            )

            test_chunk_size = self.memory_manager.calculate_optimal_chunk_size(
                self.config.n_cpgs, total_samples, safety_factor=0.6
            )

            chunk_info[n_tot] = ChunkConfiguration(
                generation_chunk_size=gen_chunk_size,
                testing_chunk_size=test_chunk_size,
                n_cnt=n_cnt,
                n_tx=n_tx,
                total_samples=total_samples,
            )

        return chunk_info

    @staticmethod
    def get_alpha_beta(
        my_mean: np.ndarray, my_var: np.ndarray
    ) -> Dict[str, np.ndarray]:
        my_mean = np.clip(my_mean, 1e-6, 1 - 1e-6)
        my_var = np.clip(my_var, 1e-10, my_mean * (1 - my_mean) - 1e-10)

        alpha = my_mean**2 * ((1 - my_mean) / my_var - 1 / my_mean)
        beta = alpha * (1 / my_mean - 1)

        alpha = np.maximum(alpha, 0.1)
        beta = np.maximum(beta, 0.1)

        return {"alpha": alpha, "beta": beta}

    @staticmethod
    def fix_boundary_values(beta_array: np.ndarray) -> np.ndarray:
        beta_fixed = beta_array.copy()

        mask_1 = beta_fixed == 1
        if np.any(mask_1):
            max_val = np.max(beta_fixed[beta_fixed != 1])
            beta_fixed[mask_1] = max_val

        mask_0 = beta_fixed == 0
        if np.any(mask_0):
            min_val = np.min(beta_fixed[beta_fixed != 0])
            beta_fixed[mask_0] = min_val

        return beta_fixed

    def generate_delta_values(
        self, a_vals: np.ndarray, b_vals: np.ndarray, tau: float
    ) -> np.ndarray:
        a_std = a_vals / tau
        b_std = b_vals / tau
        delta = truncnorm.rvs(a_std, b_std, loc=0, scale=tau, size=len(a_vals))
        return delta

    def simulate_single_run(
        self,
        n_cnt: int,
        n_tx: int,
        tau_val: float,
        k_val: int,
        meth_para: Dict[str, np.ndarray],
        cpg_on_array: int,
        chunk_config: ChunkConfiguration,
    ) -> Dict[str, float]:
        chunk_size = chunk_config.generation_chunk_size

        cpg_idx = np.random.choice(cpg_on_array, size=self.config.n_cpgs, replace=True)
        changed_cpgs_idx = np.random.choice(cpg_idx, size=k_val, replace=False)

        mu_changed = meth_para["mu"][changed_cpgs_idx]
        var_changed = meth_para["var"][changed_cpgs_idx]
        sqrt_term = np.sqrt(np.maximum(0.25 - var_changed, 1e-10))
        a_vals = 0.5 - mu_changed - sqrt_term
        b_vals = 0.5 - mu_changed + sqrt_term
        delta = self.generate_delta_values(a_vals, b_vals, tau_val)

        meaningful_dm = np.abs(delta) >= self.config.detection_limit
        n_meaningful = np.sum(meaningful_dm)

        if n_meaningful == 0:
            log.warn("No meaningful DM CpGs found!")
            return self._create_empty_result(delta)

        mu_unchanged = meth_para["mu"][cpg_idx]
        mu_changed_full = mu_unchanged.copy()

        changed_positions, meaningful_positions, negligible_positions = (
            self._apply_deltas(
                cpg_idx, changed_cpgs_idx, delta, meaningful_dm, mu_changed_full
            )
        )

        mu_unchanged = np.clip(mu_unchanged, 1e-6, 1 - 1e-6)
        mu_changed_full = np.clip(mu_changed_full, 1e-6, 1 - 1e-6)
        var_vals = np.clip(meth_para["var"][cpg_idx], 1e-10, 0.24)

        all_pvals = self._process_simulation_chunks(
            mu_unchanged,
            mu_changed_full,
            var_vals,
            n_cnt,
            n_tx,
            chunk_size,
            chunk_config,
        )

        final_fdr = StatisticalUtilities.calculate_fdr(all_pvals)
        dm_test = {"pval": all_pvals, "fdr": final_fdr}

        self.memory_manager.cleanup_memory()

        result = self._calculate_simulation_metrics(
            dm_test,
            changed_positions,
            meaningful_positions,
            negligible_positions,
            delta,
        )

        return result

    def _create_empty_result(self, delta: np.ndarray) -> Dict[str, float]:
        return {
            "power": np.nan,
            "delta": delta,
            "marTypeI": np.nan,
            "FDR": np.nan,
            "FDC": np.nan,
            "classicalPower": np.nan,
            "probTP": 0.0,
        }

    def _apply_deltas(
        self,
        cpg_idx: np.ndarray,
        changed_cpgs_idx: np.ndarray,
        delta: np.ndarray,
        meaningful_dm: np.ndarray,
        mu_changed_full: np.ndarray,
    ) -> Tuple[List[int], List[int], List[int]]:
        sorted_indices = np.argsort(cpg_idx)
        sorted_cpg_idx = cpg_idx[sorted_indices]
        insertion_points = np.searchsorted(sorted_cpg_idx, changed_cpgs_idx)
        valid_mask = (insertion_points < len(sorted_cpg_idx)) & (
            sorted_cpg_idx[insertion_points] == changed_cpgs_idx
        )
        valid_positions = sorted_indices[insertion_points[valid_mask]]
        valid_delta = delta[valid_mask]
        valid_meaningful = meaningful_dm[valid_mask]

        mu_changed_full[valid_positions] += valid_delta

        meaningful_positions = valid_positions[valid_meaningful].tolist()
        negligible_positions = valid_positions[~valid_meaningful].tolist()
        changed_positions = valid_positions.tolist()

        return changed_positions, meaningful_positions, negligible_positions

    def _process_simulation_chunks(
        self,
        mu_unchanged: np.ndarray,
        mu_changed_full: np.ndarray,
        var_vals: np.ndarray,
        n_cnt: int,
        n_tx: int,
        chunk_size: int,
        chunk_config: ChunkConfiguration,
    ) -> np.ndarray:
        n_chunks = (self.config.n_cpgs + chunk_size - 1) // chunk_size
        all_pvals = np.zeros(self.config.n_cpgs, dtype=np.float64)

        show_progress = n_chunks > 1
        if show_progress:
            log.debug(f"Processing simulation in {n_chunks} chunks")

        for chunk_idx in tqdm(
            range(n_chunks),
            desc="Processing chunks",
            leave=False,
            disable=not show_progress,
        ):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, self.config.n_cpgs)
            chunk_size_actual = end_idx - start_idx

            mu_unch_chunk = mu_unchanged[start_idx:end_idx]
            mu_chng_chunk = mu_changed_full[start_idx:end_idx]
            var_chunk = var_vals[start_idx:end_idx]

            params_unchanged = self.get_alpha_beta(mu_unch_chunk, var_chunk)
            params_changed = self.get_alpha_beta(mu_chng_chunk, var_chunk)

            g1_beta_chunk = np.random.beta(
                params_unchanged["alpha"][:, np.newaxis],
                params_unchanged["beta"][:, np.newaxis],
                size=(chunk_size_actual, n_cnt),
            )

            g2_beta_chunk = np.random.beta(
                params_changed["alpha"][:, np.newaxis],
                params_changed["beta"][:, np.newaxis],
                size=(chunk_size_actual, n_tx),
            )

            g1_beta_chunk = self.fix_boundary_values(g1_beta_chunk)
            g2_beta_chunk = self.fix_boundary_values(g2_beta_chunk)

            dm_test_chunk = self.test_engine.perform_test(
                g1_beta_chunk,
                g2_beta_chunk,
                n_cnt,
                n_tx,
                self.config.dm_method,
                chunk_config.testing_chunk_size,
            )

            all_pvals[start_idx:end_idx] = dm_test_chunk["pval"]

            del g1_beta_chunk, g2_beta_chunk, dm_test_chunk
            del params_unchanged, params_changed

            if n_chunks > 1 and (chunk_idx + 1) % 3 == 0:
                self.memory_manager.cleanup_memory()

        return all_pvals

    def _calculate_simulation_metrics(
        self,
        dm_test: Dict[str, np.ndarray],
        changed_positions: List[int],
        meaningful_positions: List[int],
        negligible_positions: List[int],
        delta: np.ndarray,
    ) -> Dict[str, float]:
        confusion_matrix = self._calculate_confusion_matrix(
            dm_test, changed_positions, meaningful_positions, negligible_positions
        )

        metrics = self._calculate_performance_metrics(confusion_matrix)
        result = metrics.copy()
        result["delta"] = delta

        return result

    def _calculate_confusion_matrix(
        self,
        dm_test: Dict[str, np.ndarray],
        changed_positions: List[int],
        meaningful_positions: List[int],
        negligible_positions: List[int],
    ) -> Dict[str, int]:
        detected_mask = dm_test["fdr"] < self.config.fdr_threshold

        changed_mask = np.zeros(self.config.n_cpgs, dtype=bool)
        meaningful_mask = np.zeros(self.config.n_cpgs, dtype=bool)
        negligible_mask = np.zeros(self.config.n_cpgs, dtype=bool)

        if len(changed_positions) > 0:
            changed_mask[changed_positions] = True
        if len(meaningful_positions) > 0:
            meaningful_mask[meaningful_positions] = True
        if len(negligible_positions) > 0:
            negligible_mask[negligible_positions] = True

        unchanged_mask = ~changed_mask

        TP = np.sum(detected_mask & meaningful_mask)
        FP = np.sum(detected_mask & unchanged_mask)
        NP = np.sum(detected_mask & negligible_mask)
        FN = np.sum(~detected_mask & meaningful_mask)
        TN = np.sum(~detected_mask & unchanged_mask)
        NN = np.sum(~detected_mask & negligible_mask)

        total_detected = np.sum(detected_mask)
        total_meaningful = len(meaningful_positions)
        total_negligible = len(negligible_positions)
        total_unchanged = self.config.n_cpgs - len(changed_positions)

        return {
            "TP": int(TP),
            "FP": int(FP),
            "NP": int(NP),
            "FN": int(FN),
            "TN": int(TN),
            "NN": int(NN),
            "total_detected": int(total_detected),
            "total_meaningful": total_meaningful,
            "total_negligible": total_negligible,
            "total_unchanged": total_unchanged,
        }

    def _calculate_performance_metrics(
        self, confusion_matrix: Dict[str, int]
    ) -> Dict[str, float]:
        TP = confusion_matrix["TP"]
        FP = confusion_matrix["FP"]
        NP = confusion_matrix["NP"]
        TN = confusion_matrix["TN"]

        total_meaningful = confusion_matrix["total_meaningful"]
        total_unchanged = confusion_matrix["total_unchanged"]
        total_negligible = confusion_matrix["total_negligible"]
        total_detected = confusion_matrix["total_detected"]

        mar_power = TP / total_meaningful if total_meaningful > 0 else np.nan
        mar_type_i = FP / total_unchanged if total_unchanged > 0 else np.nan
        fdr = FP / total_detected if total_detected > 0 else np.nan
        fdc = FP / TP if TP > 0 else np.nan

        total_changed = total_meaningful + total_negligible
        classical_power = (NP + TP) / total_changed if total_changed > 0 else np.nan

        prob_tp = 1.0 if TP > 0 else 0.0

        sensitivity = mar_power
        specificity = TN / total_unchanged if total_unchanged > 0 else np.nan
        precision = TP / total_detected if total_detected > 0 else np.nan

        if precision > 0 and sensitivity > 0:
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        else:
            f1_score = 0.0

        return {
            "power": mar_power,
            "marTypeI": mar_type_i,
            "FDR": fdr,
            "FDC": fdc,
            "classicalPower": classical_power,
            "probTP": prob_tp,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1_score,
        }


class ParameterEstimator:
    def __init__(self, config: PowerAnalysisConfig) -> None:
        self.config = config

    def find_tau_parameter(
        self,
        target_dm_cpgs: int,
        target_delta: float,
        meth_para: Dict[str, np.ndarray],
        cpg_on_array: int,
    ) -> Dict[str, float]:
        log.debug(
            f"Finding tau for target_delta={target_delta}, target_dm_cpgs={target_dm_cpgs}"
        )

        tau = 1.0
        tau_steps = 1.0
        look_for_tau = True
        cnt = 0
        max_cnt = 100

        tau_pbar = tqdm(
            total=max_cnt,
            desc=f"Finding tau (target Δ={target_delta})",
            unit="iter",
            leave=False,
        )

        delta = None

        while cnt < max_cnt and look_for_tau:
            percentiles = []

            for i in range(100):
                cpg_idx_4_tau = np.random.choice(
                    cpg_on_array, size=self.config.n_cpgs, replace=True
                )

                mu_vals = meth_para["mu"][cpg_idx_4_tau]
                var_vals = meth_para["var"][cpg_idx_4_tau]

                sqrt_term = np.sqrt(np.maximum(0.25 - var_vals, 1e-10))
                a_vals = 0.5 - mu_vals - sqrt_term
                b_vals = 0.5 - mu_vals + sqrt_term

                delta = self._generate_delta_values(a_vals, b_vals, tau)
                percentiles.append(np.percentile(np.abs(delta), 99.99))

            mean_percentile = np.mean(percentiles)
            condition1 = (
                mean_percentile < target_delta - 0.5 * self.config.detection_limit
            )
            condition2 = (
                mean_percentile > target_delta + 0.5 * self.config.detection_limit
            )

            if condition1 and tau >= 1:
                tau = tau + 1
            elif condition1 and tau < 1:
                tau_steps = 0.5 * tau_steps
                tau = tau + tau_steps
            elif condition2:
                tau_steps = 0.5 * tau_steps
                tau = tau - tau_steps
            else:
                look_for_tau = False

            cnt += 1
            tau_pbar.update(1)

        tau_pbar.close()

        if cnt == max_cnt:
            log.warn(f"Max iterations reached in tau finding. Using tau={tau:.6f}")

        if delta is None:
            cpg_idx_4_tau = np.random.choice(
                cpg_on_array, size=self.config.n_cpgs, replace=True
            )
            mu_vals = meth_para["mu"][cpg_idx_4_tau]
            var_vals = meth_para["var"][cpg_idx_4_tau]
            sqrt_term = np.sqrt(np.maximum(0.25 - var_vals, 1e-10))
            a_vals = 0.5 - mu_vals - sqrt_term
            b_vals = 0.5 - mu_vals + sqrt_term
            delta = self._generate_delta_values(a_vals, b_vals, tau)

        truly_dm_perc = np.mean(np.abs(delta) > self.config.detection_limit)

        if truly_dm_perc == 0:
            log.warn("No deltas above detection limit! Adjusting parameters...")
            truly_dm_perc = 0.01

        target_k = round((1 / truly_dm_perc) * target_dm_cpgs)
        k = min(target_k, self.config.n_cpgs)

        log.info(
            f"Tau finding completed: tau={tau:.4f}, truly_dm_perc={truly_dm_perc:.3f}, K={k} (took {cnt} iterations)"
        )

        return {"tau": tau, "K": k}

    def calculate_k_parameter(
        self,
        target_dm_cpgs: int,
        meth_para: Dict[str, np.ndarray],
        cpg_on_array: int,
        tau: float,
    ) -> int:
        cpg_idx_4_tau = np.random.choice(
            cpg_on_array, size=self.config.n_cpgs, replace=True
        )

        mu_vals = meth_para["mu"][cpg_idx_4_tau]
        var_vals = meth_para["var"][cpg_idx_4_tau]

        sqrt_term = np.sqrt(np.maximum(0.25 - var_vals, 1e-10))
        a_vals = 0.5 - mu_vals - sqrt_term
        b_vals = 0.5 - mu_vals + sqrt_term

        delta = self._generate_delta_values(a_vals, b_vals, tau)

        truly_dm_perc = np.mean(np.abs(delta) > self.config.detection_limit)
        target_k = (
            round(target_dm_cpgs / truly_dm_perc)
            if truly_dm_perc > 0
            else self.config.n_cpgs
        )
        k = min(target_k, self.config.n_cpgs)

        return k

    def _generate_delta_values(
        self, a_vals: np.ndarray, b_vals: np.ndarray, tau: float
    ) -> np.ndarray:
        a_std = a_vals / tau
        b_std = b_vals / tau
        delta = truncnorm.rvs(a_std, b_std, loc=0, scale=tau, size=len(a_vals))
        return delta


class ResultsManager:
    def __init__(self, config: PowerAnalysisConfig) -> None:
        self.config = config

    def format_results(
        self,
        results: Dict[str, List],
        sample_sizes: List[int],
        target_values: List[float],
    ) -> Dict[str, Any]:
        n_targets = len(target_values)
        n_samples = len(sample_sizes)

        output = {
            "meanPower": None,
            "powerArray": None,
            "deltaArray": {},
            "metric": {
                "marTypeI": None,
                "classicalPower": None,
                "FDR": None,
                "FDC": None,
                "probTP": None,
            },
        }

        for metric_name in [
            "power",
            "marTypeI",
            "classicalPower",
            "FDR",
            "FDC",
            "probTP",
        ]:
            try:
                if metric_name not in results or not results[metric_name]:
                    log.warn(f"No data found for metric: {metric_name}")
                    output = self._create_empty_metric_output(
                        output,
                        metric_name,
                        n_samples,
                        n_targets,
                        sample_sizes,
                        target_values,
                    )
                    continue

                metric_array = self._flatten_metric_results(
                    results[metric_name], n_targets, n_samples
                )

                if len(metric_array) == 0:
                    log.warn(f"Empty metric array for: {metric_name}")
                    output = self._create_empty_metric_output(
                        output,
                        metric_name,
                        n_samples,
                        n_targets,
                        sample_sizes,
                        target_values,
                    )
                    continue

                metric_reshaped = metric_array.reshape(
                    n_targets, n_samples, self.config.n_simulations
                )
                mean_metric = np.nanmean(metric_reshaped, axis=2)

                if np.all(np.isnan(mean_metric)):
                    log.warn(f"All NaN values for metric: {metric_name}")
                    output = self._create_empty_metric_output(
                        output,
                        metric_name,
                        n_samples,
                        n_targets,
                        sample_sizes,
                        target_values,
                    )
                    continue

                mean_df = pd.DataFrame(
                    mean_metric.T,
                    index=[str(ss) for ss in sample_sizes],
                    columns=[str(tv) for tv in target_values],
                )

                if metric_name == "power":
                    output["meanPower"] = mean_df
                    output["powerArray"] = metric_reshaped.transpose(2, 1, 0)
                else:
                    output["metric"][metric_name] = mean_df

            except Exception as e:
                log.error(f"Error formatting {metric_name}: {e}")
                log.debug(
                    f"Results shape for {metric_name}: {len(results.get(metric_name, []))}"
                )
                output = self._create_empty_metric_output(
                    output,
                    metric_name,
                    n_samples,
                    n_targets,
                    sample_sizes,
                    target_values,
                )

        output["deltaArray"] = self._format_delta_arrays(
            results["delta"], target_values, sample_sizes, n_targets, n_samples
        )

        return output

    def _flatten_metric_results(
        self, metric_results: List, n_targets: int, n_samples: int
    ) -> np.ndarray:
        metric_flat = []
        expected_size = n_targets * n_samples * self.config.n_simulations

        for condition_results in metric_results:
            if isinstance(condition_results, list):
                for sim_result in condition_results:
                    if isinstance(sim_result, (int, float)):
                        val = float(sim_result)
                        if np.isnan(val) or np.isinf(val):
                            val = 0.0
                        metric_flat.append(val)
                    elif hasattr(sim_result, "__len__") and len(sim_result) == 1:
                        val = float(sim_result[0])
                        if np.isnan(val) or np.isinf(val):
                            val = 0.0
                        metric_flat.append(val)
                    elif hasattr(sim_result, "__len__") and len(sim_result) > 1:
                        try:
                            val = np.nanmean(sim_result)
                            if np.isnan(val) or np.isinf(val):
                                val = 0.0
                            metric_flat.append(val)
                        except Exception:
                            metric_flat.append(0.0)
                    else:
                        metric_flat.append(0.0)
            else:
                try:
                    val = float(condition_results)
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
                    metric_flat.append(val)
                except Exception:
                    metric_flat.append(0.0)

        while len(metric_flat) < expected_size:
            metric_flat.append(0.0)

        if len(metric_flat) > expected_size:
            metric_flat = metric_flat[:expected_size]
            log.warn(f"Truncated {len(metric_flat) - expected_size} excess values")

        return np.array(metric_flat, dtype=float)

    def _create_empty_metric_output(
        self,
        output: Dict,
        metric_name: str,
        n_samples: int,
        n_targets: int,
        sample_sizes: List[int],
        target_values: List[float],
    ) -> Dict:
        empty_df = pd.DataFrame(
            np.full((n_samples, n_targets), np.nan),
            index=[str(ss) for ss in sample_sizes],
            columns=[str(tv) for tv in target_values],
        )

        if metric_name == "power":
            output["meanPower"] = empty_df
            output["powerArray"] = np.full(
                (self.config.n_simulations, n_samples, n_targets), np.nan
            )
        else:
            output["metric"][metric_name] = empty_df

        return output

    def _format_delta_arrays(
        self,
        delta_results: List,
        target_values: List[float],
        sample_sizes: List[int],
        n_targets: int,
        n_samples: int,
    ) -> Dict:
        delta_arrays = {}

        try:
            for t_idx, target_val in enumerate(target_values):
                target_key = str(target_val)
                target_deltas = {}

                for s_idx, sample_size in enumerate(sample_sizes):
                    condition_idx = t_idx * n_samples + s_idx

                    if condition_idx < len(delta_results):
                        condition_deltas = delta_results[condition_idx]
                        all_deltas = (
                            np.concatenate(condition_deltas)
                            if condition_deltas
                            else np.array([])
                        )
                        target_deltas[str(sample_size)] = all_deltas
                    else:
                        target_deltas[str(sample_size)] = np.array([])

                delta_arrays[target_key] = target_deltas

        except Exception as e:
            log.error(f"Error formatting delta arrays: {e}")
            for target_val in target_values:
                target_key = str(target_val)
                delta_arrays[target_key] = {
                    str(ss): np.array([]) for ss in sample_sizes
                }

        return delta_arrays

    def save_results(self, results: Dict[str, Any]) -> None:
        if not self.config.output_file:
            log.warn("No output file specified, skipping save")
            return

        try:
            with open(self.config.output_file, "w") as f:
                self._write_results_header(f)
                self._write_power_matrix(f, results)
                self._write_additional_metrics(f, results)
                self._write_summary_statistics(f, results)

            log.info(f"Results saved to {self.config.output_file}")

        except Exception as e:
            log.error(f"Failed to save results: {e}")

    def _write_results_header(self, f: Any) -> None:
        f.write("PowerEWAS Analysis Results\n")
        f.write("=" * 50 + "\n\n")

    def _write_power_matrix(self, f, results: Dict[str, Any]) -> None:
        f.write("Mean Power Matrix:\n")
        f.write("-" * 20 + "\n")
        if results["meanPower"] is not None:
            f.write(results["meanPower"].to_string())
        else:
            f.write("No power data available")
        f.write("\n\n")

    def _write_additional_metrics(self, f: Any, results: Dict[str, Any]) -> None:
        f.write("Additional Metrics:\n")
        f.write("-" * 20 + "\n")
        for metric_name, metric_df in results["metric"].items():
            if metric_df is not None:
                f.write(f"\n{metric_name}:\n")
                f.write(metric_df.to_string())
                f.write("\n")

    def _write_summary_statistics(self, f: Any, results: Dict[str, Any]) -> None:
        f.write("\nSummary Statistics:\n")
        f.write("-" * 20 + "\n")

        if results["meanPower"] is not None:
            power_data = results["meanPower"].values
            f.write(
                f"Power Range: {np.nanmin(power_data):.3f} - {np.nanmax(power_data):.3f}\n"
            )
            f.write(f"Mean Power: {np.nanmean(power_data):.3f}\n")
        else:
            f.write("Power statistics not available\n")

        if results["deltaArray"]:
            f.write(f"\nDelta Arrays Available: {list(results['deltaArray'].keys())}\n")
            for delta_key, delta_dict in results["deltaArray"].items():
                if isinstance(delta_dict, dict):
                    total_deltas = 0
                    total_sum = 0
                    for sample_key, delta_array in delta_dict.items():
                        if len(delta_array) > 0:
                            total_deltas += len(delta_array)
                            total_sum += np.sum(np.abs(delta_array))

                    if total_deltas > 0:
                        mean_delta = total_sum / total_deltas
                        f.write(
                            f"  {delta_key}: {total_deltas} deltas, mean |delta| = {mean_delta:.4f}\n"
                        )
                    else:
                        f.write(f"  {delta_key}: No deltas available\n")


class PowerEWASVisualizer:
    def __init__(self, results: Dict[str, Any], config: PowerAnalysisConfig) -> None:
        self.results = results
        self.config = config

        self._setup_plotting_style()
        self._extract_plotting_data()

    def _setup_plotting_style(self) -> None:
        plt.style.use("default")
        sns.set_palette("husl")

        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "accent": "#F18F01",
            "success": "#C73E1D",
            "neutral": "#6c757d",
            "light": "#f8f9fa",
            "dark": "#343a40",
        }

        plt.rcParams.update(
            {
                "figure.figsize": (10, 6),
                "font.size": 11,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.dpi": 100,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.facecolor": "white",
            }
        )

    def _extract_plotting_data(self) -> None:
        self.power_matrix = self.results.get("meanPower")
        self.metrics = self.results.get("metric", {})
        self.delta_arrays = self.results.get("deltaArray", {})
        self.power_array = self.results.get("powerArray")

        if self.power_matrix is not None:
            self.sample_sizes = [int(x) for x in self.power_matrix.index]
            self.target_values = [float(x) for x in self.power_matrix.columns]
        else:
            self.sample_sizes = []
            self.target_values = []

    def create_all_plots(
        self, save_plots: bool = True, filename: str = None
    ) -> Dict[str, plt.Figure]:
        plots = {}

        if self.power_matrix is not None:
            plots["dashboard"] = self.create_summary_dashboard(
                save=save_plots, filename=filename
            )
        else:
            log.warn("No power matrix available for plotting")

        return plots

    def create_summary_dashboard(
        self, save: bool = True, filename: str = None
    ) -> plt.Figure:
        fig = plt.figure(figsize=(22, 16), constrained_layout=False)

        plt.subplots_adjust(
            top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=0.4, wspace=0.3
        )

        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

        power_data = self.power_matrix.values.T
        has_valid_data = not np.all(np.isnan(power_data))

        if not has_valid_data:
            ax_main = fig.add_subplot(gs[:, :])
            ax_main.text(
                0.5,
                0.5,
                "No valid power data available\nCheck simulation parameters",
                ha="center",
                va="center",
                fontsize=20,
                bbox=dict(boxstyle="round,pad=1", facecolor="lightcoral", alpha=0.8),
                transform=ax_main.transAxes,
            )
            ax_main.axis("off")

            fig.suptitle(
                f"PowerEWAS Analysis Dashboard - {self.config.tissue_type} (No Data)",
                fontsize=20,
                fontweight="bold",
                y=0.95,
            )

            if save and filename:
                self._save_figure(fig, filename)
            return fig

        ax_main = fig.add_subplot(gs[0:2, 0:2])
        im = ax_main.imshow(power_data, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)

        for i in range(len(self.target_values)):
            for j in range(len(self.sample_sizes)):
                power_val = power_data[i, j]
                if not np.isnan(power_val):
                    color = "white" if power_val < 0.5 else "black"
                    ax_main.text(
                        j,
                        i,
                        f"{power_val:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontweight="bold",
                        fontsize=8,
                        rotation=90,
                    )

        ax_main.set_xticks(range(len(self.sample_sizes)))
        ax_main.set_yticks(range(len(self.target_values)))
        ax_main.set_xticklabels(self.sample_sizes)
        ax_main.set_yticklabels([f"{x:.3f}" for x in self.target_values])
        ax_main.set_xlabel("Sample Size")
        ax_main.set_ylabel("Effect Size (Δ)")
        ax_main.set_title("Power Analysis Heatmap", fontweight="bold", pad=20)

        plt.colorbar(im, ax=ax_main, shrink=0.8, label="Statistical Power")

        ax_curves = fig.add_subplot(gs[0, 2:])
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.target_values)))

        for i, target_val in enumerate(self.target_values):
            power_values = self.power_matrix.iloc[:, i].values
            valid_mask = ~np.isnan(power_values)
            if np.any(valid_mask):
                ax_curves.plot(
                    np.array(self.sample_sizes)[valid_mask],
                    power_values[valid_mask],
                    color=colors[i],
                    linewidth=2,
                    marker="o",
                    label=f"Δ = {target_val:.3f}",
                )

        ax_curves.axhline(y=0.8, color="red", linestyle="--", alpha=0.7)
        ax_curves.set_xlabel("Sample Size")
        ax_curves.set_ylabel("Power")
        ax_curves.set_title("Power Curves", pad=20)
        ax_curves.legend(fontsize=8)
        ax_curves.grid(True, alpha=0.3)
        ax_curves.set_ylim(0, 1)

        ax_rec = fig.add_subplot(gs[1, 2:])

        sample_rec_80 = []
        for target_val in self.target_values:
            power_col = self.power_matrix[str(target_val)]
            sufficient_power = power_col >= 0.8
            if sufficient_power.any():
                min_sample_size = power_col[sufficient_power].index[0]
                sample_rec_80.append(int(min_sample_size))
            else:
                sample_rec_80.append(np.nan)

        bars = ax_rec.bar(
            range(len(self.target_values)),
            sample_rec_80,
            color=self.colors["primary"],
            alpha=0.8,
        )

        for i, (bar, val) in enumerate(zip(bars, sample_rec_80)):
            if not np.isnan(val):
                ax_rec.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val + 1,
                    f"{int(val)}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        ax_rec.set_xticks(range(len(self.target_values)))
        ax_rec.set_xticklabels([f"{x:.3f}" for x in self.target_values])
        ax_rec.set_xlabel("Effect Size (Δ)")
        ax_rec.set_ylabel("Min Sample Size")
        ax_rec.set_title("Sample Size for 80% Power", pad=20)
        ax_rec.grid(True, alpha=0.3)

        ax_params = fig.add_axes([0.05, 0.13, 0.18, 0.15])
        ax_params.axis("off")

        params_text = f"""Analysis Parameters:
• Tissue Type: {self.config.tissue_type}
• Method: {self.config.dm_method}
• FDR Threshold: {self.config.fdr_threshold}
• Detection Limit: {self.config.detection_limit}
• Target DM CpGs: {self.config.target_dm_cpgs}
• Total CpGs: {self.config.n_cpgs:,}
• Simulations: {self.config.n_simulations}
• Control Proportion: {self.config.control_proportion}"""

        ax_params.text(
            0.0,
            1.0,
            params_text,
            transform=ax_params.transAxes,
            fontsize=14,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8),
        )

        ax_results = fig.add_axes([0.25, 0.13, 0.18, 0.15])
        ax_results.axis("off")

        power_values = self.power_matrix.values
        if has_valid_data:
            max_power = np.nanmax(power_values)
            min_power = np.nanmin(power_values)
            mean_power = np.nanmean(power_values)
        else:
            max_power = min_power = mean_power = np.nan

        max_power_str = f"{max_power:.3f}" if not np.isnan(max_power) else "N/A"
        min_power_str = f"{min_power:.3f}" if not np.isnan(min_power) else "N/A"
        mean_power_str = f"{mean_power:.3f}" if not np.isnan(mean_power) else "N/A"

        results_text = f"""Key Results:
• Max Power: {max_power_str}
• Min Power: {min_power_str}
• Mean Power: {mean_power_str}
• Sample Range: {min(self.sample_sizes)}-{max(self.sample_sizes)}
• Effect Range: {min(self.target_values):.3f}-{max(self.target_values):.3f}"""

        ax_results.text(
            0.0,
            1.0,
            results_text,
            transform=ax_results.transAxes,
            fontsize=14,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
        )

        ax_interpretation = fig.add_axes([0.05, 0.08, 0.38, 0.04])
        ax_interpretation.axis("off")

        interpretation_text = self._generate_interpretation_text_for_best_sample()

        ax_interpretation.text(
            0.0,
            1.0,
            interpretation_text,
            transform=ax_interpretation.transAxes,
            fontsize=14,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8),
        )

        if self.metrics:
            ax_metrics = fig.add_subplot(gs[2, 2:])

            metric_names = list(self.metrics.keys())[:4]
            metric_means = []

            for metric_name in metric_names:
                metric_data = self.metrics[metric_name]
                mean_val = np.nanmean(metric_data.values)
                metric_means.append(mean_val if not np.isnan(mean_val) else 0)

            if any(val > 0 for val in metric_means):
                y_pos = np.arange(len(metric_names))
                colors_metrics = plt.cm.Set2(np.linspace(0, 1, len(metric_names)))

                bars = ax_metrics.barh(
                    y_pos, metric_means, color=colors_metrics, alpha=0.7
                )

                max_val = max(metric_means) if metric_means else 1

                for i, (bar, val) in enumerate(zip(bars, metric_means)):
                    if val > 0:
                        if val > max_val * 0.3:
                            text_x = val * 0.5
                            text_color = "white"
                            ha = "center"
                        else:
                            text_x = val + max_val * 0.02
                            text_color = "black"
                            ha = "left"

                        ax_metrics.text(
                            text_x,
                            bar.get_y() + bar.get_height() / 2.0,
                            f"{val:.3f}",
                            ha=ha,
                            va="center",
                            fontweight="bold",
                            color=text_color,
                            fontsize=10,
                        )

                metric_name_mapping = {
                    "marTypeI": "Marginal Type I Error Rate",
                    "classicalPower": "Classical Power",
                    "FDR": "False Discovery Rate",
                    "FDC": "False Discovery Cost",
                    "probTP": "Probability of True Positive",
                }

                full_metric_names = [
                    metric_name_mapping.get(name, name.replace("_", " ").title())
                    for name in metric_names
                ]

                ax_metrics.set_yticks(y_pos)
                ax_metrics.set_yticklabels(full_metric_names, fontsize=11)
                ax_metrics.set_xlabel("Mean Value", fontsize=12)
                ax_metrics.set_title("Performance Metrics", pad=20, fontsize=14)
                ax_metrics.grid(True, alpha=0.3, axis="x")
                ax_metrics.set_xlim(0, max_val * 1.1)
            else:
                ax_metrics.text(
                    0.5,
                    0.5,
                    "No valid\nmetrics available",
                    transform=ax_metrics.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    style="italic",
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
                )
                ax_metrics.axis("off")
        else:
            ax_metrics = fig.add_subplot(gs[2, 2:])
            ax_metrics.axis("off")
            ax_metrics.text(
                0.5,
                0.5,
                "No additional\nmetrics available",
                transform=ax_metrics.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                style="italic",
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
            )

        fig.suptitle(
            f"PowerEWAS Analysis Dashboard - {self.config.tissue_type}",
            fontsize=22,
            fontweight="bold",
            y=0.96,
        )

        footer_text = (
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        fig.text(
            0.99,
            0.02,
            footer_text,
            ha="right",
            va="bottom",
            fontsize=9,
            style="italic",
            alpha=0.7,
            transform=fig.transFigure,
        )

        if save and filename:
            self._save_figure(fig, filename)

        return fig

    def _generate_interpretation_text_for_best_sample(self) -> str:
        power_values = self.power_matrix.values
        if np.all(np.isnan(power_values)):
            return "Interpretation:\n• No valid results available for interpretation"

        max_power_idx = np.unravel_index(np.nanargmax(power_values), power_values.shape)
        sample_idx, target_idx = max_power_idx

        best_sample_size = int(self.power_matrix.index[sample_idx])
        best_target_value = float(self.power_matrix.columns[target_idx])
        max_power = power_values[sample_idx, target_idx]

        interpretation_lines = ["Interpretation:"]
        interpretation_lines.append(
            f"• Best performance at sample size {best_sample_size} (Δ={best_target_value:.3f}):"
        )

        power_pct = max_power * 100
        interpretation_lines.append(
            f"  - {power_pct:.0f}% of meaningful changes detected"
        )

        if self.metrics:
            sample_str = str(best_sample_size)
            target_str = str(best_target_value)

            if "classicalPower" in self.metrics:
                try:
                    classical_power = self.metrics["classicalPower"].loc[
                        sample_str, target_str
                    ]
                    if not np.isnan(classical_power):
                        classical_pct = classical_power * 100
                        interpretation_lines.append(
                            f"  - {classical_pct:.0f}% of all changes (meaningful + small) detected"
                        )
                except (KeyError, IndexError):
                    pass

            if "marTypeI" in self.metrics:
                try:
                    mar_type_i = self.metrics["marTypeI"].loc[sample_str, target_str]
                    if not np.isnan(mar_type_i):
                        type_i_pct = mar_type_i * 100
                        control_quality = (
                            "good control" if mar_type_i <= 0.05 else "elevated rate"
                        )
                        interpretation_lines.append(
                            f"  - {type_i_pct:.1f}% false positive rate ({control_quality})"
                        )
                except (KeyError, IndexError):
                    pass

            if "FDC" in self.metrics:
                try:
                    fdc = self.metrics["FDC"].loc[sample_str, target_str]
                    if not np.isnan(fdc) and fdc > 0:
                        if fdc >= 1:
                            interpretation_lines.append(
                                f"  - {fdc:.1f} false discoveries per true discovery"
                            )
                        else:
                            true_per_false = 1 / fdc
                            interpretation_lines.append(
                                f"  - 1 false discovery per {true_per_false:.0f} true discoveries"
                            )
                except (KeyError, IndexError):
                    pass

            if "FDR" in self.metrics:
                try:
                    fdr = self.metrics["FDR"].loc[sample_str, target_str]
                    if not np.isnan(fdr) and fdr > 0:
                        fdr_pct = fdr * 100
                        interpretation_lines.append(
                            f"  - {fdr_pct:.0f}% of discoveries are false"
                        )
                    else:
                        interpretation_lines.append(
                            "  - All discoveries appear genuine"
                        )
                except (KeyError, IndexError):
                    pass

        return "\n".join(interpretation_lines)

    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        fig.savefig(
            filename, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )

        log.info(f"Saved plot: {filename}")

    def generate_report_summary(self) -> Dict[str, Any]:
        power_values = self.power_matrix.values
        max_power = np.nanmax(power_values)
        min_power = np.nanmin(power_values)
        mean_power = np.nanmean(power_values)

        summary = {
            "analysis_info": {
                "tissue_type": self.config.tissue_type,
                "method": self.config.dm_method,
                "fdr_threshold": self.config.fdr_threshold,
                "n_simulations": self.config.n_simulations,
                "sample_size_range": f"{min(self.sample_sizes)}-{max(self.sample_sizes)}",
                "effect_size_range": f"{min(self.target_values):.3f}-{max(self.target_values):.3f}",
            },
            "power_results": {
                "max_power": float(max_power) if not np.isnan(max_power) else None,
                "min_power": float(min_power) if not np.isnan(min_power) else None,
                "mean_power": float(mean_power) if not np.isnan(mean_power) else None,
                "power_80_achievable": bool(np.any(self.power_matrix.values >= 0.8)),
                "power_90_achievable": bool(np.any(self.power_matrix.values >= 0.9)),
            },
            "recommendations": {},
        }

        for threshold in [0.8, 0.9]:
            recommendations = []
            for target_val in self.target_values:
                power_col = self.power_matrix[str(target_val)]
                sufficient_power = power_col >= threshold
                if sufficient_power.any():
                    min_sample_size = power_col[sufficient_power].index[0]
                    recommendations.append(
                        {
                            "effect_size": target_val,
                            "min_sample_size": int(min_sample_size),
                        }
                    )
            summary["recommendations"][
                f"power_{int(threshold * 100)}"
            ] = recommendations

        return summary


class PowerEWAS:
    def __init__(
        self,
        min_sample_size: int,
        max_sample_size: int,
        sample_size_steps: int,
        control_proportion: float,
        output_file: str,
        target_delta: Optional[str] = None,
        delta_sd: Optional[str] = None,
        n_cpgs: int = 100000,
        target_dm_cpgs: int = 100,
        tissue_type: str = "Adult (PBMC)",
        detection_limit: float = 0.01,
        dm_method: str = "limma",
        fdr_threshold: float = 0.05,
        n_simulations: int = 50,
        seed: int = 42,
        plot: Optional[str] = None,
    ) -> None:

        target_delta_list = None
        delta_sd_list = None

        if target_delta:
            target_delta_list = self._parse_numeric_list(target_delta)
        if delta_sd:
            delta_sd_list = self._parse_numeric_list(delta_sd)

        if target_delta_list and delta_sd_list:
            raise ValueError("Please specify only one: 'target_delta' or 'delta_sd'")
        if not target_delta_list and not delta_sd_list:
            raise ValueError("Must specify either 'target_delta' or 'delta_sd'")

        self.config = PowerAnalysisConfig(
            min_sample_size=min_sample_size,
            max_sample_size=max_sample_size,
            sample_size_steps=sample_size_steps,
            control_proportion=control_proportion,
            n_cpgs=n_cpgs,
            target_dm_cpgs=target_dm_cpgs,
            tissue_type=tissue_type,
            detection_limit=detection_limit,
            dm_method=dm_method,
            fdr_threshold=fdr_threshold,
            n_simulations=n_simulations,
            output_file=output_file,
            seed=seed,
            target_delta=target_delta_list,
            delta_sd=delta_sd_list,
        )

        self.plot = plot

        np.random.seed(seed)
        log.info(f"Random seed set to {seed}")

        self.memory_manager = MemoryManager()
        self.data_manager = PowerEWASDataManager.get_instance()
        self.test_engine = StatisticalTestEngine(self.memory_manager)
        self.simulation_engine = SimulationEngine(
            self.config, self.memory_manager, self.test_engine, self.data_manager
        )
        self.parameter_estimator = ParameterEstimator(self.config)
        self.results_manager = ResultsManager(self.config)

        log.info(
            f"PowerEWAS initialized with {n_simulations} simulations and optimized chunked processing"
        )
        SystemUtils.print_system_info()

    @classmethod
    def from_args(cls, **kwargs) -> "PowerEWAS":
        target_delta = None
        delta_sd = None

        if kwargs.get("target_delta"):
            target_delta = cls._parse_numeric_list(kwargs["target_delta"])
        if kwargs.get("delta_sd"):
            delta_sd = cls._parse_numeric_list(kwargs["delta_sd"])

        if target_delta and delta_sd:
            raise ValueError("Please specify only one: 'target_delta' or 'delta_sd'")
        if not target_delta and not delta_sd:
            raise ValueError("Must specify either 'target_delta' or 'delta_sd'")

        config = PowerAnalysisConfig(
            min_sample_size=kwargs["min_sample_size"],
            max_sample_size=kwargs["max_sample_size"],
            sample_size_steps=kwargs["sample_size_steps"],
            control_proportion=kwargs["control_proportion"],
            n_cpgs=kwargs["n_cpgs"],
            target_dm_cpgs=kwargs["target_dm_cpgs"],
            tissue_type=kwargs["tissue_type"],
            detection_limit=kwargs["detection_limit"],
            dm_method=kwargs["dm_method"],
            fdr_threshold=kwargs["fdr_threshold"],
            n_simulations=kwargs["n_simulations"],
            output_file=kwargs["output"],
            seed=kwargs["seed"],
            target_delta=target_delta,
            delta_sd=delta_sd,
        )

        instance = cls.__new__(cls)
        instance.config = config

        np.random.seed(config.seed)
        log.info(f"Random seed set to {config.seed}")

        instance.memory_manager = MemoryManager()
        instance.data_manager = PowerEWASDataManager.get_instance()
        instance.test_engine = StatisticalTestEngine(instance.memory_manager)
        instance.simulation_engine = SimulationEngine(
            config, instance.memory_manager, instance.test_engine, instance.data_manager
        )
        instance.parameter_estimator = ParameterEstimator(config)
        instance.results_manager = ResultsManager(config)

        log.info(
            f"PowerEWAS initialized with {config.n_simulations} simulations and optimized chunked processing"
        )
        SystemUtils.print_system_info()

        return instance

    @staticmethod
    def _parse_numeric_list(value_str: str) -> List[float]:
        try:
            return [float(x.strip()) for x in value_str.split(",")]
        except ValueError as e:
            log.error(f"Failed to parse numeric list '{value_str}': {e}")
            raise ValueError(f"Invalid numeric list format: {value_str}")

    def run_power_analysis(self) -> Dict[str, Any]:
        log.info("Starting PowerEWAS analysis")

        meth_para = self.data_manager.load_dataset(self.config.tissue_type)
        cpg_on_array = len(meth_para["mu"])

        tot_sample_sizes = list(
            range(
                self.config.min_sample_size,
                self.config.max_sample_size + 1,
                self.config.sample_size_steps,
            )
        )
        log.info(f"Sample sizes to test: {tot_sample_sizes}")

        tau, K, target_values = self._estimate_parameters(meth_para, cpg_on_array)

        log.info("Running simulation")
        results = self._run_simulations(
            tot_sample_sizes, target_values, tau, K, meth_para, cpg_on_array
        )

        formatted_results = self.results_manager.format_results(
            results, tot_sample_sizes, target_values
        )
        self.results_manager.save_results(formatted_results)

        log.info("PowerEWAS analysis completed successfully")
        return formatted_results

    def _estimate_parameters(
        self, meth_para: Dict[str, np.ndarray], cpg_on_array: int
    ) -> Tuple[List[float], List[int], List[float]]:
        if self.config.delta_sd is None:
            log.info("Finding tau parameters...")
            tau = []
            K = []
            target_values = self.config.target_delta

            for d in self.config.target_delta:
                tau_result = self.parameter_estimator.find_tau_parameter(
                    self.config.target_dm_cpgs, d, meth_para, cpg_on_array
                )
                tau.append(tau_result["tau"])
                K.append(tau_result["K"])
        else:
            tau = self.config.delta_sd
            K = []
            target_values = self.config.delta_sd

            for t in tau:
                k_val = self.parameter_estimator.calculate_k_parameter(
                    self.config.target_dm_cpgs, meth_para, cpg_on_array, t
                )
                K.append(k_val)

        log.debug(f"K parameters: {K}")
        return tau, K, target_values

    def _run_simulations(
        self,
        tot_sample_sizes: List[int],
        target_values: List[float],
        tau: List[float],
        K: List[int],
        meth_para: Dict[str, np.ndarray],
        cpg_on_array: int,
    ) -> Dict[str, List]:
        all_results = {
            "power": [],
            "delta": [],
            "marTypeI": [],
            "classicalPower": [],
            "FDR": [],
            "FDC": [],
            "probTP": [],
        }

        total_iterations = (
            len(target_values) * len(tot_sample_sizes) * self.config.n_simulations
        )

        with monitor_resources(interval=10.0):
            with tqdm(
                total=total_iterations, desc="Power simulation", unit="sim"
            ) as pbar:
                for d_idx, (target_val, tau_val, k_val) in enumerate(
                    zip(target_values, tau, K)
                ):
                    for n_tot in tot_sample_sizes:
                        chunk_config = self.simulation_engine.chunk_configurations[
                            n_tot
                        ]
                        n_cnt = chunk_config.n_cnt
                        n_tx = chunk_config.n_tx

                        sim_results = {
                            "power": [],
                            "delta": [],
                            "marTypeI": [],
                            "classicalPower": [],
                            "FDR": [],
                            "FDC": [],
                            "probTP": [],
                        }

                        for sim in range(self.config.n_simulations):
                            sim_result = self.simulation_engine.simulate_single_run(
                                n_cnt=n_cnt,
                                n_tx=n_tx,
                                tau_val=tau_val,
                                k_val=k_val,
                                meth_para=meth_para,
                                cpg_on_array=cpg_on_array,
                                chunk_config=chunk_config,
                            )

                            for key in sim_results.keys():
                                sim_results[key].append(sim_result[key])

                            pbar.update(1)

                            if (sim + 1) % 10 == 0:
                                self.memory_manager.cleanup_memory()

                        for key in all_results.keys():
                            all_results[key].append(sim_results[key])

                        del sim_results
                        self.memory_manager.cleanup_memory()

        return all_results

    def run(self) -> Dict[str, Any]:
        results = self.run_power_analysis()

        print("\n" + "=" * 60)
        print("PowerEWAS Analysis Complete!")
        print("=" * 60)

        if results.get("meanPower") is not None:
            power_matrix = results["meanPower"]
            print("\nPower Analysis Summary:")
            print(f"Sample sizes tested: {list(power_matrix.index)}")
            print(f"Target values: {list(power_matrix.columns)}")
            print(
                f"Power range: {power_matrix.min().min():.3f} - {power_matrix.max().max():.3f}"
            )
            print(f"Mean power: {power_matrix.mean().mean():.3f}")

        print("\nMean Power Matrix:")
        print(results["meanPower"].round(2))

        if self.plot and results.get("meanPower") is not None:
            log.info("Creating visualization plot...")
            try:
                visualizer = PowerEWASVisualizer(results, self.config)
                visualizer.create_all_plots(save_plots=True, filename=self.plot)

                log.success(f"Saved plot to: {self.plot}")

                summary_report = visualizer.generate_report_summary()
                results["summary_report"] = summary_report

            except Exception as e:
                log.error(f"Error creating plots: {e}")
                log.debug("Continuing without plots...")

        elif not self.plot:
            log.info("No plot filename specified, skipping visualization")

        log.success(f"Results saved to: {self.config.output_file}")


options = [
    OptionConfig(
        flags=["-i", "--min_sample_size"], type=int, default=10, required=False
    ),
    OptionConfig(
        flags=["-a", "--max_sample_size"], type=int, default=100, required=False
    ),
    OptionConfig(
        flags=["-j", "--sample_size_steps"], type=int, default=10, required=False
    ),
    OptionConfig(
        flags=["-c", "--control_proportion"], type=float, default=0.5, required=False
    ),
    OptionConfig(
        flags=["-d", "--target_delta"], type=str, default=None, required=False
    ),
    OptionConfig(flags=["-e", "--delta_sd"], type=str, default=None, required=False),
    OptionConfig(flags=["-g", "--n_cpgs"], type=int, default=100000, required=False),
    OptionConfig(
        flags=["-r", "--target_dm_cpgs"], type=int, default=100, required=False
    ),
    OptionConfig(
        flags=["-t", "--tissue_type"],
        type=str,
        default="Adult (PBMC)",
        required=False,
        choices=[
            "Saliva",
            "Lymphoma",
            "Placenta",
            "Liver",
            "Colon",
            "Blood adult",
            "Blood 5 year olds",
            "Blood newborns",
            "Cord-blood (whole blood)",
            "Cord-blood (PBMC)",
            "Adult (PBMC)",
            "Sperm",
        ],
    ),
    OptionConfig(
        flags=["-q", "--detection_limit"], type=float, default=0.01, required=False
    ),
    OptionConfig(
        flags=["-m", "--dm_method"],
        type=str,
        default="limma",
        required=False,
        choices=["limma", "dmpFinder", "ttest"],
    ),
    OptionConfig(flags=["-f", "--fdr_threshold"], type=float, default=0.05),
    OptionConfig(flags=["-n", "--n_simulations"], type=int, default=50, required=False),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
    OptionConfig(flags=["-s", "--seed"], type=int, default=42, required=False),
    OptionConfig(flags=["-p", "--plot"], type=str, default=None, required=False),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="PowerEWAS")
    opt = framework.run()

    power_analysis = PowerEWAS(
        min_sample_size=opt.min_sample_size,
        max_sample_size=opt.max_sample_size,
        sample_size_steps=opt.sample_size_steps,
        control_proportion=opt.control_proportion,
        target_delta=opt.target_delta,
        delta_sd=opt.delta_sd,
        n_cpgs=opt.n_cpgs,
        target_dm_cpgs=opt.target_dm_cpgs,
        tissue_type=opt.tissue_type,
        detection_limit=opt.detection_limit,
        dm_method=opt.dm_method,
        fdr_threshold=opt.fdr_threshold,
        n_simulations=opt.n_simulations,
        output_file=opt.output,
        seed=opt.seed,
        plot=opt.plot,
    )

    power_analysis.run()
