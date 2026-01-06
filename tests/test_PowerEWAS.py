#!/usr/bin/env python
import contextlib
import matplotlib.pyplot as plt
from types import SimpleNamespace
from typing import Dict
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from PowerEWAS import (
    ChunkConfiguration,
    MemoryManager,
    ParameterEstimator,
    PowerAnalysisConfig,
    PowerEWAS,
    PowerEWASDataManager,
    PowerEWASVisualizer,
    ResultsManager,
    SimulationEngine,
    StatisticalTestEngine,
    StatisticalUtilities,
)
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.fixture(autouse=True)
def patch_system_info(monkeypatch):
    fake_info = {
        "cpu_name": "Test CPU",
        "physical_cores": 4,
        "logical_cores": 8,
        "allocated_cores": None,
        "effective_cores": 4,
        "environment": "Test",
        "ram_total_gb": 32.0,
        "ram_available_gb": 16.0,
        "memory_source": "Mock",
        "training_cores": 4,
        "platform": "Linux",
        "architecture": "x86_64",
    }
    monkeypatch.setattr(
        "PowerEWAS.SystemUtils.get_system_info", lambda: fake_info.copy()
    )
    return fake_info


@pytest.fixture
def mock_methylation_params() -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    mu = np.clip(rng.beta(2.0, 5.0, size=500), 1e-3, 1 - 1e-3)
    var = np.clip(rng.gamma(shape=2.0, scale=0.002, size=500), 1e-6, 0.24)
    return {"mu": mu, "var": var}


@pytest.fixture
def simple_config() -> PowerAnalysisConfig:
    return PowerAnalysisConfig(
        min_sample_size=8,
        max_sample_size=12,
        sample_size_steps=4,
        control_proportion=0.5,
        n_cpgs=20,
        target_dm_cpgs=4,
        tissue_type="Adult (PBMC)",
        detection_limit=0.01,
        dm_method="limma",
        fdr_threshold=0.05,
        n_simulations=3,
        output_file=None,
        seed=123,
        target_delta=[0.1],
        delta_sd=None,
    )


def test_parse_numeric_list_valid():
    assert PowerEWAS._parse_numeric_list("0.1,0.2, 0.3") == [0.1, 0.2, 0.3]


def test_parse_numeric_list_invalid():
    with pytest.raises(ValueError):
        PowerEWAS._parse_numeric_list("0.1;foo")


def test_memory_manager_chunk_cache_behaviour():
    manager = MemoryManager()
    manager.clear_cache()

    chunk = manager.calculate_optimal_chunk_size(
        5000, 10, dtype_size=8, safety_factor=0.6
    )
    assert chunk >= 1000
    calculated_chunk = manager.calculate_optimal_chunk_size(
        5000, 10, dtype_size=8, safety_factor=0.6
    )
    assert calculated_chunk == chunk
    cached_chunk = manager.get_cached_chunk_size(
        5000, 10, dtype_size=8, safety_factor=0.6
    )
    assert cached_chunk == chunk

    manager.clear_cache()
    assert manager._chunk_size_cache == {}
    manager.cleanup_memory()


def test_power_ewas_requires_effect_specification():
    with patch(
        "PowerEWAS.PowerEWASDataManager.get_instance", return_value=MagicMock()
    ), patch("PowerEWAS.SystemUtils.print_system_info"):
        with pytest.raises(
            ValueError, match="Must specify either 'target_delta' or 'delta_sd'"
        ):
            PowerEWAS(
                min_sample_size=8,
                max_sample_size=12,
                sample_size_steps=4,
                control_proportion=0.5,
                output_file=None,
            )


def test_power_ewas_initializes_with_target_delta():
    with patch(
        "PowerEWAS.PowerEWASDataManager.get_instance", return_value=MagicMock()
    ), patch("PowerEWAS.SystemUtils.print_system_info"):
        power = PowerEWAS(
            min_sample_size=10,
            max_sample_size=20,
            sample_size_steps=5,
            control_proportion=0.5,
            target_delta="0.1,0.2",
            n_cpgs=50,
            target_dm_cpgs=5,
            n_simulations=4,
            output_file=None,
        )
    assert power.config.target_delta == [0.1, 0.2]
    assert power.config.delta_sd is None
    assert isinstance(power.memory_manager, MemoryManager)


def test_power_ewas_from_args_parses_lists():
    with patch(
        "PowerEWAS.PowerEWASDataManager.get_instance", return_value=MagicMock()
    ), patch("PowerEWAS.SystemUtils.print_system_info"):
        instance = PowerEWAS.from_args(
            min_sample_size=8,
            max_sample_size=12,
            sample_size_steps=2,
            control_proportion=0.5,
            target_delta="0.05,0.1",
            delta_sd=None,
            n_cpgs=30,
            target_dm_cpgs=5,
            tissue_type="Adult (PBMC)",
            detection_limit=0.01,
            dm_method="limma",
            fdr_threshold=0.05,
            n_simulations=2,
            output=None,
            seed=42,
        )
    assert instance.config.target_delta == [0.05, 0.1]
    assert instance.config.delta_sd is None
    assert instance.config.min_sample_size == 8


def test_data_manager_caches_datasets(mock_methylation_params):
    manager = PowerEWASDataManager.get_instance()
    PowerEWASDataManager._data_cache.clear()
    with patch.object(
        manager, "_load_from_experiment_hub", return_value=mock_methylation_params
    ) as mock_loader:
        first = manager.load_dataset("Adult (PBMC)")
        second = manager.load_dataset("Adult (PBMC)")
    assert mock_loader.call_count == 1
    assert first is second
    PowerEWASDataManager._data_cache.clear()


def test_data_manager_load_dataset_invalid_tissue():
    manager = PowerEWASDataManager.get_instance()
    PowerEWASDataManager._data_cache.clear()
    with pytest.raises(ValueError):
        manager.load_dataset("Unknown Tissue")


def test_data_manager_load_dataset_wrapped_error():
    manager = PowerEWASDataManager.get_instance()
    PowerEWASDataManager._data_cache.clear()
    with patch.object(
        manager, "_load_from_experiment_hub", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError, match="Could not load"):
            manager.load_dataset("Adult (PBMC)")


def test_extract_from_enhanced_hub_primary(monkeypatch):
    manager = PowerEWASDataManager.get_instance()
    sample = {"block": {"mu": [0.1, 0.2], "var": [0.01, 0.02]}}
    monkeypatch.setattr(
        "PowerEWAS.ExperimentHub.get_numeric_arrays", lambda *args, **kwargs: {}
    )
    result = manager._extract_from_enhanced_hub(sample, "Adult (PBMC)")
    assert np.allclose(result["mu"], [0.1, 0.2])
    assert np.allclose(result["var"], [0.01, 0.02])


def test_extract_from_enhanced_hub_numeric_fallback(monkeypatch):
    manager = PowerEWASDataManager.get_instance()
    numeric = {"mu": np.array([0.1, 0.2]), "var": np.array([0.01, 0.02])}
    monkeypatch.setattr(
        "PowerEWAS.ExperimentHub.get_numeric_arrays", lambda *args, **kwargs: numeric
    )
    result = manager._extract_from_enhanced_hub({}, "Adult (PBMC)")
    assert np.allclose(result["mu"], numeric["mu"])
    assert np.allclose(result["var"], numeric["var"])


def test_extract_from_enhanced_hub_failure(monkeypatch):
    manager = PowerEWASDataManager.get_instance()
    monkeypatch.setattr(
        "PowerEWAS.ExperimentHub.get_numeric_arrays", lambda *args, **kwargs: {}
    )
    with pytest.raises(ValueError):
        manager._extract_from_enhanced_hub({"bad": object()}, "Adult (PBMC)")


def test_extract_from_object_handles_namespace():
    manager = PowerEWASDataManager.get_instance()
    payload = SimpleNamespace(mu=[0.1, 0.2], var=[0.01, 0.02])
    result = manager._extract_from_object(payload)
    assert np.allclose(result["mu"], [0.1, 0.2])
    assert np.allclose(result["var"], [0.01, 0.02])


def test_validate_dataset_catches_mismatch():
    manager = PowerEWASDataManager.get_instance()
    with pytest.raises(ValueError):
        manager._validate_dataset(
            {"mu": np.array([0.1, 0.2]), "var": np.array([0.01])}, "Adult (PBMC)"
        )


def test_statistical_utilities_beta_to_m_and_fdr():
    beta_vals = np.array([0.2, 0.5, 0.8])
    m_vals = StatisticalUtilities.beta_to_m_value(beta_vals)
    assert np.all(np.isfinite(m_vals))
    p_values = np.array([0.01, 0.03, 0.2])
    fdr = StatisticalUtilities.calculate_fdr(p_values)
    assert np.all((fdr >= 0) & (fdr <= 1))


def test_statistical_test_engine_perform_test_ttest():
    rng = np.random.default_rng(42)
    group1 = rng.beta(2, 5, size=(50, 6))
    group2 = rng.beta(2, 5, size=(50, 6))
    engine = StatisticalTestEngine(MemoryManager())
    result = engine.perform_test(
        group1, group2, n_cnt=3, n_tx=3, method="t-test (equal var)", chunk_size=128
    )
    assert "pval" in result
    assert isinstance(result["pval"], np.ndarray)
    assert result["pval"].shape == (50,)
    assert np.all((result["pval"] >= 0) & (result["pval"] <= 1))


def test_statistical_test_engine_other_methods():
    rng = np.random.default_rng(0)
    data1 = rng.beta(2, 5, size=(5, 4))
    data2 = rng.beta(2, 5, size=(5, 4))
    engine = StatisticalTestEngine(MemoryManager())
    for method in ["t-test (unequal var)", "Wilcox rank sum", "CPGassoc"]:
        result = engine.perform_test(
            data1, data2, n_cnt=4, n_tx=4, method=method, chunk_size=16
        )
        assert result["pval"].shape == (5,)
        assert np.all((result["pval"] >= 0) & (result["pval"] <= 1))


def test_statistical_test_engine_limma_method():
    rng = np.random.default_rng(1)
    g1 = rng.beta(2, 5, size=(3, 4))
    g2 = rng.beta(2, 5, size=(3, 4))
    engine = StatisticalTestEngine(MemoryManager())
    result = engine.perform_test(g1, g2, n_cnt=4, n_tx=4, method="limma", chunk_size=8)
    assert result["pval"].shape == (3,)
    assert np.all((result["pval"] >= 0) & (result["pval"] <= 1))


def test_statistical_test_engine_rejects_unknown_method():
    engine = StatisticalTestEngine(MemoryManager())
    data = np.random.rand(10, 4)
    with pytest.raises(ValueError, match="Unknown method"):
        engine.perform_test(
            data, data, n_cnt=2, n_tx=2, method="unsupported", chunk_size=16
        )


def test_simulation_engine_get_alpha_beta_non_negative():
    mu = np.array([0.3, 0.5, 0.7])
    var = np.array([0.02, 0.01, 0.005])
    params = SimulationEngine.get_alpha_beta(mu, var)
    assert set(params.keys()) == {"alpha", "beta"}
    assert np.all(params["alpha"] > 0)
    assert np.all(params["beta"] > 0)


def test_simulation_engine_apply_deltas(simple_config):
    memory_manager = MemoryManager()
    test_engine = MagicMock()
    data_manager = MagicMock()
    engine = SimulationEngine(simple_config, memory_manager, test_engine, data_manager)

    cpg_idx = np.array([5, 1, 3, 2, 4])
    changed_cpgs_idx = np.array([1, 4])
    delta = np.array([0.1, -0.05])
    meaningful_dm = np.array([True, False])
    mu_changed_full = np.zeros_like(cpg_idx, dtype=float)

    changed, meaningful, negligible = engine._apply_deltas(
        cpg_idx, changed_cpgs_idx, delta, meaningful_dm, mu_changed_full
    )
    assert changed == [1, 4]
    assert meaningful == [1]
    assert negligible == [4]
    assert np.isclose(mu_changed_full[1], delta[0])
    assert np.isclose(mu_changed_full[4], delta[1])


def test_simulation_engine_create_empty_result(simple_config):
    engine = SimulationEngine(simple_config, MemoryManager(), MagicMock(), MagicMock())
    delta = np.array([0.1, -0.1])
    result = engine._create_empty_result(delta)
    assert result["probTP"] == 0.0
    assert np.all(result["delta"] == delta)
    assert np.isnan(result["power"])


def test_simulation_engine_process_simulation_chunks(simple_config, monkeypatch):
    config = PowerAnalysisConfig(
        **{**simple_config.__dict__, "n_cpgs": 4, "n_simulations": 1}
    )
    engine = SimulationEngine(config, MemoryManager(), MagicMock(), MagicMock())
    call_state = {"idx": 0}

    def fake_perform_test(*args, **kwargs):
        responses = [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
        ]
        pvals = responses[call_state["idx"]]
        call_state["idx"] += 1
        return {"pval": pvals}

    engine.test_engine.perform_test.side_effect = fake_perform_test

    chunk_config = ChunkConfiguration(
        generation_chunk_size=2, testing_chunk_size=2, n_cnt=2, n_tx=2, total_samples=4
    )

    monkeypatch.setattr(
        "PowerEWAS.SimulationEngine.get_alpha_beta",
        lambda *args, **kwargs: {"alpha": np.ones(2), "beta": np.ones(2)},
    )
    monkeypatch.setattr(
        "numpy.random.beta", lambda *args, **kwargs: np.full((2, 2), 0.5)
    )

    mu = np.full(config.n_cpgs, 0.4)
    var = np.full(config.n_cpgs, 0.02)

    pvals = engine._process_simulation_chunks(
        mu, mu + 0.01, var, n_cnt=2, n_tx=2, chunk_size=2, chunk_config=chunk_config
    )
    assert np.allclose(pvals, [0.1, 0.2, 0.3, 0.4])


def test_simulation_engine_confusion_and_metrics(simple_config):
    config = PowerAnalysisConfig(
        **{**simple_config.__dict__, "n_cpgs": 5, "n_simulations": 1}
    )
    engine = SimulationEngine(config, MemoryManager(), MagicMock(), MagicMock())
    dm_test = {"fdr": np.array([0.01, 0.2, 0.03, 0.5, 0.04])}
    changed = [0, 2]
    meaningful = [0]
    negligible = [2]
    confusion = engine._calculate_confusion_matrix(
        dm_test, changed, meaningful, negligible
    )
    assert confusion["TP"] == 1
    assert confusion["FP"] >= 0
    metrics = engine._calculate_performance_metrics(confusion)
    assert metrics["probTP"] in (0.0, 1.0)
    assert "f1_score" in metrics


def test_simulation_engine_simulate_single_run(simple_config):
    config = PowerAnalysisConfig(
        min_sample_size=4,
        max_sample_size=4,
        sample_size_steps=1,
        control_proportion=0.5,
        n_cpgs=6,
        target_dm_cpgs=1,
        tissue_type="Adult (PBMC)",
        detection_limit=0.01,
        dm_method="limma",
        fdr_threshold=0.05,
        n_simulations=1,
        output_file=None,
        seed=123,
        target_delta=[0.1],
        delta_sd=None,
    )
    memory_manager = MemoryManager()
    test_engine = MagicMock()
    data_manager = MagicMock()
    engine = SimulationEngine(config, memory_manager, test_engine, data_manager)
    engine.generate_delta_values = MagicMock(return_value=np.array([0.2]))

    def fake_perform_test(g1, g2, n_cnt, n_tx, method, chunk_size):
        return {"pval": np.linspace(0.01, 0.05, g1.shape[0])}

    test_engine.perform_test.side_effect = fake_perform_test

    chunk_config = ChunkConfiguration(
        generation_chunk_size=2, testing_chunk_size=2, n_cnt=2, n_tx=2, total_samples=4
    )
    meth_para = {
        "mu": np.linspace(0.2, 0.8, 50),
        "var": np.full(50, 0.02),
    }

    result = engine.simulate_single_run(
        n_cnt=2,
        n_tx=2,
        tau_val=0.2,
        k_val=1,
        meth_para=meth_para,
        cpg_on_array=len(meth_para["mu"]),
        chunk_config=chunk_config,
    )

    assert result["delta"].shape == (1,)
    assert 0.0 <= result["power"] <= 1 or np.isnan(result["power"])
    assert "FDR" in result


def test_parameter_estimator_calculate_k_parameter(
    simple_config, mock_methylation_params
):
    estimator = ParameterEstimator(simple_config)
    with patch.object(
        estimator,
        "_generate_delta_values",
        return_value=np.array([0.02, 0.05, 0.0, 0.03]),
    ):
        k_value = estimator.calculate_k_parameter(
            target_dm_cpgs=simple_config.target_dm_cpgs,
            meth_para=mock_methylation_params,
            cpg_on_array=len(mock_methylation_params["mu"]),
            tau=0.2,
        )
    expected = min(simple_config.n_cpgs, round(simple_config.target_dm_cpgs / 0.75))
    assert k_value == expected


def test_parameter_estimator_generate_delta_values(simple_config):
    estimator = ParameterEstimator(simple_config)
    a_vals = np.array([-0.2, -0.1])
    b_vals = np.array([0.1, 0.2])
    deltas = estimator._generate_delta_values(a_vals, b_vals, tau=0.5)
    assert deltas.shape == (2,)


def test_parameter_estimator_find_tau_parameter(
    simple_config, mock_methylation_params, monkeypatch
):
    estimator = ParameterEstimator(simple_config)

    def fake_choice(a, size=None, replace=True):
        base = np.arange(a if isinstance(a, int) else len(a))
        if size is None:
            return base[0]
        return np.resize(base, size)

    monkeypatch.setattr("numpy.random.choice", fake_choice)
    monkeypatch.setattr(
        estimator,
        "_generate_delta_values",
        lambda *args, **kwargs: np.full(simple_config.n_cpgs, 0.1),
    )

    result = estimator.find_tau_parameter(
        target_dm_cpgs=simple_config.target_dm_cpgs,
        target_delta=0.1,
        meth_para=mock_methylation_params,
        cpg_on_array=len(mock_methylation_params["mu"]),
    )
    assert result["tau"] == pytest.approx(1.0)
    assert result["K"] == simple_config.target_dm_cpgs


def test_results_manager_format_results_produces_frames(simple_config):
    results_manager = ResultsManager(simple_config)
    sample_sizes = [8, 12]
    target_values = [0.1, 0.2]

    condition_metrics = [
        [0.7, 0.71, 0.72],
        [0.5, 0.51, 0.52],
        [0.6, 0.61, 0.62],
        [0.4, 0.41, 0.42],
    ]

    metrics = {
        name: [vals[:] for vals in condition_metrics]
        for name in ["power", "marTypeI", "classicalPower", "FDR", "FDC", "probTP"]
    }

    metrics["delta"] = [
        [np.array([0.1, 0.12])],
        [np.array([0.2])],
        [np.array([0.05, 0.06])],
        [np.array([])],
    ]

    formatted = results_manager.format_results(metrics, sample_sizes, target_values)

    assert list(formatted["meanPower"].index) == [str(s) for s in sample_sizes]
    assert list(formatted["meanPower"].columns) == [str(t) for t in target_values]
    assert formatted["powerArray"].shape == (
        simple_config.n_simulations,
        len(sample_sizes),
        len(target_values),
    )
    assert formatted["metric"]["FDR"].shape == (len(sample_sizes), len(target_values))
    assert "0.1" in formatted["deltaArray"]
    assert "8" in formatted["deltaArray"]["0.1"]


def test_results_manager_flatten_metric_truncation(simple_config):
    results_manager = ResultsManager(simple_config)
    n_targets = 1
    n_samples = 1
    metric_results = [[0.5] * (simple_config.n_simulations + 2)]
    flat = results_manager._flatten_metric_results(metric_results, n_targets, n_samples)
    expected_size = n_targets * n_samples * simple_config.n_simulations
    assert len(flat) == expected_size


def test_results_manager_handles_missing_metric(simple_config):
    results_manager = ResultsManager(simple_config)
    sample_sizes = [8]
    target_values = [0.1]
    formatted = results_manager.format_results(
        {"delta": []}, sample_sizes, target_values
    )
    assert formatted["meanPower"].isna().all().all()


def test_results_manager_save_results(tmp_path, simple_config):
    results_manager = ResultsManager(simple_config)
    simple_config.output_file = tmp_path / "results.txt"
    results = {
        "meanPower": pd.DataFrame([[0.5]], index=["8"], columns=["0.1"]),
        "metric": {
            "marTypeI": pd.DataFrame([[0.1]], index=["8"], columns=["0.1"]),
            "classicalPower": pd.DataFrame([[0.6]], index=["8"], columns=["0.1"]),
            "FDR": pd.DataFrame([[0.05]], index=["8"], columns=["0.1"]),
            "FDC": pd.DataFrame([[0.02]], index=["8"], columns=["0.1"]),
            "probTP": pd.DataFrame([[1.0]], index=["8"], columns=["0.1"]),
        },
        "deltaArray": {"0.1": {"8": np.array([0.1, 0.2])}},
        "powerArray": np.array([[[0.5]]]),
    }
    results_manager.save_results(results)
    assert simple_config.output_file.exists()
    text = simple_config.output_file.read_text()
    assert "Mean Power Matrix" in text


def test_power_ewas_run_power_analysis_with_patched_components(mock_methylation_params):
    @contextlib.contextmanager
    def dummy_monitor(*args, **kwargs):
        yield {"max_cpu": 0.0, "max_memory": 0.0, "samples": 0}

    with patch(
        "PowerEWAS.PowerEWASDataManager.get_instance"
    ) as mock_get_instance, patch("PowerEWAS.monitor_resources", dummy_monitor), patch(
        "PowerEWAS.SystemUtils.print_system_info"
    ):
        data_manager = MagicMock()
        data_manager.load_dataset.return_value = mock_methylation_params
        mock_get_instance.return_value = data_manager

        power = PowerEWAS(
            min_sample_size=8,
            max_sample_size=8,
            sample_size_steps=1,
            control_proportion=0.5,
            target_delta="0.1",
            n_cpgs=10,
            target_dm_cpgs=2,
            n_simulations=2,
            output_file=None,
        )

        with patch.object(
            power.parameter_estimator,
            "find_tau_parameter",
            return_value={"tau": 0.2, "K": 3},
        ) as mock_tau, patch.object(
            power.simulation_engine,
            "simulate_single_run",
            return_value={
                "power": 0.75,
                "delta": np.array([0.1, 0.12]),
                "marTypeI": 0.05,
                "classicalPower": 0.8,
                "FDR": 0.04,
                "FDC": 0.02,
                "probTP": 1.0,
            },
        ) as mock_simulate:
            results = power.run_power_analysis()

    data_manager.load_dataset.assert_called_once_with("Adult (PBMC)")
    mock_tau.assert_called_once()
    min_sample_size_len = len([power.config.min_sample_size])
    target_delta_len = len(power.config.target_delta)
    expected_calls = power.config.n_simulations * min_sample_size_len * target_delta_len
    assert mock_simulate.call_count == expected_calls
    assert isinstance(results["meanPower"], pd.DataFrame)
    assert results["meanPower"].iloc[0, 0] == pytest.approx(0.75)
    assert "FDR" in results["metric"]


def test_power_ewas_estimate_parameters_delta_sd_branch(mock_methylation_params):
    with patch(
        "PowerEWAS.PowerEWASDataManager.get_instance", return_value=MagicMock()
    ), patch("PowerEWAS.SystemUtils.print_system_info"):
        power = PowerEWAS(
            min_sample_size=8,
            max_sample_size=8,
            sample_size_steps=1,
            control_proportion=0.5,
            delta_sd="0.05,0.1",
            n_cpgs=10,
            target_dm_cpgs=2,
            n_simulations=1,
            output_file=None,
        )

    with patch.object(
        power.parameter_estimator, "calculate_k_parameter", side_effect=[2, 3]
    ) as mock_calc:
        tau, K, targets = power._estimate_parameters(
            mock_methylation_params, len(mock_methylation_params["mu"])
        )
    assert tau == [0.05, 0.1]
    assert K == [2, 3]
    assert targets == [0.05, 0.1]
    assert mock_calc.call_count == 2


def test_visualizer_extracts_and_plots(simple_config):
    simple_config_with_output = PowerAnalysisConfig(
        **{**simple_config.__dict__, "output_file": None}
    )
    results = {
        "meanPower": pd.DataFrame(
            [[0.6, 0.7], [0.5, 0.65]], index=["8", "12"], columns=["0.1", "0.2"]
        ),
        "metric": {},
        "deltaArray": {},
        "powerArray": np.ones((simple_config_with_output.n_simulations, 2, 2)),
    }
    viz = PowerEWASVisualizer(results, simple_config_with_output)
    plots = viz.create_all_plots(save_plots=False)
    assert "dashboard" in plots
    plt.close(plots["dashboard"])


def test_visualizer_handles_missing_power_matrix(simple_config):
    results = {"meanPower": None, "metric": {}, "deltaArray": {}, "powerArray": None}
    viz = PowerEWASVisualizer(results, simple_config)
    with patch("PowerEWAS.log.warn") as mock_warn:
        plots = viz.create_all_plots(save_plots=False)
    mock_warn.assert_called()
    assert plots == {}
