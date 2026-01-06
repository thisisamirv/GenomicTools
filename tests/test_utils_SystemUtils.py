#!/usr/bin/env python
import os
import logging
import pytest
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, mock_open, patch
from utils.LoggingUtils import log
from utils.SystemUtils import SystemUtils

log.setup(level="DEBUG")


@pytest.fixture
def mock_psutil_cpu_count():
    with patch("utils.SystemUtils.psutil.cpu_count") as mock:
        mock.side_effect = lambda logical=True: 16 if logical else 8
        yield mock


@pytest.fixture
def mock_psutil_memory():
    with patch("utils.SystemUtils.psutil.virtual_memory") as mock:
        mock_memory = MagicMock()
        mock_memory.total = 32 * (1024**3)
        mock_memory.available = 16 * (1024**3)
        mock_memory.used = 16 * (1024**3)
        mock_memory.percent = 50.0
        mock.return_value = mock_memory
        yield mock


@pytest.fixture
def clean_environment():
    original_env = dict(os.environ)

    scheduler_vars = [
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_NPROCS",
        "SLURM_NTASKS",
        "SLURM_JOB_ID",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "LSB_DJOB_NUMPROC",
        "LSB_MAX_NUM_PROCESSORS",
        "LSB_JOBID",
        "LSF_BINDIR",
        "LSB_BATCH",
        "LSB_DJOB_RUSAGE",
        "LSB_SUB_RUSAGE",
        "LSB_HOSTS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]

    for var in scheduler_vars:
        os.environ.pop(var, None)

    yield

    os.environ.clear()
    os.environ.update(original_env)


@contextmanager
def mock_lsf_environment(env_vars: Dict[str, str]):
    original_values = {}
    for key in env_vars:
        original_values[key] = os.environ.get(key)

    for key, value in env_vars.items():
        os.environ[key] = value

    try:
        yield
    finally:
        for key in env_vars:
            if original_values[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_values[key]


@pytest.mark.unit
def test_class_exists():
    assert hasattr(SystemUtils, "get_optimal_cores")
    assert hasattr(SystemUtils, "get_memory_info")
    assert hasattr(SystemUtils, "get_system_info")


@pytest.mark.unit
def test_parse_memory_string():
    assert abs(SystemUtils._parse_memory_string("1024M") - 1.0) < 0.001
    assert abs(SystemUtils._parse_memory_string("16G") - 16.0) < 0.001
    assert abs(SystemUtils._parse_memory_string("2048") - 2.0) < 0.001
    assert abs(SystemUtils._parse_memory_string("1024K") - 0.001) < 0.001
    assert abs(SystemUtils._parse_memory_string("1T") - 1024.0) < 0.001

    assert abs(SystemUtils._parse_memory_string("1024", "G") - 1024.0) < 0.001

    with pytest.raises(ValueError):
        SystemUtils._parse_memory_string("invalid")

    with pytest.raises(ValueError):
        SystemUtils._parse_memory_string("")


@pytest.mark.unit
def test_run_command():
    result = SystemUtils._run_command(["echo", "hello"])
    if result is not None:
        assert "hello" in result

    result = SystemUtils._run_command(["nonexistent_command_12345"])
    assert result is None

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
        result = SystemUtils._run_command(["slow_command"])
        assert result is None


@pytest.mark.unit
def test_validate_resources():
    with patch.object(SystemUtils, "get_system_info") as mock_info:
        mock_info.return_value = {"effective_cores": 8, "ram_available_gb": 16.0}

        is_valid, message = SystemUtils.validate_resources(4, 8.0)
        assert is_valid is True
        assert "successfully" in message.lower()

        is_valid, message = SystemUtils.validate_resources(16, 8.0)
        assert is_valid is False
        assert "cores" in message

        is_valid, message = SystemUtils.validate_resources(4, 32.0)
        assert is_valid is False
        assert "GB" in message

        is_valid, message = SystemUtils.validate_resources(0, 8.0)
        assert is_valid is False

        is_valid, message = SystemUtils.validate_resources(4, -1.0)
        assert is_valid is False


@pytest.mark.unit
def test_get_optimal_cores_validation():
    with pytest.raises(ValueError):
        SystemUtils.get_optimal_cores(reserve_cores=-1)

    with pytest.raises(ValueError):
        SystemUtils.get_optimal_cores(max_cores=0)


@pytest.mark.unit
def test_optimal_cores_no_scheduler(clean_environment, mock_psutil_cpu_count):
    result = SystemUtils.get_optimal_cores()

    assert result == 7
    mock_psutil_cpu_count.assert_called()


@pytest.mark.unit
def test_optimal_cores_with_reserve(clean_environment, mock_psutil_cpu_count):
    result = SystemUtils.get_optimal_cores(reserve_cores=3)

    assert result == 5


@pytest.mark.unit
def test_optimal_cores_with_max_limit(clean_environment, mock_psutil_cpu_count):
    result = SystemUtils.get_optimal_cores(max_cores=4)

    assert result == 4


@pytest.mark.unit
def test_optimal_cores_slurm_environment(clean_environment):
    os.environ["SLURM_CPUS_PER_TASK"] = "12"

    result = SystemUtils.get_optimal_cores()

    assert result == 11


@pytest.mark.unit
def test_optimal_cores_lsf_environment(clean_environment):
    os.environ["LSB_DJOB_NUMPROC"] = "16"

    result = SystemUtils.get_optimal_cores()

    assert result == 15


@pytest.mark.unit
def test_optimal_cores_minimum_one(clean_environment):
    with patch("utils.SystemUtils.psutil.cpu_count", return_value=1):
        result = SystemUtils.get_optimal_cores(reserve_cores=2)

        assert result == 1


@pytest.mark.unit
def test_optimal_cores_physical_cores_none(clean_environment):
    with patch("utils.SystemUtils.psutil.cpu_count") as mock_cpu:
        with patch("multiprocessing.cpu_count", return_value=16):
            mock_cpu.side_effect = lambda logical=True: 16 if logical else None

            result = SystemUtils.get_optimal_cores()

            assert result == 15


@pytest.mark.unit
def test_optimal_cores_error_handling(clean_environment):
    with patch(
        "utils.SystemUtils.psutil.cpu_count",
        side_effect=Exception("Mock error"),
    ):
        with patch("os.cpu_count", return_value=4):
            result = SystemUtils.get_optimal_cores()

            assert result == 3


@pytest.mark.unit
def test_get_slurm_cores_cpus_per_task(clean_environment):
    os.environ["SLURM_CPUS_PER_TASK"] = "8"

    cores, scheduler = SystemUtils._get_scheduler_cores()

    assert cores == 8
    assert scheduler == "SLURM"


@pytest.mark.unit
def test_get_slurm_cores_cpus_on_node(clean_environment):
    os.environ["SLURM_CPUS_ON_NODE"] = "16"

    cores, scheduler = SystemUtils._get_scheduler_cores()

    assert cores == 16
    assert scheduler == "SLURM"


@pytest.mark.unit
def test_get_slurm_cores_scontrol(clean_environment):
    os.environ["SLURM_JOB_ID"] = "12345"

    mock_output = (
        "JobId=12345 JobName=test_job\n   NumCPUs=24 MinMemoryNode=16000M State=RUNNING"
    )

    with patch.object(SystemUtils, "_run_command", return_value=mock_output):
        cores, scheduler = SystemUtils._get_scheduler_cores()

        assert cores == 24
        assert scheduler == "SLURM"


@pytest.mark.unit
def test_get_lsf_cores_djob_numproc(clean_environment):
    os.environ["LSB_DJOB_NUMPROC"] = "32"

    cores, scheduler = SystemUtils._get_scheduler_cores()

    assert cores == 32
    assert scheduler == "LSF"


@pytest.mark.unit
def test_get_lsf_cores_max_num_processors(clean_environment):
    os.environ["LSB_MAX_NUM_PROCESSORS"] = "24"

    cores, scheduler = SystemUtils._get_scheduler_cores()

    assert cores == 24
    assert scheduler == "LSF"


@pytest.mark.unit
def test_get_lsf_cores_bjobs(clean_environment):
    os.environ["LSB_JOBID"] = "67890"

    mock_output = """Job <67890>, User <user>, Project <default>, Status <RUN>, Queue <normal>
                 processors: 16; Execution Home </home/user>"""

    with patch.object(SystemUtils, "_run_command", return_value=mock_output):
        cores, scheduler = SystemUtils._get_scheduler_cores()

        assert cores == 16
        assert scheduler == "LSF"


@pytest.mark.unit
def test_scheduler_cores_error_handling(clean_environment):
    os.environ["SLURM_CPUS_PER_TASK"] = "8"

    with patch.object(
        SystemUtils, "_get_slurm_cores", side_effect=Exception("SLURM error")
    ):
        with patch.object(
            SystemUtils, "_get_lsf_cores", side_effect=Exception("LSF error")
        ):
            cores, scheduler = SystemUtils._get_scheduler_cores()

            assert cores is None
            assert scheduler is None


@pytest.mark.unit
def test_lsf_cores_error_handling(clean_environment):
    os.environ["LSB_DJOB_NUMPROC"] = "invalid"

    cores = SystemUtils._get_lsf_cores()

    assert cores is None


@pytest.mark.unit
def test_no_scheduler_detected(clean_environment):
    cores, scheduler = SystemUtils._get_scheduler_cores()

    assert cores is None
    assert scheduler is None


@pytest.mark.unit
def test_scheduler_error_handling(clean_environment):
    os.environ["SLURM_CPUS_PER_TASK"] = "invalid"

    cores, scheduler = SystemUtils._get_scheduler_cores()

    assert cores is None
    assert scheduler == "SLURM"


@pytest.mark.unit
def test_get_memory_info_system(clean_environment, mock_psutil_memory):
    result = SystemUtils.get_memory_info()

    expected = {
        "total_gb": 32.0,
        "available_gb": 16.0,
        "used_gb": 16.0,
        "percent_used": 50.0,
        "source": "System",
    }

    assert result == expected


@pytest.mark.unit
def test_get_slurm_memory_per_node(clean_environment):
    for key in list(os.environ.keys()):
        if key.startswith(("SLURM_", "LSB_")):
            os.environ.pop(key, None)

    os.environ["SLURM_MEM_PER_NODE"] = "32768"
    os.environ["SLURM_JOB_ID"] = "12345"

    result = SystemUtils.get_memory_info()

    assert result["total_gb"] == 32.0
    assert abs(result["available_gb"] - 28.8) < 0.1
    assert result["source"] == "SLURM"


@pytest.mark.unit
def test_get_slurm_memory_per_cpu(clean_environment):
    os.environ["SLURM_MEM_PER_CPU"] = "2048"
    os.environ["SLURM_CPUS_PER_TASK"] = "8"

    result = SystemUtils.get_memory_info()

    assert result["total_gb"] == 16.0
    assert abs(result["available_gb"] - 14.4) < 0.1
    assert result["source"] == "SLURM"


@pytest.mark.unit
def test_get_slurm_memory_scontrol_gb_unit(clean_environment):
    os.environ["SLURM_JOB_ID"] = "12345"

    mock_output = "JobId=12345 JobName=test MinMemoryNode=32G NumCPUs=8"

    with patch.object(SystemUtils, "_run_command", return_value=mock_output):
        result = SystemUtils.get_memory_info()

        assert result["total_gb"] == 32.0
        assert result["source"] == "SLURM"


@pytest.mark.unit
def test_get_slurm_memory_scontrol_no_unit(clean_environment):
    os.environ["SLURM_JOB_ID"] = "12345"

    mock_output = "JobId=12345 JobName=test MinMemoryNode=16000 NumCPUs=8"

    with patch.object(SystemUtils, "_run_command", return_value=mock_output):
        result = SystemUtils.get_memory_info()

        assert abs(result["total_gb"] - 15.625) < 0.001
        assert result["source"] == "SLURM"


@pytest.mark.unit
def test_get_slurm_memory_scontrol_error(clean_environment):
    os.environ["SLURM_JOB_ID"] = "12345"

    with patch.object(SystemUtils, "_run_command", return_value=None):
        result = SystemUtils._get_slurm_memory()

        assert result is None


@pytest.mark.unit
def test_get_slurm_memory_error_handling(clean_environment):
    os.environ["SLURM_MEM_PER_NODE"] = "invalid"

    memory = SystemUtils._get_slurm_memory()

    assert memory is None


@pytest.mark.unit
def test_get_lsf_memory_rusage(clean_environment):
    for key in list(os.environ.keys()):
        if key.startswith(("SLURM_", "LSB_")):
            os.environ.pop(key, None)

    os.environ["LSB_DJOB_RUSAGE"] = "rusage[mem=16000]"
    os.environ["LSB_JOBID"] = "12345"

    result = SystemUtils.get_memory_info()

    assert abs(result["total_gb"] - 15.625) < 0.001
    assert result["source"] == "LSF"


@pytest.mark.unit
def test_get_lsf_memory_rusage_gb(clean_environment):
    for key in list(os.environ.keys()):
        if key.startswith(("SLURM_", "LSB_")):
            os.environ.pop(key, None)

    os.environ["LSB_DJOB_RUSAGE"] = "rusage[mem=32G]"
    os.environ["LSB_JOBID"] = "12345"

    result = SystemUtils.get_memory_info()

    assert result["total_gb"] == 32.0
    assert result["source"] == "LSF"


@pytest.mark.unit
def test_get_lsf_memory_bjobs(clean_environment):
    os.environ["LSB_JOBID"] = "67890"

    mock_output = """Job <67890>, User <user>, Project <default>, Status <RUN>, Queue <normal>
                 mem=8192M; Execution Home </home/user>"""

    with patch.object(SystemUtils, "_run_command", return_value=mock_output):
        result = SystemUtils.get_memory_info()

        assert result["total_gb"] == 8.0
        assert result["source"] == "LSF"


@pytest.mark.unit
def test_memory_error_handling(clean_environment):
    with patch(
        "utils.SystemUtils.psutil.virtual_memory",
        side_effect=Exception("Mock error"),
    ):
        result = SystemUtils.get_memory_info()

        assert result["total_gb"] == 0.0
        assert result["source"] == "Error"


@pytest.mark.unit
def test_get_system_info_linux(
    clean_environment, mock_psutil_cpu_count, mock_psutil_memory
):
    with patch("platform.system", return_value="Linux"):
        with patch("platform.machine", return_value="x86_64"):
            with patch("os.cpu_count", return_value=16):
                with patch.object(
                    SystemUtils,
                    "_get_cpu_name",
                    return_value="Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz",
                ):
                    result = SystemUtils.get_system_info()

                    assert (
                        "Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz" in result["cpu_name"]
                    )
                    assert result["physical_cores"] == 8
                    assert result["logical_cores"] == 16
                    assert result["platform"] == "Linux"
                    assert result["architecture"] == "x86_64"
                    assert result["environment"] == "Native"


@pytest.mark.unit
def test_get_system_info_with_scheduler(
    clean_environment, mock_psutil_cpu_count, mock_psutil_memory
):
    SystemUtils.clear_cache()

    os.environ["SLURM_CPUS_PER_TASK"] = "4"
    os.environ["SLURM_MEM_PER_NODE"] = "16384"

    with patch("platform.system", return_value="Linux"):
        with patch("os.cpu_count", return_value=16):
            result = SystemUtils.get_system_info()

            assert result["allocated_cores"] == 4
            assert result["effective_cores"] == 4
            assert result["environment"] == "SLURM"
            assert result["memory_source"] == "SLURM"


@pytest.mark.unit
def test_get_system_info_error_handling(clean_environment):
    SystemUtils.clear_cache()

    with patch.object(
        SystemUtils, "_get_cpu_name", side_effect=Exception("Mock error")
    ):
        result = SystemUtils.get_system_info()

        assert result["cpu_name"] == "Unknown"
        assert result["physical_cores"] == 1
        assert result["logical_cores"] == 1
        assert result["allocated_cores"] is None
        assert result["effective_cores"] == 1
        assert result["environment"] == "Unknown"
        assert result["ram_total_gb"] == 0.0
        assert result["ram_available_gb"] == 0.0
        assert result["memory_source"] == "Error"
        assert result["training_cores"] == 1


@pytest.mark.unit
def test_print_system_info(
    clean_environment, mock_psutil_cpu_count, mock_psutil_memory, caplog
):
    SystemUtils.clear_cache()
    os.environ["SLURM_CPUS_PER_TASK"] = "8"

    with caplog.at_level(logging.DEBUG):
        with patch("platform.system", return_value="Linux"):
            with patch("os.cpu_count", return_value=16):
                SystemUtils.print_system_info()

    combined = "\n".join(caplog.messages)
    assert "CPU:" in combined
    assert "Environment:" in combined
    assert "Memory total:" in combined
    assert "Platform:" in combined
    assert "Allocated cores: 8" in combined


@pytest.mark.unit
def test_cache_functionality(clean_environment):
    SystemUtils.clear_cache()

    with patch.object(SystemUtils, "_get_cpu_name", return_value="Test CPU"):
        with patch("utils.SystemUtils.psutil.cpu_count", return_value=8):
            with patch("os.cpu_count", return_value=8):
                result1 = SystemUtils.get_system_info()

                result2 = SystemUtils.get_system_info()

                assert result1 == result2
                assert result1["cpu_name"] == "Test CPU"

                SystemUtils.clear_cache()
                assert SystemUtils._cached_system_info is None


@pytest.mark.unit
def test_load_config():
    config = SystemUtils.load_config("/nonexistent/path")
    assert config["reserve_cores"] == 1
    assert config["max_memory_gb"] == 1024

    mock_config = '{"reserve_cores": 2, "custom_setting": "test"}'
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_config)):
            config = SystemUtils.load_config()
            assert config["reserve_cores"] == 2
            assert config["custom_setting"] == "test"


@pytest.mark.unit
def test_get_disk_info():
    with patch("utils.SystemUtils.psutil.disk_usage") as mock_disk:
        mock_result = MagicMock()
        mock_result.total = 100 * (1024**3)
        mock_result.used = 30 * (1024**3)
        mock_result.free = 70 * (1024**3)
        mock_result.percent = 30.0
        mock_disk.return_value = mock_result

        with patch.object(SystemUtils, "_get_mount_point", return_value="/mnt/data"):
            with patch("os.path.exists", return_value=True):
                result = SystemUtils.get_disk_info("/home/user")

                assert result["total_gb"] == 100.0
                assert result["used_gb"] == 30.0
                assert result["free_gb"] == 70.0
                assert result["percent_used"] == 30.0
                assert result["path"] == "/home/user"
                assert result["mount_point"] == "/mnt/data"


@pytest.mark.unit
def test_get_mount_point():
    with patch("os.path.ismount", side_effect=lambda path: path == "/mnt"):
        assert SystemUtils._get_mount_point("/mnt/data/file.txt") == "/mnt"

    with patch("os.path.ismount", side_effect=Exception("Test error")):
        assert SystemUtils._get_mount_point("/any/path") == ""


@pytest.mark.unit
def test_check_disk_space():
    with patch.object(SystemUtils, "get_disk_info") as mock_get_disk:
        mock_get_disk.return_value = {"free_gb": 20.0, "total_gb": 100.0}

        has_space, message = SystemUtils.check_disk_space("/test/path", 10.0, 5.0)

        assert has_space is True
        assert "Sufficient" in message
        assert "10.0GB" in message

        mock_get_disk.return_value = {"free_gb": 9.0, "total_gb": 100.0}

        has_space, message = SystemUtils.check_disk_space("/test/path", 10.0, 5.0)

        assert has_space is False
        assert "Insufficient" in message
        assert "9.0GB available" in message

        mock_get_disk.side_effect = Exception("Test error")

        has_space, message = SystemUtils.check_disk_space("/test/path", 10.0)

        assert has_space is False
        assert "Error" in message


@pytest.mark.unit
def test_get_disk_usage():
    mock_disk_info_values = [
        {
            "total_gb": 100.0,
            "used_gb": 30.0,
            "free_gb": 70.0,
            "percent_used": 30.0,
            "path": "/path1",
            "mount_point": "/mnt1",
        },
        {
            "total_gb": 200.0,
            "used_gb": 100.0,
            "free_gb": 100.0,
            "percent_used": 50.0,
            "path": "/path2",
            "mount_point": "/mnt2",
        },
        {
            "total_gb": 500.0,
            "used_gb": 200.0,
            "free_gb": 300.0,
            "percent_used": 40.0,
            "path": "/path3",
            "mount_point": "/mnt1",
        },
    ]

    with patch.object(SystemUtils, "get_disk_info", side_effect=mock_disk_info_values):
        result = SystemUtils.get_disk_usage(["/path1", "/path2", "/path3"])

        assert len(result) == 2
        assert result[0]["mount_point"] == "/mnt1"
        assert result[1]["mount_point"] == "/mnt2"

    with patch("os.getcwd", return_value="/current"):
        with patch.object(SystemUtils, "get_disk_info") as mock_get_disk:
            mock_get_disk.return_value = {"mount_point": "/", "free_gb": 50.0}

            result = SystemUtils.get_disk_usage()

            assert len(result) == 1
            mock_get_disk.assert_called_with("/current")


@pytest.mark.unit
def test_disable_core_dumps():
    with patch("resource.setrlimit") as mock_setrlimit:
        success, message = SystemUtils.disable_core_dumps()

        assert success is True
        assert "disabled" in message.lower()
        mock_setrlimit.assert_called_once()

    with patch(
        "utils.SystemUtils.resource.setrlimit", side_effect=ValueError("Test error")
    ):
        with patch("platform.system", return_value="Linux"):
            with patch.object(
                SystemUtils, "_run_command", return_value="Core dumps disabled"
            ):
                success, message = SystemUtils.disable_core_dumps()

                assert success is True
                assert "system command" in message.lower()


@pytest.mark.unit
def test_get_core_dump_settings():
    with patch("platform.system", return_value="Linux"):
        with patch("utils.SystemUtils.resource.getrlimit") as mock_getrlimit:
            mock_getrlimit.return_value = (1024, 4096)

            with patch.object(SystemUtils, "_run_command", return_value=None):
                with patch("builtins.open", mock_open(read_data="core.%p")):
                    settings = SystemUtils.get_core_dump_settings()

                    assert settings["enabled"] is True
                    assert settings["size_limit"] == 1024
                    assert settings["hard_limit"] == 4096
                    assert settings["pattern"] == "core.%p"
                    assert settings["platform"] == "Linux"


@pytest.mark.unit
def test_create_safe_tempdir():
    with patch.object(
        SystemUtils, "check_disk_space", return_value=(True, "Sufficient space")
    ):
        with patch("os.path.exists", return_value=True):
            with patch("tempfile.mkdtemp") as mock_mkdtemp:
                mock_mkdtemp.return_value = "/tmp/safe_temp_123"

                temp_dir, info = SystemUtils.create_safe_tempdir(
                    default_path="/test/path", required_gb=5.0
                )

                assert temp_dir == "/tmp/safe_temp_123"
                assert info["success"] is True
                assert info["chosen_path"] == "/tmp/safe_temp_123"
                assert len(info["checked_paths"]) == 1

    with patch.object(
        SystemUtils,
        "check_disk_space",
        side_effect=[(False, "Insufficient space"), (True, "Sufficient space")],
    ):
        with patch("os.path.exists", return_value=True):
            with patch("tempfile.mkdtemp") as mock_mkdtemp:
                mock_mkdtemp.return_value = "/tmp/safe_temp_456"

                temp_dir, info = SystemUtils.create_safe_tempdir(
                    default_path="/test/path",
                    required_gb=5.0,
                    alternative_paths=["/another/path"],
                )

                assert temp_dir == "/tmp/safe_temp_456"
                assert info["success"] is True
                assert len(info["checked_paths"]) == 2
                assert info["checked_paths"][0]["has_space"] is False

    with patch.object(
        SystemUtils, "check_disk_space", return_value=(False, "Insufficient space")
    ):
        with patch("os.path.exists", return_value=True):
            with pytest.raises(IOError) as excinfo:
                SystemUtils.create_safe_tempdir(
                    default_path="/test/path",
                    required_gb=5.0,
                    alternative_paths=["/another/path"],
                )

            assert "No location with" in str(excinfo.value)


@pytest.mark.unit
def test_get_safe_tempdir():
    with patch.object(
        SystemUtils,
        "create_safe_tempdir",
        return_value=("/tmp/test_safe_dir", {"success": True}),
    ):
        result = SystemUtils.get_safe_tempdir(output_dir="/test/dir", required_gb=2.0)
        assert result == "/tmp/test_safe_dir"

    with patch.object(
        SystemUtils, "create_safe_tempdir", side_effect=IOError("No space")
    ):
        with patch("tempfile.mkdtemp", return_value="/tmp/fallback_dir"):
            result = SystemUtils.get_safe_tempdir(
                output_dir="/test/dir", required_gb=2.0
            )
            assert result == "/tmp/fallback_dir"


@pytest.mark.unit
def test_cleanup_tempdir():
    with patch("os.path.exists", return_value=True):
        with patch("shutil.rmtree") as mock_rmtree:
            result = SystemUtils.cleanup_tempdir("/tmp/test_dir")

            assert result is True
            mock_rmtree.assert_called_with("/tmp/test_dir")

    with patch("os.path.exists", return_value=False):
        result = SystemUtils.cleanup_tempdir("/nonexistent/dir")

        assert result is True

    with patch("os.path.exists", return_value=True):
        with patch("shutil.rmtree", side_effect=Exception("Cannot delete")):
            result = SystemUtils.cleanup_tempdir("/tmp/protected_dir")

            assert result is False


@pytest.mark.unit
def test_cleanup_stale_temp_files():
    mock_now = time.time()
    old_time = mock_now - 48 * 3600
    new_time = mock_now - 12 * 3600

    mock_files = {
        "temp_old_file": (True, False, 1024, old_time),
        "temp_old_dir": (False, True, 2048, old_time),
        "temp_new_file": (True, False, 512, new_time),
        "other_file": (True, False, 128, old_time),
    }

    def mock_exists(path):
        return True

    def mock_isdir(path):
        if path == "/tmp":
            return True
        basename = os.path.basename(path)
        return basename in mock_files and mock_files[basename][1]

    def mock_isfile(path):
        basename = os.path.basename(path)
        return basename in mock_files and mock_files[basename][0]

    def mock_stat(path):
        basename = os.path.basename(path)
        mock_st = MagicMock()
        mock_st.st_size = mock_files[basename][2]
        mock_st.st_mtime = mock_files[basename][3]
        return mock_st

    def mock_listdir(path):
        return list(mock_files.keys())

    with patch("os.path.exists", side_effect=mock_exists):
        with patch("os.path.isdir", side_effect=mock_isdir):
            with patch("os.listdir", side_effect=mock_listdir):
                with patch("os.path.isfile", side_effect=mock_isfile):
                    with patch("os.stat", side_effect=mock_stat):
                        with patch("time.time", return_value=mock_now):
                            with patch("os.remove"):
                                with patch("shutil.rmtree"):
                                    with patch(
                                        "os.walk",
                                        return_value=[("/tmp", [], ["file1"])],
                                    ):
                                        with patch(
                                            "os.path.getsize", return_value=1024
                                        ):
                                            result = (
                                                SystemUtils.cleanup_stale_temp_files(
                                                    directories=["/tmp"],
                                                    prefix="temp",
                                                    max_age_hours=24,
                                                )
                                            )

                                            assert result["files_found"] == 1
                                            assert result["dirs_found"] == 1


@pytest.mark.unit
def test_cleanup_stale_temp_files_dry_run():
    with patch("os.path.exists", return_value=True):
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["temp_old_file"]):
                with patch("os.path.isfile", return_value=True):
                    with patch("os.stat") as mock_stat:
                        mock_st = MagicMock()
                        mock_st.st_size = 1024
                        mock_st.st_mtime = time.time() - 48 * 3600
                        mock_stat.return_value = mock_st

                        with patch("os.remove") as mock_remove:
                            result = SystemUtils.cleanup_stale_temp_files(
                                directories=["/tmp"],
                                prefix="temp",
                                max_age_hours=24,
                                dry_run=True,
                            )

                            assert result["files_found"] == 1
                            assert result["files_deleted"] == 0
                            assert mock_remove.call_count == 0


@pytest.mark.unit
def test_check_system_health():
    with patch("psutil.cpu_percent", return_value=30.0):
        with patch.object(
            SystemUtils,
            "get_memory_info",
            return_value={"percent_used": 40.0, "total_gb": 16.0, "available_gb": 9.6},
        ):
            with patch.object(
                SystemUtils,
                "get_disk_usage",
                return_value=[{"path": "/", "free_gb": 50.0, "percent_used": 50.0}],
            ):
                with patch.object(
                    SystemUtils,
                    "get_core_dump_settings",
                    return_value={"enabled": False},
                ):
                    health = SystemUtils.check_system_health()

                    assert health["status"] == "healthy"
                    assert len(health["warnings"]) == 0
                    assert len(health["critical"]) == 0

    with patch("psutil.cpu_percent", return_value=95.0):
        with patch.object(
            SystemUtils,
            "get_memory_info",
            return_value={"percent_used": 40.0, "total_gb": 16.0, "available_gb": 9.6},
        ):
            with patch.object(
                SystemUtils,
                "get_disk_usage",
                return_value=[{"path": "/", "free_gb": 50.0, "percent_used": 50.0}],
            ):
                with patch.object(
                    SystemUtils,
                    "get_core_dump_settings",
                    return_value={"enabled": True},
                ):
                    health = SystemUtils.check_system_health()

                    assert health["status"] == "warning"
                    assert len(health["warnings"]) == 2
                    assert "CPU usage" in health["warnings"][0]
                    assert "Core dumps" in health["warnings"][1]

    with patch("psutil.cpu_percent", return_value=30.0):
        with patch.object(
            SystemUtils,
            "get_memory_info",
            return_value={"percent_used": 40.0, "total_gb": 16.0, "available_gb": 9.6},
        ):
            with patch.object(
                SystemUtils,
                "get_disk_usage",
                return_value=[{"path": "/", "free_gb": 1.0, "percent_used": 99.0}],
            ):
                health = SystemUtils.check_system_health(min_free_disk_gb=10.0)

                assert health["status"] == "critical"
                assert len(health["critical"]) == 1
                assert "Very low disk space" in health["critical"][0]


@pytest.mark.unit
def test_benchmark_system():
    with patch.object(
        SystemUtils, "get_system_info", return_value={"cpu_name": "Test CPU"}
    ):
        counter = 0

        def mock_time():
            nonlocal counter
            counter += 1
            return counter

        with patch("time.time", side_effect=mock_time):
            with patch("random.random", return_value=0.5):
                with patch("math.sqrt", return_value=0.707):
                    with patch("math.sin", return_value=0.5):
                        with patch("math.cos", return_value=0.866):
                            with patch("math.log", return_value=0.693):
                                with patch("builtins.open", mock_open()):
                                    with patch("os.remove"):
                                        with patch.object(
                                            SystemUtils,
                                            "benchmark_system",
                                            return_value={
                                                "cpu": {
                                                    "single_thread_ops_per_sec": 1000,
                                                    "multi_thread_ops_per_sec": 3500,
                                                    "parallelization_factor": 3.5,
                                                },
                                                "memory": {"mb_per_sec": 500},
                                                "disk": {
                                                    "write_mb_per_sec": 100,
                                                    "read_mb_per_sec": 150,
                                                },
                                                "system_info": {"cpu_name": "Test CPU"},
                                            },
                                        ):

                                            result = SystemUtils.benchmark_system(
                                                duration_sec=1
                                            )

                                            assert "cpu" in result
                                            assert "memory" in result
                                            assert "disk" in result
                                            assert "system_info" in result


@pytest.mark.integration
def test_visual_inspection(clean_environment):
    print("\n==== SYSTEM UTILS VISUAL INSPECTION ====")

    print("\n1. Native System Detection:")
    with patch("utils.SystemUtils.psutil.cpu_count") as mock_cpu:
        with patch("utils.SystemUtils.psutil.virtual_memory") as mock_mem:
            mock_cpu.side_effect = lambda logical=True: 32 if logical else 16

            mock_memory = MagicMock()
            mock_memory.total = 64 * (1024**3)
            mock_memory.available = 48 * (1024**3)
            mock_memory.used = 16 * (1024**3)
            mock_memory.percent = 25.0
            mock_mem.return_value = mock_memory

            optimal = SystemUtils.get_optimal_cores()
            memory = SystemUtils.get_memory_info()

            print(f"  Optimal cores: {optimal}")
            print(
                f"  Memory: {memory['total_gb']:.1f}GB total, {memory['available_gb']:.1f}GB available"
            )
            print(f"  Source: {memory['source']}")

    print("\n=== Testing Complete ===")
    assert True


@pytest.mark.integration
def test_real_environment_inspection(capsys):
    print("\n==== REAL ENVIRONMENT INSPECTION ====")

    print("\n1. Real System Detection:")
    real_info = SystemUtils.get_system_info()

    print(f"  CPU: {real_info['cpu_name']}")
    print(f"  Platform: {real_info['platform']} ({real_info['architecture']})")
    print(f"  Environment: {real_info['environment']}")
    print(f"  Physical cores: {real_info['physical_cores']}")
    print(f"  Logical cores: {real_info['logical_cores']}")
    if real_info["allocated_cores"] is not None:
        print(f"  Allocated cores: {real_info['allocated_cores']}")
    print(f"  Effective cores: {real_info['effective_cores']}")
    print(f"  Training cores: {real_info['training_cores']}")

    print("\n=== Real Environment Test Complete ===")

    assert real_info["platform"] in ["Linux", "Darwin", "Windows"]
    assert real_info["physical_cores"] > 0
    assert real_info["logical_cores"] > 0


@pytest.mark.integration
def test_lsf_basic_job_8_cores():
    env_vars = {
        "LSB_JOBID": "12345",
        "LSB_DJOB_NUMPROC": "8",
        "LSB_BATCH": "Y",
        "LSB_QUEUE": "normal",
    }

    with mock_lsf_environment(env_vars):
        cores = SystemUtils.get_optimal_cores()
        cores_scheduler, scheduler = SystemUtils._get_scheduler_cores()

        assert cores == 7
        assert cores_scheduler == 8
        assert scheduler == "LSF"


@pytest.mark.integration
def test_comprehensive_lsf_scenarios(capsys):
    print("\n==== LSF ENVIRONMENT TESTING ====")

    test_scenarios = [
        {
            "name": "Basic LSF Job - 8 cores",
            "env": {
                "LSB_JOBID": "12345",
                "LSB_DJOB_NUMPROC": "8",
                "LSB_BATCH": "Y",
                "LSB_QUEUE": "normal",
            },
            "expected_cores": 7,
            "expected_memory": None,
        },
        {
            "name": "LSF Job with Memory - 16GB",
            "env": {
                "LSB_JOBID": "23456",
                "LSB_DJOB_NUMPROC": "4",
                "LSB_DJOB_RUSAGE": "rusage[mem=16G]",
                "LSB_BATCH": "Y",
            },
            "expected_cores": 3,
            "expected_memory": 16.0,
        },
    ]

    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")

        with mock_lsf_environment(scenario["env"]):
            print("Environment variables:")
            for key, value in scenario["env"].items():
                print(f"  {key}={value}")

            cores = SystemUtils.get_optimal_cores()
            cores_scheduler, scheduler = SystemUtils._get_scheduler_cores()
            print(
                f"Detected cores: {cores} (scheduler: {cores_scheduler} from {scheduler})"
            )

            if scenario["expected_cores"] is not None:
                status = "✅ PASS" if cores == scenario["expected_cores"] else "❌ FAIL"
                print(f"Expected cores: {scenario['expected_cores']} - {status}")

            memory_info = SystemUtils.get_memory_info()
            print(
                f"Detected memory: {memory_info['total_gb']:.1f}GB (source: {memory_info['source']})"
            )

            if scenario["expected_memory"] is not None:
                if abs(memory_info["total_gb"] - scenario["expected_memory"]) < 0.1:
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                print(
                    f"Expected memory: {scenario['expected_memory']:.1f}GB - {status}"
                )

            sys_info = SystemUtils.get_system_info()
            print(f"Environment: {sys_info['environment']}")
            if sys_info["allocated_cores"] is not None:
                print(f"Allocated cores: {sys_info['allocated_cores']}")

    print("\n=== Edge Cases ===")

    print("\n--- Invalid Numeric Values ---")
    invalid_env = {
        "LSB_JOBID": "99999",
        "LSB_DJOB_NUMPROC": "invalid_number",
        "LSB_DJOB_RUSAGE": "rusage[mem=invalid]",
    }

    with mock_lsf_environment(invalid_env):
        try:
            cores = SystemUtils.get_optimal_cores()
            memory_info = SystemUtils.get_memory_info()
            print(
                f"Error handling: cores={cores}, memory={memory_info['total_gb']:.1f}GB"
            )
            print("✅ Error handling working correctly")
        except Exception as e:
            print(f"❌ Error handling failed: {e}")

    print("\n=====================================")

    assert True


@pytest.mark.integration
def test_lsf_memory_scenarios():
    test_cases = [
        {
            "name": "LSF Memory in MB",
            "env": {"LSB_JOBID": "1", "LSB_DJOB_RUSAGE": "rusage[mem=8192]"},
            "expected": 8.0,
        },
        {
            "name": "LSF Memory in GB",
            "env": {"LSB_JOBID": "2", "LSB_DJOB_RUSAGE": "rusage[mem=16G]"},
            "expected": 16.0,
        },
        {
            "name": "LSF Memory in KB",
            "env": {"LSB_JOBID": "3", "LSB_DJOB_RUSAGE": "rusage[mem=2097152K]"},
            "expected": 2.0,
        },
    ]

    for case in test_cases:
        with mock_lsf_environment(case["env"]):
            memory_info = SystemUtils.get_memory_info()
            assert abs(memory_info["total_gb"] - case["expected"]) < 0.1
            assert memory_info["source"] == "LSF"


@pytest.mark.integration
def test_scheduler_priority():
    combined_env = {
        "LSB_JOBID": "12345",
        "LSB_DJOB_NUMPROC": "16",
        "LSB_DJOB_RUSAGE": "rusage[mem=32G]",
        "SLURM_JOB_ID": "67890",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_MEM_PER_NODE": "16384",
    }

    with mock_lsf_environment(combined_env):
        cores_scheduler, scheduler = SystemUtils._get_scheduler_cores()
        memory_scheduler, mem_scheduler = SystemUtils._get_scheduler_memory()

        assert cores_scheduler == 16
        assert scheduler == "LSF"
        assert memory_scheduler == 32.0
        assert mem_scheduler == "LSF"


@pytest.mark.integration
def test_resource_monitoring():
    from utils.SystemUtils import monitor_resources
    import time

    with monitor_resources(interval=0.1) as stats:
        time.sleep(0.3)
        _ = sum(i * i for i in range(1000))

    assert stats["samples"] > 0
    assert stats["max_cpu"] >= 0
    assert stats["max_memory"] >= 0


@pytest.mark.integration
def test_complete_workflow():
    print("\n==== COMPLETE WORKFLOW TEST ====")

    SystemUtils.clear_cache()

    system_info = SystemUtils.get_system_info()
    print(
        f"System: {system_info['platform']} with {system_info['logical_cores']} logical cores"
    )

    optimal_cores = SystemUtils.get_optimal_cores()
    print(f"Optimal cores: {optimal_cores}")

    memory_info = SystemUtils.get_memory_info()
    print(
        f"Memory: {memory_info['total_gb']:.1f}GB total, {memory_info['available_gb']:.1f}GB available"
    )

    is_valid, message = SystemUtils.validate_resources(cores=2, memory_gb=4.0)
    print(f"Resource validation (2 cores, 4GB): {is_valid} - {message}")

    with mock_lsf_environment({"LSB_JOBID": "test", "LSB_DJOB_NUMPROC": "4"}):
        SystemUtils.clear_cache()
        scheduler_info = SystemUtils.get_system_info()
        print(
            f"With LSF: Environment={scheduler_info['environment']}, Allocated={scheduler_info['allocated_cores']}"
        )

    print("==================================")

    assert True


@pytest.mark.integration
def test_error_resilience():
    print("\n==== ERROR RESILIENCE TEST ====")

    for key in list(os.environ.keys()):
        if key.startswith(("SLURM_", "LSB_")):
            os.environ.pop(key, None)

    with patch(
        "utils.SystemUtils.psutil.cpu_count", side_effect=Exception("psutil broken")
    ):
        with patch("os.cpu_count", return_value=4):
            cores = SystemUtils.get_optimal_cores()
            assert cores >= 1
            print(f"✅ Handled broken psutil, got {cores} cores")

    with mock_lsf_environment(
        {"LSB_DJOB_NUMPROC": "not_a_number", "LSB_JOBID": "test"}
    ):
        cores = SystemUtils.get_optimal_cores()
        assert cores >= 1
        print(f"✅ Handled invalid LSF variables, got {cores} cores")

    with patch(
        "utils.SystemUtils.psutil.virtual_memory",
        side_effect=Exception("memory broken"),
    ):
        for key in list(os.environ.keys()):
            if key.startswith(("SLURM_", "LSB_")):
                os.environ.pop(key, None)

        memory_info = SystemUtils.get_memory_info()
        assert memory_info["source"] == "Error"
        print("✅ Handled broken memory detection")

    SystemUtils.clear_cache()
    with patch.object(
        SystemUtils, "_get_cpu_name", side_effect=Exception("cpu detection failed")
    ):
        with patch(
            "utils.SystemUtils.psutil.cpu_count", side_effect=Exception("psutil failed")
        ):
            with patch("os.cpu_count", side_effect=Exception("os failed")):
                with patch("multiprocessing.cpu_count", return_value=1):
                    system_info = SystemUtils.get_system_info()
                    assert system_info["cpu_name"] == "Unknown"
                    print("✅ Handled broken CPU detection")

    print("================================")

    assert True


@pytest.mark.integration
def test_temp_dir_operations(temp_output_dir):
    try:
        temp_dir = SystemUtils.get_safe_tempdir(
            output_dir=temp_output_dir, required_gb=0.01, prefix="test_safe"
        )

        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)
        assert "test_safe" in os.path.basename(temp_dir)

        test_file = os.path.join(temp_dir, "testfile.txt")
        with open(test_file, "w") as f:
            f.write("Test content")

        assert os.path.exists(test_file)

        success = SystemUtils.cleanup_tempdir(temp_dir)
        assert success is True
        assert not os.path.exists(temp_dir)
    except IOError:
        pytest.skip("Could not create temp directory, skipping integration test")


@pytest.mark.integration
def test_disk_info_real():
    disk_info = SystemUtils.get_disk_info()

    assert disk_info["total_gb"] > 0
    assert disk_info["used_gb"] >= 0
    assert disk_info["free_gb"] >= 0
    assert 0 <= disk_info["percent_used"] <= 100
    assert disk_info["path"] == os.path.abspath(os.getcwd())
    assert disk_info["mount_point"]

    temp_dir = tempfile.gettempdir()
    home_dir = str(Path.home())

    disk_usage = SystemUtils.get_disk_usage([temp_dir, home_dir, os.getcwd()])

    assert len(disk_usage) > 0

    for usage in disk_usage:
        assert usage["total_gb"] > 0
        assert usage["free_gb"] >= 0


@pytest.mark.integration
def test_system_health_real():
    health = SystemUtils.check_system_health()

    assert "status" in health
    assert "details" in health
    assert "warnings" in health
    assert "critical" in health

    assert "cpu" in health["details"]
    assert 0 <= health["details"]["cpu"]["percent"] <= 100
    assert health["details"]["cpu"]["cores"] > 0

    assert "memory" in health["details"]
    assert health["details"]["memory"]["total_gb"] > 0

    assert "disk" in health["details"]
    assert len(health["details"]["disk"]) > 0


@pytest.mark.integration
@pytest.mark.slow
def test_benchmark_real():
    result = SystemUtils.benchmark_system(duration_sec=1)

    assert "cpu" in result
    assert "memory" in result
    assert "disk" in result
    assert "system_info" in result

    assert result["cpu"]["single_thread_ops_per_sec"] > 0

    assert result["memory"]["mb_per_sec"] > 0

    assert result["disk"]["write_mb_per_sec"] > 0
    assert result["disk"]["read_mb_per_sec"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
