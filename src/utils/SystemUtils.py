#!/usr/bin/env python
# Import required modules
import json
import math
import multiprocessing
import os
import platform
import psutil
import random
import re
import resource
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from .LoggingUtils import log


class SystemConfig:
    LSF_ENV_VARS = [
        "LSB_JOBID",
        "LSB_DJOB_NUMPROC",
        "LSB_MAX_NUM_PROCESSORS",
        "LSB_HOSTS",
        "LSB_BATCH",
        "LSB_QUEUE",
    ]
    SLURM_ENV_VARS = [
        "SLURM_JOB_ID",
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_NPROCS",
        "SLURM_NTASKS",
    ]
    SLURM_MEMORY_VARS = ["SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"]
    LSF_MEMORY_VARS = ["LSB_DJOB_RUSAGE", "LSB_SUB_RUSAGE"]
    SUBPROCESS_TIMEOUT = 10
    MAX_MEMORY_GB = 1024
    MIN_CORES = 1
    SCHEDULER_MEMORY_OVERHEAD = 0.1
    SYSTEM_MEMORY_RESERVE = 0.05
    CACHE_TIMEOUT = 60


class SystemUtils:
    _cache_timeout: int = SystemConfig.CACHE_TIMEOUT
    _last_cache_time: float = 0
    _cached_system_info: Optional[Dict[str, Any]] = None
    _cache_lock = threading.RLock()

    @staticmethod
    def check_system_health(
        min_free_disk_gb: float = 5.0,
        max_cpu_percent: float = 90.0,
        max_memory_percent: float = 90.0,
        check_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform a system health check."""
        health = {
            "status": "healthy",
            "warnings": [],
            "critical": [],
            "details": {},
        }

        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            health["details"]["cpu"] = {
                "percent": cpu_percent,
                "cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
            }

            if cpu_percent > max_cpu_percent:
                health["status"] = "warning"
                health["warnings"].append(f"High CPU usage: {cpu_percent:.1f}%")

            memory = SystemUtils.get_memory_info()
            health["details"]["memory"] = memory

            if memory["percent_used"] > max_memory_percent:
                health["status"] = "warning"
                health["warnings"].append(
                    f"High memory usage: {memory['percent_used']:.1f}%"
                )

            if check_paths is None:
                paths = [os.getcwd(), str(Path.home()), tempfile.gettempdir()]
            else:
                paths = []
                for path in check_paths:
                    try:
                        resolved_path = os.path.realpath(os.path.abspath(path))
                        paths.append(resolved_path)
                    except Exception as e:
                        log.warn(f"Could not resolve path {path}: {e}")
                        paths.append(path)

                default_paths = [os.getcwd(), str(Path.home()), tempfile.gettempdir()]
                for default_path in default_paths:
                    resolved_default = os.path.realpath(os.path.abspath(default_path))
                    if resolved_default not in paths:
                        paths.append(resolved_default)

            disk_info = SystemUtils.get_disk_usage(paths)
            health["details"]["disk"] = disk_info

            for disk in disk_info:
                free_gb = disk.get("free_gb", 0)
                path = disk.get("path", "unknown")
                if free_gb < min_free_disk_gb:
                    if free_gb < (min_free_disk_gb / 2):
                        health["status"] = "critical"
                        health["critical"].append(
                            f"Very low disk space on {path}: {free_gb:.1f}GB free"
                        )
                    else:
                        if health["status"] != "critical":
                            health["status"] = "warning"
                        health["warnings"].append(
                            f"Low disk space on {path}: {free_gb:.1f}GB free"
                        )

            core_settings = SystemUtils.get_core_dump_settings()
            health["details"]["core_dumps"] = core_settings

            if core_settings.get("enabled", False):
                health["warnings"].append(
                    "Core dumps are enabled, may consume disk space on crash"
                )

            if platform.system() != "Windows":
                try:
                    load1, load5, load15 = os.getloadavg()
                    health["details"]["load_avg"] = {
                        "1min": load1,
                        "5min": load5,
                        "15min": load15,
                    }

                    core_count = psutil.cpu_count(logical=False) or 1
                    if load5 > core_count * 1.5:
                        health["warnings"].append(
                            f"High system load: {load5:.1f} (5 min average)"
                        )
                except Exception:
                    pass

            return health
        except Exception as e:
            log.error(f"Error in system health check: {e}")
            return {
                "status": "error",
                "warnings": [f"Error performing health check: {str(e)}"],
                "critical": [],
                "details": {},
            }

    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load system utility configuration from JSON file."""
        if config_path is None:
            config_path = Path.home() / ".system_utils.json"
        else:
            config_path = Path(config_path)
        default_config: Dict[str, Any] = {
            "reserve_cores": 1,
            "max_memory_gb": SystemConfig.MAX_MEMORY_GB,
            "subprocess_timeout": SystemConfig.SUBPROCESS_TIMEOUT,
            "enable_caching": True,
            "log_level": "INFO",
        }
        if config_path.exists():
            try:
                with open(config_path) as f:
                    user_config: Dict[str, Any] = json.load(f)
                default_config.update(user_config)
                log.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                log.warn(f"Failed to load config from {config_path}: {e}")
        return default_config

    @staticmethod
    def _run_command(
        cmd: List[str], timeout: int = SystemConfig.SUBPROCESS_TIMEOUT
    ) -> Optional[str]:
        """Run a system command and return its output."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=True
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            log.warn(f"Command {' '.join(cmd)} timed out after {timeout}s")
        except subprocess.CalledProcessError as e:
            log.debug(f"Command {' '.join(cmd)} failed with exit code {e.returncode}")
        except FileNotFoundError:
            log.debug(f"Command {cmd[0]} not found")
        except Exception as e:
            log.debug(f"Unexpected error running {' '.join(cmd)}: {e}")
        return None

    @staticmethod
    def _parse_memory_string(mem_str: str, default_unit: str = "M") -> float:
        """Parse a memory string like '4G' or '512M' into gigabytes."""
        try:
            match = re.match(r"(\d+(?:\.\d+)?)([KMGT]?B?)", mem_str.upper())
            if not match:
                raise ValueError(f"Cannot parse memory string: {mem_str}")
            value = float(match.group(1))
            unit = match.group(2).replace("B", "") or default_unit
            multipliers: Dict[str, float] = {
                "K": 1 / (1024 * 1024),
                "M": 1 / 1024,
                "G": 1,
                "T": 1024,
            }
            if unit not in multipliers:
                raise ValueError(f"Unknown memory unit: {unit}")
            return value * multipliers[unit]
        except Exception as e:
            raise ValueError(f"Error parsing memory string '{mem_str}': {e}")

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_cpu_name() -> str:
        """Retrieve the CPU model name."""
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":", 1)[1].strip()
                output = SystemUtils._run_command(["lscpu"])
                if output:
                    for line in output.splitlines():
                        if "Model name" in line:
                            return line.split(":", 1)[1].strip()
            elif platform.system() == "Darwin":
                output = SystemUtils._run_command(
                    ["sysctl", "-n", "machdep.cpu.brand_string"]
                )
                if output:
                    return output
            elif platform.system() == "Windows":
                output = SystemUtils._run_command(
                    ["wmic", "cpu", "get", "name", "/value"]
                )
                if output:
                    for line in output.splitlines():
                        if line.startswith("Name="):
                            return line.split("=", 1)[1].strip()
        except Exception as e:
            log.debug(f"Failed to get CPU name: {e}")
        return f"Unknown CPU ({platform.machine()})"

    @staticmethod
    def get_optimal_cores(
        reserve_cores: int = 1, max_cores: Optional[int] = None
    ) -> int:
        """Determine the optimal number of CPU cores to use."""
        if reserve_cores < 0:
            raise ValueError("reserve_cores must be non-negative")
        if max_cores is not None and max_cores < 1:
            raise ValueError("max_cores must be at least 1")
        try:
            allocated_cores, scheduler = SystemUtils._get_scheduler_cores()
            if scheduler == "LSF" and allocated_cores is None:
                log.warn("LSF environment detected but no CPU allocation found")
                optimal_cores = SystemConfig.MIN_CORES
            elif allocated_cores is not None:
                log.info(
                    f"{scheduler} environment detected: {allocated_cores} allocated cores"
                )
                optimal_cores = max(
                    SystemConfig.MIN_CORES, allocated_cores - reserve_cores
                )
            else:
                physical_cores = psutil.cpu_count(logical=False)
                logical_cores = (
                    psutil.cpu_count(logical=True) or multiprocessing.cpu_count()
                )
                if physical_cores is None:
                    log.warn(
                        "Unable to detect physical cores, falling back to logical cores"
                    )
                    physical_cores = logical_cores
                log.info(
                    f"System cores: {physical_cores} physical, {logical_cores} logical"
                )
                optimal_cores = max(
                    SystemConfig.MIN_CORES, physical_cores - reserve_cores
                )
            if max_cores is not None:
                optimal_cores = min(optimal_cores, max_cores)
            log.info(
                f"Optimal cores for processing: {optimal_cores} (reserving {reserve_cores} core(s))"
            )
            return optimal_cores
        except Exception as e:
            log.error(f"Error detecting CPU cores: {e}")
            fallback_cores = max(
                SystemConfig.MIN_CORES, (os.cpu_count() or 1) - reserve_cores
            )
            log.warn(f"Using fallback: {fallback_cores} cores")
            return fallback_cores

    @staticmethod
    def _get_scheduler_cores() -> Tuple[Optional[int], Optional[str]]:
        """Determine allocated CPU cores from scheduler environment."""
        try:
            if any(var in os.environ for var in SystemConfig.LSF_ENV_VARS):
                lsf_cores = SystemUtils._get_lsf_cores()
                if lsf_cores is not None:
                    return lsf_cores, "LSF"
                else:
                    return None, "LSF"
            if any(var in os.environ for var in SystemConfig.SLURM_ENV_VARS):
                slurm_cores = SystemUtils._get_slurm_cores()
                if slurm_cores is not None:
                    return slurm_cores, "SLURM"
                else:
                    return None, "SLURM"
            return None, None
        except Exception as e:
            log.debug(f"Error checking scheduler allocation: {e}")
            return None, None

    @staticmethod
    def _get_slurm_cores() -> Optional[int]:
        """Determine allocated CPU cores from SLURM environment."""
        try:
            slurm_vars = [
                "SLURM_CPUS_PER_TASK",
                "SLURM_CPUS_ON_NODE",
                "SLURM_NPROCS",
                "SLURM_NTASKS",
            ]
            for var in slurm_vars:
                if var in os.environ:
                    try:
                        cores = int(os.environ[var])
                        return cores
                    except ValueError:
                        log.debug(f"Invalid value for {var}: {os.environ[var]}")
                        continue
            if "SLURM_JOB_ID" in os.environ:
                log.warn("SLURM environment detected but no CPU allocation found")
                job_id = os.environ["SLURM_JOB_ID"]
                output = SystemUtils._run_command(["scontrol", "show", "job", job_id])
                if output:
                    for line in output.split("\n"):
                        if "NumCPUs=" in line:
                            for part in line.split():
                                if part.startswith("NumCPUs="):
                                    try:
                                        cores = int(part.split("=")[1])
                                        log.debug(
                                            f"Found NumCPUs from scontrol: {cores}"
                                        )
                                        return cores
                                    except ValueError:
                                        continue
            return None
        except Exception as e:
            log.debug(f"Error checking SLURM allocation: {e}")
            return None

    @staticmethod
    def _get_lsf_cores() -> Optional[int]:
        """Determine allocated CPU cores from LSF environment."""
        try:
            if "LSB_DJOB_NUMPROC" in os.environ:
                try:
                    cores = int(os.environ["LSB_DJOB_NUMPROC"])
                    log.debug(f"Found LSB_DJOB_NUMPROC: {cores}")
                    return cores
                except ValueError:
                    log.debug(
                        f"Error parsing LSB_DJOB_NUMPROC: {os.environ['LSB_DJOB_NUMPROC']}"
                    )
            if "LSB_MAX_NUM_PROCESSORS" in os.environ:
                try:
                    cores = int(os.environ["LSB_MAX_NUM_PROCESSORS"])
                    log.debug(f"Found LSB_MAX_NUM_PROCESSORS: {cores}")
                    return cores
                except ValueError:
                    log.debug(
                        f"Error parsing LSB_MAX_NUM_PROCESSORS: {os.environ['LSB_MAX_NUM_PROCESSORS']}"
                    )
            if "LSB_HOSTS" in os.environ:
                try:
                    hosts = os.environ["LSB_HOSTS"].split()
                    cores = len(hosts)
                    if cores > 0:
                        log.debug(f"Found {cores} hosts in LSB_HOSTS")
                        return cores
                    else:
                        log.debug("LSB_HOSTS is empty - no core allocation found")
                except Exception as e:
                    log.debug(f"Error parsing LSB_HOSTS: {e}")

            if "LSB_JOBID" in os.environ:
                job_id = os.environ["LSB_JOBID"]
                log.debug(f"Attempting to query job {job_id} with bjobs")

                try:
                    output = SystemUtils._run_command(
                        ["bjobs", "-l", job_id], timeout=5
                    )
                    if output:
                        proc_match = re.search(
                            r"(?:Processors|processors)[:\s]+(\d+)",
                            output,
                            re.IGNORECASE,
                        )
                        if proc_match:
                            cores = int(proc_match.group(1))
                            log.debug(f"Found {cores} processors from bjobs output")
                            return cores
                except Exception as e:
                    log.debug(f"bjobs command failed: {e}")

                log.debug(
                    "bjobs failed, attempting to infer core count from LSF environment"
                )

                if "LSB_QUEUE" in os.environ:
                    log.debug(f"Running in LSF queue: {os.environ['LSB_QUEUE']}")

                if "LSB_BATCH" in os.environ:
                    log.debug(f"LSF batch mode: {os.environ['LSB_BATCH']}")

                log.warn(
                    "LSF environment detected but no CPU allocation found, assuming 1 core"
                )
                return 1

            return None
        except Exception as e:
            log.debug(f"Error checking LSF allocation: {e}")
            return None

    @staticmethod
    def _get_scheduler_memory() -> Tuple[Optional[float], Optional[str]]:
        """Determine allocated memory from scheduler environment."""
        try:
            if any(var in os.environ for var in SystemConfig.LSF_ENV_VARS):
                lsf_memory = SystemUtils._get_lsf_memory()
                if lsf_memory is not None:
                    return lsf_memory, "LSF"
                else:
                    return None, "LSF"
            if any(var in os.environ for var in SystemConfig.SLURM_ENV_VARS):
                slurm_memory = SystemUtils._get_slurm_memory()
                if slurm_memory is not None:
                    return slurm_memory, "SLURM"
                else:
                    return None, "SLURM"
            return None, None
        except Exception as e:
            log.debug(f"Error checking scheduler memory allocation: {e}")
            return None, None

    @staticmethod
    def _get_slurm_memory() -> Optional[float]:
        """Determine allocated memory from SLURM environment."""
        try:
            for var in SystemConfig.SLURM_MEMORY_VARS:
                if var in os.environ:
                    try:
                        mem_mb = int(os.environ[var])
                        if var == "SLURM_MEM_PER_CPU":
                            cores, _ = SystemUtils._get_scheduler_cores()
                            if cores:
                                mem_mb *= cores
                        mem_gb = mem_mb / 1024.0
                        return mem_gb
                    except ValueError:
                        log.debug(f"Invalid value for {var}: {os.environ[var]}")
                        continue
            if "SLURM_JOB_ID" in os.environ:
                job_id = os.environ["SLURM_JOB_ID"]
                output = SystemUtils._run_command(["scontrol", "show", "job", job_id])
                if output:
                    for line in output.split("\n"):
                        for part in line.split():
                            if "MinMemory" in part and "=" in part:
                                try:
                                    mem_str = part.split("=")[1]
                                    mem_gb = SystemUtils._parse_memory_string(
                                        mem_str, "M"
                                    )
                                    return mem_gb
                                except (ValueError, IndexError):
                                    continue
            return None
        except Exception as e:
            log.debug(f"Error checking SLURM memory allocation: {e}")
            return None

    @staticmethod
    def _get_lsf_memory() -> Optional[float]:
        """Determine allocated memory from LSF environment."""
        try:
            for var in SystemConfig.LSF_MEMORY_VARS:
                if var in os.environ:
                    rusage = os.environ[var]
                    mem_match = re.search(
                        r"mem=(\d+(?:\.\d+)?)([GMK]?B?)", rusage, re.IGNORECASE
                    )
                    if mem_match:
                        try:
                            mem_str = f"{mem_match.group(1)}{mem_match.group(2) or 'M'}"
                            mem_gb = SystemUtils._parse_memory_string(mem_str, "M")
                            return mem_gb
                        except ValueError as e:
                            log.debug(f"Error parsing LSF memory value: {e}")
            if "LSB_JOBID" in os.environ:
                job_id = os.environ["LSB_JOBID"]
                output = SystemUtils._run_command(["bjobs", "-l", job_id])
                if output:
                    mem_match = re.search(
                        r"mem=(\d+(?:\.\d+)?)([GMK]?B?)",
                        output,
                        re.IGNORECASE,
                    )
                    if mem_match:
                        try:
                            mem_str = f"{mem_match.group(1)}{mem_match.group(2) or 'M'}"
                            mem_gb = SystemUtils._parse_memory_string(mem_str, "M")
                            return mem_gb
                        except ValueError as e:
                            log.debug(f"Error parsing bjobs memory value: {e}")
            return None
        except Exception as e:
            log.debug(f"Error checking LSF memory allocation: {e}")
            return None

    @staticmethod
    def get_memory_info() -> Dict[str, Union[float, str]]:
        """Get system memory information."""
        try:
            scheduler_memory, scheduler = SystemUtils._get_scheduler_memory()
            if scheduler_memory is not None:
                available = scheduler_memory * (
                    1.0 - SystemConfig.SCHEDULER_MEMORY_OVERHEAD
                )
                return {
                    "total_gb": scheduler_memory,
                    "available_gb": available,
                    "used_gb": scheduler_memory - available,
                    "percent_used": SystemConfig.SCHEDULER_MEMORY_OVERHEAD * 100,
                    "source": scheduler,
                }
            else:
                memory = psutil.virtual_memory()
                return {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent_used": memory.percent,
                    "source": "System",
                }
        except Exception as e:
            log.error(f"Error getting memory info: {e}")
            return {
                "total_gb": 0.0,
                "available_gb": 0.0,
                "used_gb": 0.0,
                "percent_used": 0.0,
                "source": "Error",
            }

    @staticmethod
    def get_disk_info(path: Optional[str] = None) -> Dict[str, Union[float, str]]:
        """Get disk usage information for the specified path."""
        try:
            if path is None:
                path = os.getcwd()

            path = os.path.realpath(os.path.abspath(path))

            if not os.path.exists(path):
                log.warn(f"Path does not exist: {path}")
                path = os.path.dirname(path)
                if not os.path.exists(path):
                    log.warn(
                        f"Parent path does not exist: {path}, using root directory"
                    )
                    path = os.path.abspath(os.sep)

            disk_usage = psutil.disk_usage(path)
            return {
                "total_gb": disk_usage.total / (1024**3),
                "used_gb": disk_usage.used / (1024**3),
                "free_gb": disk_usage.free / (1024**3),
                "percent_used": disk_usage.percent,
                "path": path,
                "mount_point": SystemUtils._get_mount_point(path),
            }
        except Exception as e:
            log.error(f"Error getting disk info for {path}: {e}")
            return {
                "total_gb": 0.0,
                "used_gb": 0.0,
                "free_gb": 0.0,
                "percent_used": 0.0,
                "path": path or "unknown",
                "mount_point": "",
                "error": str(e),
            }

    @staticmethod
    def _get_mount_point(path: str) -> str:
        """Get the mount point for the specified path."""
        try:
            path = os.path.abspath(path)
            while not os.path.ismount(path):
                parent = os.path.dirname(path)
                if parent == path:
                    break
                path = parent
            return path
        except Exception as e:
            log.debug(f"Error getting mount point: {e}")
            return ""

    @staticmethod
    def check_disk_space(
        path: str, required_gb: float, buffer_percent: float = 5.0
    ) -> Tuple[bool, str]:
        """Check if there is sufficient disk space at the specified path."""
        try:
            resolved_path = os.path.realpath(os.path.abspath(path))
            disk_info = SystemUtils.get_disk_info(resolved_path)
            free_gb = disk_info["free_gb"]
            required_with_buffer = required_gb * (1 + buffer_percent / 100)

            if free_gb >= required_with_buffer:
                return (
                    True,
                    f"Sufficient disk space: {free_gb:.1f}GB available, {required_gb:.1f}GB required",
                )
            else:
                buffer_gb = required_gb * buffer_percent / 100
                return False, (
                    f"Insufficient disk space at {resolved_path}: {free_gb:.1f}GB available, "
                    f"{required_gb:.1f}GB required (plus {buffer_gb:.1f}GB buffer)"
                )
        except Exception as e:
            return False, f"Error checking disk space: {e}"

    @staticmethod
    def get_disk_usage(
        paths: Optional[List[str]] = None,
    ) -> List[Dict[str, Union[float, str]]]:
        """Get disk usage information for the specified paths."""
        if not paths:
            paths = [os.getcwd()]

        try:
            mount_points = set()
            result = []

            for path in paths:
                resolved_path = os.path.realpath(os.path.abspath(path))
                disk_info = SystemUtils.get_disk_info(resolved_path)
                mount_point = disk_info.get("mount_point", resolved_path)

                if mount_point not in mount_points:
                    mount_points.add(mount_point)
                    result.append(disk_info)

            return result
        except Exception as e:
            log.error(f"Error getting disk usage: {e}")
            return [{"error": str(e)}]

    @staticmethod
    def validate_resources(cores: int, memory_gb: float) -> Tuple[bool, str]:
        """Validate requested CPU cores and memory against system availability."""
        try:
            system_info = SystemUtils.get_system_info()
            max_cores = system_info.get("effective_cores", 1)
            if cores > max_cores:
                return False, f"Requested {cores} cores but only {max_cores} available"
            if cores < 1:
                return False, "Must request at least 1 core"
            available_memory = system_info.get("ram_available_gb", 0)
            if memory_gb > available_memory:
                return (
                    False,
                    f"Requested {memory_gb:.1f}GB but only {available_memory:.1f}GB available",
                )
            if memory_gb < 0:
                return False, "Memory request must be positive"
            memory_per_core = memory_gb / cores if cores > 0 else memory_gb
            if memory_per_core > 32:
                log.info(f"High memory-to-core ratio: {memory_per_core:.1f}GB per core")
            return True, "Resources validated successfully"
        except Exception as e:
            return False, f"Error validating resources: {e}"

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Retrieve and cache system information."""
        current_time = time.time()
        with SystemUtils._cache_lock:
            condition1 = SystemUtils._cached_system_info is not None
            condition2 = (
                current_time - SystemUtils._last_cache_time < SystemUtils._cache_timeout
            )
            if condition1 and condition2:
                return SystemUtils._cached_system_info.copy()
        try:
            cpu_info = SystemUtils._get_cpu_name()
            scheduler_cores, scheduler = SystemUtils._get_scheduler_cores()
            logical_cores = os.cpu_count() or multiprocessing.cpu_count()
            physical_cores = psutil.cpu_count(logical=False)
            memory_info = SystemUtils.get_memory_info()
            if scheduler_cores is not None:
                effective_cores = scheduler_cores
                environment = scheduler
            else:
                effective_cores = physical_cores or logical_cores
                environment = "Native"
            training_cores = effective_cores
            if "OMP_NUM_THREADS" in os.environ:
                try:
                    training_cores = int(os.environ["OMP_NUM_THREADS"])
                except ValueError:
                    log.debug("Invalid OMP_NUM_THREADS value")
            elif "MKL_NUM_THREADS" in os.environ:
                try:
                    training_cores = int(os.environ["MKL_NUM_THREADS"])
                except ValueError:
                    log.debug("Invalid MKL_NUM_THREADS value")
            system_info: Dict[str, Any] = {
                "cpu_name": cpu_info,
                "physical_cores": physical_cores,
                "logical_cores": logical_cores,
                "allocated_cores": scheduler_cores,
                "effective_cores": effective_cores,
                "environment": environment,
                "ram_total_gb": memory_info["total_gb"],
                "ram_available_gb": memory_info["available_gb"],
                "memory_source": memory_info["source"],
                "training_cores": training_cores,
                "platform": platform.system(),
                "architecture": platform.machine(),
            }
            SystemUtils._cached_system_info = system_info.copy()
            SystemUtils._last_cache_time = current_time
            return system_info
        except Exception as e:
            log.error(f"Failed to retrieve system information: {e}")
            return {
                "cpu_name": "Unknown",
                "physical_cores": 1,
                "logical_cores": 1,
                "allocated_cores": None,
                "effective_cores": 1,
                "environment": "Unknown",
                "ram_total_gb": 0.0,
                "ram_available_gb": 0.0,
                "memory_source": "Error",
                "training_cores": 1,
                "platform": platform.system(),
                "architecture": platform.machine(),
            }

    @staticmethod
    def print_system_info() -> None:
        """Print system information to the log."""
        info = SystemUtils.get_system_info()
        log.debug(f"CPU: {info['cpu_name']}")
        log.debug(f"Environment: {info['environment']}")
        if info["allocated_cores"] is not None:
            log.debug(
                f"Allocated cores: {info['allocated_cores']} ({info['environment']})"
            )
        log.debug(
            f"System cores: {info['physical_cores']} physical, {info['logical_cores']} logical"
        )
        log.debug(f"Effective cores: {info['effective_cores']}")
        log.debug(
            f"Memory total: {info['ram_total_gb']:.1f} GB, available: {info['ram_available_gb']:.1f} GB"
        )
        log.debug(f"Memory source: {info['memory_source']}")
        log.debug(f"Platform: {info['platform']} ({info['architecture']})")
        log.debug(f"Training using: {info['training_cores']} threads")

    @staticmethod
    def clear_cache() -> None:
        """Clear cached system information."""
        SystemUtils._cached_system_info = None
        SystemUtils._last_cache_time = 0
        SystemUtils._get_cpu_name.cache_clear()

    @staticmethod
    def disable_core_dumps() -> Tuple[bool, str]:
        """Disable core dumps on the system."""
        try:
            try:
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                log.info("Core dumps disabled via Python resource module")
                return True, "Core dumps disabled"
            except (ValueError, resource.error) as e:
                log.warn(f"Failed to disable core dumps via resource module: {e}")

            if platform.system() != "Windows":
                result = SystemUtils._run_command(["bash", "-c", "ulimit -c 0"])
                if result is not None:
                    log.info("Core dumps disabled via ulimit command")
                    return True, "Core dumps disabled via system command"

                result = SystemUtils._run_command(["sh", "-c", "ulimit -c 0"])
                if result is not None:
                    log.info("Core dumps disabled via sh ulimit command")
                    return True, "Core dumps disabled via system command"

            if platform.system() == "Windows":
                log.info("Core dumps not applicable on Windows systems")
                return True, "Core dumps not applicable on this system"

            return False, "Failed to disable core dumps"
        except Exception as e:
            log.error(f"Error disabling core dumps: {e}")
            return False, f"Error: {str(e)}"

    @staticmethod
    def get_core_dump_settings() -> Dict[str, Any]:
        """Retrieve core dump settings from the system."""
        settings = {
            "enabled": True,
            "size_limit": "unknown",
            "pattern": "unknown",
            "platform": platform.system(),
        }

        try:
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
                settings["size_limit"] = soft
                settings["hard_limit"] = hard
                settings["enabled"] = soft > 0
            except (ValueError, resource.error, AttributeError) as e:
                log.debug(f"Could not get resource limits: {e}")

            if platform.system() == "Linux":
                try:
                    with open("/proc/sys/kernel/core_pattern", "r") as f:
                        settings["pattern"] = f.read().strip()
                except (IOError, PermissionError) as e:
                    log.debug(f"Could not read core pattern: {e}")

                try:
                    with open("/proc/sys/kernel/core_uses_pid", "r") as f:
                        settings["uses_pid"] = f.read().strip() == "1"
                except (IOError, PermissionError):
                    settings["uses_pid"] = "unknown"

            if platform.system() != "Windows":
                result = SystemUtils._run_command(["bash", "-c", "ulimit -c"])
                if result is None:
                    result = SystemUtils._run_command(["sh", "-c", "ulimit -c"])

                if result is not None:
                    try:
                        if result.strip() == "0":
                            settings["enabled"] = False
                        elif result.strip() == "unlimited":
                            settings["size_limit"] = "unlimited"
                        else:
                            settings["size_limit"] = int(result.strip())
                    except ValueError:
                        pass
        except Exception as e:
            log.error(f"Error getting core dump settings: {e}")
            settings["error"] = str(e)

        return settings

    @staticmethod
    def create_safe_tempdir(
        default_path: Optional[str] = None,
        required_gb: float = 1.0,
        prefix: str = "temp",
        buffer_percent: float = 10.0,
        alternative_paths: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Create a temporary directory in a location with sufficient disk space."""
        if default_path is None:
            default_path = os.getcwd()

        if alternative_paths is None:
            alternative_paths = [
                os.path.join(str(Path.home()), "temp"),
                os.path.join(str(Path.home())),
                tempfile.gettempdir(),
                "/tmp" if platform.system() != "Windows" else "C:\\Temp",
            ]

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique_prefix = f"{prefix}_{timestamp}"

        all_paths = [default_path] + alternative_paths

        result_info = {
            "checked_paths": [],
            "success": False,
            "chosen_path": None,
            "required_gb": required_gb,
            "buffer_percent": buffer_percent,
        }

        for path in all_paths:
            try:
                if not os.path.exists(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                        log.debug(f"Created directory: {path}")
                    except (IOError, PermissionError) as e:
                        log.debug(f"Cannot create directory {path}: {e}")
                        result_info["checked_paths"].append(
                            {"path": path, "error": str(e)}
                        )
                        continue

                has_space, message = SystemUtils.check_disk_space(
                    path, required_gb, buffer_percent
                )
                path_info = {"path": path, "has_space": has_space, "message": message}
                result_info["checked_paths"].append(path_info)

                if has_space:
                    try:
                        temp_dir = tempfile.mkdtemp(prefix=unique_prefix, dir=path)
                        result_info["success"] = True
                        result_info["chosen_path"] = temp_dir
                        log.debug(f"Created temporary directory: {temp_dir}")
                        return temp_dir, result_info
                    except (IOError, PermissionError) as e:
                        path_info["error"] = str(e)
                        log.warn(f"Failed to create temp directory in {path}: {e}")
            except Exception as e:
                log.warn(f"Error checking path {path}: {e}")
                result_info["checked_paths"].append({"path": path, "error": str(e)})

        log.error(
            "Could not find location with sufficient disk space for temporary directory"
        )
        raise IOError(f"No location with {required_gb}GB space available.")

    @staticmethod
    def get_safe_tempdir(
        output_dir: Optional[str] = None,
        required_gb: float = 1.0,
        prefix: str = "temp",
    ) -> str:
        """Get a safe temporary directory, falling back to system temp if needed."""
        try:
            temp_dir, _ = SystemUtils.create_safe_tempdir(
                default_path=output_dir, required_gb=required_gb, prefix=prefix
            )
            return temp_dir
        except IOError as e:
            log.warn(f"Failed to find safe temp location: {e}")
            log.warn("Using system temp directory as last resort")
            return tempfile.mkdtemp(prefix=prefix)

    @staticmethod
    def cleanup_tempdir(temp_dir: str, silent: bool = False) -> bool:
        """Remove the specified temporary directory."""
        if not temp_dir or not os.path.exists(temp_dir):
            return True

        try:
            shutil.rmtree(temp_dir)
            return True
        except Exception as e:
            if not silent:
                log.warn(f"Failed to clean up temporary directory {temp_dir}: {e}")
            return False

    @staticmethod
    def cleanup_stale_temp_files(
        directories: Optional[List[str]] = None,
        prefix: str = "temp",
        max_age_hours: int = 24,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Clean up stale temporary files and directories."""
        if directories is None:
            directories = [
                os.path.join(str(Path.home()), "temp"),
                tempfile.gettempdir(),
            ]

        result = {
            "scanned_dirs": directories,
            "files_found": 0,
            "files_deleted": 0,
            "dirs_found": 0,
            "dirs_deleted": 0,
            "errors": [],
            "deleted_paths": [],
            "space_reclaimed_bytes": 0,
            "dry_run": dry_run,
        }

        max_age_seconds = max_age_hours * 3600
        current_time = time.time()

        for directory in directories:
            try:
                if not os.path.exists(directory) or not os.path.isdir(directory):
                    result["errors"].append(f"Directory does not exist: {directory}")
                    continue

                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)

                    if not item.startswith(prefix):
                        continue

                    try:
                        item_stat = os.stat(item_path)
                        item_age = current_time - item_stat.st_mtime

                        if item_age < max_age_seconds:
                            continue

                        if os.path.isfile(item_path):
                            result["files_found"] += 1
                            result["space_reclaimed_bytes"] += item_stat.st_size
                            if not dry_run:
                                os.remove(item_path)
                                result["files_deleted"] += 1
                            result["deleted_paths"].append(item_path)
                        elif os.path.isdir(item_path):
                            result["dirs_found"] += 1
                            dir_size = 0
                            for root, dirs, files in os.walk(item_path):
                                for f in files:
                                    try:
                                        file_path = os.path.join(root, f)
                                        dir_size += os.path.getsize(file_path)
                                    except Exception:
                                        pass

                            result["space_reclaimed_bytes"] += dir_size
                            if not dry_run:
                                shutil.rmtree(item_path)
                                result["dirs_deleted"] += 1
                            result["deleted_paths"].append(item_path)
                    except Exception as e:
                        result["errors"].append(
                            f"Error processing {item_path}: {str(e)}"
                        )
            except Exception as e:
                result["errors"].append(
                    f"Error scanning directory {directory}: {str(e)}"
                )

        result["space_reclaimed_gb"] = result["space_reclaimed_bytes"] / (1024**3)
        return result

    @staticmethod
    def benchmark_system(duration_sec: int = 5) -> Dict[str, Any]:
        """Benchmark system CPU, memory, and disk performance."""
        results = {
            "cpu": {},
            "memory": {},
            "disk": {},
            "system_info": SystemUtils.get_system_info(),
        }

        log.info("Starting system benchmark...")

        def cpu_benchmark() -> float:
            start = time.time()
            iterations = 0
            while time.time() - start < duration_sec:
                for _ in range(10000):
                    x = random.random() * 1000
                    math.sqrt(x)
                    math.sin(x) * math.cos(x)
                    math.log(x + 1)
                iterations += 10000
            elapsed = time.time() - start
            return iterations / elapsed if elapsed > 0 else 0.0

        log.info("Benchmarking CPU (single-threaded)...")
        ops_per_sec = cpu_benchmark()
        results["cpu"]["single_thread_ops_per_sec"] = ops_per_sec

        log.info("Benchmarking CPU (multi-threaded)...")
        cores = min(4, multiprocessing.cpu_count())
        multi_ops = 0.0
        try:
            with ProcessPoolExecutor(max_workers=cores) as executor:
                futures = [executor.submit(cpu_benchmark) for _ in range(cores)]
                multi_ops = sum(f.result() for f in futures)
        except Exception as e:
            log.debug(f"Multi-threaded benchmark failed: {e}")
        results["cpu"]["multi_thread_ops_per_sec"] = multi_ops
        results["cpu"]["parallelization_factor"] = multi_ops / (ops_per_sec or 1)

        log.info("Benchmarking memory...")
        start = time.time()
        iterations = 0
        while time.time() - start < duration_sec:
            size = 10 * 1024 * 1024
            data = bytearray(size)
            for i in range(0, size, 1024):
                data[i] = i % 256
            iterations += 1

        memory_mb_per_sec = (iterations * 10) / duration_sec
        results["memory"]["mb_per_sec"] = memory_mb_per_sec

        log.info("Benchmarking disk...")
        temp_file = None
        try:
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"benchmark_{os.getpid()}.tmp")

            start = time.time()
            total_mb = 0
            chunk_size = 1024 * 1024
            chunk = b"x" * chunk_size
            with open(temp_file, "wb") as f:
                while time.time() - start < duration_sec:
                    f.write(chunk)
                    total_mb += 1
            write_mb_per_sec = total_mb / duration_sec
            results["disk"]["write_mb_per_sec"] = write_mb_per_sec

            start = time.time()
            total_mb = 0
            with open(temp_file, "rb") as f:
                while time.time() - start < duration_sec:
                    data = f.read(chunk_size)
                    if not data:
                        f.seek(0)
                        continue
                    total_mb += 1
            read_mb_per_sec = total_mb / duration_sec
            results["disk"]["read_mb_per_sec"] = read_mb_per_sec

        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

        log.info("Benchmark complete")
        return results

    @staticmethod
    def configure_safe_environment() -> Dict[str, bool]:
        """Configure system environment for safe operation."""
        results = {}

        success, _ = SystemUtils.disable_core_dumps()
        results["core_dumps_disabled"] = success

        try:
            memory_info = SystemUtils.get_memory_info()
            available_bytes = int(memory_info["available_gb"] * 0.8 * 1024**3)
            try:
                if hasattr(resource, "RLIMIT_AS"):
                    _, hard = resource.getrlimit(resource.RLIMIT_AS)
                    new_limit = (
                        min(available_bytes, hard)
                        if hard != resource.RLIM_INFINITY
                        else available_bytes
                    )
                    resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
                    results["memory_limit_set"] = True
                else:
                    results["memory_limit_set"] = False
            except (ValueError, resource.error):
                results["memory_limit_set"] = False
        except Exception:
            results["memory_limit_set"] = False

        if platform.system() == "Linux":
            try:
                if hasattr(os, "sched_getaffinity") and hasattr(
                    os, "sched_setaffinity"
                ):
                    current_affinity = os.sched_getaffinity(0)
                    os.sched_setaffinity(0, current_affinity)
                    results["cpu_affinity_configurable"] = True
                else:
                    results["cpu_affinity_configurable"] = False
            except Exception:
                results["cpu_affinity_configurable"] = False
        else:
            results["cpu_affinity_configurable"] = False

        return results


@contextmanager
def monitor_resources(
    interval: float = 1.0,
) -> Generator[Dict[str, Union[float, int]], None, None]:
    """Context manager to monitor CPU and memory usage during a code block."""
    monitoring = True
    stats: Dict[str, Union[float, int]] = {
        "max_cpu": 0.0,
        "max_memory": 0.0,
        "samples": 0,
    }

    def monitor() -> None:
        while monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                stats["max_cpu"] = max(stats["max_cpu"], cpu_percent)
                stats["max_memory"] = max(stats["max_memory"], memory_info.percent)
                stats["samples"] += 1
                time.sleep(interval)
            except Exception as e:
                log.debug(f"Error monitoring resources: {e}")
                break

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        yield stats
    finally:
        monitoring = False
        monitor_thread.join(timeout=1.0)
        if stats["samples"] > 0:
            log.info(
                f"Resource usage - Max CPU: {stats['max_cpu']:.1f}%, "
                f"Max Memory: {stats['max_memory']:.1f}%"
            )
