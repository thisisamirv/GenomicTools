#!/usr/bin/env python
# Import required modules
import logging
import logging.handlers
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Union

try:
    from colorlog import ColoredFormatter

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False
    ColoredFormatter = logging.Formatter


class LoggerError(Exception):
    pass


def _configure_worker_loggers() -> None:
    """Configure loggers for worker processes to suppress noisy logs."""
    try:
        for logger_name in [
            "matplotlib",
            "matplotlib.font_manager",
            "PIL",
            "h5py",
            "h5py._conv",
            "numpy",
            "scipy",
        ]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        for logger_name in [
            "numba",
            "numba.core.ssa",
            "numba.core.byteflow",
            "numba.core.interpreter",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        for logger_name in [
            "requests",
            "urllib3",
            "urllib3.connectionpool",
            "requests.packages.urllib3",
            "requests.packages.urllib3.connectionpool",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    except Exception:
        pass


class Logger:
    _instance: Optional["Logger"] = None
    _initialized: bool = False
    _lock = threading.RLock()

    def __new__(cls) -> "Logger":
        """Singleton implementation to ensure only one Logger instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the Logger instance."""
        if not Logger._initialized:
            with Logger._lock:
                if not Logger._initialized:
                    self.manager: Optional[mp.Manager] = None
                    self.mp_queue: Optional[mp.Queue] = None
                    self.listener: Optional[threading.Thread] = None
                    self.hidden_log_path: Optional[Path] = None
                    self.user_log_path: Optional[Path] = None
                    self._setup_default()
                    Logger._initialized = True

    def _setup_default(self) -> None:
        """Set up default logging configuration."""
        self.setup(level=logging.WARNING, file=None)

    def format_log_message(self, message: str = "", *args: Any) -> str:
        """Format log message with additional arguments."""
        if not args:
            return message
        try:
            formatted_args = []
            for arg in args:
                if hasattr(arg, "to_string") and hasattr(arg, "shape"):
                    formatted_args.append(f"\n{arg.to_string()}")
                elif hasattr(arg, "__len__") and not isinstance(arg, str):
                    if len(arg) > 10:
                        formatted_args.append(
                            f"{type(arg).__name__}({len(arg)} items): {str(arg[:5])}...{str(arg[-5:])}"
                        )
                    else:
                        formatted_args.append(str(arg))
                else:
                    formatted_args.append(str(arg))
            if any("\n" in arg for arg in formatted_args):
                return f"{message}\n{' '.join(formatted_args)}"
            else:
                return f"{message} {' '.join(formatted_args)}"
        except Exception as e:
            return f"{message} [Error formatting arguments: {e}]"

    def debug(self, message: str, *args: Any) -> None:
        """Log a debug message."""
        formatted_message = self.format_log_message(message, *args)
        logging.debug(formatted_message)

    def info(self, message: str, *args: Any) -> None:
        """Log an info message."""
        formatted_message = self.format_log_message(message, *args)
        logging.info(formatted_message)

    def warning(self, message: str, *args: Any) -> None:
        """Log a warning message."""
        formatted_message = self.format_log_message(message, *args)
        logging.warning(formatted_message)

    def warn(self, message: str, *args: Any) -> None:
        """Log a warning message (alias for warning)."""
        self.warning(message, *args)

    def error(self, message: str, *args: Any) -> None:
        """Log an error message."""
        formatted_message = self.format_log_message(message, *args)
        logging.error(formatted_message)

    def critical(self, message: str, *args: Any) -> None:
        """Log a critical message."""
        formatted_message = self.format_log_message(message, *args)
        logging.critical(formatted_message)

    def success(self, message: str, *args: Any) -> None:
        """Log a success message (printed in green with a checkmark)."""
        formatted_message = self.format_log_message(message, *args)
        print(f"\033[32m✓ {formatted_message}\033[0m")

    def start_multiprocessing_logging(self) -> None:
        """Start multiprocessing logging listener thread."""
        if self.is_in_test_environment():
            self.manager = None
            self.mp_queue = None
            self.listener = None
            return
        try:
            ctx = mp.get_context("spawn")
            self.mp_queue = ctx.Queue(maxsize=2000)
            self.manager = None

            self.listener = threading.Thread(target=self._listener_thread, daemon=True)
            self.listener.start()

            time.sleep(0.2)

        except Exception as e:
            print(f"Failed to start multiprocessing logging: {e}", file=sys.stderr)
            self.manager = None
            self.mp_queue = None
            self.listener = None

    def stop_multiprocessing_logging(self) -> None:
        """Stop multiprocessing logging listener thread."""
        if self.is_in_test_environment():
            self.mp_queue = None
            self.manager = None
            self.listener = None
            return

        if self.mp_queue is not None:
            try:
                for _ in range(3):
                    try:
                        self.mp_queue.put(None, timeout=1.0)
                        break
                    except Exception:
                        pass
            except Exception:
                pass

        if self.listener is not None and self.listener.is_alive():
            try:
                self.listener.join(timeout=5.0)
                if self.listener.is_alive():
                    print(
                        "Warning: Log listener thread did not shut down cleanly",
                        file=sys.stderr,
                    )
            except Exception:
                pass

        self.mp_queue = None
        self.manager = None
        self.listener = None

    def _listener_thread(self) -> None:
        """Listener thread for processing log records from multiprocessing queue."""
        consecutive_errors = 0
        max_consecutive_errors = 10

        while True:
            try:
                if self.mp_queue is None:
                    break

                try:
                    record = self.mp_queue.get(timeout=3.0)
                except queue.Empty:
                    consecutive_errors = 0
                    continue
                except (EOFError, OSError, BrokenPipeError):
                    break
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors > max_consecutive_errors:
                        print(f"Too many queue errors: {e}", file=sys.stderr)
                        break
                    continue

                if record is None:
                    break

                consecutive_errors = 0

                try:
                    if hasattr(record, "name") and hasattr(record, "levelno"):
                        logger = logging.getLogger(record.name)
                        logger.handle(record)
                    else:
                        print(f"Invalid log record: {record}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing log record: {e}", file=sys.stderr)
                    continue

            except Exception as e:
                consecutive_errors += 1
                print(f"Error in log listener: {e}", file=sys.stderr)
                if consecutive_errors > max_consecutive_errors:
                    print(
                        "Too many consecutive errors, shutting down listener",
                        file=sys.stderr,
                    )
                    break
                time.sleep(0.1)

    def is_in_test_environment(self) -> bool:
        """Check if the code is running in a test environment (pytest or unittest)."""
        in_pytest = "pytest" in sys.modules and hasattr(
            sys.modules.get("pytest", None), "main"
        )
        in_pytest_env = "PYTEST_CURRENT_TEST" in os.environ

        running_unittest = False
        if "unittest" in sys.modules:
            import unittest

            running_unittest = hasattr(unittest, "_running_tests") and getattr(
                unittest, "_running_tests", False
            )
            if not running_unittest:
                import inspect

                for frame_info in inspect.stack():
                    if "unittest" in frame_info.filename and (
                        "main" in frame_info.function or "run" in frame_info.function
                    ):
                        running_unittest = True
                        break

        return in_pytest or in_pytest_env or running_unittest

    def setup(
        self,
        level: Union[int, str] = logging.INFO,
        file: Optional[Union[str, Path]] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        """Set up logging configuration."""
        self.hidden_log_path = None
        self.user_log_path = None
        if isinstance(level, str):
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "WARN": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            level = level_map.get(level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.DEBUG)
        console_handler = self._create_console_handler(level)
        root_logger.addHandler(console_handler)
        if file:
            file_handler = self._create_file_handler(
                file, level, max_bytes, backup_count
            )
            root_logger.addHandler(file_handler)
            try:
                user_path = Path(file).resolve()
            except Exception:
                user_path = Path(file)
            self.user_log_path = user_path
            os.environ["GT_LOG_FILE"] = str(user_path)
        else:
            os.environ.pop("GT_LOG_FILE", None)
            self.user_log_path = None
        hidden_log_path = self._setup_hidden_debug_log(max_bytes, backup_count)
        if hidden_log_path:
            self.hidden_log_path = hidden_log_path
            os.environ["GT_LAST_LOG"] = str(hidden_log_path)
        else:
            os.environ.pop("GT_LAST_LOG", None)
        self._configure_external_loggers()

    def _create_console_handler(self, level: int) -> logging.StreamHandler:
        """Create console logging handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = self._build_console_formatter()
        handler.setFormatter(formatter)
        return handler

    def _build_console_formatter(self) -> logging.Formatter:
        """Build console log formatter, using color if available."""
        if HAS_COLORLOG:
            return ColoredFormatter(
                "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
                datefmt=None,
                reset=True,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "white",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
                secondary_log_colors={},
                style="%",
            )
        else:
            return logging.Formatter(
                "%(levelname)-8s %(message)s", datefmt=None, style="%"
            )

    def _create_file_handler(
        self,
        file_path: Union[str, Path],
        level: int,
        max_bytes: int,
        backup_count: int,
        colored: bool = False,
    ) -> logging.Handler:
        """Create file logging handler with rotation."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setLevel(level)
        if colored:
            formatter = self._build_console_formatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                style="%",
            )
        handler.setFormatter(formatter)
        return handler

    def _configure_external_loggers(self) -> None:
        """Configure external loggers to suppress noisy logs."""
        for logger_name in [
            "matplotlib",
            "matplotlib.font_manager",
            "PIL",
            "h5py",
            "h5py._conv",
            "numpy",
            "scipy",
        ]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        for logger_name in [
            "numba",
            "numba.core.ssa",
            "numba.core.byteflow",
            "numba.core.interpreter",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        for logger_name in [
            "requests",
            "urllib3",
            "urllib3.connectionpool",
            "requests.packages.urllib3",
            "requests.packages.urllib3.connectionpool",
        ]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        self._suppress_warnings()

    def _suppress_warnings(self) -> None:
        """Suppress specific warnings from external libraries."""
        try:
            import urllib3
            from requests.packages.urllib3.exceptions import InsecureRequestWarning

            urllib3.disable_warnings()
            warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        except ImportError:
            pass
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
        warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

    @staticmethod
    def child_init(queue: mp.Queue) -> None:
        """Initialize logging for child processes using a multiprocessing queue."""
        try:
            if queue is not None:
                handler = logging.handlers.QueueHandler(queue)
                root = logging.getLogger()

                for h in root.handlers[:]:
                    h.close()
                    root.removeHandler(h)

                root.addHandler(handler)
                root.setLevel(logging.DEBUG)
                root.propagate = False

                _configure_worker_loggers()

            else:
                logging.basicConfig(
                    level=logging.INFO,
                    format=f"Worker-{os.getpid()}: %(levelname)s - %(message)s",
                    stream=sys.stdout,
                )

        except Exception as e:
            print(f"Worker logging setup failed: {e}", file=sys.stderr)
            try:
                logging.basicConfig(
                    level=logging.INFO,
                    format=f"Worker-{os.getpid()}: %(levelname)s - %(message)s",
                )
            except Exception:
                pass

    def get_queue(self) -> Optional[mp.Queue]:
        """Get the multiprocessing logging queue."""
        return self.mp_queue

    def _setup_hidden_debug_log(
        self, max_bytes: int, backup_count: int
    ) -> Optional[Path]:
        """Set up hidden debug log file for detailed logging."""
        if self.is_in_test_environment():
            return None
        try:
            hidden_path = self._determine_hidden_log_path()
            self._clear_hidden_log(hidden_path)
            handler = self._create_file_handler(
                hidden_path, logging.DEBUG, max_bytes, backup_count, colored=True
            )
            handler.setLevel(logging.DEBUG)
            logging.getLogger().addHandler(handler)
            return hidden_path
        except Exception:
            return None

    def _determine_hidden_log_path(self) -> Path:
        """Determine the path for the hidden debug log file."""
        target = Path.home() / ".local" / "bin" / "GenomicTools" / "LAST_LOG.log"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            fallback = Path.cwd() / ".GenomicTools"
            fallback.mkdir(parents=True, exist_ok=True)
            target = fallback / "LAST_LOG.log"
        return target

    def _clear_hidden_log(self, path: Path) -> None:
        """Clear the hidden debug log file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            path.write_text("", encoding="utf-8")
        except Exception:
            try:
                with path.open("w", encoding="utf-8") as handle:
                    handle.write("")
            except Exception:
                pass


log = Logger()

if hasattr(sys, "ps1"):
    log.setup(level=logging.WARNING, file=None)
