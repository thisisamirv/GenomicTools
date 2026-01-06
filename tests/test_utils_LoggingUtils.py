#!/usr/bin/env python
import io
import logging
import os
import queue
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from utils.LoggingUtils import Logger


@pytest.fixture(autouse=True)
def restore_env():
    saved = {key: os.environ.get(key) for key in ("GT_LOG_FILE", "GT_LAST_LOG")}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def clean_root_logger():
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)
    for handler in original_handlers:
        root.addHandler(handler)
    root.setLevel(original_level)


@pytest.fixture
def fresh_logger(clean_root_logger):
    original_instance = Logger._instance
    original_initialized = Logger._initialized
    Logger._instance = None
    Logger._initialized = False
    try:
        logger = Logger()
        yield logger
    finally:
        Logger._instance = original_instance
        Logger._initialized = original_initialized


def test_logger_singleton(fresh_logger):
    assert Logger() is fresh_logger


def test_format_log_message_basic(fresh_logger):
    assert fresh_logger.format_log_message("Hello") == "Hello"
    assert fresh_logger.format_log_message("Count:", 3) == "Count: 3"


def test_format_log_message_dataframe(fresh_logger):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    message = fresh_logger.format_log_message("Data:", df)
    assert message.startswith("Data:\n")
    assert "A" in message and "B" in message


def test_format_log_message_long_sequence(fresh_logger):
    data = list(range(20))
    message = fresh_logger.format_log_message("Seq:", data)
    assert "Seq:" in message
    assert "list(20 items)" in message
    assert "[0, 1, 2, 3, 4]" in message
    assert "[15, 16, 17, 18, 19]" in message


def test_format_log_message_handles_errors(fresh_logger):
    class BadObject:
        def __str__(self):
            raise ValueError("bad")

    message = fresh_logger.format_log_message("Object:", BadObject())
    assert "Error formatting arguments" in message


def test_format_log_message_special_objects(fresh_logger):
    class ArrayLike:
        def __array__(self):
            return np.array([1, 2, 3])

    class Columnar:
        def __init__(self):
            self.columns = ["x", "y"]
            self.index = [0, 1]

        def __iter__(self):
            return iter([{"x": 1, "y": 2}, {"x": 3, "y": 4}])

    class IterableOnly:
        def __iter__(self):
            yield ArrayLike()
            yield "text"

    class AttributeObject:
        def __init__(self):
            self.value = 42
            self.array_like = ArrayLike()

        def method(self):
            return "ignore"

    array_message = fresh_logger.format_log_message("ArrayLike:", ArrayLike())
    assert array_message.startswith("ArrayLike:")
    assert isinstance(array_message, str)

    column_message = fresh_logger.format_log_message("Columnar:", Columnar())
    assert column_message.startswith("Columnar:")
    assert isinstance(column_message, str)

    iterable_message = fresh_logger.format_log_message("Iter:", IterableOnly())
    assert iterable_message.startswith("Iter:")

    attr_message = fresh_logger.format_log_message("Attrs:", AttributeObject())
    assert attr_message.startswith("Attrs:")
    assert isinstance(attr_message, str)


def test_logging_methods_delegate_to_standard_library(fresh_logger):
    with patch("logging.info") as mock_info, patch(
        "logging.warning"
    ) as mock_warning, patch("logging.error") as mock_error, patch(
        "logging.debug"
    ) as mock_debug, patch(
        "logging.critical"
    ) as mock_critical:
        fresh_logger.info("info", 1)
        fresh_logger.warning("warn")
        fresh_logger.error("error")
        fresh_logger.debug("debug")
        fresh_logger.critical("critical")

        mock_info.assert_called_once()
        mock_warning.assert_called_once()
        mock_error.assert_called_once()
        mock_debug.assert_called_once()
        mock_critical.assert_called_once()


def test_success_prints_colored_output(fresh_logger):
    with patch("sys.stdout", new_callable=io.StringIO) as stdout:
        fresh_logger.success("done", 123)
        output = stdout.getvalue()
        assert "✓ done 123" in output
        assert output.startswith("\033[32m")
        assert output.rstrip().endswith("\033[0m")


def test_setup_console_only(fresh_logger, clean_root_logger):
    fresh_logger.setup(level="INFO", file=None)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
    assert handlers and handlers[0].level == logging.INFO
    assert "GT_LOG_FILE" not in os.environ


def test_setup_with_file_and_hidden_log(tmp_path, fresh_logger):
    logger = fresh_logger
    log_file = tmp_path / "app.log"
    hidden_file = tmp_path / "hidden.log"

    with patch.object(
        logger, "is_in_test_environment", return_value=False
    ), patch.object(logger, "_determine_hidden_log_path", return_value=hidden_file):
        logger.setup(level="DEBUG", file=log_file)
        logger.info("message")

    for handler in logging.getLogger().handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    assert log_file.exists()
    assert "message" in log_file.read_text()
    assert os.environ["GT_LOG_FILE"] == str(log_file.resolve())
    assert os.environ["GT_LAST_LOG"] == str(hidden_file)
    assert hidden_file.exists()


def test_configure_external_loggers(fresh_logger):
    target_levels = [
        ("matplotlib", logging.ERROR),
        ("h5py", logging.ERROR),
        ("numba", logging.WARNING),
        ("requests", logging.WARNING),
    ]
    originals = {name: logging.getLogger(name).level for name, _ in target_levels}
    try:
        for name, _ in target_levels:
            logging.getLogger(name).setLevel(logging.NOTSET)

        fresh_logger._configure_external_loggers()

        for name, level in target_levels:
            assert logging.getLogger(name).level == level
    finally:
        for name, level in originals.items():
            logging.getLogger(name).setLevel(level)


def test_start_multiprocessing_logging_in_test_env(fresh_logger):
    fresh_logger.start_multiprocessing_logging()
    assert fresh_logger.mp_queue is None
    assert fresh_logger.listener is None


def test_start_stop_multiprocessing_logging_spawn_path(fresh_logger):
    class DummyContext:
        def Queue(self, maxsize):
            return queue.Queue(maxsize)

    class DummyThread:
        def __init__(self, *args, **kwargs):
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

        def join(self, timeout=None):
            self.started = False

    with patch.object(
        fresh_logger, "is_in_test_environment", return_value=False
    ), patch("multiprocessing.get_context", return_value=DummyContext()), patch(
        "threading.Thread",
        side_effect=lambda target, daemon: DummyThread(target, daemon),
    ), patch(
        "time.sleep", return_value=None
    ):
        fresh_logger.start_multiprocessing_logging()
        assert isinstance(fresh_logger.mp_queue, queue.Queue)
        assert fresh_logger.listener is not None

        fresh_logger.stop_multiprocessing_logging()
        assert fresh_logger.mp_queue is None
        assert fresh_logger.listener is None


def test_child_init_with_queue():
    mock_root = MagicMock()
    mock_root.handlers = [MagicMock()]

    queue_handler_instance = MagicMock()

    def get_logger(name=None):
        return mock_root if not name else MagicMock()

    with patch(
        "logging.handlers.QueueHandler", return_value=queue_handler_instance
    ), patch("logging.getLogger", side_effect=get_logger):
        log_queue = queue.Queue()
        Logger.child_init(log_queue)

    mock_root.removeHandler.assert_called()
    mock_root.addHandler.assert_called_with(queue_handler_instance)
    mock_root.setLevel.assert_called_with(logging.DEBUG)
    assert mock_root.propagate is False


def test_child_init_without_queue():
    with patch("logging.basicConfig") as mock_basic_config:
        Logger.child_init(None)
    mock_basic_config.assert_called()


def test_suppress_warnings(fresh_logger):
    logger = fresh_logger
    with patch("warnings.filterwarnings") as mock_filter, patch.dict(
        sys.modules,
        {
            "urllib3": MagicMock(),
            "requests": MagicMock(),
            "requests.packages.urllib3.exceptions": MagicMock(),
        },
    ):
        logger._suppress_warnings()
    assert mock_filter.called


def test_listener_thread_processing(fresh_logger):
    class DummyQueue:
        def __init__(self, values):
            self.values = list(values)
            self.empty_raised = False

        def get(self, timeout=None):
            if not self.empty_raised:
                self.empty_raised = True
                raise queue.Empty()
            if not self.values:
                raise AssertionError("Queue get called too many times")
            return self.values.pop(0)

    valid_record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )
    invalid_record = object()
    sentinel = None

    dummy_queue = DummyQueue([valid_record, invalid_record, sentinel])
    fresh_logger.mp_queue = dummy_queue

    mock_logger = MagicMock()
    with patch("logging.getLogger", return_value=mock_logger), patch(
        "sys.stderr", new_callable=io.StringIO
    ) as stderr:
        fresh_logger._listener_thread()

    mock_logger.handle.assert_called_once_with(valid_record)
    assert "Invalid log record" in stderr.getvalue()


def test_is_in_test_environment_detection(fresh_logger):
    logger = fresh_logger
    assert logger.is_in_test_environment() is True

    bare_logger = object.__new__(Logger)
    with patch.dict(os.environ, {}, clear=True), patch.dict(
        sys.modules, {}, clear=True
    ):
        assert bare_logger.is_in_test_environment() is False


def test_format_log_message_edge_cases(fresh_logger):
    long_text = "x" * 200
    message = fresh_logger.format_log_message("Long:", long_text)
    assert long_text in message

    unicode_text = "🚀 Привет"
    assert fresh_logger.format_log_message(unicode_text) == unicode_text

    assert fresh_logger.format_log_message("Empty list:", []) == "Empty list: []"
    assert fresh_logger.format_log_message("Empty dict:", {}) == "Empty dict: {}"
