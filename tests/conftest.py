#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import pytest
import sys
import tempfile

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(TEST_DIR, "../src"))
DATA_DIR = os.path.join(TEST_DIR, "data")
OUTPUT_DIR = os.path.join(TEST_DIR, "output")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")


@pytest.fixture(autouse=True)
def configure_logging_for_tests():
    from utils.LoggingUtils import log

    log.setup(level="DEBUG", file=None)
    yield


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture(scope="session")
def test_dir():
    return TEST_DIR


@pytest.fixture(scope="session")
def src_dir():
    return SRC_DIR


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR


@pytest.fixture(scope="session")
def output_dir():
    return OUTPUT_DIR


def compare_dataframes(df1, df2, precision=3):
    assert set(df1.columns) == set(df2.columns), "DataFrame columns don't match"
    assert len(df1) == len(df2), "DataFrame row counts don't match"
    for col in df1.columns:
        if pd.api.types.is_numeric_dtype(df1[col]):
            assert np.allclose(
                df1[col], df2[col], atol=10**-precision
            ), f"Column {col} values don't match"
        else:
            assert df1[col].equals(df2[col]), f"Column {col} values don't match"
    return True
