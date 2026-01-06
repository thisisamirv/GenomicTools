#!/usr/bin/env python
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import gc

from utils.ExperimentHub import (
    ExperimentHub,
    ExperimentHubOptions,
    HubQuery,
    RDataConverter,
)


def _create_test_database(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE resources (
            id INTEGER PRIMARY KEY,
            ah_id TEXT,
            title TEXT,
            preparerclass TEXT,
            dataprovider TEXT,
            species TEXT,
            genome TEXT,
            description TEXT,
            rdatadateadded TEXT,
            rdatadateremoved TEXT,
            status_id INTEGER,
            location_prefix_id INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE location_prefixes (
            id INTEGER PRIMARY KEY,
            location_prefix TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE rdatapaths (
            resource_id INTEGER,
            rdatapath TEXT,
            rdataclass TEXT,
            dispatchclass TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE statuses (
            id INTEGER PRIMARY KEY,
            status TEXT
        )
        """
    )

    cur.execute(
        """
        INSERT INTO location_prefixes (id, location_prefix)
        VALUES (1, 'https://hub.example.org/')
        """
    )
    cur.execute("INSERT INTO statuses (id, status) VALUES (1, 'public')")
    cur.execute(
        """
        INSERT INTO resources (
            id, ah_id, title, preparerclass, dataprovider, species,
            genome, description, rdatadateadded, rdatadateremoved,
            status_id, location_prefix_id
        ) VALUES (
            1, 'EH123', 'Test Resource', 'TestPackage', 'Provider',
            'Homo sapiens', 'hg38', 'Test description',
            '2024-01-01', NULL, 1, 1
        )
        """
    )
    cur.execute(
        """
        INSERT INTO rdatapaths (resource_id, rdatapath, rdataclass, dispatchclass)
        VALUES (1, 'data/EH123.rdata', 'matrix', 'Rda')
        """
    )

    conn.commit()
    return conn


@pytest.fixture
def hub_with_db(tmp_path):
    db_path = tmp_path / "experimenthub.sqlite3"
    conn = _create_test_database(db_path)
    ExperimentHub.clear_cache()
    hub = ExperimentHub(cache=str(tmp_path), local_hub=True, ask=False)
    yield hub
    hub.session.close()
    hub._hub_data = None
    conn.close()
    gc.collect()
    ExperimentHub.clear_cache()


def test_experiment_hub_options_respect_environment(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_HUB_CACHE", "/tmp/custom_cache")
    monkeypatch.setenv("EXPERIMENT_HUB_URL", "https://custom.hub")
    options = ExperimentHubOptions()
    assert options.get("CACHE") == "/tmp/custom_cache"
    assert options.get("URL") == "https://custom.hub"


def test_setup_database_prompt_denied(tmp_path):
    cache_dir = tmp_path / "cache"
    with patch("builtins.input", return_value="n"), patch(
        "utils.ExperimentHub.requests.Session.get"
    ) as mock_get:
        with pytest.raises(RuntimeError, match="Cannot proceed without metadata"):
            ExperimentHub(cache=str(cache_dir), ask=True, hub="https://example.org")
    mock_get.assert_not_called()


def test_setup_database_downloads_metadata(tmp_path):
    cache_dir = tmp_path / "cache"
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content.return_value = [b"sqlite-data"]

    with patch(
        "utils.ExperimentHub.requests.Session.get", return_value=mock_response
    ) as mock_get:
        hub = ExperimentHub(cache=str(cache_dir), ask=False, hub="https://example.org")

    mock_get.assert_called_once()
    assert hub.db_file.exists()
    assert hub.db_file.read_bytes() == b"sqlite-data"
    hub.session.close()
    ExperimentHub.clear_cache()


def test_rdata_converter_handles_nested_structures():
    nested = {
        b"name": [1, 2, 3],
        "frame": pd.DataFrame({"a": [1, 2]}),
        "bytes_list": [b"value1", b"value2"],
    }
    converted = RDataConverter.convert_r_object(nested, "root")

    assert isinstance(converted, dict)
    assert "name" in converted
    assert converted["name"] == [1, 2, 3]
    assert isinstance(converted["frame"], pd.DataFrame)


def test_rdata_converter_extract_numeric_data():
    data = {
        "matrix": np.array([[1, 2], [3, 4]]),
        "frame": pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        "nested": {"values": [np.array([5, 6]), "ignore"]},
    }
    extracted = RDataConverter.extract_numeric_data(data)

    assert "matrix" in extracted
    assert "frame.a" in extracted
    assert "nested.values[0]" in extracted
    np.testing.assert_array_equal(extracted["matrix"], np.array([[1, 2], [3, 4]]))


def test_rdata_converter_special_object_paths():

    class ArrayLike:
        def __array__(self):
            return np.array([7, 8, 9])

    class Columnar:
        def __init__(self):
            self.columns = ["foo", "bar"]
            self.index = [0, 1]

        def __iter__(self):
            return iter([{"foo": 1, "bar": 2}, {"foo": 3, "bar": 4}])

    class IterableOnly:
        def __iter__(self):
            yield ArrayLike()
            yield "skip"

    class AttributeObject:
        def __init__(self):
            self.value = 42
            self.array_like = ArrayLike()

        def method(self):
            return "ignored"

    array_result = RDataConverter.convert_r_object(ArrayLike(), "array")
    np.testing.assert_array_equal(array_result, np.array([7, 8, 9]))

    column_result = RDataConverter.convert_r_object(Columnar(), "columnar")
    assert isinstance(column_result, pd.DataFrame)
    assert list(column_result.columns) == ["foo", "bar"]

    iterable_result = RDataConverter.convert_r_object(IterableOnly(), "iterable")
    assert isinstance(iterable_result, list)
    np.testing.assert_array_equal(iterable_result[0], np.array([7, 8, 9]))
    assert iterable_result[1] == "skip"

    attr_result = RDataConverter.convert_r_object(AttributeObject(), "attrs")
    assert attr_result["value"] == 42
    np.testing.assert_array_equal(attr_result["array_like"], np.array([7, 8, 9]))


def test_hub_query_indexing_and_show():
    data = pd.DataFrame(
        [
            {
                "ah_id": "EH123",
                "title": "Demo",
                "species": "human",
                "preparerclass": "Pkg",
                "rdataclass": "matrix",
            }
        ]
    )
    mock_hub = MagicMock()
    mock_hub.__getitem__.side_effect = lambda ah_id: f"resource_{ah_id}"
    query = HubQuery(mock_hub, data)

    assert len(query) == 1
    assert query[0] == "resource_EH123"
    assert query["EH123"] == "resource_EH123"

    preview = query.show()
    assert isinstance(preview, pd.DataFrame)
    assert "ah_id" in preview.columns


def test_experiment_hub_query_returns_hubquery(hub_with_db):
    result = hub_with_db.query()
    assert isinstance(result, HubQuery)
    assert len(result) == 1


def test_experiment_hub_getitem_uses_loader(hub_with_db):
    mock_result = {"data": {"values": np.array([1, 2, 3])}}
    with patch.object(
        hub_with_db, "_download_and_load", return_value=mock_result
    ) as loader:
        resource = hub_with_db["EH123"]
        assert resource == mock_result
        loader.assert_called_once()


def test_experiment_hub_get_numeric_data(hub_with_db):
    mock_result = {"data": {"values": np.array([10, 20, 30])}}
    with patch.object(hub_with_db, "_download_and_load", return_value=mock_result):
        numeric = hub_with_db.get_numeric_data("EH123")
        assert "values" in numeric
        np.testing.assert_array_equal(numeric["values"], np.array([10, 20, 30]))


def test_experiment_hub_removed_resource(tmp_path):
    db_path = tmp_path / "experimenthub.sqlite3"
    conn = _create_test_database(db_path)
    conn.execute(
        "UPDATE resources SET rdatadateremoved='2024-03-01' WHERE ah_id='EH123'"
    )
    conn.commit()
    conn.close()

    ExperimentHub.clear_cache()
    hub = ExperimentHub(cache=str(tmp_path), local_hub=True, ask=False)
    with pytest.raises(KeyError):
        _ = hub["EH123"]
    hub.session.close()
    hub._hub_data = None
    ExperimentHub.clear_cache()


def test_experiment_hub_local_hub_missing_file(tmp_path):
    db_path = tmp_path / "experimenthub.sqlite3"
    conn = _create_test_database(db_path)
    conn.close()

    ExperimentHub.clear_cache()
    hub = ExperimentHub(cache=str(tmp_path), local_hub=True, ask=False)
    with pytest.raises(FileNotFoundError):
        _ = hub["EH123"]
    hub.session.close()
    hub._hub_data = None
    ExperimentHub.clear_cache()


def test_experiment_hub_load_rdata_success(tmp_path):
    db_path = tmp_path / "experimenthub.sqlite3"
    conn = _create_test_database(db_path)
    conn.close()

    cache_file = tmp_path / "EH123.rdata"
    cache_file.write_bytes(b"mock rdata content")

    ExperimentHub.clear_cache()
    hub = ExperimentHub(cache=str(tmp_path), local_hub=True, ask=False)
    with patch(
        "utils.ExperimentHub.rdata.parser.parse_file", return_value={"object": "raw"}
    ), patch(
        "utils.ExperimentHub.rdata.conversion.convert",
        return_value={"matrix": np.array([[1, 2], [3, 4]])},
    ):
        resource = hub["EH123"]

    assert resource["source"] == "rdata"
    assert "data" in resource and "matrix" in resource["data"]
    np.testing.assert_array_equal(
        resource["data"]["matrix"], np.array([[1, 2], [3, 4]])
    )
    assert "raw_data" in resource and "matrix" in resource["raw_data"]
    np.testing.assert_array_equal(
        resource["raw_data"]["matrix"], np.array([[1, 2], [3, 4]])
    )

    hub.session.close()
    hub._hub_data = None
    ExperimentHub.clear_cache()


def test_experiment_hub_load_rdata_failure_returns_error(tmp_path):
    db_path = tmp_path / "experimenthub.sqlite3"
    conn = _create_test_database(db_path)
    conn.close()

    cache_file = tmp_path / "EH123.rdata"
    cache_file.write_bytes(b"bad content")

    ExperimentHub.clear_cache()
    hub = ExperimentHub(cache=str(tmp_path), local_hub=True, ask=False)
    with patch(
        "utils.ExperimentHub.rdata.parser.parse_file",
        side_effect=RuntimeError("parse error"),
    ):
        resource = hub["EH123"]

    assert "error" in resource
    assert "parse error" in resource["error"]

    hub.session.close()
    hub._hub_data = None
    ExperimentHub.clear_cache()


def test_experiment_hub_helpers(tmp_path):
    db_path = tmp_path / "experimenthub.sqlite3"
    conn = _create_test_database(db_path)
    conn.close()

    cache_dir = Path(tmp_path)
    (cache_dir / "EH123.rdata").write_bytes(b"cached")
    (cache_dir / "EH124.rdata").write_bytes(b"other")

    ExperimentHub.clear_cache()
    hub = ExperimentHub(cache=str(tmp_path), local_hub=True, ask=False)

    info = hub.get_resource_info("EH123")
    assert info["ah_id"] == "EH123"
    assert info["preparerclass"] == "TestPackage"

    cached = hub.list_cached_resources()
    assert "EH123" in cached

    assert hub.clear_cached_resource("EH123") is True
    assert "EH123" not in hub.list_cached_resources()

    all_packages = hub.package()
    assert "TestPackage" in all_packages
    assert hub.package("EH123") == "TestPackage"

    package_query = hub.get_package_resources("TestPackage")
    assert isinstance(package_query, HubQuery)
    assert len(hub) == 1
    assert "ExperimentHub" in repr(hub)

    hub.session.close()
    hub._hub_data = None
    ExperimentHub.clear_cache()


def test_experiment_hub_unsupported_file_type(tmp_path):
    db_path = tmp_path / "experimenthub.sqlite3"
    conn = _create_test_database(db_path)
    conn.execute("UPDATE rdatapaths SET rdatapath='data/EH123.txt' WHERE resource_id=1")
    conn.commit()
    conn.close()

    cache_file = tmp_path / "EH123.txt"
    cache_file.write_text("some data")

    ExperimentHub.clear_cache()
    hub = ExperimentHub(cache=str(tmp_path), local_hub=True, ask=False)
    result = hub["EH123"]
    assert "suggested_packages" in result
    assert "pandas" in result["suggested_packages"]

    hub.session.close()
    hub._hub_data = None
    ExperimentHub.clear_cache()


def test_classmethod_cached_hub(tmp_path):
    db_path = tmp_path / "experimenthub.sqlite3"
    conn = _create_test_database(db_path)
    ExperimentHub.clear_cache()

    hub1 = ExperimentHub.get_cached_hub(cache=str(tmp_path), local_hub=True, ask=False)
    hub2 = ExperimentHub.get_cached_hub(cache=str(tmp_path), local_hub=True, ask=False)

    assert hub1 is hub2
    hub1.session.close()
    hub1._hub_data = None
    conn.close()
    gc.collect()
    ExperimentHub.clear_cache()
