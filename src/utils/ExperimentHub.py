#!/usr/bin/env python
# Import required modules
import numpy as np
import os
import pandas as pd
import rdata
import requests
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin
from .LoggingUtils import log


class RDataConverter:
    @staticmethod
    def convert_r_object(obj: Any, name: str = "object") -> Any:
        """Recursively convert R objects to Python equivalents."""
        if obj is None:
            return None

        if isinstance(obj, np.ndarray):
            return obj

        if isinstance(obj, pd.DataFrame):
            return obj

        if isinstance(obj, (int, float, str, bool, list, tuple)):
            return obj

        if isinstance(obj, dict):
            converted_dict: Dict[str, Any] = {}
            for key, value in obj.items():
                if isinstance(key, bytes):
                    key = key.decode("utf-8", errors="ignore")
                converted_dict[str(key)] = RDataConverter.convert_r_object(
                    value, f"{name}${key}"
                )
            return converted_dict

        if hasattr(obj, "__array__"):
            try:
                arr = np.asarray(obj)
                return arr
            except Exception:
                pass

        if hasattr(obj, "columns") and hasattr(obj, "index"):
            try:
                df = pd.DataFrame(obj)
                return df
            except Exception:
                pass

        if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
            try:
                converted_list: List[Any] = []
                for i, item in enumerate(obj):
                    converted_list.append(
                        RDataConverter.convert_r_object(item, f"{name}[{i}]")
                    )
                return converted_list
            except Exception:
                pass

        if hasattr(obj, "__dict__"):
            try:
                attr_dict: Dict[str, Any] = {}
                for attr_name in dir(obj):
                    if not attr_name.startswith("_"):
                        try:
                            attr_value = getattr(obj, attr_name)
                            if not callable(attr_value):
                                attr_dict[attr_name] = RDataConverter.convert_r_object(
                                    attr_value, f"{name}.{attr_name}"
                                )
                        except Exception:
                            pass
                if attr_dict:
                    return attr_dict
            except Exception:
                pass

        return obj

    @staticmethod
    def extract_numeric_data(
        obj: Any, prefer_matrix: bool = True
    ) -> Dict[str, np.ndarray]:
        """Extract numeric arrays from the given object."""
        numeric_data: Dict[str, np.ndarray] = {}

        def extract_recursive(data: Any, prefix: str = "") -> None:
            if isinstance(data, np.ndarray):
                key = prefix or "array"
                numeric_data[key] = data

            elif isinstance(data, pd.DataFrame):
                for col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        key = f"{prefix}.{col}" if prefix else col
                        numeric_data[key] = data[col].values

            elif isinstance(data, dict):
                for key, value in data.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    extract_recursive(value, new_prefix)

            elif isinstance(data, (list, tuple)):
                for i, item in enumerate(data):
                    new_prefix = f"{prefix}[{i}]" if prefix else f"item_{i}"
                    extract_recursive(item, new_prefix)

        extract_recursive(obj)
        return numeric_data


class ExperimentHubOptions:

    def __init__(self) -> None:
        self._options: Dict[str, Any] = self._get_defaults()

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default options from environment variables or use hardcoded defaults."""
        default_cache = Path.home() / ".cache" / "R" / "ExperimentHub"

        env_list = [
            ("URL", "EXPERIMENT_HUB_URL"),
            ("CACHE", "EXPERIMENT_HUB_CACHE"),
            ("PROXY", "EXPERIMENT_HUB_PROXY"),
            ("LOCAL", "EXPERIMENT_HUB_LOCAL"),
            ("ASK", "EXPERIMENT_HUB_ASK"),
            ("FORCE_DOWNLOAD", "EXPERIMENT_HUB_FORCE_DOWNLOAD"),
        ]

        env = {key: os.getenv(name) for key, name in env_list}

        defaults: Dict[str, Any] = {
            "URL": env["URL"] or "https://experimenthub.bioconductor.org",
            "CACHE": env["CACHE"] or str(default_cache),
            "PROXY": env["PROXY"],
            "LOCAL": (env["LOCAL"] or "false").lower() == "true",
            "ASK": (env["ASK"] or "true").lower() == "true",
            "FORCE_DOWNLOAD": (env["FORCE_DOWNLOAD"] or "false").lower() == "true",
        }

        return defaults

    def get(self, key: str) -> Any:
        """Get the value of the specified option."""
        return self._options.get(key.upper())


class HubQuery:
    def __init__(self, hub: "ExperimentHub", data: pd.DataFrame) -> None:
        self.hub: "ExperimentHub" = hub
        self.data: pd.DataFrame = data

    def __len__(self) -> int:
        """Get the number of entries in the query result."""
        return len(self.data)

    def __getitem__(self, key: Union[int, str]) -> Any:
        """Retrieve the resource by index or ah_id."""
        if isinstance(key, int):
            ah_id = self.data.iloc[key]["ah_id"]
            return self.hub[ah_id]
        elif isinstance(key, str):
            return self.hub[key]
        else:
            raise TypeError("Key must be string or int")

    def show(self, n: int = 10) -> pd.DataFrame:
        """Display the first n entries of the query result."""
        cols = ["ah_id", "title", "species", "preparerclass", "rdataclass"]
        return self.data[cols].head(n)

    def load_resource(self, ah_id: str) -> Any:
        """Load the resource with the specified ah_id."""
        return self.hub[ah_id]

    def get_numeric_arrays(self, ah_id: str) -> Dict[str, np.ndarray]:
        """Get numeric arrays from the resource with the specified ah_id."""
        return self.hub.get_numeric_data(ah_id)


class ExperimentHub:

    _HUB_CACHE: Dict[str, "ExperimentHub"] = {}

    def __init__(
        self,
        hub: Optional[str] = None,
        cache: Optional[str] = None,
        proxy: Optional[str] = None,
        local_hub: Optional[bool] = None,
        ask: Optional[bool] = None,
        auto_convert: bool = True,
        return_raw_on_error: bool = True,
    ) -> None:

        self.options: ExperimentHubOptions = ExperimentHubOptions()
        self.hub_url: str = hub or self.options.get("URL")
        self.cache_dir: Path = Path(cache or self.options.get("CACHE"))
        self.proxy: Optional[str] = proxy or self.options.get("PROXY")
        self.local_hub: bool = (
            local_hub if local_hub is not None else self.options.get("LOCAL")
        )
        self.ask: bool = ask if ask is not None else self.options.get("ASK")
        self.force_download: bool = self.options.get("FORCE_DOWNLOAD")
        self.auto_convert: bool = auto_convert
        self.return_raw_on_error: bool = return_raw_on_error
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session: requests.Session = requests.Session()
        if self.proxy:
            self.session.proxies = {"http": self.proxy, "https": self.proxy}

        self.db_file: Path = self.cache_dir / "experimenthub.sqlite3"
        self._hub_data: Optional[pd.DataFrame] = None
        self._setup_database()

    @classmethod
    def get_cached_hub(cls, **kwargs: Any) -> "ExperimentHub":
        """Get a cached ExperimentHub instance based on the provided parameters."""
        cache_key = str(sorted(kwargs.items()))

        if cache_key not in cls._HUB_CACHE:
            cls._HUB_CACHE[cache_key] = cls(**kwargs)
        return cls._HUB_CACHE[cache_key]

    @classmethod
    def load_resource(cls, ah_id: str, **hub_kwargs: Any) -> Any:
        """Load a resource from ExperimentHub by its ah_id."""
        hub = cls.get_cached_hub(**hub_kwargs)
        return hub[ah_id]

    @classmethod
    def get_numeric_arrays(cls, ah_id: str, **hub_kwargs: Any) -> Dict[str, np.ndarray]:
        """Get numeric arrays from a resource in ExperimentHub by its ah_id."""
        hub = cls.get_cached_hub(**hub_kwargs)
        return hub.get_numeric_data(ah_id)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cached ExperimentHub instances."""
        cls._HUB_CACHE.clear()

    def _setup_database(self) -> None:
        """Ensure the local metadata database is available."""
        if self.local_hub and self.db_file.exists():
            return

        if not self.db_file.exists():
            if self.ask:
                response = (
                    input("Download ExperimentHub metadata? (y/n): ").strip().lower()
                )
                if response != "y":
                    raise RuntimeError("Cannot proceed without metadata")

            metadata_url = f"{self.hub_url}/metadata/experimenthub.sqlite3"

            try:
                response = self.session.get(metadata_url, timeout=30)
                response.raise_for_status()

                with open(self.db_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            except Exception as e:
                raise ConnectionError(f"Failed to download metadata: {e}")

    def _load_hub_data(self) -> pd.DataFrame:
        """Load the hub metadata from the local database."""
        if self._hub_data is not None:
            return self._hub_data

        query = """
        SELECT r.ah_id, r.title, r.preparerclass, r.dataprovider,
               r.species, r.genome, r.description, r.rdatadateadded,
               r.rdatadateremoved, l.location_prefix, p.rdatapath,
               p.rdataclass, p.dispatchclass, s.status
        FROM resources r
        LEFT JOIN location_prefixes l ON r.location_prefix_id = l.id
        LEFT JOIN rdatapaths p ON r.id = p.resource_id
        LEFT JOIN statuses s ON r.status_id = s.id
        WHERE r.rdatadateremoved IS NULL
        ORDER BY r.ah_id
        """

        conn = sqlite3.connect(self.db_file)
        log.debug(f"Opened sqlite connection {id(conn)} for hub data")
        try:
            self._hub_data = pd.read_sql_query(query, conn)
        finally:
            conn.close()
            log.debug(f"Closed sqlite connection {id(conn)} for hub data")

        return self._hub_data

    def query(self, pattern: Optional[str] = None, **filters: Any) -> HubQuery:
        """Query the hub metadata with optional pattern and filters."""
        data = self._load_hub_data().copy()

        if pattern:
            mask1 = data["title"].str.contains(pattern, na=False, case=False)
            mask2 = data["description"].str.contains(pattern, na=False, case=False)
            mask3 = data["preparerclass"].str.contains(pattern, na=False, case=False)
            mask4 = data["species"].str.contains(pattern, na=False, case=False)
            mask = mask1 | mask2 | mask3 | mask4
            data = data[mask]

        for field, value in filters.items():
            if field in data.columns:
                mask = (
                    data[field]
                    .astype(str)
                    .str.contains(str(value), na=False, case=False)
                )
                data = data[mask]

        return HubQuery(self, data)

    def __getitem__(self, ah_id: str) -> Any:
        """Retrieve the resource with the specified ah_id."""
        hub_data = self._load_hub_data()
        resource_data = hub_data[hub_data["ah_id"] == ah_id]

        if len(resource_data) == 0:
            raise KeyError(f"Resource {ah_id} not found")

        resource_info = resource_data.iloc[0]

        if pd.notna(resource_info["rdatadateremoved"]):
            raise ValueError(
                f"Resource {ah_id} was removed on {resource_info['rdatadateremoved']}"
            )

        return self._download_and_load(resource_info)

    def _download_and_load(self, resource_info: pd.Series) -> Any:
        """Download the resource if not cached and load it."""
        ah_id: str = resource_info["ah_id"]

        rdatapath: str = resource_info.get("rdatapath", f"{ah_id}.rds")
        file_extension: str = os.path.splitext(rdatapath)[1] or ".rds"
        cache_file: Path = self.cache_dir / f"{ah_id}{file_extension}"

        if not cache_file.exists():
            log.debug(
                f"Local hub: {self.local_hub}, resource {ah_id} not found in cache — attempting download..."
            )
            base_url: str = resource_info.get("location_prefix") or self.hub_url
            if pd.notna(resource_info.get("rdatapath")):
                download_url: str = urljoin(base_url, resource_info["rdatapath"])
            else:
                preparerclass: str = resource_info.get("preparerclass", "unknown")
                download_url: str = urljoin(base_url, f"{preparerclass}/{ah_id}.rds")

            try:
                response = self.session.get(download_url, timeout=60)
                response.raise_for_status()

                with open(cache_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                raise ConnectionError(f"Failed to download {ah_id}: {e}")
        else:
            log.info(f"Using cached file for {ah_id}")

        return self._load_and_process_resource(cache_file, resource_info)

    def _load_and_process_resource(
        self, cache_file: Path, resource_info: pd.Series
    ) -> Any:
        """Load and process the resource from the cached file."""
        ah_id: str = resource_info["ah_id"]
        file_extension: str = cache_file.suffix.lower()

        if file_extension in [".rdata", ".rda"]:
            try:
                parsed = rdata.parser.parse_file(cache_file)
                converted = rdata.conversion.convert(parsed)

                if self.auto_convert:
                    processed_data: Dict[str, Any] = {}
                    for key, value in converted.items():
                        processed_data[key] = RDataConverter.convert_r_object(
                            value, f"{ah_id}.{key}"
                        )

                    return {
                        "data": processed_data,
                        "raw_data": converted,
                        "source": "rdata",
                        "file_path": str(cache_file),
                        "resource_info": resource_info.to_dict(),
                    }
                else:
                    return converted

            except Exception as e:
                log.error(f"⚠️  Failed to load RData: {e}")
                if self.return_raw_on_error:
                    return {
                        "file_path": str(cache_file),
                        "resource_info": resource_info.to_dict(),
                        "error": str(e),
                    }
                else:
                    raise
        else:
            log.error(f"⚠️  Unsupported file type: {file_extension}")
            return {
                "file_path": str(cache_file),
                "resource_info": resource_info.to_dict(),
                "suggested_packages": self._get_suggested_packages(file_extension),
            }

    def _get_suggested_packages(self, file_extension: str) -> List[str]:
        """Get suggested packages for unsupported file types."""
        suggestions: Dict[str, List[str]] = {
            ".rdata": ["rdata"],
            ".rda": ["rdata"],
            ".csv": ["pandas"],
            ".tsv": ["pandas"],
            ".txt": ["pandas", "numpy"],
            ".h5": ["h5py", "tables"],
            ".hdf5": ["h5py", "tables"],
        }
        return suggestions.get(file_extension, [])

    def get_numeric_data(self, ah_id: str) -> Dict[str, np.ndarray]:
        """Get numeric arrays from the resource with the specified ah_id."""
        resource = self[ah_id]

        if isinstance(resource, dict) and "data" in resource:
            return RDataConverter.extract_numeric_data(resource["data"])
        else:
            return RDataConverter.extract_numeric_data(resource)

    def get_resource_info(self, ah_id: str) -> Dict[str, Any]:
        """Get metadata information for the resource with the specified ah_id."""
        hub_data = self._load_hub_data()
        resource_data = hub_data[hub_data["ah_id"] == ah_id]

        if len(resource_data) == 0:
            raise KeyError(f"Resource {ah_id} not found")

        return resource_data.iloc[0].to_dict()

    def list_cached_resources(self) -> List[str]:
        """List all cached resource ah_ids."""
        if not self.cache_dir.exists():
            return []

        cached_files: List[str] = []
        for file_path in self.cache_dir.glob("EH*"):
            if file_path.is_file():
                cached_files.append(file_path.stem)

        return sorted(cached_files)

    def clear_cached_resource(self, ah_id: str) -> bool:
        """Remove the cached file for the specified ah_id."""
        pattern = f"{ah_id}.*"
        removed: bool = False

        for file_path in self.cache_dir.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                removed = True

        return removed

    def package(
        self, resources: Optional[Union[str, List[str]]] = None
    ) -> Union[str, List[str]]:
        """Get the preparerclass (package name) for the specified resources."""
        hub_data = self._load_hub_data()

        if resources is None:
            return hub_data["preparerclass"].unique().tolist()

        if isinstance(resources, str):
            resources = [resources]

        result: List[Optional[str]] = []
        for ah_id in resources:
            matching = hub_data[hub_data["ah_id"] == ah_id]
            if len(matching) > 0:
                result.append(matching.iloc[0]["preparerclass"])
            else:
                result.append(None)

        return result[0] if len(result) == 1 else result

    def get_package_resources(self, package_name: str) -> HubQuery:
        """Get all resources for the specified package (preparerclass)."""
        return self.query(preparerclass=package_name)

    def __len__(self) -> int:
        """Get the total number of resources in the hub."""
        return len(self._load_hub_data())

    def __repr__(self) -> str:
        """Return a string representation of the ExperimentHub instance."""
        return f"ExperimentHub(resources={len(self)}, cache='{self.cache_dir}')"
