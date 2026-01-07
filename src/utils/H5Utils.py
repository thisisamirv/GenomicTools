#!/usr/bin/env python
# Import required modules
import h5py
import numpy as np
import os
import pandas as pd
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Any, List, Iterable, Iterator, Callable, Dict, Union, Tuple
from .AliasUtils import AliasUtils
from .LoggingUtils import log


@dataclass
class H5Config:
    cache_enabled: bool = True
    strict_validation: bool = False
    default_sample_prefix: str = "sample_"
    default_marker_prefix: str = "marker_"
    max_cache_size: int = 1000
    chunk_size: int = 10000


class DataValidator:

    @staticmethod
    def validate_data_shape(data: Any, expected_dims: int = 2) -> None:
        """Validate that data has the expected number of dimensions and positive sizes."""
        if not hasattr(data, "shape"):
            raise ValueError("Data must have shape attribute")

        if len(data.shape) != expected_dims:
            raise ValueError(f"Expected {expected_dims}D data, got {len(data.shape)}D")

        if any(dim <= 0 for dim in data.shape):
            raise ValueError("Data dimensions must be positive")

    @staticmethod
    def validate_sample_data_alignment(
        data_shape: Any, sample_names: Iterable[Any]
    ) -> bool:
        """Check if the data shape aligns with the number of sample names."""
        if len(data_shape) < 2:
            return False

        return data_shape[1] == len(list(sample_names)) or data_shape[0] == len(
            list(sample_names)
        )


class ChromosomeMapper:
    def __init__(self, h5_file: Any) -> None:
        self.h5_file = h5_file
        self._chromosome_cache: Dict[str, Optional[str]] = {}

    def map_chromosome_name(self, chromosome: str) -> Optional[str]:
        """Map a requested chromosome name to the actual name in the HDF5 file."""
        if chromosome in self._chromosome_cache:
            return self._chromosome_cache[chromosome]

        mapped = self._do_mapping(chromosome)
        self._chromosome_cache[chromosome] = mapped
        return mapped

    def _do_mapping(self, chromosome: str) -> Optional[str]:
        """Perform the actual mapping logic for chromosome names."""
        try:
            if chromosome in self.h5_file:
                return chromosome

            for h5_chr in self.h5_file.keys():
                if h5_chr.lower() == chromosome.lower():
                    return h5_chr

            if chromosome.lower().startswith("chr"):
                chr_num = chromosome[3:]
                for h5_chr in self.h5_file.keys():
                    if h5_chr.upper() == f"CHR{chr_num}":
                        return h5_chr

            requested_base = AliasUtils.strip_numeric_suffix(chromosome)
            requested_suffix = chromosome.replace(requested_base, "")

            for h5_chr in self.h5_file.keys():
                if AliasUtils.find_keys({h5_chr: h5_chr}, "Metadata"):
                    continue

                h5_base = AliasUtils.strip_numeric_suffix(h5_chr)
                h5_suffix = h5_chr.replace(h5_base, "")

                condition1 = h5_base.lower() == requested_base.lower()
                condition2 = h5_suffix.upper() == requested_suffix.upper()
                if condition1 and condition2:
                    return h5_chr

            return None
        except Exception as e:
            log.error(f"Error mapping chromosome name {chromosome}: {e}")
            return None


class DataTypeDetector:

    @classmethod
    def detect_data_type(cls, chrom_group: Any) -> Optional[str]:
        """Detect whether the data in the chromosome group is genotype or methylation."""
        if AliasUtils.find_keys(chrom_group, "Genotype"):
            log.debug("Found Genotype data using AliasUtils")
            return "Genotype"

        if AliasUtils.find_keys(chrom_group, "RSID"):
            log.debug("Found RSID, indicating Genotype data")
            return "Genotype"

        if AliasUtils.find_keys(chrom_group, "Methylation"):
            log.debug("Found Methylation data using AliasUtils")
            return "Methylation"

        if AliasUtils.find_keys(chrom_group, "ProbeList"):
            log.debug("Found ProbeList, indicating Methylation data")
            return "Methylation"

        genotype_indicators = ["A1", "A2", "INFO", "MAF", "HWE"]
        genotype_score = sum(
            1
            for indicator in genotype_indicators
            if AliasUtils.find_keys(chrom_group, indicator)
        )

        methylation_indicators = ["CGID"]
        methylation_score = sum(
            1
            for indicator in methylation_indicators
            if AliasUtils.find_keys(chrom_group, indicator)
        )

        if genotype_score > methylation_score and genotype_score > 0:
            log.debug(
                f"Detected genotype data based on indicators (score: {genotype_score})"
            )
            return "Genotype"
        elif methylation_score > 0:
            log.debug(
                f"Detected methylation data based on indicators (score: {methylation_score})"
            )
            return "Methylation"

        log.error("Cannot determine data type from chromosome group")
        return None


class ChunkedDataReader:

    def __init__(self, chunk_size: int = 10000) -> None:
        self.chunk_size = chunk_size

    def read_in_chunks(
        self, dataset: Any, indices: Optional[List[int]] = None
    ) -> Iterator[Any]:
        """Generator to read dataset in chunks based on provided indices."""
        if indices is None:
            indices = list(range(dataset.shape[0]))

        for i in range(0, len(indices), self.chunk_size):
            chunk_plus_i = i + self.chunk_size
            chunk_indices = indices[i:chunk_plus_i]
            yield dataset[chunk_indices]


class BaseH5Utils:
    def __init__(self, config: Optional[H5Config] = None) -> None:
        self.config = config or H5Config()

    @staticmethod
    def _decode_if_bytes(data: Any) -> str:
        """Decode bytes to UTF-8 string if necessary, stripping null bytes."""
        if isinstance(data, bytes):
            try:
                return data.decode("utf-8").rstrip('\x00').strip()
            except UnicodeDecodeError:
                try:
                    return data.decode("latin-1").rstrip('\x00').strip()
                except UnicodeDecodeError:
                    return str(data).rstrip('\x00').strip()
        return str(data).strip()

    @staticmethod
    def _decode_array(array: Iterable[Any]) -> List[str]:
        """Decode an array of bytes to strings if necessary, stripping null bytes."""
        arr_list = list(array)
        if not arr_list:
            return []
        first = arr_list[0]
        # Detect type once, apply to all
        if isinstance(first, bytes):
            # Try utf-8 first for all, fallback per-element only on failure
            try:
                return [x.decode("utf-8").rstrip('\x00').strip() for x in arr_list]
            except UnicodeDecodeError:
                return [BaseH5Utils._decode_if_bytes(x) for x in arr_list]
        elif isinstance(first, str):
            return [x.rstrip('\x00').strip() for x in arr_list]
        return [str(x).strip() for x in arr_list]

    @staticmethod
    def _normalize_sample_id_list(sample_ids: Iterable[Any]) -> List[str]:
        """Normalize sample IDs to strings, converting numeric IDs appropriately."""
        id_list = list(sample_ids)
        if not id_list:
            return []
        
        first = id_list[0]
        # Fast path: if first element is already a clean string, likely all are
        if isinstance(first, str) and not first.replace('.', '').replace('-', '').isdigit():
            return id_list
        
        # Otherwise, normalize each
        normalized: List[str] = []
        for sid in id_list:
            sid_str = str(sid)
            try:
                float_val = float(sid_str)
                if float_val.is_integer():
                    normalized.append(str(int(float_val)))
                else:
                    normalized.append(sid_str)
            except (ValueError, TypeError):
                normalized.append(sid_str)
        return normalized

    @staticmethod
    def _convert_sample_ids(raw_sample_ids: Iterable[Any]) -> List[str]:
        """Convert raw sample IDs from HDF5 to normalized string format."""
        raw_list = list(raw_sample_ids)
        if len(raw_list) == 0:
            return []

        first_element = raw_list[0]
        if isinstance(first_element, (bytes, str)):
            decoded = BaseH5Utils._decode_array(raw_list)
        else:
            decoded = [str(int(sid)) for sid in raw_list]

        normalized: List[str] = []
        for item in decoded:
            try:
                float_val = float(item)
                if float_val.is_integer():
                    normalized.append(str(int(float_val)))
                else:
                    normalized.append(item)
            except (ValueError, TypeError):
                normalized.append(item)

        return normalized

    @staticmethod
    def _get_sample_path(h5_file: Any, data_type: Optional[str] = None) -> str:
        """Determine the path to the sample list in the HDF5 file based on data type."""
        metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
        if not metadata_key:
            raise ValueError("Could not find metadata group in HDF5 file")

        metadata_group = h5_file[metadata_key]

        if data_type:
            canonical_type = AliasUtils.get_field(data_type)
            log.debug(f"Canonical type for '{data_type}': {canonical_type}")
        else:
            canonical_type = None

        if canonical_type == "Methylation" or data_type == "Methylation":
            sample_key = AliasUtils.find_keys(metadata_group, "SampleList")
            log.debug(f"Looking for SampleList in metadata, found: {sample_key}")
            if sample_key:
                return f"/{metadata_key}/{sample_key}"
        elif canonical_type == "Genotype" or data_type == "Genotype":
            sample_key = AliasUtils.find_keys(metadata_group, "IID")
            if sample_key:
                return f"/{metadata_key}/{sample_key}"
        else:
            samplelist_key = AliasUtils.find_keys(metadata_group, "SampleList")
            log.debug(f"Auto-detect: Looking for SampleList, found: {samplelist_key}")
            if samplelist_key:
                return f"/{metadata_key}/{samplelist_key}"

            iid_key = AliasUtils.find_keys(metadata_group, "IID")
            log.debug(f"Auto-detect: Looking for IID, found: {iid_key}")
            if iid_key:
                return f"/{metadata_key}/{iid_key}"

        raise ValueError(
            f"Could not find sample list in metadata for data type: {data_type}"
        )

    def _get_sample_names(
        self, h5_file: Any, data_type: Optional[str], data_shape: Any
    ) -> List[str]:
        """Retrieve sample names from HDF5 file or generate defaults."""
        try:
            sample_path = self._get_sample_path(h5_file, data_type)
            sample_names = self._convert_sample_ids(h5_file[sample_path][:])
            return sample_names
        except (ValueError, KeyError, OSError) as e:
            log.debug(f"Could not get sample names from HDF5: {e}")
            n_samples = (
                data_shape[1]
                if hasattr(data_shape, "__len__") and len(data_shape) > 1
                else 0
            )
            return [f"{self.config.default_sample_prefix}{i}" for i in range(n_samples)]

    @staticmethod
    def _find_dataset(
        group: Any, field_name: str, required: bool = True
    ) -> Optional[Any]:
        """Find and return a dataset from the group using aliases."""
        found_key = AliasUtils.find_keys(group, field_name)
        if found_key:
            log.debug(f"Found dataset '{found_key}' for field '{field_name}'")
            return group[found_key][:]

        if required:
            log.warn(f"Required field '{field_name}' not found in group")
        return None


class DataProcessor:
    def __init__(
        self,
        config: Optional[H5Config] = None,
        chunked_reader: Optional[ChunkedDataReader] = None,
    ) -> None:
        self.config = config or H5Config()
        self.chunked_reader = chunked_reader
        self.validator = DataValidator()

    def _find_dataset_by_type(
        self, chrom_group: Any, data_type: str
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Find the appropriate dataset in the chromosome group based on data type."""
        key = AliasUtils.find_keys(chrom_group, data_type)
        if key:
            data = self._read_dataset_safely(chrom_group[key])
            log.debug(
                f"Found dataset '{key}' with shape {getattr(data, 'shape', None)}"
            )
            return data, key

        largest_size = 0
        largest_key: Optional[str] = None

        for key in chrom_group.keys():
            dataset = chrom_group[key]
            condition1 = hasattr(dataset, "shape")
            condition2 = len(getattr(dataset, "shape", [])) == 2
            condition3 = getattr(dataset, "shape", (0, 0))[0] > 0
            condition4 = getattr(dataset, "shape", (0, 0))[1] > 0
            if condition1 and condition2 and condition3 and condition4:
                try:
                    current_size = dataset.shape[0] * dataset.shape[1]
                    condition1 = current_size > largest_size
                    condition2 = hasattr(dataset, "dtype")
                    condition3 = np.issubdtype(dataset.dtype, np.number)
                    if condition1 and condition2 and condition3:
                        largest_size = current_size
                        largest_key = key
                        log.debug(
                            f"Candidate dataset '{key}' with shape {getattr(dataset, 'shape', None)}"
                        )
                except (AttributeError, TypeError) as e:
                    log.debug(f"Skipping dataset {key} due to incompatible format: {e}")
                    continue

        if largest_key:
            data = self._read_dataset_safely(chrom_group[largest_key])
            log.debug(
                f"Using largest dataset '{largest_key}' with shape {getattr(data, 'shape', None)}"
            )
            return data, largest_key

        return None, None

    def _read_dataset_safely(self, dataset: Any) -> Any:
        """Read dataset into memory, using chunked reading if necessary."""
        if self.chunked_reader and hasattr(dataset, "shape"):
            if len(getattr(dataset, "shape", [])) >= 2:
                if dataset.shape[0] > self.config.chunk_size:
                    log.debug(
                        f"Using chunked reading for large dataset (rows: {dataset.shape[0]})"
                    )
                    return self._read_chunked_to_memory(dataset)
            else:
                total_size = np.prod(getattr(dataset, "shape", ()))
                if total_size > self.config.chunk_size:
                    log.debug(
                        f"Using chunked reading for large dataset (size: {total_size})"
                    )
                    return self._read_chunked_to_memory(dataset)

        return dataset[:]

    def _read_chunked_to_memory(self, dataset: Any) -> Any:
        """Read a large dataset into memory using chunked reading."""
        try:
            shape = dataset.shape
            dtype = dataset.dtype

            result = np.empty(shape, dtype=dtype)

            start_idx = 0
            for chunk in self.chunked_reader.read_in_chunks(dataset):
                chunk_size = chunk.shape[0]
                end_idx = start_idx + chunk_size

                result[start_idx:end_idx] = chunk
                start_idx = end_idx

                progress_interval = self.config.chunk_size * 10
                if end_idx % progress_interval == 0 or end_idx == shape[0]:
                    log.debug(
                        f"Loaded {end_idx}/{shape[0]} rows ({100 * end_idx / shape[0]:.1f}%)"
                    )

            log.debug(f"Successfully loaded {shape[0]} rows using chunked reading")
            return result

        except Exception as e:
            log.error(f"Error in chunked reading, falling back to direct read: {e}")
            return dataset[:]

    def _create_dataframe_safely(
        self, data: Any, sample_names: List[str]
    ) -> pd.DataFrame:
        """Create a pandas DataFrame from data and sample names, handling shape issues."""
        if self.config.strict_validation:
            self.validator.validate_data_shape(data, expected_dims=2)
            if not self.validator.validate_sample_data_alignment(
                data.shape, sample_names
            ):
                log.warn(
                    f"Sample names ({len(sample_names)}) don't align with data shape {getattr(data, 'shape', None)}"
                )

        try:
            n_rows, n_cols = data.shape
            n_sample_names = len(sample_names)

            log.debug(f"Data shape: {data.shape}, Sample names: {n_sample_names}")

            if n_cols == n_sample_names:
                log.debug("Using data as markers x samples")
                return pd.DataFrame(data, columns=sample_names)
            elif n_rows == n_sample_names:
                log.debug(
                    "Transposing data from samples x markers to markers x samples"
                )
                return pd.DataFrame(data.T, columns=sample_names)
            else:
                log.warn(
                    f"Dimension mismatch: data shape {getattr(data, 'shape', None)}, samples {n_sample_names}"
                )
                return self._create_dataframe_fallback(data, sample_names)

        except (ValueError, TypeError) as e:
            log.debug(f"Failed to create DataFrame directly: {e}")
            return self._create_dataframe_fallback(data, sample_names)

    def _create_dataframe_fallback(
        self, data: Any, sample_names: List[str]
    ) -> pd.DataFrame:
        """Fallback method to create DataFrame when direct method fails."""
        try:
            df = pd.DataFrame(data)

            if hasattr(data, "shape") and len(getattr(data, "shape", [])) > 1:
                n_rows, n_cols = data.shape
                n_names = len(sample_names)

                if n_cols <= n_names:
                    column_mapping = {i: sample_names[i] for i in range(n_cols)}
                    df = df.rename(columns=column_mapping)
                elif n_rows <= n_names:
                    log.debug("Attempting transpose in fallback")
                    df = pd.DataFrame(data.T)
                    column_mapping = {
                        i: sample_names[i] for i in range(min(df.shape[1], n_names))
                    }
                    df = df.rename(columns=column_mapping)
                else:
                    column_mapping: Dict[int, str] = {}
                    for i in range(n_cols):
                        if i < n_names:
                            column_mapping[i] = sample_names[i]
                        else:
                            column_mapping[i] = (
                                f"{self.config.default_sample_prefix}{i}"
                            )
                    df = df.rename(columns=column_mapping)

            return df

        except Exception as e:
            log.error(f"Fallback DataFrame creation also failed: {e}")
            return pd.DataFrame(data)

    def process_genotype_data(
        self, h5_file: Any, chrom_group: Any, data: Any, sample_names: List[str]
    ) -> pd.DataFrame:
        """Process genotype data and return a DataFrame with annotations."""
        log.debug(
            f"Processing genotype data: shape {getattr(data, 'shape', None)}, samples {len(sample_names)}"
        )

        df = self._create_dataframe_safely(data, sample_names)

        rsid = BaseH5Utils._find_dataset(chrom_group, "RSID", required=False)
        bp = BaseH5Utils._find_dataset(chrom_group, "BP", required=False)
        a1 = BaseH5Utils._find_dataset(chrom_group, "A1", required=False)
        a2 = BaseH5Utils._find_dataset(chrom_group, "A2", required=False)
        info = BaseH5Utils._find_dataset(chrom_group, "INFO", required=False)

        n_markers = df.shape[0]

        df["RSID"] = (
            [BaseH5Utils._decode_if_bytes(r) for r in rsid[:n_markers]]
            if rsid is not None and len(rsid) >= n_markers
            else [f"{self.config.default_marker_prefix}{i}" for i in range(n_markers)]
        )

        df["BP"] = (
            bp[:n_markers]
            if bp is not None and len(bp) >= n_markers
            else np.arange(n_markers)
        )

        df["A1"] = (
            [BaseH5Utils._decode_if_bytes(a) for a in a1[:n_markers]]
            if a1 is not None and len(a1) >= n_markers
            else ["A"] * n_markers
        )

        df["A2"] = (
            [BaseH5Utils._decode_if_bytes(a) for a in a2[:n_markers]]
            if a2 is not None and len(a2) >= n_markers
            else ["B"] * n_markers
        )

        if info is not None and len(info) >= n_markers:
            df["INFO"] = info[:n_markers]

        return df

    def process_methylation_data(
        self, h5_file: Any, chrom_group: Any, data: Any, sample_names: List[str]
    ) -> pd.DataFrame:
        """Process methylation data and return a DataFrame with annotations."""
        log.debug(
            f"Processing methylation data: shape {getattr(data, 'shape', None)}, samples {len(sample_names)}"
        )

        df = self._create_dataframe_safely(data, sample_names)

        cgid = BaseH5Utils._find_dataset(chrom_group, "ProbeList", required=False)
        if cgid is None:
            cgid = BaseH5Utils._find_dataset(chrom_group, "CGID", required=False)

        n_probes = df.shape[0]

        df["CGID"] = (
            [BaseH5Utils._decode_if_bytes(c) for c in cgid[:n_probes]]
            if cgid is not None and len(cgid) >= n_probes
            else [
                f"{self.config.default_marker_prefix}{i:08d}" for i in range(n_probes)
            ]
        )

        try:
            if hasattr(data, "size") and data.size > 0:
                max_val = np.nanmax(data)
                min_val = np.nanmin(data)
                if max_val <= 1.0 and min_val >= 0.0:
                    df["DataType"] = "Beta"
                elif min_val < 0:
                    df["DataType"] = "M-value"
                else:
                    df["DataType"] = "Unknown"
            else:
                df["DataType"] = "Unknown"
        except (TypeError, ValueError) as e:
            log.debug(f"Could not determine methylation data type: {e}")
            df["DataType"] = "Unknown"

        bp = BaseH5Utils._find_dataset(chrom_group, "BP", required=False)
        chr_pos = BaseH5Utils._find_dataset(chrom_group, "CHR", required=False)

        if bp is not None and len(bp) >= n_probes:
            df["BP"] = bp[:n_probes]
        if chr_pos is not None and len(chr_pos) >= n_probes:
            df["CHR"] = chr_pos[:n_probes]

        return df


class ChromosomeReader(BaseH5Utils):
    def __init__(self, config: Optional[H5Config] = None) -> None:
        super().__init__(config)
        self.data_processor = DataProcessor(
            self.config, ChunkedDataReader(self.config.chunk_size)
        )

    def read_chromosome_data(
        self,
        h5_file: Any,
        chromosome: str,
        data_type: Optional[str],
        chromosome_mapper: ChromosomeMapper,
        data_type_detector: DataTypeDetector,
    ) -> Optional[pd.DataFrame]:
        """Read and process data for a specific chromosome from the HDF5 file."""
        try:
            chrom = chromosome_mapper.map_chromosome_name(chromosome)
            if chrom is None:
                log.warn(f"Chromosome {chromosome} not found in H5 file")
                return None

            log.debug(
                f"Mapped requested chromosome '{chromosome}' to actual chromosome '{chrom}'"
            )
            chrom_group = h5_file[chrom]

            if data_type is None:
                data_type = data_type_detector.detect_data_type(chrom_group)
                if data_type is None:
                    log.error(f"Could not detect data type for chromosome {chrom}")
                    return None

            data, dataset_key = self.data_processor._find_dataset_by_type(
                chrom_group, data_type
            )
            if data is None:
                log.error(f"No {data_type.lower()} data found for chromosome {chrom}")
                return None

            sample_names = self._get_sample_names(
                h5_file, data_type, getattr(data, "shape", None)
            )

            genotype_aliases = AliasUtils.get_aliases("Genotype")
            methylation_aliases = AliasUtils.get_aliases("Methylation")

            data_type_lower = data_type.lower() if isinstance(data_type, str) else ""

            if data_type_lower in [alias.lower() for alias in genotype_aliases]:
                return self.data_processor.process_genotype_data(
                    h5_file, chrom_group, data, sample_names
                )
            elif data_type_lower in [alias.lower() for alias in methylation_aliases]:
                return self.data_processor.process_methylation_data(
                    h5_file, chrom_group, data, sample_names
                )
            else:
                log.error(f"Unsupported data type: {data_type}")
                return None

        except Exception as e:
            log.error(f"Error reading chromosome data: {e}")
            return None


class SampleIndexer(BaseH5Utils):
    def get_sample_indices(
        self, h5_file: Any, sample_ids: Iterable[Any], sample_path: Optional[str] = None
    ) -> Optional[List[int]]:
        """Get indices of specified sample IDs in the HDF5 file."""
        try:
            if sample_ids is None:
                return None

            if sample_path is None:
                sample_path = self._get_sample_path(h5_file)

            if sample_path not in h5_file:
                log.warn(f"Sample path {sample_path} not found in HDF5 file")

                metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
                if metadata_key:
                    if "samplelist" in sample_path.lower():
                        iid_key = AliasUtils.find_keys(h5_file[metadata_key], "IID")
                        alt_path = f"/{metadata_key}/{iid_key}" if iid_key else None
                    else:
                        samplelist_key = AliasUtils.find_keys(
                            h5_file[metadata_key], "SampleList"
                        )
                        alt_path = (
                            f"/{metadata_key}/{samplelist_key}"
                            if samplelist_key
                            else None
                        )

                    if alt_path and alt_path in h5_file:
                        log.info(f"Using alternative sample path: {alt_path}")
                        sample_path = alt_path
                    else:
                        raise ValueError("Sample list not found in HDF5 file")

            raw_sample_names = h5_file[sample_path][:]
            sample_names = self._convert_sample_ids(raw_sample_names)

            sample_ids_normalized = self._normalize_sample_id_list(sample_ids)

            sample_to_idx = {sample: idx for idx, sample in enumerate(sample_names)}
            indices = [
                sample_to_idx[sample_id]
                for sample_id in sample_ids_normalized
                if sample_id in sample_to_idx
            ]

            log.debug(
                f"Found {len(indices)} matching samples out of {len(sample_ids_normalized)} requested"
            )

            if len(indices) != len(sample_ids_normalized):
                missing_samples = [
                    s for s in sample_ids_normalized if s not in sample_names
                ]
                if missing_samples:
                    log.warn(
                        f"Some samples not found: {', '.join(missing_samples[:5])}"
                    )

            return indices

        except Exception as e:
            log.error(f"Error getting sample indices: {e}")
            return None


class MarkerIndexer(BaseH5Utils):
    def get_marker_indices(
        self,
        h5_file: Any,
        chromosome: str,
        marker_ids: Optional[Iterable[Any]] = None,
        data_type: Optional[str] = None,
        chromosome_mapper: Optional[ChromosomeMapper] = None,
        data_type_detector: Optional[DataTypeDetector] = None,
    ) -> Optional[List[int]]:
        """Get indices of specified marker IDs for a chromosome in the HDF5 file."""
        try:
            if data_type is None and data_type_detector:
                actual_chromosome = (
                    chromosome_mapper.map_chromosome_name(chromosome)
                    if chromosome_mapper
                    else chromosome
                )
                if actual_chromosome and actual_chromosome in h5_file:
                    data_type = data_type_detector.detect_data_type(
                        h5_file[actual_chromosome]
                    )

                if data_type is None:
                    log.error(f"Could not auto-detect data type for {chromosome}")
                    return None

            if data_type is None:
                log.error(f"Could not auto-detect data type for {chromosome}")
                return None

            if not isinstance(data_type, str):
                log.error(f"Invalid data type detected: {data_type}")
                return None

            data_type_lower = data_type.lower()

            if chromosome_mapper:
                actual_chromosome = chromosome_mapper.map_chromosome_name(chromosome)
            else:
                actual_chromosome = chromosome

            if actual_chromosome is None:
                log.warn(f"Chromosome {chromosome} not found in H5 file")
                return None

            chr_group = h5_file[actual_chromosome]

            methylation_aliases = AliasUtils.get_aliases("Methylation")
            genotype_aliases = AliasUtils.get_aliases("Genotype")

            if data_type_lower in [alias.lower() for alias in methylation_aliases]:
                marker_key = AliasUtils.find_keys(chr_group, "ProbeList")
                marker_type = "probes"
            elif data_type_lower in [alias.lower() for alias in genotype_aliases]:
                marker_key = AliasUtils.find_keys(chr_group, "RSID")
                marker_type = "SNPs"
            else:
                log.error(f"Unsupported data type: {data_type}")
                return None

            if not marker_key:
                log.error(f"No {marker_type} found for chromosome {actual_chromosome}")
                return None

            marker_list_raw = chr_group[marker_key][:]
            marker_list = self._decode_array(marker_list_raw)

            if marker_ids:
                # Use set for O(1) lookup instead of list O(n)
                marker_ids_set = set(str(m) for m in marker_ids)
                indices = [i for i, m in enumerate(marker_list) if m in marker_ids_set]

                if not indices:
                    log.debug(
                        f"Found 0 of {len(marker_ids_set)} requested {marker_type} in {actual_chromosome}"
                    )
                    return None

                log.debug(
                    f"Found {len(indices)} of {len(marker_ids_set)} requested {marker_type} in {actual_chromosome}"
                )
            else:
                indices = list(range(len(marker_list)))
                log.debug(
                    f"Using all {len(indices)} {marker_type} in {actual_chromosome}"
                )

            return indices

        except Exception as e:
            log.error(f"Error getting marker indices: {e}")
            return None


class ChromosomeLister(BaseH5Utils):
    def get_chromosome_list(self, h5_file: Any) -> Optional[List[str]]:
        """Get a sorted list of chromosomes in the HDF5 file."""
        try:
            chr_list: List[str] = []
            for key in h5_file.keys():
                if not AliasUtils.find_keys({key: key}, "Metadata"):
                    chr_list.append(key)

            if len(chr_list) == 0:
                raise ValueError("No chromosomes found in HDF5 file")

            def sort_key(chrom_name: str) -> int:
                base_name = AliasUtils.strip_numeric_suffix(chrom_name)
                numeric_part = chrom_name.replace(base_name, "")

                try:
                    return int(numeric_part)
                except ValueError:
                    priority_map = {"X": 23, "Y": 24, "M": 25, "MT": 25}
                    return priority_map.get(numeric_part.upper(), 999)

            chr_list = sorted(chr_list, key=sort_key)
            log.debug(f"Found {len(chr_list)} chromosomes: {', '.join(chr_list)}")
            return chr_list

        except Exception as e:
            log.error(f"Error getting chromosome list: {e}")
            return None


class CachedH5Utils:
    def __init__(self, h5_file: Any, config: Optional[H5Config] = None, owns_file: bool = False) -> None:
        self.h5_file = h5_file
        self._owns_file = owns_file
        self.config = config or H5Config()
        self._cache: Optional[OrderedDict] = (
            OrderedDict() if self.config.cache_enabled else None
        )

        self.chromosome_mapper = ChromosomeMapper(h5_file)
        self.data_type_detector = DataTypeDetector()
        self.chromosome_reader = ChromosomeReader(config)
        self.sample_indexer = SampleIndexer(config)
        self.marker_indexer = MarkerIndexer(config)
        self.chromosome_lister = ChromosomeLister(config)

        self._metadata_key = AliasUtils.find_keys(h5_file, "Metadata")
        self._data_type: Optional[str] = None

    def _get_from_cache_or_compute(
        self, cache_key: str, compute_func: Callable[[], Any]
    ) -> Any:
        """Retrieve value from cache or compute it if not present."""
        if not self.config.cache_enabled or self._cache is None:
            return compute_func()

        if cache_key in self._cache:
            result = self._cache.pop(cache_key)
            self._cache[cache_key] = result
            return result

        try:
            result = compute_func()
        except Exception:
            raise

        if len(self._cache) >= self.config.max_cache_size:
            self._cache.popitem(last=False)

        self._cache[cache_key] = result
        return result

    def get_data_type(self) -> Optional[str]:
        """Get the data type (Genotype or Methylation) of the HDF5 file."""
        if self._data_type is None:
            chromosomes = self.get_chromosomes()
            if chromosomes:
                chrom_group = self.h5_file[chromosomes[0]]
                self._data_type = self.data_type_detector.detect_data_type(chrom_group)
        return self._data_type

    def get_metadata_key(self) -> Optional[str]:
        """Get the metadata group key in the HDF5 file."""
        return self._metadata_key

    def __enter__(self) -> "CachedH5Utils":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit context manager and close HDF5 file if it owns it."""
        if self._owns_file and hasattr(self.h5_file, "close"):
            self.h5_file.close()
        return False

    def get_chromosomes(self) -> Optional[List[str]]:
        """Get a list of chromosomes in the HDF5 file."""
        return self._get_from_cache_or_compute(
            "chromosomes",
            lambda: self.chromosome_lister.get_chromosome_list(self.h5_file),
        )

    def read_chromosome(self, chromosome: str, data_type: Optional[str] = None) -> Any:
        """Read data for a specific chromosome from the HDF5 file."""
        cache_key = f"chromosome_{chromosome}_{data_type or 'auto'}"
        return self._get_from_cache_or_compute(
            cache_key,
            lambda: self.chromosome_reader.read_chromosome_data(
                self.h5_file,
                chromosome,
                data_type,
                self.chromosome_mapper,
                self.data_type_detector,
            ),
        )

    def get_sample_indices(
        self, sample_ids: Iterable[Any], data_type: Optional[str] = None
    ) -> Optional[List[int]]:
        """Get indices of specified sample IDs in the HDF5 file."""
        if sample_ids is None:
            return None

        if data_type is None:
            data_type = self.get_data_type()

        sample_ids_normalized = BaseH5Utils._normalize_sample_id_list(sample_ids)

        sample_key = str(sorted(sample_ids_normalized))
        cache_key = f"sample_indices_{sample_key}_{data_type}"

        return self._get_from_cache_or_compute(
            cache_key,
            lambda: self.sample_indexer.get_sample_indices(
                self.h5_file, sample_ids_normalized
            ),
        )

    def get_marker_indices(
        self,
        chromosome: str,
        marker_ids: Optional[Iterable[Any]] = None,
        data_type: Optional[str] = None,
    ) -> Optional[List[int]]:
        """Get indices of specified marker IDs for a chromosome in the HDF5 file."""
        if data_type is None:
            data_type = self.get_data_type()

        marker_key = str(sorted(map(str, marker_ids))) if marker_ids else "all"
        cache_key = f"marker_indices_{chromosome}_{marker_key}_{data_type}"

        return self._get_from_cache_or_compute(
            cache_key,
            lambda: self.marker_indexer.get_marker_indices(
                self.h5_file,
                chromosome,
                marker_ids,
                data_type,
                self.chromosome_mapper,
                self.data_type_detector,
            ),
        )

    def get_data_info(self) -> Optional[Dict[str, Any]]:
        """Get summary information about the data in the HDF5 file."""
        return self._get_from_cache_or_compute(
            "data_info", lambda: self._compute_data_info()
        )

    def _compute_data_info(self) -> Optional[Dict[str, Any]]:
        """Compute summary information about the data in the HDF5 file."""
        try:
            info: Dict[str, Any] = {
                "chromosomes": self.get_chromosomes(),
                "n_chromosomes": 0,
                "data_type": None,
                "n_samples": 0,
                "sample_path": None,
            }

            if info["chromosomes"]:
                info["n_chromosomes"] = len(info["chromosomes"])

                first_chr = info["chromosomes"][0]
                chrom_group = self.h5_file[first_chr]
                info["data_type"] = self.data_type_detector.detect_data_type(
                    chrom_group
                )

                try:
                    sample_path = BaseH5Utils._get_sample_path(
                        self.h5_file, info["data_type"]
                    )
                    info["sample_path"] = sample_path
                    if sample_path in self.h5_file:
                        info["n_samples"] = len(self.h5_file[sample_path])
                        log.debug(f"Found {info['n_samples']} samples at {sample_path}")
                except Exception as e:
                    log.warn(f"Could not get sample information: {e}")

            return info

        except Exception as e:
            log.error(f"Error getting data info: {e}")
            return None

    def validate_file_structure(self) -> bool:
        """Validate the structure of the HDF5 file."""
        try:
            info = self.get_data_info()
            if info is None:
                return False

            if info["n_chromosomes"] == 0:
                log.error("No chromosomes found in file")
                return False

            if info["data_type"] is None:
                log.error("Cannot determine data type")
                return False

            if info["n_samples"] == 0:
                log.error("No samples found in file")
                return False

            log.info(
                f"File validation passed: {info['data_type']} data with "
                f"{info['n_chromosomes']} chromosomes and {info['n_samples']} samples"
            )
            return True

        except Exception as e:
            log.error(f"Error validating file structure: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        if self._cache is not None:
            self._cache.clear()
            log.debug("Cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        if self._cache is None:
            return {"enabled": False, "size": 0, "max_size": 0}

        return {
            "enabled": True,
            "size": len(self._cache),
            "max_size": self.config.max_cache_size,
            "keys": list(self._cache.keys()),
        }


class ChunkedH5Utils(CachedH5Utils):
    def __init__(self, h5_file: Any, config: Optional[H5Config] = None) -> None:
        super().__init__(h5_file, config)
        self.chunked_reader = ChunkedDataReader(self.config.chunk_size)

        self.chromosome_reader.data_processor.chunked_reader = self.chunked_reader

    def read_chromosome_chunked(
        self,
        chromosome: str,
        data_type: Optional[str] = None,
        chunk_callback: Optional[Callable[[Any], None]] = None,
    ) -> Optional[Any]:
        """Read data for a specific chromosome in chunks from the HDF5 file."""
        try:
            chrom = self.chromosome_mapper.map_chromosome_name(chromosome)
            if chrom is None:
                return None

            if data_type is None:
                chrom_group = self.h5_file[chrom]
                data_type = self.data_type_detector.detect_data_type(chrom_group)

            chrom_group = self.h5_file[chrom]

            dataset_key = AliasUtils.find_keys(chrom_group, data_type)
            if not dataset_key:
                log.error(f"Could not find dataset for data type {data_type}")
                return None

            dataset_obj = chrom_group[dataset_key]

            if chunk_callback:
                for chunk in self.chunked_reader.read_in_chunks(dataset_obj):
                    chunk_callback(chunk)
                return None
            else:
                return self.read_chromosome(chromosome, data_type)

        except Exception as e:
            log.error(f"Error reading chromosome data in chunks: {e}")
            return None


class H5UtilsFactory:
    @staticmethod
    def create_utils(
        file_path: str,
        mode: str = "r",
        config: Optional[H5Config] = None,
        enable_caching: bool = True,
        enable_chunking: bool = False,
    ) -> Union[CachedH5Utils, ChunkedH5Utils]:
        """Create an H5Utils instance from a file path."""
        if mode == "r" and not os.path.exists(file_path):
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        try:
            h5_file = h5py.File(file_path, mode)
        except (OSError, IOError, ValueError) as e:
            raise ValueError(f"Failed to open HDF5 file '{file_path}': {e}")

        if config is None:
            config = H5Config()
        else:
            config = H5Config(
                cache_enabled=config.cache_enabled,
                strict_validation=config.strict_validation,
                default_sample_prefix=config.default_sample_prefix,
                default_marker_prefix=config.default_marker_prefix,
                max_cache_size=config.max_cache_size,
                chunk_size=config.chunk_size,
            )

        config.cache_enabled = enable_caching

        if enable_chunking:
            return ChunkedH5Utils(h5_file, config, owns_file=True)
        else:
            return CachedH5Utils(h5_file, config, owns_file=True)

    @staticmethod
    def create_utils_from_file(
        h5_file: h5py.File,
        config: Optional[H5Config] = None,
        enable_caching: bool = True,
        enable_chunking: bool = False,
    ) -> Union[CachedH5Utils, ChunkedH5Utils]:
        """Create an H5Utils instance from an existing h5py File object."""
        if config is None:
            config = H5Config()
        else:
            config = H5Config(
                cache_enabled=config.cache_enabled,
                strict_validation=config.strict_validation,
                default_sample_prefix=config.default_sample_prefix,
                default_marker_prefix=config.default_marker_prefix,
                max_cache_size=config.max_cache_size,
                chunk_size=config.chunk_size,
            )

        config.cache_enabled = enable_caching

        if enable_chunking:
            return ChunkedH5Utils(h5_file, config)
        else:
            return CachedH5Utils(h5_file, config)
