#!/usr/bin/env python
# Import required modules
import h5py
import numpy as np
import os
import pandas as pd
import re
import sys
from typing import Dict, Iterable, List, Optional
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.LoggingUtils import log


class ViewHDF5:
    def __init__(self, input_file: str, missing_analysis: bool = False) -> None:
        self.input_file = input_file
        self.missing_analysis = missing_analysis
        self.data_type = None
        self.chr_keys = None

    def find_dataset_key(self, group: h5py.Group, field: str) -> str:
        try:
            aliases = AliasUtils.get_aliases(field)
        except (AttributeError, KeyError) as e:
            log.warn(f"Could not get aliases for field '{field}': {e}")
            aliases = [field]
        aliases = list(set(aliases))
        lower_aliases = {a.lower(): a for a in aliases}
        for gk in group.keys():
            lower_gk = gk.lower()
            if lower_gk in lower_aliases:
                return gk
        raise ValueError(
            f"No matching key found for {field} in group. Tried: {aliases}"
        )

    def get_chr_prefixes(self) -> List[str]:
        try:
            return [alias.lower() for alias in AliasUtils.get_aliases("CHR")]
        except (AttributeError, KeyError) as e:
            log.warn(f"Could not get CHR aliases: {e}")
            return ["chr", "chromosome"]

    def detect_chr_groups(self, keys: Iterable[str]) -> List[str]:
        prefixes = self.get_chr_prefixes()
        chr_groups: List[str] = []
        for key in keys:
            key_lower = key.lower()
            for prefix in prefixes:
                prefix_len = len(prefix)
                if key_lower.startswith(prefix) and key_lower[prefix_len:].isdigit():
                    chr_groups.append(key)
                    break
        return chr_groups

    def _detect_data_type(self, h5_file: h5py.File) -> Optional[str]:
        try:
            chr_groups = self.detect_chr_groups(h5_file.keys())
            if not chr_groups:
                if len(h5_file.keys()) == 0:
                    log.warn("HDF5 file appears to be empty")
                    return None
                log.warn("No chromosome groups found in the HDF5 file")
                return "unknown"
            first_chr = chr_groups[0]
            chr_group = h5_file[first_chr]
            genotype_indicators = 0
            genotype_fields = ["RSID", "A1", "A2", "BP", "Genotype"]
            for field in genotype_fields:
                try:
                    self.find_dataset_key(chr_group, field)
                    genotype_indicators += 1
                except ValueError:
                    continue
            has_genotype_data = False
            has_rsid = False
            try:
                self.find_dataset_key(chr_group, "Genotype")
                has_genotype_data = True
            except ValueError:
                pass
            try:
                self.find_dataset_key(chr_group, "RSID")
                has_rsid = True
            except ValueError:
                pass
            if has_genotype_data or has_rsid or genotype_indicators >= 2:
                log.debug("Detected genotype data format")
                return "genotype"
            methylation_indicators = 0
            methylation_fields = ["ProbeList", "Methylation"]
            for field in methylation_fields:
                try:
                    self.find_dataset_key(chr_group, field)
                    methylation_indicators += 1
                except ValueError:
                    continue
            if methylation_indicators >= 1:
                log.debug("Detected methylation data format")
                return "methylation"
            log.warn("Unknown data format in HDF5 file")
            return "unknown"
        except Exception as e:
            log.error(f"Error detecting data type: {e}")
            return None

    def view_structure(self, h5_file: h5py.File) -> None:
        log.info("Analyzing HDF5 file structure")
        print("=" * 80)
        print("HDF5 FILE STRUCTURE")
        print("=" * 80)
        data = []
        for name, obj in h5_file.items():
            group = "/"
            log.debug(f"Processing item: {name} in group: {group}")
            if isinstance(obj, h5py.Group):
                log.debug(f"Item: {name} is a group")
                data.append([group, name, "H5I_GROUP", "", ""])
                for sub_name, sub_obj in obj.items():
                    sub_group = f"/{name}"
                    log.debug(
                        f"Processing sub-item: {sub_name} in sub-group: {sub_group}"
                    )
                    if isinstance(sub_obj, h5py.Group):
                        log.debug(f"Sub-item: {sub_name} is a group")
                        data.append([sub_group, sub_name, "H5I_GROUP", "", ""])
                        for nested_name, nested_obj in sub_obj.items():
                            nested_group = f"/{name}/{sub_name}"
                            if isinstance(nested_obj, h5py.Dataset):
                                dclass = (
                                    "STRING"
                                    if nested_obj.dtype.char == "S"
                                    else nested_obj.dtype.name.upper()
                                )
                                dim = " x ".join(map(str, nested_obj.shape[::-1]))
                                data.append(
                                    [
                                        nested_group,
                                        nested_name,
                                        "H5I_DATASET",
                                        dclass,
                                        dim,
                                    ]
                                )
                    elif isinstance(sub_obj, h5py.Dataset):
                        log.debug(f"Sub-item: {sub_name} is a dataset")
                        dclass = (
                            "STRING"
                            if sub_obj.dtype.char == "S"
                            else sub_obj.dtype.name.upper()
                        )
                        dim = " x ".join(map(str, sub_obj.shape))
                        data.append([sub_group, sub_name, "H5I_DATASET", dclass, dim])
            elif isinstance(obj, h5py.Dataset):
                log.debug(f"Item: {name} is a dataset")
                dclass = "STRING" if obj.dtype.char == "S" else obj.dtype.name.upper()
                dim = " x ".join(map(str, obj.shape))
                data.append([group, name, "H5I_DATASET", dclass, dim])
        log.debug("Converting data to pandas DataFrame")
        df = pd.DataFrame(data, columns=["group", "name", "otype", "dclass", "dim"])
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        print(df)
        print()

    def count_data_elements(self, h5_file: h5py.File) -> None:
        print("=" * 80)
        print("DATA SUMMARY")
        print("=" * 80)
        if self.data_type == "unknown" or self.data_type is None:
            print("Cannot analyze data: Unknown or unsupported file format")
            return
        if self.data_type == "methylation":
            marker_field = "ProbeList"
            sample_field = "SampleList"
            entity_name = "probes"
        else:
            marker_field = "RSID"
            sample_field = "IID"
            entity_name = "RSIDs"
        log.info(f"Counting {entity_name} and samples")
        total_markers = 0
        chromosome_counts: Dict[str, int] = {}
        for grp in sorted(self.chr_keys):
            try:
                marker_key = self.find_dataset_key(h5_file[grp], marker_field)
                markers = h5_file[grp][marker_key][:]
                num_markers = len(markers)
                total_markers += num_markers
                chromosome_counts[grp] = num_markers
                log.debug(f"Chromosome {grp}: {num_markers} {entity_name}")
            except ValueError as e:
                log.warn(
                    f"Marker key not found in {grp}: {e}. Trying to infer from data dimensions."
                )
                data_field = (
                    "Methylation" if self.data_type == "methylation" else "Genotype"
                )
                try:
                    data_key = self.find_dataset_key(h5_file[grp], data_field)
                    data_shape = h5_file[grp][data_key].shape
                    num_markers = data_shape[0]
                    total_markers += num_markers
                    chromosome_counts[grp] = num_markers
                    log.debug(
                        f"Chromosome {grp}: {num_markers} {entity_name} (inferred from data shape)"
                    )
                except ValueError:
                    log.warn(f"Cannot determine marker count for {grp}. Skipping.")
                    continue
        num_samples = 0
        metadata_grp = None
        try:
            metadata_aliases = AliasUtils.get_aliases("Metadata")
            for key in h5_file.keys():
                if key in metadata_aliases:
                    metadata_grp = h5_file[key]
                    break
        except (AttributeError, KeyError) as e:
            log.warn(f"Could not get Metadata aliases: {e}")
            for key in h5_file.keys():
                if key.lower() in ["metadata", "meta_data", "metaData", "meta"]:
                    metadata_grp = h5_file[key]
                    break
        if metadata_grp is None:
            log.warn("No metadata group found. Trying to infer sample count from data.")
            if self.chr_keys:
                try:
                    data_field = (
                        "Methylation" if self.data_type == "methylation" else "Genotype"
                    )
                    data_key = self.find_dataset_key(
                        h5_file[self.chr_keys[0]], data_field
                    )
                    data_shape = h5_file[self.chr_keys[0]][data_key].shape
                    num_samples = data_shape[1] if len(data_shape) > 1 else 1
                    log.info(f"Inferred sample count from data shape: {num_samples}")
                except ValueError:
                    log.error("Cannot determine sample count")
        else:
            try:
                sample_key = self.find_dataset_key(metadata_grp, sample_field)
                samples = metadata_grp[sample_key][:]
                num_samples = len(samples)
            except ValueError as e:
                log.error(f"No sample list found in metadata: {e}")
                alt_fields = (
                    ["SampleList", "IID"] if sample_field != "SampleList" else ["IID"]
                )
                for alt_field in alt_fields:
                    try:
                        alt_key = self.find_dataset_key(metadata_grp, alt_field)
                        samples = metadata_grp[alt_key][:]
                        num_samples = len(samples)
                        log.info(f"Using alternative sample field: {alt_field}")
                        break
                    except ValueError:
                        continue
                if num_samples == 0:
                    for key in metadata_grp.keys():
                        if "sample" in key.lower() or "id" in key.lower():
                            try:
                                samples = metadata_grp[key][:]
                                num_samples = len(samples)
                                log.info(f"Using fallback sample field: {key}")
                                break
                            except ValueError:
                                continue
        print(f"Data Type: {self.data_type.title()}")
        print(f"Number of {entity_name}: {total_markers:,}")
        print(f"Number of samples: {num_samples:,}")
        print(f"Total data points: {total_markers * num_samples:,}")
        if len(chromosome_counts) > 1:
            print("\nBreakdown by chromosome:")
            for chr_name in sorted(
                chromosome_counts.keys(),
                key=lambda x: (
                    int(re.search(r"\d+", x).group())
                    if re.search(r"\d+", x)
                    else float("inf")
                ),
            ):
                print(f"  {chr_name}: {chromosome_counts[chr_name]:,} {entity_name}")
        print()

    def count_missing_values(self, h5_file: h5py.File) -> None:
        print("=" * 80)
        print("MISSING VALUES ANALYSIS")
        print("=" * 80)
        if self.data_type == "unknown" or self.data_type is None:
            print("Cannot analyze missing values: Unknown or unsupported file format")
            return
        log.info("Counting missing values")
        if self.data_type == "methylation":
            data_field = "Methylation"
            missing_check = np.isnan
            data_name = "beta values"
        else:
            data_field = "Genotype"

            def missing_check(x):
                return (x == -1) | np.isnan(x)

            data_name = "genotype calls"
        total_missing_values = 0
        total_data_points = 0
        chromosome_stats: Dict[str, Dict[str, float]] = {}
        for grp in sorted(self.chr_keys):
            try:
                data_key = self.find_dataset_key(h5_file[grp], data_field)
                data = h5_file[grp][data_key]
                if data.size > 1e8:
                    log.debug(f"Processing {grp} in chunks due to large size")
                    chunk_size = 1000000
                    missing_count = 0
                    total_elements = data.size
                    processed = 0
                    while processed < total_elements:
                        if len(data.shape) == 2:
                            rows, cols = data.shape
                            chunk_rows = min(
                                chunk_size // cols, rows - processed // cols
                            )
                            if chunk_rows == 0:
                                chunk_rows = 1
                            start_row = processed // cols
                            end_row = min(start_row + chunk_rows, rows)
                            chunk = data[start_row:end_row, :]
                        else:
                            start = processed
                            end = min(start + chunk_size, total_elements)
                            chunk = data[start:end]
                        missing_count += np.sum(missing_check(chunk))
                        processed += chunk.size
                else:
                    data_array = data[:]
                    missing_count = np.sum(missing_check(data_array))
                data_points = data.size
                total_data_points += data_points
                total_missing_values += missing_count
                missing_percent = (
                    (missing_count / data_points * 100) if data_points > 0 else 0
                )
                chromosome_stats[grp] = {
                    "missing": missing_count,
                    "total": data_points,
                    "percent": missing_percent,
                }
                print(
                    f"{grp}: {missing_count:,} missing {data_name} ({missing_percent:.2f}%)"
                )
            except ValueError as e:
                log.warn(f"Data key not found in {grp}: {e}. Skipping {grp}.")
                continue
        total_missing_percent = (
            (total_missing_values / total_data_points * 100)
            if total_data_points > 0
            else 0
        )
        print("\nOVERALL SUMMARY:")
        print(f"Total missing {data_name}: {total_missing_values:,}")
        print(f"Total data points: {total_data_points:,}")
        print(f"Overall missing rate: {total_missing_percent:.2f}%")
        if chromosome_stats:
            max_missing_chr = max(
                chromosome_stats.items(), key=lambda x: x[1]["percent"]
            )
            min_missing_chr = min(
                chromosome_stats.items(), key=lambda x: x[1]["percent"]
            )
            print(
                f"\nHighest missing rate: {max_missing_chr[0]} ({max_missing_chr[1]['percent']:.2f}%)"
            )
            print(
                f"Lowest missing rate: {min_missing_chr[0]} ({min_missing_chr[1]['percent']:.2f}%)"
            )
        print()

    def analyze_file(self) -> bool:
        try:
            log.info(f"Starting analysis of HDF5 file: {self.input_file}")
            if not os.path.exists(self.input_file):
                log.error(f"Input file does not exist: {self.input_file}")
                return False
            with h5py.File(self.input_file, "r") as h5_file:
                log.debug(f"Successfully opened HDF5 file: {self.input_file}")
                self.view_structure(h5_file)
                self.data_type = self._detect_data_type(h5_file)
                if self.data_type is None:
                    log.error("Failed to detect data type")
                    return False
                self.chr_keys = self.detect_chr_groups(h5_file.keys())
                log.debug(f"Chromosome keys found: {self.chr_keys}")
                self.count_data_elements(h5_file)

                if self.missing_analysis:
                    if self.chr_keys:
                        self.count_missing_values(h5_file)
                    else:
                        print("=" * 80)
                        print("MISSING VALUES ANALYSIS")
                        print("=" * 80)
                        print("No chromosome data found for missing value analysis")
                        print()
                else:
                    log.info(
                        "Skipping missing values analysis (use -m/--missing_analysis to enable)"
                    )
            return True
        except Exception as e:
            log.error(f"Error analyzing HDF5 file: {e}")
            sys.exit(1)


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(
        flags=["-m", "--missing_analysis"], type=bool, default=False, required=False
    ),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="ViewHDF5")
    opt = framework.run()
    analyzer = ViewHDF5(opt.input, opt.missing_analysis)
    success = analyzer.analyze_file()
