#!/usr/bin/env python
# Import required modules
import functools
import numpy as np
import os
import pandas as pd
import re
import requests
import tempfile
from contextlib import contextmanager
from bs4 import BeautifulSoup
from intervaltree import IntervalTree
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from tqdm import tqdm
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.ConvertGeneID import ConvertGeneID
from utils.DownloadAndExtract import DownloadAndExtract
from utils.LoggingUtils import log

F = TypeVar("F", bound=Callable[..., Any])


def handle_errors(
    default_return: Any = None, error_message: str = "Operation failed"
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.error(f"{error_message}: {e}")
                return default_return

        return cast(F, wrapper)

    return decorator


@contextmanager
def temp_file_operation(suffix: str = ".csv") -> Iterator[str]:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            yield temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def validate_input_requirements(
    data: pd.DataFrame,
    required_columns: Sequence[str],
    context: str = "",
    use_aliases: bool = True,
) -> bool:
    if not use_aliases:
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            log.error(f"{context} missing required columns: {missing}")
            log.debug(f"Available columns: {list(data.columns)}")
            return False
        return True

    found_columns = []
    missing_columns = []
    for required_col in required_columns:
        found_col = AliasUtils.find_keys(dict.fromkeys(data.columns), required_col)
        if found_col:
            found_columns.append(found_col)
        else:
            missing_columns.append(required_col)

    if missing_columns:
        if context != "Silent check":
            log.error(f"{context} missing required columns: {missing_columns}")
            log.debug(f"Available columns: {list(data.columns)}")
        return False
    return True


def ensure_analysis_columns(df: pd.DataFrame, analysis_type: str) -> pd.DataFrame:
    common_gene_fields = [
        "CHR",
        "GENE",
        "GENE_ID",
        "TSS",
        "TSS_DIST",
        "NEAREST_GENE_DIST",
        "STRAND",
    ]
    required_columns = {
        "EWAS_ARRAY": [
            "CGID",
            "GENE",
            "STRAND",
            "CHR",
            "CPG_REGION",
            "BP",
        ],
        "EWAS": [
            "CGID",
            *common_gene_fields,
            "CPG_REGION",
            "BIOTYPE",
            "BP",
        ],
        "GWAS": [
            "RSID",
            *common_gene_fields,
            "BP",
            "REGULATORY_REGION",
            "REG_GENE",
            "REG_DISTANCE",
        ],
    }

    if analysis_type in required_columns:
        for col in required_columns[analysis_type]:
            if col not in df.columns:
                df[col] = None
    return df


class ChromosomeUtils:
    def __init__(self, genome_version: str) -> None:
        self.genome_version = genome_version
        self._chromosome_sizes = self.get_chromosome_sizes()

    @handle_errors(default_return={}, error_message="Error getting chromosome sizes")
    def get_chromosome_sizes(self) -> Dict[str, int]:
        if self.genome_version == "hg38":
            return {
                "1": 248956422,
                "2": 242193529,
                "3": 198295559,
                "4": 190214555,
                "5": 181538259,
                "6": 170805979,
                "7": 159345973,
                "8": 145138636,
                "9": 138394717,
                "10": 133797422,
                "11": 135086622,
                "12": 133275309,
                "13": 114364328,
                "14": 107043718,
                "15": 101991189,
                "16": 90338345,
                "17": 83257441,
                "18": 80373285,
                "19": 58617616,
                "20": 64444167,
                "21": 46709983,
                "22": 50818468,
                "X": 156040895,
                "Y": 57227415,
            }
        elif self.genome_version == "hg19":
            return {
                "1": 249250621,
                "2": 243199373,
                "3": 198022430,
                "4": 191154276,
                "5": 180915260,
                "6": 171115067,
                "7": 159138663,
                "8": 146364022,
                "9": 141213431,
                "10": 135534747,
                "11": 135006516,
                "12": 133851895,
                "13": 115169878,
                "14": 107349540,
                "15": 102531392,
                "16": 90354753,
                "17": 81195210,
                "18": 78077248,
                "19": 59128983,
                "20": 63025520,
                "21": 48129895,
                "22": 51304566,
                "X": 155270560,
                "Y": 59373566,
            }
        else:
            log.error(f"Invalid genome assembly: {self.genome_version}")
            return {}

    def standardize_chromosome_format(
        self, chrom: Any, remove_prefix: bool = False
    ) -> Any:
        if pd.isna(chrom):
            return chrom

        chrom = str(chrom).upper().strip()

        if chrom.startswith("CHR"):
            chrom_num = chrom[3:]
        else:
            chrom_num = chrom

        if chrom_num == "23":
            chrom_num = "X"
        elif chrom_num == "24":
            chrom_num = "Y"

        if remove_prefix:
            return chrom_num
        else:
            return f"CHR{chrom_num}"

    @handle_errors(
        default_return=False, error_message="Error validating chromosome column"
    )
    def validate_chromosome_column(
        self, df: pd.DataFrame, col_name: str = "CHR"
    ) -> bool:
        if col_name not in df.columns:
            found_chr_col = AliasUtils.find_keys(dict.fromkeys(df.columns), "CHR")
            if found_chr_col:
                col_name = found_chr_col
            else:
                log.error("No chromosome column found in dataframe")
                return False

        unique_chroms = df[col_name].unique()
        invalid_chroms = [
            chrom for chrom in unique_chroms if not self.is_valid_chromosome(chrom)
        ]

        if invalid_chroms:
            log.warn(f"Invalid chromosomes found: {invalid_chroms}")
            return False

        return True

    @handle_errors(default_return=False, error_message="Error validating chromosome")
    def is_valid_chromosome(self, chrom: Any) -> bool:
        if pd.isna(chrom):
            return False

        std_chrom = self.standardize_chromosome_format(chrom, remove_prefix=True)
        return std_chrom in self._chromosome_sizes


class DataProcessor:
    def __init__(self, analysis_type: str, genome_version: str = "hg38") -> None:
        self.analysis_type = analysis_type
        self.genome_version = genome_version
        self.data_standardizer = DataStandardizer(genome_version=genome_version)

    @handle_errors(
        default_return=pd.DataFrame(),
        error_message="Error extracting chromosome and position",
    )
    def extract_chr_pos(
        self, df: pd.DataFrame, analysis_type: Optional[str] = None
    ) -> pd.DataFrame:
        df = df.copy()
        analysis = analysis_type if analysis_type else self.analysis_type

        if analysis == "EWAS" and "CGID" in df.columns:
            df["CHR"] = df["CGID"].str.replace(r"_.*$", "", regex=True)
            df["BP"] = pd.to_numeric(
                df["CGID"].str.replace(r"^[^_]+_", "", regex=True), errors="coerce"
            )

            if "CHR" in df.columns:
                df = self.data_standardizer.standardize_chromosomes(df)

        return df

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error adding distance columns"
    )
    def add_distance_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in ["TSS_DIST", "NEAREST_GENE_DIST", "REG_DISTANCE"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                except Exception:
                    log.debug(f"Failed to convert {col} to Int64 type")

        for col in [
            "GENE",
            "GENE_ID",
            "TSS",
            "TSS_DIST",
            "NEAREST_GENE_DIST",
            "STRAND",
        ]:
            if col not in df.columns:
                df[col] = pd.NA

        return df

    @handle_errors(
        default_return=pd.DataFrame(),
        error_message="Error processing single chromosome",
    )
    def process_single_chromosome(
        self, chrom: str, df_data_work: pd.DataFrame, df_annot: pd.DataFrame
    ) -> pd.DataFrame:
        if df_data_work.empty:
            return pd.DataFrame(columns=df_data_work.columns)

        df_data_work = df_data_work.copy()

        if df_annot.empty:
            return ensure_analysis_columns(df_data_work, self.analysis_type)

        ew_has_strand = "STRAND" in df_data_work.columns
        annot_has_strand = "STRAND" in df_annot.columns

        if ew_has_strand and annot_has_strand:
            return self.process_stranded_data(df_data_work, df_annot)
        else:
            genomic_annotator = GenomicAnnotator(self.genome_version)
            gene_positions = np.vstack(
                [df_annot[["START"]].values, df_annot[["END"]].values]
            )
            position_to_gene_idx = np.hstack(
                [np.arange(len(df_annot)), np.arange(len(df_annot))]
            )
            kdtree = KDTree(gene_positions)
            return genomic_annotator.assign_nearest(
                df_data_work, df_annot, kdtree, position_to_gene_idx
            )

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error processing stranded data"
    )
    def process_stranded_data(
        self, ewas_df: pd.DataFrame, annot_df: pd.DataFrame
    ) -> pd.DataFrame:
        mask = ewas_df["STRAND"].notna()
        annotated_parts = []
        genomic_annotator = GenomicAnnotator(self.genome_version)

        if mask.any():
            for strand_val in ewas_df.loc[mask, "STRAND"].unique():
                ewas_strand = ewas_df[ewas_df["STRAND"] == strand_val].copy()
                annot_strand = annot_df[annot_df["STRAND"] == strand_val].copy()

                if annot_strand.empty:
                    gene_positions = np.vstack(
                        [annot_df[["START"]].values, annot_df[["END"]].values]
                    )
                    position_to_gene_idx = np.hstack(
                        [np.arange(len(annot_df)), np.arange(len(annot_df))]
                    )
                    kdtree = KDTree(gene_positions)
                    annotated_parts.append(
                        genomic_annotator.assign_nearest(
                            ewas_strand, annot_df, kdtree, position_to_gene_idx
                        )
                    )
                else:
                    gene_positions = np.vstack(
                        [annot_strand[["START"]].values, annot_strand[["END"]].values]
                    )
                    position_to_gene_idx = np.hstack(
                        [np.arange(len(annot_strand)), np.arange(len(annot_strand))]
                    )
                    kdtree = KDTree(gene_positions)
                    annotated_parts.append(
                        genomic_annotator.assign_nearest(
                            ewas_strand, annot_strand, kdtree, position_to_gene_idx
                        )
                    )

            if (~mask).any():
                ewas_no_strand = ewas_df[~mask].copy()
                gene_positions = np.vstack(
                    [annot_df[["START"]].values, annot_df[["END"]].values]
                )
                position_to_gene_idx = np.hstack(
                    [np.arange(len(annot_df)), np.arange(len(annot_df))]
                )
                kdtree = KDTree(gene_positions)
                annotated_parts.append(
                    genomic_annotator.assign_nearest(
                        ewas_no_strand, annot_df, kdtree, position_to_gene_idx
                    )
                )

            return pd.concat(annotated_parts, ignore_index=True)
        else:
            gene_positions = np.vstack(
                [annot_df[["START"]].values, annot_df[["END"]].values]
            )
            position_to_gene_idx = np.hstack(
                [np.arange(len(annot_df)), np.arange(len(annot_df))]
            )
            kdtree = KDTree(gene_positions)
            return genomic_annotator.assign_nearest(
                ewas_df, annot_df, kdtree, position_to_gene_idx
            )


class DataStandardizer:
    def __init__(
        self, genome_version: str = "hg38", chip: Optional[str] = None
    ) -> None:
        self.genome_version = genome_version
        self.chromosome_utils = ChromosomeUtils(genome_version)
        self.chip = chip
        self.last_column_mapping: Dict[str, str] = {}

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error standardizing input"
    )
    def standardize_input(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        ci_column_map = AliasUtils.find_all_standardized_ci_columns(df)
        column_mappings = {}
        all_logical_fields = list(AliasUtils.ALIASES.keys())

        for col in df.columns:
            if col in column_mappings:
                continue
            for logical_field in all_logical_fields:
                found_field = AliasUtils.find_keys({col: True}, logical_field)
                if found_field:
                    if logical_field not in column_mappings.values():
                        column_mappings[col] = logical_field
                        break

        combined_mapping = {**column_mappings, **ci_column_map}
        original_combined = combined_mapping.copy()
        if original_combined:
            df = df.rename(columns=original_combined)
            loggable_mapping = {k: v for k, v in original_combined.items() if k != v}
            if loggable_mapping:
                log.debug(f"Applied column mappings: {loggable_mapping}")
        self.last_column_mapping = original_combined or {}

        if "CHR" in df.columns:
            df = self.standardize_chromosomes(df)

        df = self.standardize_data_types(df)

        return df

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error standardizing annotation"
    )
    def standardize_annotation(
        self, df: pd.DataFrame, annotation_type: str
    ) -> pd.DataFrame:
        df = df.copy()

        if annotation_type == "array_450k":
            return self.standardize_array_manifest(df, "450k")
        elif annotation_type == "array_EPIC":
            return self.standardize_array_manifest(df, "EPIC")
        elif annotation_type == "ensembl_genes":
            return self.standardize_ensembl_annotation(df, regions=False)
        elif annotation_type == "ensembl_regions":
            return self.standardize_ensembl_annotation(df, regions=True)
        elif annotation_type == "cpg_islands":
            return self.standardize_cpg_islands(df)

        df = self.standardize_chromosomes(df)
        df = self.standardize_data_types(df)
        return df

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error standardizing chromosomes"
    )
    def standardize_chromosomes(
        self,
        df: pd.DataFrame,
        chr_column: Optional[str] = "CHR",
        target_format: str = "with_prefix",
        filter_autosomal: bool = False,
    ) -> pd.DataFrame:
        df = df.copy()

        if chr_column is None:
            chr_field = None
            for possible_field in ["CHR", "chr", "Chromosome", "chromosome", "CHROM"]:
                if possible_field in df.columns:
                    chr_field = possible_field
                    break
        else:
            chr_field = chr_column

        if chr_field is None:
            log.error("No chromosome field found in dataframe")
            return df

        if target_format == "with_prefix":
            df[chr_field] = df[chr_field].apply(
                lambda x: (
                    self.chromosome_utils.standardize_chromosome_format(
                        x, remove_prefix=False
                    )
                    if pd.notna(x)
                    else x
                )
            )
        elif target_format == "without_prefix":
            df[chr_field] = df[chr_field].apply(
                lambda x: (
                    self.chromosome_utils.standardize_chromosome_format(
                        x, remove_prefix=True
                    )
                    if pd.notna(x)
                    else x
                )
            )

        if filter_autosomal:
            if target_format == "with_prefix":
                autosomal_chromosomes = [f"CHR{i}" for i in range(1, 23)]
            else:
                autosomal_chromosomes = [str(i) for i in range(1, 23)]
            df = df[df[chr_field].isin(autosomal_chromosomes)]

        return df

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error standardizing data types"
    )
    def standardize_data_types(
        self, df: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return df

        df = df.copy()

        for col in ["START", "END", "BP", "TSS"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if df[col].notna().all():
                        df[col] = df[col].astype(np.int64)
                except Exception:
                    log.debug(f"Failed to coerce/cast column {col} to numeric/integer")
                    continue

        for col in ["TSS_DIST", "NEAREST_GENE_DIST", "REG_DISTANCE"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception:
                    log.debug(f"Failed to coerce column {col} to numeric")
                    continue

        for col in ["CHR", "STRAND", "CPG_REGION", "BIOTYPE", "REGULATORY_REGION"]:
            if col in df.columns:
                try:
                    unique_count = df[col].nunique(dropna=True)
                    total = len(df)
                    if total > 0 and unique_count < (total * 0.5):
                        df[col] = df[col].astype("category")
                except Exception:
                    log.debug(f"Failed to convert column {col} to category")

        return df

    @handle_errors(
        default_return=pd.DataFrame(),
        error_message="Error standardizing array manifest",
    )
    def standardize_array_manifest(
        self,
        manifest: Optional[pd.DataFrame],
        array_type: str = "450k",
        source: str = "downloaded",
    ) -> pd.DataFrame:
        if manifest is None or manifest.empty:
            return pd.DataFrame()

        manifest = manifest.copy()

        if source == "downloaded":
            if array_type == "450k":
                required_mappings = {
                    "CGID": ["IlmnID"],
                    "CHR": ["CHR"],
                    "BP": ["MAPINFO"],
                }
                optional_mappings = {
                    "PROBE_TYPE": ["Infinium_Design_Type"],
                    "STRAND": ["Strand"],
                    "CPG_REGION": ["Relation_to_UCSC_CpG_Island"],
                    "GENE": ["UCSC_RefGene_Name"],
                    "BIOTYPE": ["UCSC_RefGene_Group"],
                    "SNP_ID": ["Probe_SNPs"],
                }
                manifest = self.map_columns(
                    manifest, required_mappings, optional_mappings
                )
                position_col = "BP"
            else:
                available_columns = list(manifest.columns)
                id_col = "IlmnID"

                if self.genome_version == "hg38":
                    chr_col = "CHR_hg38" if "CHR_hg38" in available_columns else "CHR"
                    pos_col = (
                        "Start_hg38" if "Start_hg38" in available_columns else "MAPINFO"
                    )
                    end_col = (
                        "End_hg38" if "End_hg38" in available_columns else "MAPINFO"
                    )
                    strand_col = (
                        "Strand_hg38"
                        if "Strand_hg38" in available_columns
                        else "Strand"
                    )
                else:
                    chr_col = "CHR"
                    pos_col = "MAPINFO"
                    end_col = "MAPINFO"
                    strand_col = "Strand"

                possible_cols = [
                    id_col,
                    "Infinium_Design_Type",
                    chr_col,
                    pos_col,
                    end_col,
                    strand_col,
                    "Relation_to_UCSC_CpG_Island",
                    "UCSC_RefGene_Name",
                    "UCSC_RefGene_Group",
                    "SNP_ID",
                ]
                cols_to_use = [col for col in possible_cols if col in available_columns]
                manifest = manifest[cols_to_use].copy()

                column_mapping = {
                    id_col: "CGID",
                    chr_col: "CHR",
                    pos_col: "START",
                }
                if end_col in cols_to_use:
                    column_mapping[end_col] = "END"
                if strand_col in cols_to_use:
                    column_mapping[strand_col] = "STRAND"
                if "Infinium_Design_Type" in cols_to_use:
                    column_mapping["Infinium_Design_Type"] = "PROBE_TYPE"
                if "Relation_to_UCSC_CpG_Island" in cols_to_use:
                    column_mapping["Relation_to_UCSC_CpG_Island"] = "CPG_REGION"
                if "UCSC_RefGene_Name" in cols_to_use:
                    column_mapping["UCSC_RefGene_Name"] = "GENE"
                if "UCSC_RefGene_Group" in cols_to_use:
                    column_mapping["UCSC_RefGene_Group"] = "BIOTYPE"
                if "SNP_ID" in cols_to_use:
                    column_mapping["SNP_ID"] = "SNP_ID"

                manifest = manifest.rename(columns=column_mapping)

                if "START" in manifest.columns:
                    manifest["BP"] = manifest["START"].copy()
                elif pos_col in manifest.columns:
                    manifest["BP"] = manifest[pos_col].copy()

                position_col = "START"
        else:
            if array_type == "450k":
                rename_dict = {
                    "IlmnID": "CGID",
                    "CHR": "CHR",
                    "MAPINFO": "BP",
                }
                if "Infinium_Design_Type" in manifest.columns:
                    rename_dict["Infinium_Design_Type"] = "PROBE_TYPE"
                if "STRAND" in manifest.columns:
                    rename_dict["STRAND"] = "STRAND"
                if "Relation_to_UCSC_CpG_Island" in manifest.columns:
                    rename_dict["Relation_to_UCSC_CpG_Island"] = "CPG_REGION"
                if "UCSC_RefGENE_Name" in manifest.columns:
                    rename_dict["UCSC_RefGENE_Name"] = "GENE"
                if "UCSC_RefGENE_Group" in manifest.columns:
                    rename_dict["UCSC_RefGENE_Group"] = "BIOTYPE"
                if "Probe_SNPs" in manifest.columns:
                    rename_dict["Probe_SNPs"] = "SNP_ID"

                manifest = manifest.rename(columns=rename_dict)
                position_col = "BP"
            else:
                if self.genome_version == "hg38":
                    start_col = (
                        "Start_hg38" if "Start_hg38" in manifest.columns else "MAPINFO"
                    )
                    end_col = (
                        "End_hg38" if "End_hg38" in manifest.columns else "MAPINFO"
                    )
                    chr_col = "CHR_hg38" if "CHR_hg38" in manifest.columns else "CHR"
                    strand_col = (
                        "Strand_hg38" if "Strand_hg38" in manifest.columns else "Strand"
                    )
                else:
                    start_col = "MAPINFO"
                    end_col = "MAPINFO"
                    chr_col = "CHR"
                    strand_col = "Strand"

                required_cols = ["IlmnID", chr_col, start_col]
                optional_cols = [
                    end_col,
                    strand_col,
                    "Infinium_Design_Type",
                    "Relation_to_UCSC_CpG_Island",
                    "UCSC_RefGENE_Name",
                    "UCSC_RefGENE_Group",
                    "SNP_ID",
                ]
                cols_to_keep = [
                    col
                    for col in required_cols + optional_cols
                    if col in manifest.columns
                ]
                manifest = manifest[cols_to_keep].copy()

                rename_dict = {"IlmnID": "CGID", chr_col: "CHR", start_col: "START"}
                if end_col in manifest.columns:
                    rename_dict[end_col] = "END"
                if strand_col in manifest.columns:
                    rename_dict[strand_col] = "STRAND"
                if "Infinium_Design_Type" in manifest.columns:
                    rename_dict["Infinium_Design_Type"] = "PROBE_TYPE"
                if "Relation_to_UCSC_CpG_Island" in manifest.columns:
                    rename_dict["Relation_to_UCSC_CpG_Island"] = "CPG_REGION"
                if "UCSC_RefGENE_Name" in manifest.columns:
                    rename_dict["UCSC_RefGENE_Name"] = "GENE"
                if "UCSC_RefGENE_Group" in manifest.columns:
                    rename_dict["UCSC_RefGENE_Group"] = "BIOTYPE"
                if "SNP_ID" in manifest.columns:
                    rename_dict["SNP_ID"] = "SNP_ID"

                manifest = manifest.rename(columns=rename_dict)
                manifest["BP"] = manifest["START"]
                position_col = "START"

        if position_col in manifest.columns:
            manifest[position_col] = pd.to_numeric(
                manifest[position_col], errors="coerce"
            )
            manifest.dropna(subset=[position_col], inplace=True)

        if "CPG_REGION" in manifest.columns:
            manifest["CPG_REGION"] = (
                manifest["CPG_REGION"]
                .replace(
                    {
                        "N_Shelf": "Shelf",
                        "S_Shelf": "Shelf",
                        "N_Shore": "Shore",
                        "S_Shore": "Shore",
                    }
                )
                .fillna("Open Sea")
            )

        if "CHR" in manifest.columns:
            manifest = self.standardize_chromosomes(
                manifest,
                chr_column="CHR",
                target_format="with_prefix",
                filter_autosomal=True,
            )

        if array_type == "EPIC":
            if "END" in manifest.columns:
                end_cols = [col for col in manifest.columns if col == "END"]
                if len(end_cols) > 1:
                    log.debug(
                        f"Found {len(end_cols)} duplicate END columns, keeping first one"
                    )
                    manifest = manifest.loc[:, ~manifest.columns.duplicated()]
                manifest["END"] = pd.to_numeric(manifest["END"], errors="coerce")
                manifest.dropna(subset=["END"], inplace=True)
            elif "START" in manifest.columns:
                manifest["END"] = pd.to_numeric(manifest["START"], errors="coerce")
                manifest.dropna(subset=["END"], inplace=True)

        manifest = self.standardize_data_types(manifest)
        return manifest.drop_duplicates().reset_index(drop=True)

    @handle_errors(
        default_return=pd.DataFrame(),
        error_message="Error standardizing ENSEMBL annotation",
    )
    def standardize_ensembl_annotation(
        self, df: Optional[pd.DataFrame], regions: bool = False
    ) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return df

        df = df.copy()

        chr_field = self.find_annotation_field(df, "CHR")
        gene_field = self.find_annotation_field(df, "GENE")
        gene_id_field = self.find_annotation_field(df, "GENE_ID")
        tss_field = self.find_annotation_field(df, "TSS")
        start_field = self.find_annotation_field(df, "START")
        end_field = self.find_annotation_field(df, "END")
        strand_field = self.find_annotation_field(df, "STRAND")

        if not all([chr_field, start_field, end_field]):
            log.error("Missing required fields in ENSEMBL annotation data")
            return df

        column_mapping = {
            chr_field: "CHR",
            start_field: "START",
            end_field: "END",
        }

        if gene_field:
            column_mapping[gene_field] = "GENE"
        if gene_id_field:
            column_mapping[gene_id_field] = "GENE_ID"
        if tss_field:
            column_mapping[tss_field] = "TSS"
        if strand_field:
            column_mapping[strand_field] = "STRAND"

        if regions:
            biotype_field = self.find_annotation_field(df, "BIOTYPE")
            if biotype_field:
                column_mapping[biotype_field] = "BIOTYPE"

        df = df.rename(columns=column_mapping)
        df = self.standardize_chromosomes(df, target_format="with_prefix")
        df = self.standardize_data_types(df)

        return df

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error standardizing CpG islands"
    )
    def standardize_cpg_islands(
        self, df: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return df

        df = df.copy()

        chr_field = self.find_annotation_field(df, "CHR")
        start_field = self.find_annotation_field(df, "START")
        end_field = self.find_annotation_field(df, "END")
        region_field = self.find_annotation_field(df, "CPG_REGION")

        if not all([chr_field, start_field, end_field]):
            log.error("Missing required fields in CpG islands annotation data")
            return df

        column_mapping = {
            chr_field: "CHR",
            start_field: "START",
            end_field: "END",
        }

        if region_field:
            column_mapping[region_field] = "CPG_REGION"

        df = df.rename(columns=column_mapping)

        if "CPG_REGION" not in df.columns:
            df["CPG_REGION"] = "Island"

        df = self.standardize_chromosomes(df, target_format="with_prefix")
        df = self.standardize_data_types(df)

        return df

    @handle_errors(
        default_return=pd.DataFrame(),
        error_message="Error standardizing output columns",
    )
    def standardize_output_columns(
        self, df: Optional[pd.DataFrame], analysis_type: Optional[str] = None
    ) -> pd.DataFrame:
        if df is None or len(df) == 0:
            log.error("Empty dataframe passed to standardize_output_columns")
            return pd.DataFrame()

        df = df.copy()
        df = self._resolve_duplicate_columns(df)

        column_mappings = {}
        for col in df.columns:
            for logical_field in AliasUtils.ALIASES.keys():
                if AliasUtils.find_keys({col: True}, logical_field):
                    if logical_field not in column_mappings.values():
                        column_mappings[col] = logical_field
                        break

        if column_mappings:
            df = df.rename(columns=column_mappings)

        if analysis_type:
            df = ensure_analysis_columns(df, analysis_type)

        if analysis_type == "GWAS":
            for col in ["CPG_REGION", "PROBE_TYPE"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    log.debug(f"Removed GWAS-incompatible column {col} from output")

        df = self.standardize_data_types(df)

        if analysis_type and hasattr(self, "chip") and self.chip in ["450k", "EPIC"]:
            df = ensure_analysis_columns(df, "EWAS_ARRAY")
            methylseq_only_columns = [
                "GENE_ID",
                "TSS",
                "TSS_DIST",
                "BIOTYPE",
                "NEAREST_GENE_DIST",
            ]
            for col in methylseq_only_columns:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    log.debug(
                        f"Removed MethylSeq-specific column {col} from array standardization"
                    )

        return df

    def _resolve_duplicate_columns(
        self, df: pd.DataFrame, prefer_y: bool = False
    ) -> pd.DataFrame:
        df = df.copy()
        cols = list(df.columns)

        for col in cols:
            if (col.endswith("_x") or col.endswith("_y")) and col[:-2] not in cols:
                base_col = col[:-2]
                if col.endswith("_x") and f"{base_col}_y" not in cols:
                    df[base_col] = df[col]
                    df = df.drop(col, axis=1)
                elif col.endswith("_y") and f"{base_col}_x" not in cols:
                    df[base_col] = df[col]
                    df = df.drop(col, axis=1)

        cols = list(df.columns)

        if prefer_y:
            for col in cols:
                if col.endswith("_x"):
                    base_col = col[:-2]
                    y_col = f"{base_col}_y"
                    if y_col in cols:
                        df[base_col] = df[y_col].fillna(df[col])
                        df = df.drop([col, y_col], axis=1)
        else:
            for col in cols:
                if col.endswith("_x"):
                    base_col = col[:-2]
                    y_col = f"{base_col}_y"
                    if y_col in cols:
                        df[base_col] = df[col].fillna(df[y_col])
                        df = df.drop([col, y_col], axis=1)

        df = df.loc[:, ~df.columns.duplicated()]

        return df

    @handle_errors(default_return=None, error_message="Error finding annotation field")
    def find_annotation_field(
        self, df: pd.DataFrame, logical_field: str
    ) -> Optional[str]:
        found_field = AliasUtils.find_keys(dict.fromkeys(df.columns), logical_field)
        if found_field:
            log.debug(
                f"Found field {logical_field} as column {found_field} using AliasUtils"
            )
            return found_field

        log.error(f"Annotation data missing required field: {logical_field}")
        log.debug(f"Available columns: {list(df.columns)}")
        return None

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error merging annotation data"
    )
    def merge_annotation(
        self,
        data: pd.DataFrame,
        annotation_data: pd.DataFrame,
        on: str,
        annotation_fields: Sequence[str],
    ) -> pd.DataFrame:
        if on not in data.columns:
            log.error(f"Merge key {on} not found in main dataset")
            return data

        merge_cols = [on] + [
            col for col in annotation_fields if col in annotation_data.columns
        ]

        if len(merge_cols) <= 1:
            log.error("No annotation columns found for merging")
            return data

        result = pd.merge(data, annotation_data[merge_cols], on=on, how="left")
        result = self._resolve_duplicate_columns(result, prefer_y=True)

        return result

    @handle_errors(default_return=pd.DataFrame(), error_message="Error mapping columns")
    def map_columns(
        self,
        df: pd.DataFrame,
        required_mappings: Mapping[str, Sequence[str]],
        optional_mappings: Optional[Mapping[str, Sequence[str]]] = None,
        silent: bool = False,
    ) -> pd.DataFrame:
        if optional_mappings is None:
            optional_mappings = {}

        column_mapping = {}

        for logical_name, possible_names in required_mappings.items():
            found = False
            for name in possible_names:
                if name in df.columns:
                    column_mapping[name] = logical_name
                    found = True
                    break
            if not found and not silent:
                log.error(f"Required column not found: {logical_name}")
                log.debug(f"Expected one of: {possible_names}")

        for logical_name, possible_names in optional_mappings.items():
            for name in possible_names:
                if name in df.columns:
                    column_mapping[name] = logical_name
                    break

        return df.rename(columns=column_mapping) if column_mapping else df

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error mapping array columns"
    )
    def map_array_columns(
        self, manifest: pd.DataFrame, array_type: str
    ) -> pd.DataFrame:
        required_mappings = {}

        cgid_col = AliasUtils.find_keys(dict.fromkeys(manifest.columns), "CGID")
        if cgid_col:
            required_mappings["IlmnID"] = [cgid_col]

        chr_col = AliasUtils.find_keys(dict.fromkeys(manifest.columns), "CHR")
        if chr_col:
            required_mappings["CHR"] = [chr_col]

        strand_col = AliasUtils.find_keys(dict.fromkeys(manifest.columns), "STRAND")
        if strand_col:
            required_mappings["STRAND"] = [strand_col]

        if array_type == "450k":
            pos_col = AliasUtils.find_keys(dict.fromkeys(manifest.columns), "BP")
            if pos_col:
                required_mappings["MAPINFO"] = [pos_col]
        else:
            start_col = AliasUtils.find_keys(dict.fromkeys(manifest.columns), "START")
            if start_col:
                required_mappings["Start"] = [start_col]
            end_col = AliasUtils.find_keys(dict.fromkeys(manifest.columns), "END")
            if end_col:
                required_mappings["End"] = [end_col]

        optional_mappings = {}

        region_col = AliasUtils.find_keys(dict.fromkeys(manifest.columns), "CPG_REGION")
        if region_col:
            optional_mappings["Relation_to_UCSC_CpG_Island"] = [region_col]

        gene_col = AliasUtils.find_keys(dict.fromkeys(manifest.columns), "GENE")
        if gene_col:
            optional_mappings["UCSC_RefGene_Name"] = [gene_col]

        region_type_col = AliasUtils.find_keys(
            dict.fromkeys(manifest.columns), "BIOTYPE"
        )
        if region_type_col:
            optional_mappings["UCSC_RefGene_Group"] = [region_type_col]

        return self.map_columns(manifest, required_mappings, optional_mappings)


class BaseAnnotator:
    def __init__(self, genome_version: str, reference: Optional[str] = None) -> None:
        self.genome_version = genome_version
        self.reference = reference
        self.data_standardizer = DataStandardizer(genome_version=genome_version)
        self.chromosome_utils = ChromosomeUtils(genome_version)
        self.resource_manager = ResourceManager(
            reference, genome_version, self.data_standardizer
        )
        self.resource_manager.chromosome_utils = self.chromosome_utils

    def process_by_chromosome(
        self,
        data: pd.DataFrame,
        annotation: pd.DataFrame,
        process_func: Callable[[Any, pd.DataFrame, pd.DataFrame], pd.DataFrame],
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        chroms = list(data["CHR"].unique())
        log.info(f"Processing {len(chroms)} chromosomes in parallel")
        try:
            results = Parallel(n_jobs=n_jobs, verbose=0, backend="threading")(
                delayed(process_func)(
                    chrom,
                    data[data["CHR"] == chrom],
                    annotation[annotation["CHR"] == chrom],
                )
                for chrom in chroms
            )
        except Exception as e:
            log.warn(
                f"Parallel processing failed ({e}), falling back to serial processing"
            )
            results = []
            for chrom in chroms:
                try:
                    res = process_func(
                        chrom,
                        data[data["CHR"] == chrom],
                        annotation[annotation["CHR"] == chrom],
                    )
                    results.append(res)
                except Exception as ex:
                    log.error(f"Processing chromosome {chrom} failed: {ex}")
                    results.append(pd.DataFrame())

        if not results or all(
            (isinstance(df, pd.DataFrame) and df.empty) or df is None for df in results
        ):
            return pd.DataFrame()
        dfs = [df for df in results if isinstance(df, pd.DataFrame) and not df.empty]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)


class ArrayAnnotator(BaseAnnotator):
    def __init__(
        self, array_type: str, genome_version: str, reference: Optional[str] = None
    ) -> None:
        super().__init__(genome_version, reference)
        self.array_type = array_type
        self.data_standardizer.chip = array_type

    @handle_errors(default_return=False, error_message="Error getting array annotation")
    def get_array_annotation(
        self, output_path: str, array_type: Optional[str] = None
    ) -> bool:
        if array_type is None:
            array_type = self.array_type

        log.info(f"Getting {array_type.upper()} array annotation")

        got = self.resource_manager.download_and_process_array(output_path, array_type)
        if got and os.path.exists(output_path):
            try:
                manifest_df = pd.read_csv(output_path, low_memory=False)
                self._cached_manifest = manifest_df
                log.debug(
                    f"Cached array manifest ({array_type}) with {len(manifest_df)} rows"
                )
            except Exception as e:
                log.debug(f"Could not read/cache manifest at {output_path}: {e}")
        return got

    def get_cached_manifest(self) -> Optional[pd.DataFrame]:
        return getattr(self, "_cached_manifest", None)

    @handle_errors(
        default_return=pd.DataFrame(),
        error_message="Error processing methylseq annotation",
    )
    def process_methylseq(
        self, data: pd.DataFrame, annotation_data: pd.DataFrame, analysis_type: str
    ) -> pd.DataFrame:
        id_column = "CGID"
        if id_column not in data.columns:
            log.error(f"Input data missing required {id_column} column")
            return ensure_analysis_columns(data, analysis_type)

        data_processor = DataProcessor(analysis_type)

        df_data_work = data.copy()
        df_data_work = df_data_work[
            ~df_data_work["CGID"].str.contains("random|hap", na=False)
        ]

        if "CHR" not in df_data_work.columns or "BP" not in df_data_work.columns:
            df_data_work = data_processor.extract_chr_pos(df_data_work)

        df_annot = self._prepare_annotation_data(annotation_data)

        total_positions = len(df_data_work)
        log.debug(f"Total positions to annotate: {total_positions}")

        pbar = tqdm(total=total_positions, desc="Annotating genes")

        def process_chrom_func(
            chrom: Any, data_chrom: pd.DataFrame, annot_chrom: pd.DataFrame
        ) -> pd.DataFrame:
            result = data_processor.process_single_chromosome(
                chrom, data_chrom, annot_chrom
            )
            pbar.update(len(data_chrom))
            return result

        processed_results = self.process_by_chromosome(
            df_data_work, df_annot, process_chrom_func
        )

        pbar.close()

        if not processed_results.empty:
            processed_results = processed_results.drop_duplicates(subset=["CGID"])
            result_columns = [id_column]
            for col in [
                "GENE",
                "GENE_ID",
                "TSS",
                "TSS_DIST",
                "NEAREST_GENE_DIST",
                "STRAND",
            ]:
                if col in processed_results.columns:
                    result_columns.append(col)
            final_results = pd.merge(
                data, processed_results[result_columns], on=id_column, how="left"
            )
            return ensure_analysis_columns(final_results, analysis_type)
        else:
            return ensure_analysis_columns(data, analysis_type)

    def _prepare_annotation_data(self, annotation_data: pd.DataFrame) -> pd.DataFrame:
        df_annot = annotation_data.copy()

        expected_cols = ["CHR", "GENE", "START", "END"]
        if all(col in df_annot.columns for col in expected_cols):
            log.debug("Annotation data already in standardized format")
            return df_annot

        found_cols = {}
        for field in ["CHR", "GENE", "GENE_ID", "TSS", "START", "END", "STRAND"]:
            found = self.data_standardizer.find_annotation_field(df_annot, field)
            if found and found != field:
                found_cols[found] = field

        if found_cols:
            df_annot = df_annot.rename(columns=found_cols)
            log.debug(f"Applied additional column mapping: {found_cols}")

        return df_annot

    def get_expected_array_columns(self, array_type: str) -> List[str]:
        tail = ["STRAND", "PROBE_TYPE", "CPG_REGION", "GENE", "BIOTYPE"]
        if array_type == "450k":
            return ["CGID", "CHR", "BP"] + tail
        return ["CGID", "CHR", "START", "END"] + tail + ["SNP_ID"]


class GenomicAnnotator(BaseAnnotator):
    def __init__(
        self,
        genome_version: str,
        protein_coding: bool = True,
        reference: Optional[str] = None,
    ) -> None:
        super().__init__(genome_version, reference)
        self.protein_coding = protein_coding
        self._cached_gtf_path = None
        self._cached_gtf_dir = None

    @handle_errors(default_return=False, error_message="ENSEMBL annotation failed")
    def get_ensembl_annotation(self, output: str, regions: bool = False) -> bool:
        log.debug(f"Getting ENSEMBL annotation, regions={regions}")

        condition1 = self.reference is not None
        condition2 = os.path.exists(self.reference)
        condition3 = self.reference.endswith(".gtf")
        condition4 = self.reference.endswith(".gtf.gz")
        condition3_4 = condition3 or condition4
        if condition1 and condition2 and condition3_4:
            gtf_path = self.reference
            log.debug(f"Using user-provided GTF: {gtf_path}")

            if regions:
                self.create_genomic_regions(
                    gtf_path, output, protein_coding=self.protein_coding
                )
            else:
                self.create_gene_annotation(
                    gtf_path, output, protein_coding=self.protein_coding
                )
            return True

        rm_cached = getattr(self.resource_manager, "_cached_gtf_path", None)
        if rm_cached and os.path.exists(rm_cached):
            log.debug(f"Using ResourceManager's cached GTF: {rm_cached}")

            if regions:
                self.create_genomic_regions(
                    rm_cached, output, protein_coding=self.protein_coding
                )
            else:
                self.create_gene_annotation(
                    rm_cached, output, protein_coding=self.protein_coding
                )
            return True

        if self._cached_gtf_path is not None and os.path.exists(self._cached_gtf_path):
            log.debug(f"Using GenomicAnnotator's cached GTF: {self._cached_gtf_path}")

            if regions:
                self.create_genomic_regions(
                    self._cached_gtf_path, output, protein_coding=self.protein_coding
                )
            else:
                self.create_gene_annotation(
                    self._cached_gtf_path, output, protein_coding=self.protein_coding
                )
            return True

        log.debug("No cached GTF found, downloading...")
        success = self.resource_manager.download_and_process_ensembl(
            output, regions=regions
        )

        if not success:
            log.error("Failed to download ENSEMBL annotations")
            return False

        rm_cached = getattr(self.resource_manager, "_cached_gtf_path", None)
        if not rm_cached or not os.path.exists(rm_cached):
            log.error("ResourceManager did not cache GTF path after download")
            return False

        self._cached_gtf_path = rm_cached
        self._cached_gtf_dir = getattr(self.resource_manager, "_cached_gtf_dir", None)

        log.debug(f"Using downloaded GTF: {rm_cached}")

        if regions:
            self.create_genomic_regions(
                rm_cached, output, protein_coding=self.protein_coding
            )
        else:
            self.create_gene_annotation(
                rm_cached, output, protein_coding=self.protein_coding
            )

        return True

    @handle_errors(
        default_return=False, error_message="ENSEMBL regulatory annotation failed"
    )
    def get_ensembl_regulatory_annotation(self, output: str) -> bool:
        log.debug("Getting ENSEMBL regulatory regions annotation")
        success = self.resource_manager.get_resource("regulatory", output)

        if success:
            condition1 = self.reference is not None
            condition2 = os.path.exists(self.reference)
            condition3 = self.reference.endswith(".gtf")
            condition4 = self.reference.endswith(".gtf.gz")
            condition3_4 = condition3 or condition4
            if condition1 and condition2 and condition3_4:
                self.create_regulatory_regions(
                    self.reference, output, protein_coding=self.protein_coding
                )
            else:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    gtf_url, gtf_filename = (
                        self.resource_manager.get_gtf_url_and_filename()
                    )
                    local_gtf = os.path.join(tmpdirname, gtf_filename)

                    gtf_path = local_gtf
                    if not os.path.exists(gtf_path):
                        if not self.resource_manager._download_file(
                            gtf_url,
                            gtf_path,
                            f"Downloading {self.genome_version} ENSEMBL annotations",
                        ):
                            return False

                    if not os.path.exists(gtf_path):
                        base_path, ext = os.path.splitext(gtf_path)
                        if ext == ".gz" and os.path.exists(base_path):
                            gtf_path = base_path
                        elif ext == ".gz" and os.path.exists(gtf_path[:-3]):
                            gtf_path = gtf_path[:-3]
                        else:
                            log.error(
                                f"Downloaded GTF file not found at expected location: {gtf_path}"
                            )
                            return False

                    self.create_regulatory_regions(
                        gtf_path, output, protein_coding=self.protein_coding
                    )

        return success

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error processing genomic regions"
    )
    def process_genomic_regions(
        self, data: pd.DataFrame, annotation_data: pd.DataFrame
    ) -> pd.DataFrame:
        if not validate_input_requirements(data, ["CGID"], "Genomic regions input"):
            return ensure_analysis_columns(data.copy(), "EWAS")

        data_processor = DataProcessor("EWAS")

        df_data_work = data.copy()
        df_data_work = df_data_work[
            ~df_data_work["CGID"].str.contains("random|hap", na=False)
        ]

        if "CHR" not in df_data_work.columns or "BP" not in df_data_work.columns:
            df_data_work = data_processor.extract_chr_pos(df_data_work, "EWAS")

        df_annot = self._prepare_annotation_data(
            annotation_data, ["CHR", "START", "END", "BIOTYPE"]
        )

        biotype_priority = {
            "5' UTR": 1,
            "3' UTR": 2,
            "promoter": 3,
            "TSS": 4,
            "exon": 5,
            "intron": 6,
            "intergenic": 7,
        }

        total_positions = len(df_data_work)
        log.debug(f"Total positions to annotate: {total_positions}")

        pbar = tqdm(total=total_positions, desc="Annotating genomic regions")

        def process_chrom_func(
            chrom: Any, data_chrom: pd.DataFrame, annot_chrom: pd.DataFrame
        ) -> pd.DataFrame:
            if annot_chrom.empty or data_chrom.empty:
                pbar.update(len(data_chrom))
                return pd.DataFrame({"CGID": data_chrom["CGID"], "BIOTYPE": None})

            tree = IntervalTree()

            starts = annot_chrom["START"].values
            ends = annot_chrom["END"].values
            biotypes = (
                annot_chrom["BIOTYPE"].values
                if "BIOTYPE" in annot_chrom.columns
                else np.array(["unknown"] * len(annot_chrom))
            )

            for idx in range(len(annot_chrom)):
                tree.addi(starts[idx], ends[idx] + 1, biotypes[idx])

            positions = data_chrom["BP"].values
            cgids = data_chrom["CGID"].values

            result_biotypes = np.empty(len(positions), dtype=object)

            for i in range(len(positions)):
                pos = positions[i]
                overlaps = tree.at(pos)

                if overlaps:
                    matched_biotypes = [interval.data for interval in overlaps]
                    unique_biotypes = list(set(matched_biotypes))
                    result_biotypes[i] = min(
                        unique_biotypes, key=lambda x: biotype_priority.get(x, 100)
                    )
                else:
                    result_biotypes[i] = None

                pbar.update(1)

            return pd.DataFrame({"CGID": cgids, "BIOTYPE": result_biotypes})

        biotype_results = self.process_by_chromosome(
            df_data_work, df_annot, process_chrom_func
        )

        pbar.close()

        if not biotype_results.empty:
            final_results = pd.merge(data, biotype_results, on="CGID", how="left")
            return ensure_analysis_columns(final_results, "EWAS")
        else:
            return ensure_analysis_columns(data.copy(), "EWAS")

    def _prepare_annotation_data(
        self, annotation_data: pd.DataFrame, required_fields: Sequence[str]
    ) -> pd.DataFrame:
        df_annot = annotation_data.copy()

        missing_fields = [
            field for field in required_fields if field not in df_annot.columns
        ]

        if missing_fields:
            found_cols = {}
            for field in missing_fields:
                found = self.data_standardizer.find_annotation_field(df_annot, field)
                if found:
                    found_cols[found] = field

            if found_cols:
                df_annot = df_annot.rename(columns=found_cols)
                log.debug(f"Applied column mapping for annotation data: {found_cols}")

        return df_annot

    @handle_errors(
        default_return=False, error_message="Creating gene annotation from GTF"
    )
    def create_gene_annotation(
        self, gtf_path: str, output: str, protein_coding: Optional[bool] = None
    ) -> bool:
        if protein_coding is None:
            protein_coding = self.protein_coding

        log.debug("Creating gene annotation")
        genes = []
        gene_tss = {}

        with open(gtf_path, "r") as file:
            for line in tqdm(file, desc="Parsing GTF for genes"):
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9:
                    continue

                chrom, source, feature, start, end, score, strand, frame, attributes = (
                    fields
                )

                if feature != "gene":
                    continue

                gene_id = re.search(r'gene_id "([^"]+)"', attributes)
                gene_name = re.search(r'gene_name "([^"]+)"', attributes)
                gene_biotype = re.search(r'gene_biotype "([^"]+)"', attributes)

                if not (gene_id and gene_name):
                    continue

                gene_id = gene_id.group(1)
                gene_name = gene_name.group(1)
                gene_biotype = gene_biotype.group(1) if gene_biotype else "unknown"

                if protein_coding and gene_biotype != "protein_coding":
                    continue

                tss = int(start) if strand == "+" else int(end)

                genes.append(
                    {
                        "CHR": chrom,
                        "GENE_ID": gene_id,
                        "GENE": gene_name,
                        "BIOTYPE": gene_biotype,
                        "START": int(start),
                        "END": int(end),
                        "STRAND": strand,
                        "TSS": tss,
                    }
                )

                gene_tss[gene_id] = tss

        df_genes = pd.DataFrame(genes)

        if not df_genes.empty:
            df_genes = self.data_standardizer.standardize_chromosomes(
                df_genes, chr_column="CHR", target_format="with_prefix"
            )
            df_genes.to_csv(output, index=False)
            log.info(f"Created gene annotation with {len(df_genes)} genes")
        else:
            log.warn("No genes found in GTF file")
            pd.DataFrame(
                columns=[
                    "CHR",
                    "GENE_ID",
                    "GENE",
                    "BIOTYPE",
                    "START",
                    "END",
                    "STRAND",
                    "TSS",
                ]
            ).to_csv(output, index=False)
        return True

    @handle_errors(
        default_return=False, error_message="Creating genomic regions from GTF"
    )
    def create_genomic_regions(
        self, gtf_path: str, output: str, protein_coding: Optional[bool] = None
    ) -> bool:
        if protein_coding is None:
            protein_coding = self.protein_coding
        log.debug("Creating genomic regions annotation")
        regions = []
        gene_info = {}
        gene_exons = {}

        utr5_aliases = set(a.lower() for a in AliasUtils.get_aliases("5' UTR"))
        utr3_aliases = set(a.lower() for a in AliasUtils.get_aliases("3' UTR"))
        exon_aliases = set(a.lower() for a in AliasUtils.get_aliases("Exon"))

        log.debug(f"5' UTR aliases: {utr5_aliases}")
        log.debug(f"3' UTR aliases: {utr3_aliases}")
        log.debug(f"Exon aliases: {exon_aliases}")

        with open(gtf_path, "r") as file:
            for line in tqdm(file, desc="Parsing GTF for gene info"):
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9:
                    continue
                chrom, source, feature, start, end, score, strand, frame, attributes = (
                    fields
                )

                if feature == "gene":
                    gene_id = re.search(r'gene_id "([^"]+)"', attributes)
                    gene_name = re.search(r'gene_name "([^"]+)"', attributes)
                    gene_biotype = re.search(r'gene_biotype "([^"]+)"', attributes)
                    if gene_id and gene_name and gene_biotype:
                        gene_id = gene_id.group(1)
                        gene_name = gene_name.group(1)
                        gene_biotype = gene_biotype.group(1)
                        if protein_coding and gene_biotype != "protein_coding":
                            continue
                        gene_info[gene_id] = {
                            "name": gene_name,
                            "BIOTYPE": gene_biotype,
                            "CHR": chrom,
                            "START": int(start),
                            "END": int(end),
                            "STRAND": strand,
                        }
                        gene_exons[gene_id] = []

        with open(gtf_path, "r") as file:
            for line in tqdm(file, desc="Identifying genomic regions"):
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9:
                    continue
                chrom, source, feature, start, end, score, strand, frame, attributes = (
                    fields
                )
                start, end = int(start), int(end)

                gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
                if not gene_id_match:
                    continue
                gene_id = gene_id_match.group(1)
                if gene_id not in gene_info:
                    continue

                gene = gene_info[gene_id]
                feature_lower = feature.lower()

                if feature == "gene":
                    if strand == "+":
                        promoter_start = max(1, gene["START"] - 2000)
                        promoter_end = gene["START"]
                    else:
                        promoter_start = gene["END"]
                        promoter_end = gene["END"] + 2000
                    regions.append(
                        {
                            "CHR": chrom,
                            "START": promoter_start,
                            "END": promoter_end,
                            "BIOTYPE": "promoter",
                            "GENE_ID": gene_id,
                            "GENE": gene["name"],
                        }
                    )

                elif feature_lower in exon_aliases or feature == "exon":
                    regions.append(
                        {
                            "CHR": chrom,
                            "START": start,
                            "END": end,
                            "BIOTYPE": "exon",
                            "GENE_ID": gene_id,
                            "GENE": gene["name"],
                        }
                    )
                    gene_exons[gene_id].append((start, end))

                elif feature_lower in utr5_aliases:
                    regions.append(
                        {
                            "CHR": chrom,
                            "START": start,
                            "END": end,
                            "BIOTYPE": "5' UTR",
                            "GENE_ID": gene_id,
                            "GENE": gene["name"],
                        }
                    )

                elif feature_lower in utr3_aliases:
                    regions.append(
                        {
                            "CHR": chrom,
                            "START": start,
                            "END": end,
                            "BIOTYPE": "3' UTR",
                            "GENE_ID": gene_id,
                            "GENE": gene["name"],
                        }
                    )

                elif feature == "start_codon":
                    if strand == "+":
                        tss_start = max(1, start - 1000)
                        tss_end = end + 1000
                    else:
                        tss_start = max(1, start - 1000)
                        tss_end = end + 1000
                    regions.append(
                        {
                            "CHR": chrom,
                            "START": tss_start,
                            "END": tss_end,
                            "BIOTYPE": "TSS",
                            "GENE_ID": gene_id,
                            "GENE": gene["name"],
                        }
                    )

        log.debug("Identifying introns")
        for gene_id, exon_list in gene_exons.items():
            if len(exon_list) < 2:
                continue

            gene = gene_info[gene_id]
            exon_list_sorted = sorted(exon_list, key=lambda x: x[0])

            for i in range(len(exon_list_sorted) - 1):
                exon_end = exon_list_sorted[i][1]
                next_exon_start = exon_list_sorted[i + 1][0]

                if next_exon_start > exon_end + 1:
                    regions.append(
                        {
                            "CHR": gene["CHR"],
                            "START": exon_end + 1,
                            "END": next_exon_start - 1,
                            "BIOTYPE": "intron",
                            "GENE_ID": gene_id,
                            "GENE": gene["name"],
                        }
                    )

        log.debug("Identifying intergenic regions")
        self.add_intergenic_regions(regions, gene_info)

        df_regions = pd.DataFrame(regions)
        if not df_regions.empty:
            unique_biotypes = df_regions["BIOTYPE"].unique()
            biotype_counts = df_regions["BIOTYPE"].value_counts()
            log.info(f"Found biotypes: {sorted(unique_biotypes)}")
            log.info(f"Biotype counts:\n{biotype_counts}")

            df_regions = self.data_standardizer.standardize_chromosomes(
                df_regions, chr_column="CHR", target_format="with_prefix"
            )
            df_regions.to_csv(output, index=False)
            log.debug(
                f"Created genomic regions annotation with {len(df_regions)} entries"
            )
        else:
            log.error("No genomic regions found")
            pd.DataFrame(
                columns=["CHR", "START", "END", "BIOTYPE", "GENE_ID", "GENE"]
            ).to_csv(output, index=False)
        return True

    def add_intergenic_regions(
        self, regions: List[Dict[str, Any]], gene_info: Dict[str, Dict[str, Any]]
    ) -> None:
        chrom_sizes = self.chromosome_utils.get_chromosome_sizes()
        chromosome_genes = {}

        for gene_id, gene in gene_info.items():
            chrom = gene["CHR"]
            if chrom not in chromosome_genes:
                chromosome_genes[chrom] = []
            chromosome_genes[chrom].append((gene["START"], gene["END"], gene_id))

        for chrom, genes in chromosome_genes.items():
            if len(genes) == 0:
                continue

            chrom_number = self.chromosome_utils.standardize_chromosome_format(
                chrom, remove_prefix=True
            )

            if chrom_number not in chrom_sizes:
                continue

            chrom_end = chrom_sizes[chrom_number]
            genes.sort(key=lambda x: x[0])
            prev_end = 1

            for start, end, gene_id in genes:
                if start > prev_end + 1:
                    regions.append(
                        {
                            "CHR": chrom,
                            "START": prev_end + 1,
                            "END": start - 1,
                            "BIOTYPE": "intergenic",
                            "GENE_ID": None,
                            "GENE": None,
                        }
                    )
                prev_end = max(prev_end, end)

            if prev_end < chrom_end:
                regions.append(
                    {
                        "CHR": chrom,
                        "START": prev_end + 1,
                        "END": chrom_end,
                        "BIOTYPE": "intergenic",
                        "GENE_ID": None,
                        "GENE": None,
                    }
                )

    @handle_errors(
        default_return=False, error_message="Creating regulatory regions from GTF"
    )
    def create_regulatory_regions(
        self, gtf_path: str, output: str, protein_coding: bool = True
    ) -> bool:
        log.debug("Creating regulatory regions annotation")
        regulatory_regions = []
        gene_info = {}

        with open(gtf_path, "r") as file:
            for line in tqdm(file, desc="Parsing GTF for regulatory regions"):
                if line.startswith("#"):
                    continue

                fields = line.strip().split("\t")
                if len(fields) < 9:
                    continue

                chrom, source, feature, start, end, score, strand, frame, attributes = (
                    fields
                )

                if feature == "gene":
                    gene_id = re.search(r'gene_id "([^"]+)"', attributes)
                    gene_name = re.search(r'gene_name "([^"]+)"', attributes)
                    gene_biotype = re.search(r'gene_biotype "([^"]+)"', attributes)

                    if gene_id and gene_name and gene_biotype:
                        gene_id = gene_id.group(1)
                        gene_name = gene_name.group(1)
                        gene_biotype = gene_biotype.group(1)

                        if protein_coding and gene_biotype != "protein_coding":
                            continue

                        gene_info[gene_id] = {
                            "name": gene_name,
                            "biotype": gene_biotype,
                            "chr": chrom,
                            "start": int(start),
                            "end": int(end),
                            "strand": strand,
                        }

        for gene_id, gene in gene_info.items():
            chrom = gene["chr"]
            start = gene["start"]
            end = gene["end"]
            strand = gene["strand"]
            gene_name = gene["name"]

            if strand == "+":
                promoter_start = max(1, start - 2000)
                promoter_end = start
                tss_pos = start
            else:
                promoter_start = end
                promoter_end = end + 2000
                tss_pos = end

            regulatory_regions.append(
                {
                    "CHR": chrom,
                    "START": promoter_start,
                    "END": promoter_end,
                    "REGULATORY_TYPE": "promoter",
                    "GENE_ID": gene_id,
                    "GENE": gene_name,
                    "DISTANCE_TO_TSS": 0,
                }
            )

            if strand == "+":
                enh_up_start = max(1, start - 10000)
                enh_up_end = start - 2001
            else:
                enh_up_start = end + 2001
                enh_up_end = end + 10000

            if enh_up_start < enh_up_end:
                regulatory_regions.append(
                    {
                        "CHR": chrom,
                        "START": enh_up_start,
                        "END": enh_up_end,
                        "REGULATORY_TYPE": "enhancer",
                        "GENE_ID": gene_id,
                        "GENE": gene_name,
                        "DISTANCE_TO_TSS": abs(
                            (enh_up_start + enh_up_end) // 2 - tss_pos
                        ),
                    }
                )

            if strand == "+":
                enh_down_start = end + 2001
                enh_down_end = end + 10000
            else:
                enh_down_start = max(1, start - 10000)
                enh_down_end = start - 2001

            if enh_down_start < enh_down_end:
                regulatory_regions.append(
                    {
                        "CHR": chrom,
                        "START": enh_down_start,
                        "END": enh_down_end,
                        "REGULATORY_TYPE": "enhancer",
                        "GENE_ID": gene_id,
                        "GENE": gene_name,
                        "DISTANCE_TO_TSS": abs(
                            (enh_down_start + enh_down_end) // 2 - tss_pos
                        ),
                    }
                )

            regulatory_regions.append(
                {
                    "CHR": chrom,
                    "START": start,
                    "END": end,
                    "REGULATORY_TYPE": "gene_body",
                    "GENE_ID": gene_id,
                    "GENE": gene_name,
                    "DISTANCE_TO_TSS": abs((start + end) // 2 - tss_pos),
                }
            )

        df_regulatory = pd.DataFrame(regulatory_regions)
        if not df_regulatory.empty:
            df_regulatory = self.data_standardizer.standardize_chromosomes(
                df_regulatory, chr_column="CHR", target_format="with_prefix"
            )
            df_regulatory = df_regulatory.sort_values(["CHR", "START"])
            df_regulatory.to_csv(output, index=False)
            log.info(
                f"Created regulatory regions annotation with {len(df_regulatory)} regions"
            )
        else:
            log.warn("No regulatory regions found")
            pd.DataFrame(
                columns=[
                    "CHR",
                    "START",
                    "END",
                    "REGULATORY_TYPE",
                    "GENE_ID",
                    "GENE",
                    "DISTANCE_TO_TSS",
                ]
            ).to_csv(output, index=False)
        return True

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error processing genotype data"
    )
    def process_genotype(
        self, data: pd.DataFrame, annotation_data: pd.DataFrame
    ) -> pd.DataFrame:
        log.debug("Starting genotype data processing")
        if not all(col in data.columns for col in ["RSID", "CHR", "BP"]):
            log.error("Genotype data missing required columns (RSID, CHR, BP)")
            return data

        if annotation_data is None or annotation_data.empty:
            log.error("No annotation data provided for genotype processing")
            return data

        log.debug(f"Input data columns: {list(data.columns)}")
        log.debug(f"Annotation data columns: {list(annotation_data.columns)}")
        log.debug(f"Annotation data shape: {annotation_data.shape}")

        if all(
            col in annotation_data.columns for col in ["CHR", "GENE", "START", "END"]
        ):
            df_annot = annotation_data.copy()
            log.debug("Using pre-standardized annotation data")
            log.debug(
                f"Sample annotation CHR values: {df_annot['CHR'].head(5).tolist()}"
            )

            df_annot = self.data_standardizer.standardize_chromosomes(
                df_annot, target_format="with_prefix"
            )

            log.debug(
                f"Standardized annotation CHR values: {df_annot['CHR'].head(5).tolist()}"
            )
        else:
            log.debug("Preparing annotation data")
            df_annot = self._prepare_annotation_data(
                annotation_data,
                ["CHR", "GENE", "START", "END", "GENE_ID", "TSS", "STRAND"],
            )

        if df_annot is None or df_annot.empty:
            log.error("Annotation data preparation failed or returned empty")
            result = data.copy()
            for col in [
                "GENE",
                "GENE_ID",
                "TSS",
                "TSS_DIST",
                "NEAREST_GENE_DIST",
                "STRAND",
            ]:
                result[col] = None
            return result

        log.debug(f"Prepared annotation data shape: {df_annot.shape}")
        log.debug(f"Prepared annotation columns: {list(df_annot.columns)}")

        chroms = list(data["CHR"].unique())
        result_frames = []

        for chrom in chroms:
            log.debug(f"Processing chromosome {chrom}")
            genotype_chrom = data[data["CHR"] == chrom].copy()
            annot_chrom = df_annot[df_annot["CHR"] == chrom].copy()

            log.debug(
                f"Chromosome {chrom}: {len(genotype_chrom)} variants, {len(annot_chrom)} annotations"
            )

            if annot_chrom.empty:
                log.warn(f"No annotation data for chromosome {chrom}")
                for col in [
                    "GENE",
                    "GENE_ID",
                    "TSS",
                    "TSS_DIST",
                    "NEAREST_GENE_DIST",
                    "STRAND",
                ]:
                    if col not in genotype_chrom.columns:
                        genotype_chrom[col] = None
                result_frames.append(genotype_chrom)
                continue

            if genotype_chrom.empty:
                log.warn(f"No genotype data for chromosome {chrom}")
                continue

            try:
                log.debug(f"Calling assign_nearest for chromosome {chrom}")
                annotated = self.assign_nearest(genotype_chrom, annot_chrom)
                log.debug(
                    f"assign_nearest returned {len(annotated)} rows with columns: {list(annotated.columns)}"
                )
                result_frames.append(annotated)
            except Exception as e:
                log.error(f"assign_nearest failed for chromosome {chrom}: {e}")
                for col in [
                    "GENE",
                    "GENE_ID",
                    "TSS",
                    "TSS_DIST",
                    "NEAREST_GENE_DIST",
                    "STRAND",
                ]:
                    if col not in genotype_chrom.columns:
                        genotype_chrom[col] = None
                result_frames.append(genotype_chrom)

        if not result_frames:
            log.error("No result frames generated")
            result = data.copy()
            for col in [
                "GENE",
                "GENE_ID",
                "TSS",
                "TSS_DIST",
                "NEAREST_GENE_DIST",
                "STRAND",
            ]:
                result[col] = None
            return result

        processed_results = pd.concat(result_frames, ignore_index=True)
        log.debug(
            f"Concatenated results: {len(processed_results)} rows, columns: {list(processed_results.columns)}"
        )

        expected_annotation_cols = [
            "GENE",
            "GENE_ID",
            "TSS",
            "TSS_DIST",
            "NEAREST_GENE_DIST",
            "STRAND",
        ]
        for col in expected_annotation_cols:
            if col not in processed_results.columns:
                processed_results[col] = None

        return processed_results

    @handle_errors(
        default_return=pd.DataFrame(),
        error_message="Error processing regulatory regions",
    )
    def process_regulatory_regions(
        self, data: pd.DataFrame, annotation_data: pd.DataFrame
    ) -> pd.DataFrame:
        if not all(col in data.columns for col in ["RSID", "CHR", "BP"]):
            log.error("Genotype data missing required columns (RSID, CHR, BP)")
            data["REGULATORY_REGION"] = None
            return data

        df_annot = self._prepare_annotation_data(
            annotation_data,
            ["CHR", "START", "END", "REGULATORY_TYPE", "GENE", "DISTANCE_TO_TSS"],
        )

        if not all(col in df_annot.columns for col in ["CHR", "START", "END"]):
            log.error("Missing required fields in regulatory annotation data")
            data["REGULATORY_REGION"] = None
            return data

        df_annot = self.data_standardizer.standardize_chromosomes(
            df_annot, target_format="with_prefix"
        )

        regulatory_priority = {
            "promoter": 1,
            "enhancer": 2,
            "gene_body": 3,
            "intergenic": 4,
        }

        def process_chrom_func(
            chrom: Any, data_chrom: pd.DataFrame, annot_chrom: pd.DataFrame
        ) -> pd.DataFrame:
            if data_chrom.empty or annot_chrom.empty:
                return pd.DataFrame(
                    {
                        "RSID": data_chrom["RSID"],
                        "REGULATORY_REGION": None,
                        "REG_GENE": None,
                        "REG_DISTANCE": None,
                    }
                )

            results = []
            for _, variant in data_chrom.iterrows():
                pos = variant["BP"]
                overlapping = annot_chrom[
                    (annot_chrom["START"] <= pos) & (annot_chrom["END"] >= pos)
                ]

                if not overlapping.empty:
                    if "REGULATORY_TYPE" in overlapping.columns:
                        best_match = overlapping.loc[
                            overlapping["REGULATORY_TYPE"]
                            .map(lambda x: regulatory_priority.get(x, 999))
                            .idxmin()
                        ]
                        reg_type = best_match["REGULATORY_TYPE"]
                        reg_gene = best_match.get("GENE", None)
                        reg_distance = best_match.get("DISTANCE_TO_TSS", None)
                    else:
                        reg_type = "regulatory"
                        reg_gene = overlapping.iloc[0].get("GENE", None)
                        reg_distance = overlapping.iloc[0].get("DISTANCE_TO_TSS", None)
                else:
                    reg_type = None
                    reg_gene = None
                    reg_distance = None

                results.append(
                    {
                        "RSID": variant["RSID"],
                        "REGULATORY_REGION": reg_type,
                        "REG_GENE": reg_gene,
                        "REG_DISTANCE": reg_distance,
                    }
                )

            return pd.DataFrame(results)

        regulatory_results = self.process_by_chromosome(
            data, df_annot, process_chrom_func
        )

        if not regulatory_results.empty:
            final_results = pd.merge(data, regulatory_results, on="RSID", how="left")
            return final_results
        else:
            data["REGULATORY_REGION"] = None
            data["REG_GENE"] = None
            data["REG_DISTANCE"] = None
            return data

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Error assigning nearest genes"
    )
    def assign_nearest(
        self,
        ewas_df: pd.DataFrame,
        annot_df: pd.DataFrame,
        kdtree: Optional[KDTree] = None,
        position_to_gene_idx: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if not validate_input_requirements(ewas_df, ["BP"], "Gene annotation input"):
            return ensure_analysis_columns(ewas_df, "EWAS")

        valid_mask = ewas_df["BP"].notna()
        if not valid_mask.any():
            log.warn("No valid positions found for gene annotation")
            return ensure_analysis_columns(ewas_df.copy(), "EWAS")

        if kdtree is None or position_to_gene_idx is None:
            gene_positions = np.vstack(
                [annot_df[["START"]].values, annot_df[["END"]].values]
            )
            kdtree = KDTree(gene_positions)
            n_genes = len(annot_df)
            position_to_gene_idx = np.concatenate(
                [np.arange(n_genes), np.arange(n_genes)]
            )

        valid_positions = ewas_df.loc[valid_mask, "BP"].values.reshape(-1, 1)
        nearest_gene_distances, indices = kdtree.query(valid_positions, k=1)
        gene_indices = position_to_gene_idx[indices].flatten()

        result = ewas_df.copy()
        result["GENE"] = None
        result["GENE_ID"] = None
        result["STRAND"] = None if "STRAND" not in result.columns else result["STRAND"]
        result["TSS"] = None
        result["TSS_DIST"] = None
        result["NEAREST_GENE_DIST"] = None

        result.loc[valid_mask, "GENE"] = annot_df.iloc[gene_indices]["GENE"].values

        if "GENE_ID" in annot_df.columns:
            result.loc[valid_mask, "GENE_ID"] = annot_df.iloc[gene_indices][
                "GENE_ID"
            ].values

        if "STRAND" in annot_df.columns and "STRAND" not in ewas_df.columns:
            result.loc[valid_mask, "STRAND"] = annot_df.iloc[gene_indices][
                "STRAND"
            ].values

        probe_positions = ewas_df.loc[valid_mask, "BP"].values
        result.loc[valid_mask, "NEAREST_GENE_DIST"] = nearest_gene_distances.flatten()

        if "TSS" in annot_df.columns:
            result.loc[valid_mask, "TSS"] = annot_df.iloc[gene_indices]["TSS"].values
            tss_positions = annot_df.iloc[gene_indices]["TSS"].values
            result.loc[valid_mask, "TSS_DIST"] = np.abs(probe_positions - tss_positions)
        else:
            gene_starts = annot_df.iloc[gene_indices]["START"].values
            gene_ends = annot_df.iloc[gene_indices]["END"].values

            if "STRAND" in annot_df.columns:
                strands = annot_df.iloc[gene_indices]["STRAND"].values
                tss_positions = np.where(strands == "+", gene_starts, gene_ends)
                result.loc[valid_mask, "TSS"] = tss_positions
                result.loc[valid_mask, "TSS_DIST"] = np.abs(
                    probe_positions - tss_positions
                )
            else:
                result.loc[valid_mask, "TSS"] = gene_starts
                result.loc[valid_mask, "TSS_DIST"] = np.abs(
                    probe_positions - gene_starts
                )

        return result


class CpGIslandAnnotator(BaseAnnotator):
    def __init__(self, genome_version: str, reference: Optional[str] = None) -> None:
        super().__init__(genome_version, reference)

    @handle_errors(
        default_return=False,
        error_message="Error downloading and processing CpG islands",
    )
    def _download_and_process_islands(self, output: str) -> bool:
        chrom_sizes = self.chromosome_utils.get_chromosome_sizes()

        with temp_file_operation(suffix=".txt.gz") as local_gz:
            url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{self.genome_version}/database/cpgIslandExt.txt.gz"
            if not self.resource_manager._download_file(
                url, local_gz, f"Downloading {self.genome_version} CpG islands"
            ):
                pd.DataFrame(columns=["CHR", "START", "END", "CPG_REGION"]).to_csv(
                    output, index=False
                )
                return False

            extracted = os.path.splitext(local_gz)[0]
            try:
                if os.path.exists(extracted) and os.path.getsize(extracted) > 0:
                    log.info(f"Reading extracted CpG islands file: {extracted}")
                    manifest = pd.read_csv(
                        extracted, sep="\t", comment="#", header=None, low_memory=False
                    )
                else:
                    log.error(f"Downloaded file not found or empty: {local_gz}")
                    return False
            except Exception as e:
                log.error(f"Failed to read CpG islands file: {e}")
                return False

            manifest = manifest.iloc[:, 1:4]
            manifest.columns = ["CHR", "START", "END"]
            manifest["CPG_REGION"] = "Island"

            shores_shelves = self.create_shores_and_shelves(manifest)
            combined = pd.concat([manifest] + shores_shelves, ignore_index=True)

            combined = self.data_standardizer.standardize_chromosomes(
                combined,
                chr_column="CHR",
                target_format="without_prefix",
                filter_autosomal=True,
            )

            autosomal_chromosomes = [str(i) for i in range(1, 23)]
            combined = self.add_open_sea_regions(
                combined, chrom_sizes, autosomal_chromosomes
            )
            combined.to_csv(output, index=False)
            return True

    @handle_errors(
        default_return=pd.DataFrame(), error_message="Processing CpG islands"
    )
    def process_islands(
        self, data: pd.DataFrame, annotation_data: pd.DataFrame
    ) -> pd.DataFrame:
        analysis_type = "GWAS" if "RSID" in data.columns else "EWAS"
        id_column = "RSID" if analysis_type == "GWAS" else "CGID"

        if not validate_input_requirements(data, [id_column], "CpG islands annotation"):
            return ensure_analysis_columns(data.copy(), analysis_type)

        df_data = data.copy()
        if analysis_type == "EWAS" and (
            "CHR" not in df_data.columns or "BP" not in df_data.columns
        ):
            data_processor = DataProcessor("EWAS")
            df_data = data_processor.extract_chr_pos(df_data)

        df_annot = self._prepare_annotation_data(annotation_data)

        total_positions = len(df_data)
        log.debug(f"Total positions to annotate: {total_positions}")

        pbar = tqdm(total=total_positions, desc="Annotating CpG islands")

        def process_chrom_func(
            chrom: Any, data_chrom: pd.DataFrame, annot_chrom: pd.DataFrame
        ) -> pd.DataFrame:
            if data_chrom.empty:
                pbar.update(len(data_chrom))
                return pd.DataFrame({id_column: [], "CPG_REGION": []})

            if annot_chrom.empty:
                result = pd.DataFrame(
                    {
                        id_column: data_chrom[id_column].values,
                        "CPG_REGION": ["Open Sea"] * len(data_chrom),
                    }
                )
                pbar.update(len(data_chrom))
                return result

            tree = IntervalTree()

            starts = annot_chrom["START"].values
            ends = annot_chrom["END"].values
            cpg_regions = (
                annot_chrom["CPG_REGION"].values
                if "CPG_REGION" in annot_chrom.columns
                else np.array(["Island"] * len(annot_chrom))
            )

            region_priority = {
                "Island": 1,
                "Shore": 2,
                "Shelf": 3,
                "Open Sea": 4,
            }

            for idx in range(len(annot_chrom)):
                tree.addi(starts[idx], ends[idx] + 1, cpg_regions[idx])

            positions = data_chrom["BP"].values
            ids = data_chrom[id_column].values

            result_regions = np.empty(len(positions), dtype=object)

            for i in range(len(positions)):
                pos = positions[i]
                overlaps = tree.at(pos)

                if overlaps:
                    matched_regions = [interval.data for interval in overlaps]
                    result_regions[i] = min(
                        matched_regions, key=lambda x: region_priority.get(x, 999)
                    )
                else:
                    result_regions[i] = "Open Sea"

                pbar.update(1)

            return pd.DataFrame({id_column: ids, "CPG_REGION": result_regions})

        results = self.process_by_chromosome(df_data, df_annot, process_chrom_func)

        pbar.close()

        if not results.empty:
            final_results = pd.merge(
                data, results[[id_column, "CPG_REGION"]], on=id_column, how="left"
            )
            return ensure_analysis_columns(final_results, analysis_type)
        else:
            log.warn("No CpG island results generated")
            data["CPG_REGION"] = "Open Sea"
            return ensure_analysis_columns(data, analysis_type)

    def _prepare_annotation_data(self, annotation_data: pd.DataFrame) -> pd.DataFrame:
        df_annot = annotation_data.copy()

        if not all(col in df_annot.columns for col in ["CHR", "START", "END"]):
            found_cols = {}
            for field in ["CHR", "START", "END", "CPG_REGION"]:
                if field not in df_annot.columns:
                    found = self.data_standardizer.find_annotation_field(
                        df_annot, field
                    )
                    if found:
                        found_cols[found] = field

            if found_cols:
                df_annot = df_annot.rename(columns=found_cols)
                log.debug(f"Applied column mapping for annotation data: {found_cols}")

        return df_annot

    def create_shores_and_shelves(
        self, islands: Optional[pd.DataFrame]
    ) -> List[pd.DataFrame]:
        if islands is None or islands.empty:
            empty_df = pd.DataFrame(columns=["CHR", "START", "END", "CPG_REGION"])
            return [empty_df.copy() for _ in range(4)]

        north_shore = islands.copy()
        north_shore["START"] = (north_shore["START"] - 2000).clip(lower=1)
        north_shore["END"] = north_shore["START"] + 1999
        north_shore = north_shore[north_shore["START"] < north_shore["END"]].copy()
        north_shore["CPG_REGION"] = "Shore"

        south_shore = islands.copy()
        south_shore["START"] = south_shore["END"] + 1
        south_shore["END"] = south_shore["END"] + 2000
        south_shore = south_shore[south_shore["START"] < south_shore["END"]].copy()
        south_shore["CPG_REGION"] = "Shore"

        north_shelf = islands.copy()
        north_shelf["START"] = (north_shelf["START"] - 4000).clip(lower=1)
        north_shelf["END"] = north_shelf["START"] + 1999
        north_shelf = north_shelf[north_shelf["START"] < north_shelf["END"]].copy()
        north_shelf["CPG_REGION"] = "Shelf"

        south_shelf = islands.copy()
        south_shelf["START"] = south_shelf["END"] + 2001
        south_shelf["END"] = south_shelf["END"] + 4000
        south_shelf = south_shelf[south_shelf["START"] < south_shelf["END"]].copy()
        south_shelf["CPG_REGION"] = "Shelf"

        return [north_shore, south_shore, north_shelf, south_shelf]

    def add_open_sea_regions(
        self,
        regions: pd.DataFrame,
        chrom_sizes: Dict[str, int],
        autosomal_chromosomes: Sequence[str],
    ) -> pd.DataFrame:
        open_sea_records = []
        for chrom in autosomal_chromosomes:
            chrom_size = chrom_sizes.get(chrom)
            if chrom_size is None:
                log.warn(f"No size information for chromosome {chrom}")
                continue

            chr_intervals = regions[regions["CHR"] == chrom].sort_values("START")
            current = 0

            if chr_intervals.empty:
                open_sea_records.append(
                    {
                        "CHR": chrom,
                        "START": 0,
                        "END": chrom_size,
                        "CPG_REGION": "Open Sea",
                    }
                )
                continue

            for _, row in chr_intervals.iterrows():
                if row["START"] > current:
                    open_sea_records.append(
                        {
                            "CHR": chrom,
                            "START": current,
                            "END": row["START"] - 1,
                            "CPG_REGION": "Open Sea",
                        }
                    )
                current = max(current, row["END"] + 1)

            if current <= chrom_size:
                open_sea_records.append(
                    {
                        "CHR": chrom,
                        "START": current,
                        "END": chrom_size,
                        "CPG_REGION": "Open Sea",
                    }
                )

        if open_sea_records:
            open_sea_df = pd.DataFrame(open_sea_records)
            return pd.concat([regions, open_sea_df], ignore_index=True)
        return regions


class ResourceManager:
    def __init__(
        self,
        reference_path: Optional[str] = None,
        genome_version: str = "hg38",
        data_standardizer: Optional["DataStandardizer"] = None,
    ) -> None:
        self.reference_path = reference_path
        self.genome_version = genome_version
        self.data_standardizer = data_standardizer
        self.resource_cache = {}
        self.chromosome_utils = None
        self._cached_gtf_path = None
        self._cached_gtf_dir = None

    def _create_shore_shelf(
        self, islands: pd.DataFrame, region_type: str, distance: int, label: str
    ) -> pd.DataFrame:
        df = islands.copy()
        if region_type.startswith("north"):
            df["START"] = (df["START"] - distance).clip(lower=1)
            df["END"] = df["START"] + distance - 1
        else:
            df["START"] = df["END"] + 1
            df["END"] = df["END"] + distance
        df = df[df["START"] < df["END"]].copy()
        df["CPG_REGION"] = label
        return df

    def get_resource(self, resource_type: str, output_path: str, **kwargs: Any) -> bool:
        if self.reference_path and os.path.exists(self.reference_path):
            log.info(f"Using user-provided reference file for {resource_type}")
            return self._process_user_reference(resource_type, output_path, **kwargs)
        else:
            return self._download_resource(resource_type, output_path, **kwargs)

    def _process_user_reference(
        self, resource_type: str, output_path: str, **kwargs: Any
    ) -> bool:
        try:
            if resource_type == "array":
                return self.process_user_reference_array(
                    output_path, kwargs.get("array_type", "450k")
                )
            elif resource_type == "ensembl":
                return self.process_user_reference_ensembl(
                    output_path, kwargs.get("regions", False)
                )
            elif resource_type == "islands":
                return self.process_user_reference_islands(output_path)
            elif resource_type == "regulatory":
                return self.process_user_reference_regulatory(output_path)
            else:
                log.error(f"Unknown resource type: {resource_type}")
                return False
        except Exception as e:
            log.error(f"Error processing user reference file for {resource_type}: {e}")
            log.info("Falling back to downloading resource...")
            return self._download_resource(resource_type, output_path, **kwargs)

    def _download_resource(
        self, resource_type: str, output_path: str, **kwargs: Any
    ) -> bool:
        try:
            if resource_type == "array":
                return self.download_and_process_array(
                    output_path, kwargs.get("array_type", "450k")
                )
            elif resource_type == "ensembl":
                return self.download_and_process_ensembl(
                    output_path, kwargs.get("regions", False)
                )
            elif resource_type == "islands":
                return self.download_and_process_islands(output_path)
            elif resource_type == "regulatory":
                return self.download_and_process_regulatory(output_path)
            else:
                log.error(f"Unknown resource type: {resource_type}")
                return False
        except Exception as e:
            log.error(f"Error downloading resource for {resource_type}: {e}")
            return False

    def _load_user_reference(self, file_path: str) -> Optional[pd.DataFrame]:
        try:
            if file_path.endswith(".csv"):
                return pd.read_csv(file_path, low_memory=False)
            elif file_path.endswith(".txt") or file_path.endswith(".tsv"):
                try:
                    return pd.read_csv(file_path, low_memory=False)
                except Exception:
                    return pd.read_csv(file_path, sep="\t", low_memory=False)
            else:
                return pd.read_csv(
                    file_path, sep=None, engine="python", low_memory=False
                )
        except Exception as e:
            log.error(f"Failed to load user reference file: {e}")
            return None

    def process_user_reference_array(self, output_path: str, array_type: str) -> bool:
        try:
            manifest = self._load_user_reference(self.reference_path)
            if manifest is None:
                return False

            log.info(f"Loaded user reference file with {len(manifest)} records")
            log.debug(f"Reference file columns: {list(manifest.columns)}")

            if self.data_standardizer:
                manifest = self.data_standardizer.standardize_array_manifest(
                    manifest, array_type, source="user"
                )

            manifest.to_csv(output_path, index=False)
            log.info(f"Processed user reference file saved to {output_path}")
            return True
        except Exception as e:
            log.error(f"Error processing user array reference: {e}")
            return self.download_and_process_array(output_path, array_type)

    def process_user_reference_ensembl(
        self, output_path: str, regions: bool = False
    ) -> bool:
        try:
            if self.reference_path.endswith(".gtf") or self.reference_path.endswith(
                ".gtf.gz"
            ):
                log.info("Processing user-provided GTF file")
                return True
            else:
                log.info("Using user-provided processed annotation file")
                annotation_data = self._load_user_reference(self.reference_path)
                if annotation_data is None:
                    return False

                log.info(
                    f"Loaded user annotation file with {len(annotation_data)} records"
                )
                log.debug(f"Annotation file columns: {list(annotation_data.columns)}")

                if self.data_standardizer:
                    if regions:
                        annotation_data = self.data_standardizer.standardize_annotation(
                            annotation_data, "ensembl_regions"
                        )
                    else:
                        annotation_data = self.data_standardizer.standardize_annotation(
                            annotation_data, "ensembl_genes"
                        )

                annotation_data.to_csv(output_path, index=False)
                log.info(f"User annotation file saved to {output_path}")
                return True
        except Exception as e:
            log.error(f"Error processing user ENSEMBL reference: {e}")
            return self.download_and_process_ensembl(output_path, regions)

    def process_user_reference_islands(self, output_path: str) -> bool:
        try:
            islands_data = self._load_user_reference(self.reference_path)
            if islands_data is None:
                return self.download_and_process_islands(output_path)

            log.info(
                f"Loaded user islands reference file with {len(islands_data)} records"
            )
            log.debug(f"Islands reference file columns: {list(islands_data.columns)}")

            chr_col = AliasUtils.find_keys(dict.fromkeys(islands_data.columns), "CHR")
            start_col = AliasUtils.find_keys(
                dict.fromkeys(islands_data.columns), "START"
            )
            end_col = AliasUtils.find_keys(dict.fromkeys(islands_data.columns), "END")
            region_col = AliasUtils.find_keys(
                dict.fromkeys(islands_data.columns), "CPG_REGION"
            )

            if not all([chr_col, start_col, end_col]):
                log.error(
                    "User islands reference file missing required columns (chromosome, start, end)"
                )
                return self.download_and_process_islands(output_path)

            column_mapping = {chr_col: "CHR", start_col: "START", end_col: "END"}
            if region_col:
                column_mapping[region_col] = "CPG_REGION"

            islands_data = islands_data.rename(columns=column_mapping)

            if "CPG_REGION" not in islands_data.columns:
                islands_data["CPG_REGION"] = "Island"

            if self.data_standardizer:
                islands_data = self.data_standardizer.standardize_annotation(
                    islands_data, "cpg_islands"
                )

            islands_data.to_csv(output_path, index=False)
            log.info(f"Processed user islands reference file saved to {output_path}")
            return True
        except Exception as e:
            log.error(f"Error processing user islands reference: {e}")
            return self.download_and_process_islands(output_path)

    def process_user_reference_regulatory(self, output_path: str) -> bool:
        try:
            if self.reference_path.endswith(".gtf") or self.reference_path.endswith(
                ".gtf.gz"
            ):
                log.info("Processing user-provided GTF file for regulatory regions")
                return True
            else:
                regulatory_data = self._load_user_reference(self.reference_path)
                if regulatory_data is None:
                    return self.download_and_process_regulatory(output_path)

                log.info(
                    f"Loaded user regulatory reference with {len(regulatory_data)} records"
                )
                log.debug(
                    f"Regulatory reference columns: {list(regulatory_data.columns)}"
                )

                if self.data_standardizer:
                    regulatory_data = self.data_standardizer.standardize_chromosomes(
                        regulatory_data, target_format="with_prefix"
                    )
                    regulatory_data = self.data_standardizer.standardize_data_types(
                        regulatory_data
                    )

                regulatory_data.to_csv(output_path, index=False)
                log.info(f"User regulatory reference saved to {output_path}")
                return True
        except Exception as e:
            log.error(f"Error processing user regulatory reference: {e}")
            return self.download_and_process_regulatory(output_path)

    def download_and_process_array(self, output_path: str, array_type: str) -> bool:
        with tempfile.TemporaryDirectory() as tmpdirname:
            if array_type == "450k":
                base_url = "https://webdata.illumina.com/downloads/productfiles/"
                file_name = "humanmethylation450/humanmethylation450_15017482_v1-2.csv"
                manifest_url = base_url + file_name
                manifest_filename = os.path.join(tmpdirname, "450k_manifest_v1.2.csv")
                skiprows = 7
            else:
                base_url = "https://webdata.illumina.com/downloads/productfiles/"
                file_name = "methylationEPIC/infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip"
                manifest_url = base_url + file_name
                manifest_filename = os.path.join(tmpdirname, "EPIC_array_v1.B5.csv.zip")
                skiprows = 7

            if not self._download_file(
                manifest_url, manifest_filename, f"Downloading {array_type} manifest"
            ):
                empty_df = pd.DataFrame({"CGID": [], "CHR": [], "BP": []})
                empty_df.to_csv(output_path, index=False)
                return False

            manifest_csv = (
                manifest_filename.replace(".zip", "")
                if array_type == "EPIC"
                else manifest_filename
            )

            try:
                manifest = pd.read_csv(
                    manifest_csv,
                    sep=",",
                    comment="#",
                    skiprows=skiprows,
                    low_memory=False,
                )

                if self.data_standardizer:
                    manifest = self.data_standardizer.standardize_array_manifest(
                        manifest, array_type=array_type, source="downloaded"
                    )

                manifest.to_csv(output_path, index=False)
                return True
            except Exception as e:
                log.error(f"Error processing downloaded array manifest: {e}")
                empty_df = pd.DataFrame({"CGID": [], "CHR": [], "BP": []})
                empty_df.to_csv(output_path, index=False)
                return False

    def get_gtf_url_and_filename(self) -> Tuple[str, str]:
        try:
            if self.genome_version == "hg38":
                base_url = "https://ftp.ensembl.org/pub/current_gtf/homo_sapiens/"
            elif self.genome_version == "hg19":
                base_url = (
                    "https://ftp.ensembl.org/pub/grch37/current/gtf/homo_sapiens/"
                )
            else:
                raise ValueError(
                    f"Invalid genome version: {self.genome_version}. Only hg38 and hg19 are supported."
                )

            response = requests.get(base_url)
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to access {base_url}. Status code: {response.status_code}"
                )

            soup = BeautifulSoup(response.text, "html.parser")
            gtf_files = [
                link.get("href")
                for link in soup.find_all("a")
                if link.get("href", "").endswith(".chr.gtf.gz")
            ]

            if not gtf_files:
                gtf_files = []
                for link in soup.find_all("a"):
                    href = link.get("href", "")
                    if href.endswith(".gtf.gz") and "chr" in href:
                        gtf_files.append(href)

            if not gtf_files:
                raise ValueError("No GTF files found at the specified URL.")

            latest_file = sorted(gtf_files)[-1]
            return f"{base_url}{latest_file}", latest_file

        except Exception as e:
            log.error(f"Error getting GTF URL: {e}")

            if self.genome_version == "hg38":
                fallback_url_base = (
                    "https://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/"
                )
                fallback_file = "Homo_sapiens.GRCh38.109.chr.gtf.gz"
                return fallback_url_base + fallback_file, fallback_file
            elif self.genome_version == "hg19":
                fallback_url_base = (
                    "https://ftp.ensembl.org/pub/grch37/current/gtf/homo_sapiens/"
                )
                fallback_file = "Homo_sapiens.GRCh37.87.chr.gtf.gz"
                return fallback_url_base + fallback_file, fallback_file
            else:
                raise ValueError(
                    f"Could not determine GTF URL for {self.genome_version}"
                )

    def download_and_process_ensembl(
        self, output_path: str, regions: bool = False
    ) -> bool:
        if self._cached_gtf_path is not None:
            if self._cached_gtf_path.endswith(".gz"):
                extracted_path = self._cached_gtf_path.replace(".gz", "")
                if os.path.exists(extracted_path):
                    log.debug(f"Using cached extracted GTF file: {extracted_path}")
                    return True
            elif os.path.exists(self._cached_gtf_path):
                log.debug(f"Using cached GTF file: {self._cached_gtf_path}")
                return True

        self._cached_gtf_dir = tempfile.TemporaryDirectory()
        gtf_url, gtf_filename = self.get_gtf_url_and_filename()

        self._cached_gtf_path = os.path.join(self._cached_gtf_dir.name, gtf_filename)

        if not self._download_file(
            gtf_url,
            self._cached_gtf_path,
            f"Downloading {self.genome_version} ENSEMBL annotations",
        ):
            self._cached_gtf_path = None
            try:
                if self._cached_gtf_dir is not None:
                    self._cached_gtf_dir.cleanup()
            except Exception:
                pass
            self._cached_gtf_dir = None
            return False

        extracted_gtf = self._cached_gtf_path.replace(".gz", "")

        if not os.path.exists(extracted_gtf):
            log.error(f"Extracted GTF file not found after download: {extracted_gtf}")
            self._cached_gtf_path = None
            try:
                if self._cached_gtf_dir is not None:
                    self._cached_gtf_dir.cleanup()
            except Exception:
                pass
            self._cached_gtf_dir = None
            return False

        self._cached_gtf_path = extracted_gtf
        log.debug(f"GTF file cached at: {self._cached_gtf_path}")

        return True

    def download_and_process_islands(self, output_path: str) -> bool:
        if not self.chromosome_utils:
            log.error("ChromosomeUtils not available for islands processing")
            return False

        chrom_sizes = self.chromosome_utils.get_chromosome_sizes()

        with tempfile.NamedTemporaryFile(suffix=".txt.gz", delete=False) as temp_file:
            local_gz = temp_file.name
            try:
                url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{self.genome_version}/database/cpgIslandExt.txt.gz"

                if not self._download_file(
                    url, local_gz, f"Downloading {self.genome_version} CpG islands"
                ):
                    pd.DataFrame(columns=["CHR", "START", "END", "CPG_REGION"]).to_csv(
                        output_path, index=False
                    )
                    return False

                extracted = os.path.splitext(local_gz)[0]

                if not os.path.exists(extracted) or os.path.getsize(extracted) == 0:
                    log.error(f"Downloaded file not found or empty: {local_gz}")
                    return False

                manifest = pd.read_csv(
                    extracted, sep="\t", comment="#", header=None, low_memory=False
                )

                manifest = manifest.iloc[:, 1:4]
                manifest.columns = ["CHR", "START", "END"]
                manifest["CPG_REGION"] = "Island"

                shores_shelves = []
                if not manifest.empty:
                    shores_shelves = [
                        self._create_shore_shelf(
                            manifest, "north_shore", 2000, "Shore"
                        ),
                        self._create_shore_shelf(
                            manifest, "south_shore", 2000, "Shore"
                        ),
                        self._create_shore_shelf(
                            manifest, "north_shelf", 4000, "Shelf"
                        ),
                        self._create_shore_shelf(
                            manifest, "south_shelf", 4000, "Shelf"
                        ),
                    ]
                combined = pd.concat([manifest] + shores_shelves, ignore_index=True)

                if self.data_standardizer:
                    combined = self.data_standardizer.standardize_chromosomes(
                        combined,
                        chr_column="CHR",
                        target_format="with_prefix",
                        filter_autosomal=True,
                    )

                autosomal_chromosomes = [f"CHR{i}" for i in range(1, 23)]
                open_sea_records = []
                for chrom in autosomal_chromosomes:
                    chrom_number = chrom.replace("CHR", "")
                    chrom_size = chrom_sizes.get(chrom_number)
                    if chrom_size is None:
                        log.warn(f"No size information for chromosome {chrom}")
                        continue

                    chr_intervals = combined[combined["CHR"] == chrom].sort_values(
                        "START"
                    )
                    current = 0

                    if chr_intervals.empty:
                        open_sea_records.append(
                            {
                                "CHR": chrom,
                                "START": 0,
                                "END": chrom_size,
                                "CPG_REGION": "Open Sea",
                            }
                        )
                        continue

                    for _, row in chr_intervals.iterrows():
                        if row["START"] > current:
                            open_sea_records.append(
                                {
                                    "CHR": chrom,
                                    "START": current,
                                    "END": row["START"] - 1,
                                    "CPG_REGION": "Open Sea",
                                }
                            )
                        current = max(current, row["END"] + 1)

                    if current <= chrom_size:
                        open_sea_records.append(
                            {
                                "CHR": chrom,
                                "START": current,
                                "END": chrom_size,
                                "CPG_REGION": "Open Sea",
                            }
                        )

                if open_sea_records:
                    combined = pd.concat(
                        [combined, pd.DataFrame(open_sea_records)], ignore_index=True
                    )

                combined.to_csv(output_path, index=False)
                return True

            finally:
                if os.path.exists(local_gz):
                    os.unlink(local_gz)
                extracted = os.path.splitext(local_gz)[0]
                if os.path.exists(extracted):
                    os.unlink(extracted)

    def download_and_process_regulatory(self, output_path: str) -> bool:
        with tempfile.TemporaryDirectory() as tmpdirname:
            gtf_url, gtf_filename = self.get_gtf_url_and_filename()
            local_gtf = os.path.join(tmpdirname, gtf_filename)

            if not self._download_file(
                gtf_url,
                local_gtf,
                f"Downloading {self.genome_version} ENSEMBL annotations for regulatory regions",
            ):
                return False

            return True

    def _download_file(self, url: str, local_path: str, description: str = "") -> bool:
        try:
            DownloadAndExtract([(url, local_path, description)])
            return True
        except Exception as e:
            log.error(f"Failed to download {description}: {e}")
            return False


class Annotator:
    def __init__(
        self,
        input_file: str,
        output: str,
        chip: str = "450k",
        protein_coding: bool = True,
        genome_version: str = "hg38",
        analysis_type: str = "auto",
        reference: Optional[str] = None,
        standardize_col_names: bool = True,
    ) -> None:
        self.input_file = input_file
        self.output = output
        self.chip = chip
        self.protein_coding = protein_coding
        self.genome_version = genome_version
        self.analysis_type = analysis_type
        self.reference = reference
        self.standardize_col_names = standardize_col_names
        self.data = None
        self.messages = []

        if self.output:
            output_dir = os.path.dirname(self.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                log.debug(f"Created output directory: {output_dir}")

        self.data_standardizer = DataStandardizer(
            genome_version=genome_version, chip=chip
        )
        self.chromosome_utils = ChromosomeUtils(genome_version)
        self.data_processor = DataProcessor(analysis_type)

        self.resource_manager = ResourceManager(
            reference, genome_version, self.data_standardizer
        )
        self.resource_manager.chromosome_utils = self.chromosome_utils

        self.array_annotator = ArrayAnnotator(chip, genome_version, reference)
        self.array_annotator.resource_manager = self.resource_manager
        self._cached_array_manifest: Optional[pd.DataFrame] = None

        self.genomic_annotator = GenomicAnnotator(
            genome_version, protein_coding, reference
        )
        self.genomic_annotator.resource_manager = self.resource_manager

        self.island_annotator = CpGIslandAnnotator(genome_version, reference)
        self.island_annotator.resource_manager = self.resource_manager

        self.annotation_methods = {
            "450k": self.annotate_450k,
            "EPIC": self.annotate_epic,
            "MethylSeq": self.annotate_methylseq,
            "Genotype": self.annotate_genotype,
        }

    @handle_errors(default_return=None, error_message="Annotation process failed")
    def annotate(self) -> Optional[pd.DataFrame]:
        log.info("Starting annotation process")

        if self.genome_version not in ["hg19", "hg38"]:
            log.error(
                f"Invalid genome version: {self.genome_version}. Only hg19 and hg38 are supported."
            )
            return None

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file does not exist: {self.input_file}")

        self.data = pd.read_csv(self.input_file, sep=",", low_memory=False)
        log.debug(
            f"Loaded input file with {len(self.data)} rows and {len(self.data.columns)} columns"
        )

        self.data = self.data_standardizer.standardize_input(self.data)
        log.info("Input data standardized and ready for use")

        if "CHR" not in self.data.columns and self.chip in ("450k", "EPIC"):
            log.info("CHR column missing; attempting to fill CHR from array manifest")
            self._fill_chr_from_manifest()

        condition1 = "CHR" in self.data.columns
        condition2 = self.chromosome_utils.validate_chromosome_column(self.data, "CHR")
        if condition1 and not condition2:
            log.warn("Input data contains invalid chromosome values")

        if self.analysis_type == "auto":
            self.analysis_type = self.detect_analysis_type(self.data)
            log.info(f"Detected analysis type: {self.analysis_type}")
        self.data_processor.analysis_type = self.analysis_type

        if not self.validate_input_data(self.data):
            log.error(f"Input data missing required ID column for {self.analysis_type}")
            return None

        if self.analysis_type == "GWAS" and self.chip != "Genotype":
            log.info("GWAS detected: changing chip type to 'Genotype' for annotation")
            self.chip = "Genotype"
            self.data_standardizer.chip = self.chip

        if self.chip not in self.annotation_methods:
            raise ValueError(
                f"Unknown chip type: {self.chip}. Available options: {list(self.annotation_methods.keys())}"
            )

        results = self.annotation_methods[self.chip]()

        if results is None or len(results) == 0:
            log.error("Annotation produced empty results")
            results = self.data.copy()

        results = self.data_standardizer.standardize_output_columns(
            results, analysis_type=self.analysis_type
        )

        results = self.finalize_results(results)

        results.to_csv(self.output, index=False)
        log.success(f"Results written to {self.output}")

    def finalize_results(self, results: pd.DataFrame) -> pd.DataFrame:
        if self.chip not in ("450k", "EPIC"):
            results = self.data_processor.add_distance_columns(results)
        else:
            for col in ["TSS", "TSS_DIST", "NEAREST_GENE_DIST", "START", "END"]:
                if col in results.columns:
                    results = results.drop(columns=[col])

            if "GENE" in results.columns:

                def _dedup_genes(val: Any) -> Any:
                    if pd.isna(val):
                        return val
                    s = str(val)
                    if ";" not in s:
                        return s.strip()
                    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
                    seen = set()
                    uniq = []
                    for p in parts:
                        if p not in seen:
                            seen.add(p)
                            uniq.append(p)
                    if not uniq:
                        return ""
                    return ";".join(uniq)

                results["GENE"] = results["GENE"].apply(_dedup_genes)

                try:
                    all_genes = set()
                    for v in results["GENE"].dropna().astype(str):
                        if not v:
                            continue
                        for g in [x.strip() for x in v.split(";") if x.strip()]:
                            all_genes.add(g)

                    if all_genes:
                        log.info(
                            f"Converting {len(all_genes)} unique gene symbols to Ensembl IDs"
                        )
                        converted = ConvertGeneID(
                            list(all_genes),
                            id_from="symbol",
                            id_to="ensembl",
                            show_progress=False,
                        )

                        gene_list = list(all_genes)
                        mapping: Dict[str, str] = {}
                        for sym, conv in zip(gene_list, converted):
                            if conv is None:
                                mapping[sym] = ""
                            elif isinstance(conv, list):
                                uniq = []
                                for item in conv:
                                    sitem = str(item).strip()
                                    if sitem and sitem not in uniq:
                                        uniq.append(sitem)
                                mapping[sym] = ";".join(uniq)
                            else:
                                mapping[sym] = str(conv).strip()

                        def _map_genes_to_ids(val: Any) -> Any:
                            if pd.isna(val):
                                return val
                            s = str(val).strip()
                            if not s:
                                return ""
                            parts = [p.strip() for p in s.split(";") if p.strip()]
                            ids = []
                            seen_ids = set()
                            for p in parts:
                                gid = mapping.get(p, "")
                                if not gid:
                                    continue
                                for sub in [
                                    x.strip() for x in gid.split(";") if x.strip()
                                ]:
                                    if sub not in seen_ids:
                                        seen_ids.add(sub)
                                        ids.append(sub)
                            return ";".join(ids) if ids else ""

                        results["GENE_ID"] = results["GENE"].apply(_map_genes_to_ids)
                    else:
                        results["GENE_ID"] = ""
                except Exception as e:
                    log.error(f"Error converting gene symbols to Ensembl IDs: {e}")
                    if "GENE_ID" not in results.columns:
                        results["GENE_ID"] = ""

        results = self.reorder_columns(results)
        results = self.sort_by_p_values(results)
        results = self.data_standardizer.standardize_data_types(results)

        try:
            if not getattr(self, "standardize_col_names", True):
                mapping = (
                    getattr(self.data_standardizer, "last_column_mapping", {}) or {}
                )
                if mapping:
                    invert_map = {v: k for k, v in mapping.items() if v and k != v}
                    if invert_map:
                        results = results.rename(columns=invert_map)
                        log.debug(
                            f"Reverted standardized column names to original: {invert_map}"
                        )
        except Exception as e:
            log.error(f"Failed to revert column mappings in finalize_results: {e}")

        return results

    def detect_analysis_type(self, df: pd.DataFrame) -> str:
        if "RSID" in df.columns:
            return "GWAS"
        elif "CGID" in df.columns:
            if self.chip in ["450k", "EPIC"]:
                return "EWAS_ARRAY"
            return "EWAS"
        else:
            cgid_col = self.data_standardizer.find_annotation_field(df, "CGID")
            rsid_col = self.data_standardizer.find_annotation_field(df, "RSID")

            if rsid_col:
                return "GWAS"
            elif cgid_col:
                if self.chip in ["450k", "EPIC"]:
                    return "EWAS_ARRAY"
                return "EWAS"

            if self.chip in ["450k", "EPIC"]:
                return "EWAS_ARRAY"
            elif self.chip == "Genotype":
                return "GWAS"
            return "EWAS"

    def validate_input_data(self, df: pd.DataFrame) -> bool:
        if self.analysis_type == "GWAS":
            return validate_input_requirements(df, ["RSID"], "GWAS input")
        else:
            return validate_input_requirements(df, ["CGID"], "EWAS input")

    def reorder_columns(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or len(df) == 0:
            return df

        df = df.copy()

        priority_columns = []

        if "RSID" in df.columns:
            priority_columns.append("RSID")
        if "CGID" in df.columns:
            priority_columns.append("CGID")

        for col in ["CHR", "BP", "START", "END"]:
            if col in df.columns:
                priority_columns.append(col)

        for col in ["PVAL", "BETA", "SE", "OR"]:
            if col in df.columns:
                priority_columns.append(col)

        for col in [
            "GENE",
            "GENE_ID",
            "STRAND",
            "TSS",
            "TSS_DIST",
            "NEAREST_GENE_DIST",
            "BIOTYPE",
            "CPG_REGION",
            "REGULATORY_REGION",
            "REG_GENE",
            "REG_DISTANCE",
        ]:
            if col in df.columns:
                priority_columns.append(col)

        remaining = [col for col in df.columns if col not in priority_columns]
        ordered_columns = priority_columns + remaining

        return df[ordered_columns]

    def sort_by_p_values(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or len(df) == 0:
            return df

        df = df.copy()

        pval_col = None
        for col_name in ["PVAL", "P", "PVALUE", "P_VALUE"]:
            if col_name in df.columns:
                pval_col = col_name
                break

        if pval_col is not None:
            try:
                df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
                return df.sort_values(by=pval_col, ascending=True).reset_index(
                    drop=True
                )
            except Exception as e:
                log.warn(f"Could not sort by p-values: {e}")

        return df

    def _ensure_array_manifest(
        self, array_type: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        if array_type is None:
            array_type = self.chip

        condition1 = self._cached_array_manifest is not None
        condition2 = not self._cached_array_manifest.empty if condition1 else False
        if condition1 and condition2:
            return self._cached_array_manifest

        cached = self.array_annotator.get_cached_manifest()
        if cached is not None and not cached.empty:
            self._cached_array_manifest = cached
            return cached

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_file = os.path.join(tmp_dir, f"{array_type}_manifest.csv")
            got = self.array_annotator.get_array_annotation(manifest_file, array_type)
            if not got:
                log.warn(f"Could not download {array_type} manifest")
                return None
            try:
                df = pd.read_csv(manifest_file, low_memory=False)
                self._cached_array_manifest = df
                return df
            except Exception as e:
                log.error(f"Failed to read downloaded manifest: {e}")
                fallback = self.array_annotator.get_cached_manifest()
                if fallback is not None and not fallback.empty:
                    self._cached_array_manifest = fallback
                    return fallback
                return None

    def _fill_chr_from_manifest(self) -> None:
        try:
            if "CHR" in self.data.columns:
                return

            if self.chip not in ("450k", "EPIC"):
                return

            if "CGID" not in self.data.columns:
                log.warn(
                    "Cannot fill CHR from manifest: CGID column not present in input"
                )
                return

            manifest = self._ensure_array_manifest(self.chip)
            if manifest is None or manifest.empty:
                log.warn("Array manifest not available; cannot fill CHR")
                return

            cgid_col = (
                AliasUtils.find_keys(dict.fromkeys(manifest.columns), "CGID") or "CGID"
            )
            chr_col = (
                AliasUtils.find_keys(dict.fromkeys(manifest.columns), "CHR") or "CHR"
            )

            if cgid_col not in manifest.columns or chr_col not in manifest.columns:
                log.warn("Downloaded manifest lacks CGID/CHR columns; cannot map CHR")
                return

            mapping = (
                manifest[[cgid_col, chr_col]]
                .drop_duplicates()
                .rename(columns={cgid_col: "CGID", chr_col: "CHR"})
            )
            mapping = self.data_standardizer.standardize_chromosomes(
                mapping, chr_column="CHR", target_format="with_prefix"
            )

            if "CHR" in self.data.columns:
                merged = pd.merge(
                    self.data,
                    mapping,
                    on="CGID",
                    how="left",
                    suffixes=("", "_from_manifest"),
                )
                merged["CHR"] = merged["CHR"].fillna(merged.get("CHR_from_manifest"))
                if "CHR_from_manifest" in merged.columns:
                    merged = merged.drop(columns=["CHR_from_manifest"])
                self.data = merged
            else:
                self.data = pd.merge(self.data, mapping, on="CGID", how="left")

            filled = self.data["CHR"].notna().sum() if "CHR" in self.data.columns else 0
            log.info(f"Filled CHR from manifest for {filled} rows (chip={self.chip})")
        except Exception as e:
            log.error(f"Error filling CHR from manifest: {e}")
            return

    @handle_errors(default_return=None, error_message="450K array annotation failed")
    def annotate_450k(self) -> Optional[pd.DataFrame]:
        log.info("Starting 450K array annotation")

        with tempfile.TemporaryDirectory() as tmp_dir:
            array_data = self._ensure_array_manifest("450k")
            if array_data is None:
                log.error("Failed to get 450K array annotation")
                return self.data
            log.debug(f"Using 450K manifest with {len(array_data)} entries")

            island_file = os.path.join(tmp_dir, "cpg_islands.csv")
            if not self.resource_manager.get_resource("islands", island_file):
                log.warn("Failed to get CpG islands annotation")

            try:
                island_data = pd.read_csv(island_file, low_memory=False)
                log.debug(
                    f"Loaded CpG islands annotation with {len(island_data)} entries"
                )
            except Exception as e:
                log.error(f"Error reading CpG islands file: {e}")
                island_data = None

            annotated_data = self.data.copy()
            merge_fields = ["CHR", "BP", "STRAND", "GENE", "CPG_REGION"]
            array_fields = [
                field for field in merge_fields if field in array_data.columns
            ]

            if array_fields:
                annotated_data = self.data_standardizer.merge_annotation(
                    annotated_data, array_data, "CGID", array_fields
                )

            condition1 = island_data is not None
            condition2 = "CPG_REGION" not in annotated_data.columns
            condition3 = annotated_data["CPG_REGION"].isna().any()
            if condition1 and (condition2 or condition3):
                annotated_data = self.island_annotator.process_islands(
                    annotated_data, island_data
                )

            return annotated_data

    @handle_errors(default_return=None, error_message="EPIC array annotation failed")
    def annotate_epic(self) -> Optional[pd.DataFrame]:
        log.info("Starting EPIC array annotation")

        with tempfile.TemporaryDirectory() as tmp_dir:
            array_data = self._ensure_array_manifest("EPIC")
            if array_data is None:
                log.error("Failed to get EPIC array annotation")
                return self.data
            log.debug(f"Using EPIC manifest with {len(array_data)} entries")

            island_file = os.path.join(tmp_dir, "cpg_islands.csv")
            if not self.resource_manager.get_resource("islands", island_file):
                log.warn("Failed to get CpG islands annotation")

            try:
                island_data = pd.read_csv(island_file, low_memory=False)
                log.debug(
                    f"Loaded CpG islands annotation with {len(island_data)} entries"
                )
            except Exception as e:
                log.error(f"Error reading CpG islands file: {e}")
                island_data = None

            annotated_data = self.data.copy()

            merge_fields = ["CHR", "BP", "START", "END", "STRAND", "GENE", "CPG_REGION"]
            array_fields = [
                field for field in merge_fields if field in array_data.columns
            ]

            if array_fields:
                annotated_data = self.data_standardizer.merge_annotation(
                    annotated_data, array_data, "CGID", array_fields
                )

            condition1 = island_data is not None
            condition2 = "CPG_REGION" not in annotated_data.columns
            condition3 = annotated_data["CPG_REGION"].isna().any()
            if condition1 and (condition2 or condition3):
                annotated_data = self.island_annotator.process_islands(
                    annotated_data, island_data
                )

            return annotated_data

    @handle_errors(default_return=None, error_message="MethylSeq annotation failed")
    def annotate_methylseq(self) -> Optional[pd.DataFrame]:
        log.info("Starting methylation sequencing annotation")

        with tempfile.TemporaryDirectory() as tmp_dir:
            gene_file = os.path.join(tmp_dir, "ensembl_genes.csv")
            if not self.genomic_annotator.get_ensembl_annotation(
                gene_file, regions=False
            ):
                log.error("Failed to get ENSEMBL gene annotation")
                return self.data

            try:
                gene_data = pd.read_csv(gene_file, low_memory=False)
                log.debug(
                    f"Loaded ENSEMBL gene annotation with {len(gene_data)} entries"
                )
            except Exception as e:
                log.error(f"Error reading ENSEMBL gene file: {e}")
                return self.data

            region_file = os.path.join(tmp_dir, "genomic_regions.csv")
            if not self.genomic_annotator.get_ensembl_annotation(
                region_file, regions=True
            ):
                log.warn("Failed to get genomic regions annotation")
                region_data = None
            else:
                try:
                    region_data = pd.read_csv(region_file, low_memory=False)
                    log.debug(f"Loaded genomic regions with {len(region_data)} entries")
                except Exception as e:
                    log.error(f"Error reading genomic regions file: {e}")
                    region_data = None

            island_file = os.path.join(tmp_dir, "cpg_islands.csv")
            if not self.resource_manager.get_resource("islands", island_file):
                log.warn("Failed to get CpG islands annotation")
                island_data = None
            else:
                try:
                    island_data = pd.read_csv(island_file, low_memory=False)
                    log.debug(f"Loaded CpG islands with {len(island_data)} entries")
                except Exception as e:
                    log.error(f"Error reading CpG islands file: {e}")
                    island_data = None

            log.info("Processing methylation sequencing annotation")
            annotated_data = self.array_annotator.process_methylseq(
                self.data, gene_data, self.analysis_type
            )

            if region_data is not None:
                log.info("Processing genomic regions annotation")
                annotated_data = self.genomic_annotator.process_genomic_regions(
                    annotated_data, region_data
                )
                annotated_data = self.data_standardizer._resolve_duplicate_columns(
                    annotated_data, prefer_y=True
                )

            if island_data is not None:
                log.info("Processing CpG islands annotation")
                annotated_data = self.island_annotator.process_islands(
                    annotated_data, island_data
                )
                annotated_data = self.data_standardizer._resolve_duplicate_columns(
                    annotated_data, prefer_y=True
                )

            return annotated_data

    @handle_errors(default_return=None, error_message="Genotype annotation failed")
    def annotate_genotype(self) -> Optional[pd.DataFrame]:
        log.info("Starting genotype data annotation")

        with tempfile.TemporaryDirectory() as tmp_dir:
            gene_file = os.path.join(tmp_dir, "ensembl_genes.csv")
            if not self.genomic_annotator.get_ensembl_annotation(
                gene_file, regions=False
            ):
                log.error("Failed to get ENSEMBL gene annotation")
                return self.data

            try:
                gene_data = pd.read_csv(gene_file, low_memory=False)
                log.debug(
                    f"Loaded ENSEMBL gene annotation with {len(gene_data)} entries"
                )
            except Exception as e:
                log.error(f"Error reading ENSEMBL gene file: {e}")
                return self.data

            reg_file = os.path.join(tmp_dir, "regulatory_regions.csv")
            if not self.genomic_annotator.get_ensembl_regulatory_annotation(reg_file):
                log.warn("Failed to get regulatory regions annotation")
                reg_data = None
            else:
                try:
                    reg_data = pd.read_csv(reg_file, low_memory=False)
                    log.debug(f"Loaded regulatory regions with {len(reg_data)} entries")
                except Exception as e:
                    log.error(f"Error reading regulatory regions file: {e}")
                    reg_data = None

            annotated_data = self.genomic_annotator.process_genotype(
                self.data, gene_data
            )

            if reg_data is not None:
                annotated_data = self.genomic_annotator.process_regulatory_regions(
                    annotated_data, reg_data
                )

            return annotated_data


options = [
    OptionConfig(flags=["-i", "--input"], type=str),
    OptionConfig(flags=["-o", "--output"], type=str, default=None),
    OptionConfig(flags=["-c", "--chip"], type=str, default="450k"),
    OptionConfig(flags=["-p", "--protein_coding"], type=bool, default=True),
    OptionConfig(flags=["-g", "--genome_version"], type=str, default="hg38"),
    OptionConfig(flags=["-t", "--analysis_type"], type=str, default="auto"),
    OptionConfig(flags=["-r", "--reference"], type=str, default=None),
    OptionConfig(flags=["-s", "--standardize_col_names"], type=bool, default=True),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="Annotator")
    opt = framework.run()

    annotator = Annotator(
        input_file=opt.input,
        output=opt.output,
        chip=opt.chip,
        protein_coding=opt.protein_coding,
        genome_version=opt.genome_version,
        analysis_type=opt.analysis_type,
        reference=opt.reference,
        standardize_col_names=opt.standardize_col_names,
    )
    annotator.annotate()
