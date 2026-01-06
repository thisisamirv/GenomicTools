#!/usr/bin/env python
# Import required modules
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Optional, Union
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.LoggingUtils import log
from utils.ParsingUtils import ParseToKeyValueDict


class DataProcessor:
    @staticmethod
    def detect_separator(file_path: str) -> str:
        if file_path.endswith((".tsv", ".txt")):
            return "\t"
        else:
            return ","

    @staticmethod
    def read_file(file_path: str, sep: Optional[str] = None) -> pd.DataFrame:
        if sep is None:
            sep = DataProcessor.detect_separator(file_path)
        return pd.read_csv(file_path, sep=sep)

    @staticmethod
    def write_file(
        df: pd.DataFrame,
        file_path: str,
        sep: Optional[str] = None,
        index: bool = False,
        header: bool = True,
    ) -> None:
        if sep is None:
            sep = DataProcessor.detect_separator(file_path)
        df.to_csv(file_path, sep=sep, index=index, header=header)

    @staticmethod
    def resolve_column_aliases(df: pd.DataFrame, column_names: List[str]) -> List[str]:
        resolved_columns: List[str] = []
        for col in column_names:
            if col in df.columns:
                resolved_columns.append(col)
            else:
                alias = AliasUtils.find_keys(df.columns, col)
                if alias:
                    log.debug(f"Resolved column alias: '{col}' -> '{alias}'")
                    resolved_columns.append(alias)
                else:
                    log.warn(f"Column '{col}' not found and no alias exists")
        return resolved_columns


class DataTransformer:
    def __init__(
        self,
        df1: str,
        output: str,
        transform: Optional[str] = None,
        filters: Optional[str] = None,
        extract: Optional[str] = None,
        convert: Optional[str] = None,
        scale: Optional[str] = None,
    ) -> None:
        self.input_file: str = df1
        self.output_file: str = output
        self.transform_params: Dict[str, str] = (
            ParseToKeyValueDict(transform) if transform else {}
        )
        self.filters: Optional[str] = filters
        self.extract: Optional[str] = extract
        self.convert: Optional[str] = convert
        self.scale: Optional[str] = scale
        self.df: Optional[pd.DataFrame] = None

    def filter_dataframe(
        self, data: pd.DataFrame, filters_dict: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        log.debug(f"Filtering data with filters: {filters_dict}")
        try:
            filtered_data = data.copy()
            for filter_name, filter_value in filters_dict.items():
                log.debug(f"Applying filter: {filter_name} with value: {filter_value}")
                column = filter_name
                if filter_name not in filtered_data.columns:
                    alias = AliasUtils.find_keys(filtered_data.columns, filter_name)
                    if alias:
                        column = alias
                        log.debug(f"Using column alias: '{filter_name}' -> '{alias}'")
                    else:
                        log.warn(f"Filter column {filter_name} not found in data")
                        return None
                if isinstance(filter_value, list):
                    filtered_data = filtered_data[
                        filtered_data[column].isin(filter_value)
                    ]
                else:
                    filtered_data = filtered_data[filtered_data[column] == filter_value]
            if filtered_data.empty:
                log.warn("No data remains after filtering")
            log.debug(f"Filtered data shape: {filtered_data.shape}")
            return filtered_data
        except Exception as e:
            log.error(f"Error filtering data: {e}")
            return None

    def extract_data(
        self, df: pd.DataFrame, extract_params: Dict[str, str]
    ) -> pd.DataFrame:
        try:
            row = extract_params.get("row", None)
            col = extract_params.get("col", None)
            unique = extract_params.get("unique", "False").lower() == "true"
            log.debug(f"Extract params - row: {row}, col: {col}, unique: {unique}")
            if col is not None and col.lower() != "none":
                log.debug(f"Extracting columns: {col}")
                columns = [col.strip() for col in col.split(",")]
                resolved_columns: List[str] = []
                for column in columns:
                    if column in df.columns:
                        resolved_columns.append(column)
                    else:
                        alias = AliasUtils.find_keys(df.columns, column)
                        if alias:
                            log.debug(f"Using column alias: '{column}' -> '{alias}'")
                            resolved_columns.append(alias)
                        else:
                            log.warn(f"Column '{column}' not found and no alias exists")
                if resolved_columns:
                    df = df[resolved_columns]
                    log.debug(f"Data shape after column extraction: {df.shape}")
                else:
                    log.warn("No valid columns found for extraction")
            if row is not None and row.lower() != "none":
                log.debug(f"Extracting rows: {row}")
                rows = [row.strip() for row in row.split(",")]
                df = df.loc[rows]
                log.debug(f"Data shape after row extraction: {df.shape}")
            if unique:
                log.debug("Removing duplicates")
                df = df.drop_duplicates()
                log.debug(f"Data shape after removing duplicates: {df.shape}")
            return df
        except Exception as e:
            log.error(f"Error extracting data: {e}")
            raise

    def scale_zero_one(
        self, scores: Union[np.ndarray, pd.Series, List[float]]
    ) -> Optional[np.ndarray]:
        try:
            scores_arr = np.array(scores)
            min_val = np.min(scores_arr)
            max_val = np.max(scores_arr)
            if max_val == min_val:
                log.warn("All values are the same, returning zeros")
                return np.zeros_like(scores_arr)
            normalized_scores = (scores_arr - min_val) / (max_val - min_val)
            return normalized_scores
        except Exception as e:
            log.error(f"Error normalizing scores: {e}")
            return None

    def z_scale_data(self, data: Union[pd.Series, np.ndarray]) -> Optional[np.ndarray]:
        try:
            if isinstance(data, pd.Series):
                data = pd.to_numeric(data, errors="coerce")
                data_array = data.values.reshape(-1, 1)
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data_array).flatten()
                return scaled_data
            else:
                scaler = StandardScaler()
                arr = np.array(data)
                if arr.ndim == 1:
                    scaled_data = scaler.fit_transform(arr.reshape(-1, 1)).flatten()
                else:
                    scaled_data = scaler.fit_transform(arr)
                return scaled_data
        except Exception as e:
            log.error(f"Error z-scaling data: {e}")
            return None

    def scale_data(
        self, df: pd.DataFrame, scale_params: Dict[str, str]
    ) -> pd.DataFrame:
        try:
            row = scale_params.get("row", None)
            col = scale_params.get("col", None)
            zero_one = scale_params.get("zero_one", "False").lower() == "true"
            z_scale = scale_params.get("z_scale", "False").lower() == "true"
            log.debug(
                f"Scale params - row: {row}, col: {col}, zero_one: {zero_one}, z_scale: {z_scale}"
            )
            if not zero_one and not z_scale:
                log.warn("No scaling method specified")
                return df
            df_copy = df.copy()
            if col is not None and col.lower() != "none":
                columns = [c.strip() for c in col.split(",")]
                resolved_columns: List[str] = []
                for column in columns:
                    if column in df_copy.columns:
                        resolved_columns.append(column)
                    else:
                        alias = AliasUtils.find_keys(df_copy.columns, column)
                        if alias:
                            log.debug(f"Using column alias: '{column}' -> '{alias}'")
                            resolved_columns.append(alias)
                        else:
                            log.warn(
                                f"Scale column '{column}' not found and no alias exists"
                            )
                for column in resolved_columns:
                    if zero_one:
                        scaled = self.scale_zero_one(df_copy[column])
                        df_copy[column] = scaled
                        log.debug(f"Applied 0-1 scaling to column: {column}")
                    elif z_scale:
                        scaled = self.z_scale_data(df_copy[column])
                        df_copy[column] = scaled
                        log.debug(f"Applied z-scaling to column: {column}")
            if row is not None and row.lower() != "none":
                rows = [r.strip() for r in row.split(",")]
                for row_name in rows:
                    if row_name in df_copy.index:
                        if zero_one:
                            scaled = self.scale_zero_one(df_copy.loc[row_name])
                            df_copy.loc[row_name] = scaled
                            log.debug(f"Applied 0-1 scaling to row: {row_name}")
                        elif z_scale:
                            scaled = self.z_scale_data(df_copy.loc[row_name])
                            df_copy.loc[row_name] = scaled
                            log.debug(f"Applied z-scaling to row: {row_name}")
                    else:
                        log.warn(f"Scale row {row_name} not found in dataframe")
            return df_copy
        except Exception as e:
            log.error(f"Error scaling data: {e}")
            raise

    def convert_values(
        self, df: pd.DataFrame, conversion_dict: Dict[Any, Any]
    ) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            for old_val, new_val in conversion_dict.items():
                df_copy = df_copy.replace(old_val, new_val)
                log.debug(f"Converted {old_val} to {new_val}")
            log.debug(f"Value conversion completed for {len(conversion_dict)} mappings")
            return df_copy
        except Exception as e:
            log.error(f"Error in convert_values: {e}")
            raise

    def transform(self) -> None:
        log.info("Starting data transformation")
        sep = self.transform_params.get("sep", ",")
        if sep in ["'\\t'", '"\\t"', "\\t"]:
            sep = "\t"
        change_sep = self.transform_params.get("change_sep", "False").lower() == "true"
        header = self.transform_params.get("header", "True").lower() == "true"
        transpose = self.transform_params.get("transpose", "False").lower() == "true"
        log.info(f"Reading input file: {self.input_file}")
        self.df = DataProcessor.read_file(self.input_file, sep)
        log.info(f"Loaded data with shape: {self.df.shape}")
        if self.filters is not None:
            log.info("Applying filters")
            filters = ParseToKeyValueDict(self.filters)
            self.df = self.filter_dataframe(self.df, filters)
            if self.df is None:
                raise ValueError("Filtering resulted in empty dataset")
            log.info(f"Data shape after filtering: {self.df.shape}")
        if self.extract is not None:
            log.info("Extracting data")
            extract_params = ParseToKeyValueDict(self.extract)
            self.df = self.extract_data(self.df, extract_params)
            log.info(f"Data shape after extraction: {self.df.shape}")
        if self.convert is not None:
            log.info("Converting values")
            convert = ParseToKeyValueDict(self.convert)
            self.df = self.convert_values(self.df, convert)
            log.info("Value conversion completed")
        if self.scale is not None:
            log.info("Scaling data")
            scale_params = ParseToKeyValueDict(self.scale)
            self.df = self.scale_data(self.df, scale_params)
            log.info("Data scaling completed")
        if transpose:
            log.info("Transposing data")
            if self.df is not None:
                self.df = self.df.T
                log.info(f"Data shape after transpose: {self.df.shape}")
        log.info(f"Writing output file: {self.output_file}")
        output_sep = sep
        if change_sep:
            output_sep = "\t" if sep == "," else ","
        if self.df is not None:
            DataProcessor.write_file(
                self.df, self.output_file, output_sep, index=False, header=header
            )
            log.success(
                f"Data transformed successfully and saved to {self.output_file}"
            )
        else:
            raise ValueError("No data available to write after transformation")


class DataSplitter:
    def __init__(self, df1: str, output: str, split: Optional[str]) -> None:
        self.input_file: str = df1
        self.output_file: str = output
        self.split_params: Dict[str, str] = ParseToKeyValueDict(split) if split else {}

    def split_data(self) -> None:
        try:
            log.info("Starting data splitting")
            stratify_var = self.split_params.get("stratify_var", None)
            train_fraction = float(self.split_params.get("train_fraction", 0.7))
            if not stratify_var:
                raise ValueError("stratify_var is required for split operation")
            log.info(f"Loading data from: {self.input_file}")
            data = DataProcessor.read_file(self.input_file)
            log.info(f"Loaded data with shape: {data.shape}")
            if "sample_id" in data.columns:
                log.info("Sorting data by sample_id")
                data = data.sort_values("sample_id")
            else:
                sample_id_alias = AliasUtils.find_keys(data.columns, "sample_id")
                if sample_id_alias:
                    log.info(f"Sorting data by {sample_id_alias} (alias for sample_id)")
                    data = data.sort_values(sample_id_alias)
                else:
                    log.warn("No 'sample_id' column or alias found. Skipping sort.")
            if stratify_var not in data.columns:
                alias = AliasUtils.find_keys(data.columns, stratify_var)
                if alias:
                    log.info(
                        f"Using alias '{alias}' for stratification variable '{stratify_var}'"
                    )
                    stratify_var = alias
                else:
                    raise ValueError(
                        f"Stratification variable '{stratify_var}' not found in data and no alias exists"
                    )
            y = data[stratify_var]
            class_counts = y.value_counts()
            log.info(f"Class distribution in {stratify_var}:")
            for class_val, count in class_counts.items():
                log.info(f"  {class_val}: {count} ({count / len(y) * 100:.1f}%)")
            log.info(f"Splitting data with train fraction: {train_fraction}")
            train_idx, test_idx = train_test_split(
                range(len(data)),
                train_size=train_fraction,
                random_state=42,
                stratify=y,
            )
            log.info(f"Train set size: {len(train_idx)}")
            log.info(f"Test set size: {len(test_idx)}")
            cols_to_keep = [col for col in data.columns if col != stratify_var]
            train_data = data.iloc[train_idx][cols_to_keep].copy()
            test_data = data.iloc[test_idx][cols_to_keep].copy()
            train_data["set"] = "train"
            train_data["target"] = y.iloc[train_idx].values
            test_data["set"] = "test"
            test_data["target"] = y.iloc[test_idx].values
            final_data = pd.concat([train_data, test_data], ignore_index=True)
            log.info(f"Saving split data to: {self.output_file}")
            DataProcessor.write_file(final_data, self.output_file)
            log.success("Data split successfully")
            log.info(f"Final data shape: {final_data.shape}")
        except Exception as e:
            log.error(f"Error in data splitting: {e}")
            sys.exit(1)


class DataMerger:
    def __init__(self, df1: str, df2: str, output: str, merge: Optional[str]) -> None:
        self.df1_path: str = df1
        self.df2_path: str = df2
        self.output_file: str = output
        self.merge_params: Dict[str, str] = ParseToKeyValueDict(merge) if merge else {}

    def merge_dataframes(self) -> pd.DataFrame:
        try:
            log.info("Starting dataframe merge")
            join_on = self.merge_params.get("join_on", None)
            overlaps = self.merge_params.get("overlaps", "drop")
            limit = self.merge_params.get("limit", None)
            if not join_on:
                raise ValueError("join_on is required for merge operation")
            log.info(f"Loading first dataframe: {self.df1_path}")
            df1 = DataProcessor.read_file(self.df1_path)
            log.info(f"First dataframe shape: {df1.shape}")
            log.info(f"Loading second dataframe: {self.df2_path}")
            df2 = DataProcessor.read_file(self.df2_path)
            log.info(f"Second dataframe shape: {df2.shape}")
            on_cols = [join_on] if isinstance(join_on, str) else list(join_on)
            log.info(f"Joining on columns: {on_cols}")
            df1_join_cols: List[str] = []
            df2_join_cols: List[str] = []
            for col in on_cols:
                if col in df1.columns:
                    df1_join_cols.append(col)
                else:
                    alias = AliasUtils.find_keys(df1.columns, col)
                    if alias:
                        log.info(
                            f"Using alias '{alias}' for join column '{col}' in first dataframe"
                        )
                        df1_join_cols.append(alias)
                    else:
                        raise ValueError(
                            f"Join column '{col}' not found in first dataframe and no alias exists"
                        )
                if col in df2.columns:
                    df2_join_cols.append(col)
                else:
                    alias = AliasUtils.find_keys(df2.columns, col)
                    if alias:
                        log.info(
                            f"Using alias '{alias}' for join column '{col}' in second dataframe"
                        )
                        df2_join_cols.append(alias)
                    else:
                        raise ValueError(
                            f"Join column '{col}' not found in second dataframe and no alias exists"
                        )
            if limit is not None and limit.lower() != "none":
                log.info(f"Limiting second dataframe to columns: {limit}")
                limit_cols = [limit] if isinstance(limit, str) else list(limit)
                resolved_limit_cols: List[str] = []
                for col in limit_cols:
                    if col in df2.columns:
                        resolved_limit_cols.append(col)
                    else:
                        alias = AliasUtils.find_keys(df2.columns, col)
                        if alias:
                            log.info(f"Using alias '{alias}' for limit column '{col}'")
                            resolved_limit_cols.append(alias)
                        else:
                            log.warn(
                                f"Limit column '{col}' not found and no alias exists"
                            )
                limit_cols = df2_join_cols + resolved_limit_cols
                df2 = df2[limit_cols]
                log.info(f"Limited second dataframe shape: {df2.shape}")
            df1_cols = set(df1.columns)
            df2_cols = set(df2.columns)
            overlap = df1_cols.intersection(df2_cols).difference(set(df1_join_cols))
            if overlap:
                log.info(f"Found overlapping columns: {list(overlap)}")
                log.info(f"Overlap handling strategy: {overlaps}")
            if df1_join_cols != df2_join_cols:
                rename_dict = {
                    df2_join_cols[i]: df1_join_cols[i]
                    for i in range(len(df1_join_cols))
                }
                df2 = df2.rename(columns=rename_dict)
                log.info(f"Renamed join columns in second dataframe: {rename_dict}")
                df2_join_cols = df1_join_cols
            if overlaps == "drop":
                df2_processed = df2.drop(columns=list(overlap), errors="ignore")
                merged = pd.merge(df1, df2_processed, on=df1_join_cols, how="left")
            elif overlaps == "fill_na":
                merged = pd.merge(
                    df1, df2, on=df1_join_cols, how="left", suffixes=("", "_df2")
                )
                for col in overlap:
                    col_df2 = f"{col}_df2"
                    if col_df2 in merged.columns:
                        merged[col] = merged[col].combine_first(merged[col_df2])
                        merged = merged.drop(columns=[col_df2])
            else:
                raise ValueError("overlaps must be 'drop' or 'fill_na'")
            merged = merged.reset_index(drop=True)
            log.info(f"Merged dataframe shape: {merged.shape}")
            if self.output_file:
                log.info(f"Saving merged dataframe to: {self.output_file}")
                DataProcessor.write_file(merged, self.output_file)
                log.success(f"Merged DataFrame saved to: {self.output_file}")
            return merged
        except Exception as e:
            log.error(f"Error in dataframe merge: {e}")
            sys.exit(1)


class TransformData:
    def __init__(
        self,
        operation: str,
        df1: Optional[str] = None,
        df2: Optional[str] = None,
        output: Optional[str] = None,
        transform: Optional[str] = None,
        filters: Optional[str] = None,
        extract: Optional[str] = None,
        convert: Optional[str] = None,
        scale: Optional[str] = None,
        split: Optional[str] = None,
        merge: Optional[str] = None,
    ) -> None:
        self.operation: str = operation
        self.df1: Optional[str] = df1
        self.df2: Optional[str] = df2
        self.output: Optional[str] = output
        self.transform: Optional[str] = transform
        self.filters: Optional[str] = filters
        self.extract: Optional[str] = extract
        self.convert: Optional[str] = convert
        self.scale: Optional[str] = scale
        self.split: Optional[str] = split
        self.merge: Optional[str] = merge

    def run(self) -> None:
        try:
            if self.operation == "transform":
                if not self.df1:
                    raise ValueError("df1 is required for transform operation")
                if not self.output:
                    raise ValueError("output is required for transform operation")
                transformer = DataTransformer(
                    df1=self.df1,
                    output=self.output,
                    transform=self.transform,
                    filters=self.filters,
                    extract=self.extract,
                    convert=self.convert,
                    scale=self.scale,
                )
                transformer.transform()
            elif self.operation == "split":
                if not all([self.df1, self.output, self.split]):
                    raise ValueError(
                        "df1, output, and split are required for split operation"
                    )
                splitter = DataSplitter(
                    df1=self.df1,
                    output=self.output,
                    split=self.split,
                )
                splitter.split_data()
            elif self.operation == "merge":
                if not all([self.df1, self.df2, self.output, self.merge]):
                    raise ValueError(
                        "df1, df2, output, and merge are required for merge operation"
                    )
                merger = DataMerger(
                    df1=self.df1,
                    df2=self.df2,
                    output=self.output,
                    merge=self.merge,
                )
                merger.merge_dataframes()
            else:
                raise ValueError(f"Unknown operation: {self.operation}")
        except Exception as e:
            log.error(f"Error in TransformData pipeline: {e}")
            raise
        finally:
            log.success("TransformData pipeline completed")


options = [
    OptionConfig(
        flags=["-op", "--operation"],
        type=str,
        required=True,
        choices=["transform", "split", "merge"],
    ),
    OptionConfig(flags=["-df1", "--df1"], type=str, default=None, required=False),
    OptionConfig(flags=["-df2", "--df2"], type=str, default=None, required=False),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
    OptionConfig(flags=["-tr", "--transform"], type=str, default=None, required=False),
    OptionConfig(flags=["-f", "--filters"], type=str, default=None, required=False),
    OptionConfig(flags=["-x", "--extract"], type=str, default=None, required=False),
    OptionConfig(flags=["-cv", "--convert"], type=str, default=None, required=False),
    OptionConfig(flags=["-sc", "--scale"], type=str, default=None, required=False),
    OptionConfig(flags=["-sp", "--split"], type=str, default=None, required=False),
    OptionConfig(flags=["-mg", "--merge"], type=str, default=None, required=False),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="TransformData")
    opt = framework.run()
    pipeline = TransformData(
        operation=opt.operation,
        df1=opt.df1,
        df2=opt.df2,
        output=opt.output,
        transform=opt.transform,
        filters=opt.filters,
        extract=opt.extract,
        convert=opt.convert,
        scale=opt.scale,
        split=opt.split,
        merge=opt.merge,
    )
    pipeline.run()
