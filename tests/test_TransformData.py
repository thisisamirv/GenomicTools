#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import pytest
import shutil
import tempfile
import TransformData
from TransformData import (
    DataProcessor,
    DataTransformer,
    DataSplitter,
    DataMerger,
    TransformData as TransformDataClass,
)
from utils.AliasUtils import AliasUtils
from utils.LoggingUtils import log

log.setup(level="DEBUG")

TransformData.AliasUtils = AliasUtils


class AliasUtils:
    @staticmethod
    def find_keys(columns, query):
        for col in columns:
            if col.lower() == query.lower():
                return col

        if query.lower() == "id" and "SampleID".lower() in [c.lower() for c in columns]:
            return next(c for c in columns if c.lower() == "sampleid")

        if query.lower() == "sampleid" and "ID".lower() in [c.lower() for c in columns]:
            return next(c for c in columns if c.lower() == "id")

        return None


@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def csv_data(temp_dir):
    data = {
        "ID": ["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"],
        "Value1": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        "Value2": [10, 20, 30, 40, 50, 60],
        "Category": ["A", "B", "A", "C", "B", "C"],
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(temp_dir, "test_data.csv")
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def tsv_data(temp_dir):
    data = {
        "SampleID": ["sample1", "sample2", "sample3", "sample4", "sample5"],
        "Measurement": [0.5, 1.5, 2.5, 3.5, 4.5],
        "Group": ["X", "Y", "X", "Z", "Y"],
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(temp_dir, "test_data.tsv")
    df.to_csv(file_path, sep="\t", index=False)
    return file_path


@pytest.fixture
def second_data(temp_dir):
    data = {
        "ID": ["sample1", "sample2", "sample6"],
        "Extra1": ["extra1", "extra2", "extra3"],
        "Extra2": [100, 200, 300],
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(temp_dir, "second_data.csv")
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def alias_data(temp_dir):
    data = {
        "SAMPLE_ID": ["sample1", "sample2", "sample3", "sample4", "sample5", "sample6"],
        "VALUE": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        "CATEGORY_TYPE": ["A", "B", "A", "C", "B", "C"],
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(temp_dir, "alias_data.csv")
    df.to_csv(file_path, index=False)
    return file_path


@pytest.mark.unit
def test_detect_separator():
    assert DataProcessor.detect_separator("file.csv") == ","
    assert DataProcessor.detect_separator("file.tsv") == "\t"
    assert DataProcessor.detect_separator("file.txt") == "\t"
    assert DataProcessor.detect_separator("file.xlsx") == ","


@pytest.mark.unit
def test_read_file(csv_data, tsv_data):
    csv_df = DataProcessor.read_file(csv_data)
    assert isinstance(csv_df, pd.DataFrame)
    assert "ID" in csv_df.columns
    assert len(csv_df) == 6

    tsv_df = DataProcessor.read_file(tsv_data)
    assert isinstance(tsv_df, pd.DataFrame)
    assert "SampleID" in tsv_df.columns
    assert len(tsv_df) == 5


@pytest.mark.unit
def test_write_file(temp_dir, csv_data):
    df = DataProcessor.read_file(csv_data)
    output_path = os.path.join(temp_dir, "output.csv")
    DataProcessor.write_file(df, output_path)
    new_df = pd.read_csv(output_path)
    assert df.equals(new_df)
    output_tsv = os.path.join(temp_dir, "output.tsv")
    DataProcessor.write_file(df, output_tsv)
    new_tsv_df = pd.read_csv(output_tsv, sep="\t")
    assert df.equals(new_tsv_df)


@pytest.mark.unit
def test_resolve_column_aliases(alias_data):
    df = DataProcessor.read_file(alias_data)
    columns = ["SAMPLE_ID", "VALUE"]
    resolved = DataProcessor.resolve_column_aliases(df, columns)
    assert resolved == columns

    TransformData.AliasUtils = AliasUtils
    alias_columns = ["sample_id", "value", "category_type"]
    resolved = DataProcessor.resolve_column_aliases(df, alias_columns)
    assert "SAMPLE_ID" in resolved
    assert "VALUE" in resolved
    assert "CATEGORY_TYPE" in resolved


@pytest.mark.unit
def test_filter_dataframe(csv_data):
    transformer = DataTransformer(df1=csv_data, output="output.csv")
    df = DataProcessor.read_file(csv_data)

    filters = {"Category": "A"}
    filtered = transformer.filter_dataframe(df, filters)
    assert len(filtered) == 2
    assert all(filtered["Category"] == "A")

    filters = {"Category": ["A", "B"]}
    filtered = transformer.filter_dataframe(df, filters)
    assert len(filtered) == 4
    assert all(filtered["Category"].isin(["A", "B"]))

    filters = {"NonExistent": "A"}
    filtered = transformer.filter_dataframe(df, filters)
    assert filtered is None


@pytest.mark.unit
def test_extract_data(csv_data):
    transformer = DataTransformer(df1=csv_data, output="output.csv")
    df = DataProcessor.read_file(csv_data)

    extract_params = {"col": "ID,Value1"}
    extracted = transformer.extract_data(df, extract_params)
    assert set(extracted.columns) == {"ID", "Value1"}

    df = pd.concat([df, df.iloc[0:2]], ignore_index=True)
    extract_params = {"unique": "True"}
    extracted = transformer.extract_data(df, extract_params)
    assert len(extracted) == 6


@pytest.mark.unit
def test_scale_data(csv_data):
    transformer = DataTransformer(df1=csv_data, output="output.csv")
    df = DataProcessor.read_file(csv_data)

    df["Value1"] = pd.to_numeric(df["Value1"])
    df["Value2"] = pd.to_numeric(df["Value2"])

    scale_params = {"col": "Value1", "zero_one": "True"}
    scaled = transformer.scale_data(df, scale_params)
    assert np.isclose(scaled["Value1"].min(), 0)
    assert np.isclose(scaled["Value1"].max(), 1)

    scale_params = {"col": "Value2", "z_scale": "True"}
    scaled = transformer.scale_data(df, scale_params)
    assert np.isclose(scaled["Value2"].mean(), 0, atol=1e-10)
    assert np.isclose(scaled["Value2"].std(), 1, atol=0.1)


@pytest.mark.unit
def test_convert_values(csv_data):
    transformer = DataTransformer(df1=csv_data, output="output.csv")
    df = DataProcessor.read_file(csv_data)
    conversion = {"A": "X", "B": "Y", "C": "Z"}
    converted = transformer.convert_values(df, conversion)
    assert all(converted[converted["Category"] == "X"]["Category"] == "X")
    assert all(converted[converted["Category"] == "Y"]["Category"] == "Y")
    assert all(converted[converted["Category"] == "Z"]["Category"] == "Z")
    assert "A" not in converted["Category"].values
    assert "B" not in converted["Category"].values
    assert "C" not in converted["Category"].values


@pytest.mark.integration
def test_transform_full_pipeline(csv_data, temp_dir):
    output_path = os.path.join(temp_dir, "transformed.csv")
    transformer = DataTransformer(
        df1=csv_data,
        output=output_path,
        transform="transpose=True",
        filters="Category=A",
        extract="col=ID,Value1,Value2",
        scale="col=Value1,Value2;zero_one=True",
    )
    transformer.transform()
    assert os.path.exists(output_path)
    result = pd.read_csv(output_path)
    assert result.shape[0] > 0
    assert result.shape[1] > 0


@pytest.mark.integration
def test_split_data(csv_data, temp_dir, monkeypatch):
    output_path = os.path.join(temp_dir, "split.csv")

    def mock_parse(input_str):
        if not input_str:
            return {}
        result = {}
        for pair in input_str.split(";"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key] = value
        return result

    monkeypatch.setattr(TransformData, "ParseToKeyValueDict", mock_parse)

    splitter = DataSplitter(
        df1=csv_data,
        output=output_path,
        split="stratify_var=Category;train_fraction=0.6",
    )
    splitter.split_data()
    assert os.path.exists(output_path)
    result = pd.read_csv(output_path)
    assert "set" in result.columns
    assert "target" in result.columns
    train_count = len(result[result["set"] == "train"])
    test_count = len(result[result["set"] == "test"])
    assert train_count == 3
    assert test_count == 3


@pytest.mark.integration
def test_split_data_with_alias(alias_data, temp_dir, monkeypatch):
    output_path = os.path.join(temp_dir, "split_alias.csv")

    def mock_parse(input_str):
        if not input_str:
            return {}
        result = {}
        for pair in input_str.split(";"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key] = value
        return result

    monkeypatch.setattr(TransformData, "ParseToKeyValueDict", mock_parse)

    splitter = DataSplitter(
        df1=alias_data,
        output=output_path,
        split="stratify_var=CATEGORY_TYPE;train_fraction=0.6",
    )
    splitter.split_data()
    assert os.path.exists(output_path)
    result = pd.read_csv(output_path)
    assert "set" in result.columns
    assert "target" in result.columns
    train_count = len(result[result["set"] == "train"])
    test_count = len(result[result["set"] == "test"])
    assert train_count == 3
    assert test_count == 3


@pytest.mark.integration
def test_merge_dataframes(csv_data, second_data, temp_dir, monkeypatch):
    output_path = os.path.join(temp_dir, "merged.csv")

    def mock_parse(input_str):
        if not input_str:
            return {}
        result = {}
        for pair in input_str.split(";"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key] = value
        return result

    monkeypatch.setattr(TransformData, "ParseToKeyValueDict", mock_parse)

    merger = DataMerger(
        df1=csv_data,
        df2=second_data,
        output=output_path,
        merge="join_on=ID;overlaps=drop",
    )
    merged = merger.merge_dataframes()
    assert os.path.exists(output_path)
    assert "Extra1" in merged.columns
    assert "Extra2" in merged.columns
    assert "Value1" in merged.columns
    assert "Value2" in merged.columns
    assert len(merged) == 6


@pytest.mark.integration
def test_merge_with_aliases(csv_data, temp_dir, monkeypatch):
    alias_df = pd.DataFrame(
        {
            "SampleID": ["sample1", "sample2", "sample6"],
            "Additional": [1000, 2000, 3000],
        }
    )
    alias_path = os.path.join(temp_dir, "alias_merge.csv")
    alias_df.to_csv(alias_path, index=False)
    output_path = os.path.join(temp_dir, "merged_alias.csv")

    def mock_parse(input_str):
        if not input_str:
            return {}
        result = {}
        for pair in input_str.split(";"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key] = value
        return result

    monkeypatch.setattr(TransformData, "ParseToKeyValueDict", mock_parse)

    merger = DataMerger(
        df1=csv_data,
        df2=alias_path,
        output=output_path,
        merge="join_on=ID;overlaps=drop",
    )
    merged = merger.merge_dataframes()
    assert os.path.exists(output_path)
    assert "Additional" in merged.columns

    sample1 = merged[merged["ID"] == "sample1"]
    assert sample1["Additional"].iloc[0] == 1000


@pytest.mark.integration
def test_transform_operation(csv_data, temp_dir):
    output_path = os.path.join(temp_dir, "transform_result.csv")
    transform_data = TransformDataClass(
        operation="transform",
        df1=csv_data,
        output=output_path,
        filters="Category=A",
        extract="col=ID,Value1",
        transform="transpose=True",
    )
    transform_data.run()
    assert os.path.exists(output_path)
    result = pd.read_csv(output_path)
    assert result.shape[0] > 0


@pytest.mark.integration
def test_split_operation(csv_data, temp_dir, monkeypatch):
    output_path = os.path.join(temp_dir, "split_result.csv")

    def mock_parse(input_str):
        if not input_str:
            return {}
        result = {}
        for pair in input_str.split(";"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key] = value
        return result

    monkeypatch.setattr(TransformData, "ParseToKeyValueDict", mock_parse)

    transform_data = TransformDataClass(
        operation="split",
        df1=csv_data,
        output=output_path,
        split="stratify_var=Category;train_fraction=0.6",
    )
    transform_data.run()
    assert os.path.exists(output_path)
    result = pd.read_csv(output_path)
    assert "set" in result.columns
    assert "target" in result.columns


@pytest.mark.integration
def test_merge_operation(csv_data, second_data, temp_dir, monkeypatch):
    output_path = os.path.join(temp_dir, "merge_result.csv")

    def mock_parse(input_str):
        if not input_str:
            return {}
        result = {}
        for pair in input_str.split(";"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                result[key] = value
        return result

    monkeypatch.setattr(TransformData, "ParseToKeyValueDict", mock_parse)

    transform_data = TransformDataClass(
        operation="merge",
        df1=csv_data,
        df2=second_data,
        output=output_path,
        merge="join_on=ID;overlaps=drop",
    )
    transform_data.run()
    assert os.path.exists(output_path)
    result = pd.read_csv(output_path)
    assert "Extra1" in result.columns
    assert "Value1" in result.columns


@pytest.mark.integration
def test_invalid_operation():
    with pytest.raises(ValueError):
        transform_data = TransformDataClass(
            operation="invalid_op", df1="dummy.csv", output="dummy_out.csv"
        )
        transform_data.run()


@pytest.mark.integration
def test_missing_required_params():
    with pytest.raises(ValueError):
        transform_data = TransformDataClass(
            operation="transform", output="dummy_out.csv"
        )
        transform_data.run()
    with pytest.raises(ValueError):
        transform_data = TransformDataClass(
            operation="split", df1="dummy.csv", split="stratify_var=X"
        )
        transform_data.run()
    with pytest.raises(ValueError):
        transform_data = TransformDataClass(
            operation="merge", df1="dummy.csv", df2="dummy2.csv", output="dummy_out.csv"
        )
        transform_data.run()
