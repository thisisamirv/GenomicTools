#!/usr/bin/env python
import h5py
import numpy as np
import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from utils.AliasUtils import AliasUtils
from utils.H5Utils import (
    BaseH5Utils,
    CachedH5Utils,
    ChromosomeLister,
    ChromosomeMapper,
    ChromosomeReader,
    DataTypeDetector,
    H5Config,
    H5UtilsFactory,
    MarkerIndexer,
    SampleIndexer,
)
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.fixture
def methyl_h5_file(data_dir):
    file_path = os.path.join(data_dir, "mcseq.h5")
    if not os.path.exists(file_path):
        pytest.skip(f"Methylation test file not found at {file_path}")
    return h5py.File(file_path, "r")


@pytest.fixture
def genotype_h5_file(data_dir):
    file_path = os.path.join(data_dir, "gen_data.h5")
    if not os.path.exists(file_path):
        pytest.skip(f"Genotype test file not found at {file_path}")
    return h5py.File(file_path, "r")


@pytest.fixture
def mock_h5_methylation():
    mock_file = MagicMock()
    mock_file.keys.return_value = ["CHR1", "CHR2", "Metadata"]

    def contains_func(item):
        return item in [
            "CHR1",
            "CHR2",
            "Metadata",
            "/Metadata/SampleList",
            "Metadata/SampleList",
            "/CHR1/ProbeList",
            "CHR1/ProbeList",
            "/CHR1/Methylation",
            "CHR1/Methylation",
            "/CHR2/ProbeList",
            "CHR2/ProbeList",
            "/CHR2/Methylation",
            "CHR2/Methylation",
        ]

    mock_file.__contains__.side_effect = contains_func

    mock_chr1 = MagicMock()
    mock_chr1.keys.return_value = ["ProbeList", "Methylation"]

    def chr1_contains_func(item):
        return item in ["ProbeList", "Methylation"]

    mock_chr1.__contains__.side_effect = chr1_contains_func

    mock_chr2 = MagicMock()
    mock_chr2.keys.return_value = ["ProbeList", "Methylation"]

    def chr2_contains_func(item):
        return item in ["ProbeList", "Methylation"]

    mock_chr2.__contains__.side_effect = chr2_contains_func

    mock_probelist = MagicMock()
    mock_probelist.shape = (2,)
    probe_array = np.array([b"probe1", b"probe2"])
    mock_probelist.__getitem__.side_effect = lambda idx: (
        probe_array if idx == slice(None) else probe_array[idx]
    )

    mock_methylation_data = MagicMock()
    mock_methylation_data.shape = (2, 3)
    methylation_array = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    mock_methylation_data.__getitem__.side_effect = lambda idx: (
        methylation_array if idx == slice(None) else methylation_array[idx]
    )

    mock_metadata = MagicMock()
    mock_metadata.keys.return_value = ["SampleList"]

    def metadata_contains_func(item):
        return item == "SampleList"

    mock_metadata.__contains__.side_effect = metadata_contains_func

    mock_samplelist = MagicMock()
    sample_array = np.array([1, 2, 3])
    mock_samplelist.__getitem__.side_effect = lambda idx: (
        sample_array if idx == slice(None) else sample_array[idx]
    )
    mock_samplelist.shape = (3,)

    def chr1_getitem(key):
        if key == "ProbeList":
            return mock_probelist
        elif key == "Methylation":
            return mock_methylation_data
        else:
            raise KeyError(f"Key not found in CHR1: {key}")

    def chr2_getitem(key):
        if key == "ProbeList":
            return mock_probelist
        elif key == "Methylation":
            return mock_methylation_data
        else:
            raise KeyError(f"Key not found in CHR2: {key}")

    mock_chr1.__getitem__.side_effect = chr1_getitem
    mock_chr2.__getitem__.side_effect = chr2_getitem

    def metadata_getitem(key):
        if key == "SampleList":
            return mock_samplelist
        else:
            raise KeyError(f"Key not found in Metadata: {key}")

    mock_metadata.__getitem__.side_effect = metadata_getitem

    def getitem_func(key):
        if key == "CHR1":
            return mock_chr1
        elif key == "CHR2":
            return mock_chr2
        elif key == "Metadata":
            return mock_metadata
        elif key in ["CHR1/ProbeList", "/CHR1/ProbeList"]:
            return mock_probelist
        elif key in ["CHR1/Methylation", "/CHR1/Methylation"]:
            return mock_methylation_data
        elif key in ["/Metadata/SampleList", "Metadata/SampleList"]:
            return mock_samplelist
        else:
            raise KeyError(f"Mock key not found: {key}")

    mock_file.__getitem__.side_effect = getitem_func
    return mock_file


@pytest.fixture
def mock_h5_genotype():
    mock = MagicMock()
    mock.keys.return_value = ["chr1", "chr2", "metadata"]

    mock_chr1 = MagicMock()
    mock_chr1.keys.return_value = [
        "snp",
        "data",
        "genotypes",
    ]

    def chr1_contains_func(item):
        return item in ["snp", "data", "genotypes"]

    mock_chr1.__contains__.side_effect = chr1_contains_func

    mock_chr2 = MagicMock()
    mock_chr2.keys.return_value = ["snp", "data", "genotypes"]

    def chr2_contains_func(item):
        return item in ["snp", "data", "genotypes"]

    mock_chr2.__contains__.side_effect = chr2_contains_func

    mock_snp = MagicMock()
    mock_snp.shape = (3,)
    snp_array = np.array([b"rs1", b"rs2", b"rs3"])
    mock_snp.__getitem__.side_effect = lambda idx: (
        snp_array if idx == slice(None) else snp_array[idx]
    )

    mock_genotype_data = MagicMock()
    mock_genotype_data.shape = (3, 3)
    genotype_array = np.array(
        [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
        ]
    )
    mock_genotype_data.__getitem__.side_effect = lambda idx: (
        genotype_array if idx == slice(None) else genotype_array[idx]
    )

    def mock_chr1_getitem(key):
        if key == "snp":
            return mock_snp
        elif key in ["data", "genotypes"]:
            return mock_genotype_data
        else:
            raise KeyError(f"Mock key not found in chr1: {key}")

    mock_chr1.__getitem__.side_effect = mock_chr1_getitem
    mock_chr2.__getitem__.side_effect = mock_chr1_getitem

    mock_metadata = MagicMock()
    mock_metadata.keys.return_value = ["iid"]

    def metadata_contains_func(item):
        return item in ["iid"]

    mock_metadata.__contains__.side_effect = metadata_contains_func

    mock_iid = MagicMock()
    iid_array = np.array([b"sample1", b"sample2", b"sample3"])
    mock_iid.__getitem__.side_effect = lambda idx: (
        iid_array if idx == slice(None) else iid_array[idx]
    )

    def getitem_func(key):
        if key == "chr1":
            return mock_chr1
        elif key == "chr2":
            return mock_chr2
        elif key == "metadata":
            return mock_metadata
        elif key in ["chr1/snp", "/chr1/snp"]:
            return mock_snp
        elif key in ["chr1/data", "/chr1/data", "chr1/genotypes", "/chr1/genotypes"]:
            return mock_genotype_data
        elif key in ["metadata/iid", "/metadata/iid"]:
            return mock_iid
        else:
            raise KeyError(f"Mock key not found: {key}")

    mock.__getitem__.side_effect = getitem_func

    def contains_func(item):
        return item in [
            "chr1",
            "chr2",
            "metadata",
            "chr1/snp",
            "/chr1/snp",
            "chr1/data",
            "/chr1/data",
            "chr1/genotypes",
            "/chr1/genotypes",
            "metadata/iid",
            "/metadata/iid",
        ]

    mock.__contains__.side_effect = contains_func

    return mock


@pytest.mark.unit
def test_decode_if_bytes():
    result = BaseH5Utils._decode_if_bytes(b"test_string")
    assert result == "test_string"

    result = BaseH5Utils._decode_if_bytes("already_string")
    assert result == "already_string"

    result = BaseH5Utils._decode_if_bytes(123)
    assert result == "123"


@pytest.mark.unit
def test_decode_array():
    input_array = [b"item1", b"item2", b"item3"]
    result = BaseH5Utils._decode_array(input_array)
    assert result == ["item1", "item2", "item3"]

    input_array = [b"item1", "item2", 123]
    result = BaseH5Utils._decode_array(input_array)
    assert result == ["item1", "item2", "123"]


@pytest.mark.unit
def test_detect_data_type_methylation(mock_h5_methylation):
    chr_group = mock_h5_methylation["CHR1"]
    result = DataTypeDetector.detect_data_type(chr_group)
    assert result == "Methylation"


@pytest.mark.unit
def test_detect_data_type_genotype(mock_h5_genotype):
    chr_group = mock_h5_genotype["chr1"]
    result = DataTypeDetector.detect_data_type(chr_group)
    assert result == "Genotype"


@pytest.mark.unit
def test_detect_data_type_unknown():
    mock_chr = MagicMock()
    mock_chr.keys.return_value = []
    mock_chr.__contains__.side_effect = lambda x: False

    result = DataTypeDetector.detect_data_type(mock_chr)
    assert result is None


@pytest.mark.unit
def test_get_sample_path_methylation(mock_h5_methylation):
    result = BaseH5Utils._get_sample_path(mock_h5_methylation, "Methylation")
    assert result == "/Metadata/SampleList"


@pytest.mark.unit
def test_get_sample_path_genotype(mock_h5_genotype):
    result = BaseH5Utils._get_sample_path(mock_h5_genotype, "Genotype")
    assert result == "/metadata/iid"


@pytest.mark.unit
def test_get_sample_path_auto_detect(mock_h5_methylation):
    result = BaseH5Utils._get_sample_path(mock_h5_methylation)
    assert result == "/Metadata/SampleList"


@pytest.mark.unit
def test_get_sample_path_not_found():
    mock_file = MagicMock()
    mock_file.keys.return_value = []
    mock_file.__contains__.side_effect = lambda key: False

    with pytest.raises(ValueError, match="Could not find metadata group"):
        BaseH5Utils._get_sample_path(mock_file)


@pytest.mark.unit
def test_read_methylation_data_real_file(methyl_h5_file):
    chromosome_reader = ChromosomeReader()
    chromosome_mapper = ChromosomeMapper(methyl_h5_file)
    data_type_detector = DataTypeDetector()
    result = chromosome_reader.read_chromosome_data(
        methyl_h5_file, "chr1", "Methylation", chromosome_mapper, data_type_detector
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "CGID" in result.columns
    assert len(result) == 5000
    assert len([col for col in result.columns if col != "CGID"]) == 11

    assert result["CGID"].dtype == "object"
    assert "DataType" in result.columns
    for col in result.columns:
        if col not in ["CGID", "DataType"]:
            assert pd.api.types.is_numeric_dtype(result[col])


@pytest.mark.unit
def test_read_genotype_data_real_file(genotype_h5_file):
    chromosome_reader = ChromosomeReader()
    chromosome_mapper = ChromosomeMapper(genotype_h5_file)
    data_type_detector = DataTypeDetector()
    result = chromosome_reader.read_chromosome_data(
        genotype_h5_file,
        "chr1",
        "Genotype",
        chromosome_mapper,
        data_type_detector,
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "RSID" in result.columns
    assert len(result) == 10000

    sample_columns = [
        col for col in result.columns if col not in ["RSID", "BP", "A1", "A2"]
    ]
    assert len(sample_columns) == 10


@pytest.mark.unit
def test_read_methylation_data_mock(mock_h5_methylation):
    chromosome_reader = ChromosomeReader()
    chromosome_mapper = ChromosomeMapper(mock_h5_methylation)
    data_type_detector = DataTypeDetector()
    result = chromosome_reader.read_chromosome_data(
        mock_h5_methylation,
        "CHR1",
        "Methylation",
        chromosome_mapper,
        data_type_detector,
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "CGID" in result.columns
    assert len(result) == 2

    assert "probe1" in result["CGID"].values
    assert "probe2" in result["CGID"].values

    sample_columns = [col for col in result.columns if col not in ["CGID", "DataType"]]
    assert len(sample_columns) == 3


@pytest.mark.unit
def test_read_genotype_data_mock(mock_h5_genotype):
    chromosome_reader = ChromosomeReader()
    chromosome_mapper = ChromosomeMapper(mock_h5_genotype)
    data_type_detector = DataTypeDetector()
    result = chromosome_reader.read_chromosome_data(
        mock_h5_genotype, "chr1", "Genotype", chromosome_mapper, data_type_detector
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "RSID" in result.columns
    assert result.iloc[0]["RSID"] == "rs1"
    assert len(result) == 3

    sample_columns = [
        col for col in result.columns if col not in ["RSID", "BP", "A1", "A2"]
    ]
    assert len(sample_columns) == 3

    assert "BP" in result.columns
    assert "A1" in result.columns
    assert "A2" in result.columns

    assert len(result.columns) == 7


@pytest.mark.unit
def test_read_chromosome_integer_sample_ids():
    mock_file = MagicMock()
    mock_file.keys.return_value = ["chr1", "metadata"]

    def contains_func(item):
        return item in [
            "chr1",
            "metadata",
            "/metadata/sampleList",
            "metadata/sampleList",
            "/chr1/probeList",
            "chr1/probeList",
            "/chr1/betas",
            "chr1/betas",
        ]

    mock_file.__contains__.side_effect = contains_func

    mock_chr1 = MagicMock()
    mock_chr1.keys.return_value = ["probeList", "betas"]

    def chr1_contains_func(item):
        return item in ["probeList", "betas"]

    mock_chr1.__contains__.side_effect = chr1_contains_func

    mock_metadata = MagicMock()
    mock_metadata.keys.return_value = ["sampleList"]

    def metadata_contains_func(item):
        return item in ["sampleList"]

    mock_metadata.__contains__.side_effect = metadata_contains_func

    mock_samplelist = MagicMock()
    mock_samplelist.shape = (3,)
    mock_samplelist.ndim = 1
    sample_array = np.array([1, 2, 3])
    mock_samplelist.__getitem__.side_effect = lambda idx: (
        sample_array if idx == slice(None) else sample_array[idx]
    )

    mock_probelist = MagicMock()
    mock_probelist.shape = (2,)
    mock_probelist.ndim = 1
    probe_array = np.array([b"probe1", b"probe2"])
    mock_probelist.__getitem__.side_effect = lambda idx: (
        probe_array if idx == slice(None) else probe_array[idx]
    )

    mock_betas = MagicMock()
    mock_betas.shape = (2, 3)
    mock_betas.ndim = 2
    betas_array = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    mock_betas.__getitem__.side_effect = lambda idx: (
        betas_array if idx == slice(None) else betas_array[idx]
    )

    def mock_chr1_getitem(key):
        if key == "probeList":
            return mock_probelist
        elif key == "betas":
            return mock_betas
        else:
            raise KeyError(f"Mock key not found in chr1: {key}")

    mock_chr1.__getitem__.side_effect = mock_chr1_getitem

    def mock_metadata_getitem(key):
        if key == "sampleList":
            return mock_samplelist
        else:
            raise KeyError(f"Mock key not found in metadata: {key}")

    mock_metadata.__getitem__.side_effect = mock_metadata_getitem

    def getitem_func(key):
        if key == "chr1":
            return mock_chr1
        elif key == "metadata":
            return mock_metadata
        elif key in ["/metadata/sampleList", "metadata/sampleList"]:
            return mock_samplelist
        elif key in ["/chr1/probeList", "chr1/probeList"]:
            return mock_probelist
        elif key in ["/chr1/betas", "chr1/betas"]:
            return mock_betas
        else:
            raise KeyError(f"Mock key not found: {key}")

    mock_file.__getitem__.side_effect = getitem_func

    chromosome_reader = ChromosomeReader()
    chromosome_mapper = ChromosomeMapper(mock_file)
    data_type_detector = DataTypeDetector()

    result = chromosome_reader.read_chromosome_data(
        mock_file, "chr1", "Methylation", chromosome_mapper, data_type_detector
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert "CGID" in result.columns
    assert len(result) == 2

    sample_columns = [col for col in result.columns if col not in ["CGID", "DataType"]]
    assert len(sample_columns) == 3

    expected_patterns = [
        ["1", "2", "3"],
        ["sample_0", "sample_1", "sample_2"],
    ]

    found_pattern = False
    for pattern in expected_patterns:
        if all(col in result.columns for col in pattern):
            found_pattern = True
            break

    assert found_pattern, f"Expected sample columns not found. Got: {sample_columns}"


@pytest.mark.unit
def test_read_chromosome_not_found():
    mock_file = MagicMock()
    mock_file.keys.return_value = ["chr1", "metadata"]

    def contains_func(item):
        return item in ["chr1", "metadata"]

    mock_file.__contains__.side_effect = contains_func

    chromosome_reader = ChromosomeReader()
    chromosome_mapper = ChromosomeMapper(mock_h5_methylation)
    data_type_detector = DataTypeDetector()
    result = chromosome_reader.read_chromosome_data(
        mock_h5_methylation,
        "chr999",
        "methylation",
        chromosome_mapper,
        data_type_detector,
    )

    assert result is None


@pytest.mark.unit
def test_read_chromosome_exception_handling():
    mock_file = MagicMock()
    mock_file.keys.return_value = ["chr1"]

    mock_file.__contains__.side_effect = Exception("Test exception")

    chromosome_reader = ChromosomeReader()
    chromosome_mapper = ChromosomeMapper(mock_h5_methylation)
    data_type_detector = DataTypeDetector()
    result = chromosome_reader.read_chromosome_data(
        mock_h5_methylation,
        "chr1",
        "methylation",
        chromosome_mapper,
        data_type_detector,
    )
    assert result is None


@pytest.mark.unit
def test_get_sample_indices_real_methylation(methyl_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(methyl_h5_file)

    df = h5utils.read_chromosome("chr1")
    sample_columns = [col for col in df.columns if col != "CGID"]
    sample_ids = sample_columns[:3]

    sample_indexer = SampleIndexer()
    indices = sample_indexer.get_sample_indices(methyl_h5_file, sample_ids)

    assert indices is not None
    assert len(indices) == 3
    assert all(isinstance(idx, (int, np.integer)) for idx in indices)


@pytest.mark.unit
def test_get_sample_indices_real_genotype(genotype_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(genotype_h5_file)

    df = h5utils.read_chromosome("chr1")
    sample_columns = [
        col for col in df.columns if col not in ["RSID", "BP", "A1", "A2"]
    ]
    sample_ids = sample_columns[:3]

    sample_indexer = SampleIndexer()
    indices = sample_indexer.get_sample_indices(genotype_h5_file, sample_ids)

    assert indices is not None
    assert len(indices) == 3
    assert indices == [0, 1, 2]


@pytest.mark.unit
def test_get_sample_indices_mock(mock_h5_methylation):
    sample_indexer = SampleIndexer()
    indices = sample_indexer.get_sample_indices(mock_h5_methylation, ["1", "3"])

    assert indices is not None
    assert len(indices) == 2
    assert 0 in indices
    assert 2 in indices


@pytest.mark.unit
def test_get_sample_indices_not_found(mock_h5_methylation):
    sample_indexer = SampleIndexer()
    indices = sample_indexer.get_sample_indices(
        mock_h5_methylation, ["nonexistent_sample"]
    )

    assert indices is not None
    assert len(indices) == 0


@pytest.mark.unit
def test_get_sample_indices_exception_handling():
    mock_file = MagicMock()
    mock_file.keys.return_value = ["metadata"]
    mock_file.__contains__.side_effect = Exception("Test exception")

    sample_indexer = SampleIndexer()
    result = sample_indexer.get_sample_indices(mock_file, ["sample1"])
    assert result is None


@pytest.mark.unit
def test_get_marker_indices_real_methylation(methyl_h5_file):
    marker_indexer = MarkerIndexer()
    chromosome_mapper = ChromosomeMapper(methyl_h5_file)
    data_type_detector = DataTypeDetector()

    h5utils = CachedH5Utils(methyl_h5_file)
    df = h5utils.read_chromosome("chr1")
    probe_ids = df["CGID"][:5].tolist()

    indices = marker_indexer.get_marker_indices(
        methyl_h5_file,
        "CHR1",
        probe_ids,
        "Methylation",
        chromosome_mapper,
        data_type_detector,
    )

    assert indices is not None
    assert len(indices) == 5
    assert indices == [0, 1, 2, 3, 4]


@pytest.mark.unit
def test_get_marker_indices_real_genotype(genotype_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(genotype_h5_file)

    df = h5utils.read_chromosome("chr1")
    snp_ids = df["RSID"][:3].tolist()

    marker_indexer = MarkerIndexer()
    chromosome_mapper = ChromosomeMapper(genotype_h5_file)
    data_type_detector = DataTypeDetector()
    indices = marker_indexer.get_marker_indices(
        genotype_h5_file,
        "CHR1",
        snp_ids,
        "Genotype",
        chromosome_mapper,
        data_type_detector,
    )

    assert indices is not None
    assert len(indices) == 3
    assert indices == [0, 1, 2]


@pytest.mark.unit
def test_get_marker_indices_mock(mock_h5_methylation):
    marker_indexer = MarkerIndexer()
    chromosome_mapper = ChromosomeMapper(mock_h5_methylation)
    data_type_detector = DataTypeDetector()

    indices = marker_indexer.get_marker_indices(
        mock_h5_methylation,
        "chr1",
        ["probe1", "probe3"],
        "Methylation",
        chromosome_mapper,
        data_type_detector,
    )

    assert indices is not None
    assert len(indices) == 1
    assert indices[0] == 0

    indices2 = marker_indexer.get_marker_indices(
        mock_h5_methylation,
        "chr1",
        ["probe1", "probe2"],
        "Methylation",
        chromosome_mapper,
        data_type_detector,
    )
    assert indices2 is not None
    assert len(indices2) == 2
    assert set(indices2) == {0, 1}


@pytest.mark.unit
def test_get_marker_indices_not_found(mock_h5_methylation):
    indices = MarkerIndexer.get_marker_indices(
        mock_h5_methylation, "chr1", ["nonexistent"], "methylation"
    )

    assert indices is None


@pytest.mark.unit
def test_get_marker_indices_exception_handling():
    mock_file = MagicMock()

    mock_file.__contains__.side_effect = Exception("Test exception")

    result = MarkerIndexer.get_marker_indices(
        mock_file, "chr1", ["marker1"], "methylation"
    )
    assert result is None


@pytest.mark.unit
def test_get_chromosome_list_real_methylation(methyl_h5_file):
    chromosome_lister = ChromosomeLister()
    chromosomes = chromosome_lister.get_chromosome_list(methyl_h5_file)

    assert chromosomes is not None
    assert len(chromosomes) == 1
    assert "CHR1" in chromosomes
    assert "chr22" not in chromosomes
    assert "metadata" not in chromosomes

    assert chromosomes[0] == "CHR1"


@pytest.mark.unit
def test_get_chromosome_list_real_genotype(genotype_h5_file):
    chromosome_lister = ChromosomeLister()
    chromosomes = chromosome_lister.get_chromosome_list(genotype_h5_file)

    assert chromosomes is not None
    assert len(chromosomes) == 1
    assert "CHR1" in chromosomes
    assert "metadata" not in chromosomes


@pytest.mark.unit
def test_get_chromosome_list_mock(mock_h5_methylation):
    chromosome_lister = ChromosomeLister()
    chromosomes = chromosome_lister.get_chromosome_list(mock_h5_methylation)

    assert chromosomes is not None
    assert len(chromosomes) == 2
    assert "CHR1" in chromosomes
    assert "CHR2" in chromosomes
    assert "metadata" not in chromosomes


@pytest.mark.unit
def test_get_chromosome_list_empty():
    mock_file = MagicMock()
    mock_file.keys.return_value = ["metadata"]

    chromosome_lister = ChromosomeLister()
    chromosomes = chromosome_lister.get_chromosome_list(mock_file)

    assert chromosomes is None


@pytest.mark.unit
def test_get_chromosome_list_special_sorting():
    mock_file = MagicMock()
    mock_file.keys.return_value = ["chr2", "chr10", "chr1", "chrX", "chrY", "chrMT"]

    chromosome_lister = ChromosomeLister()
    chromosomes = chromosome_lister.get_chromosome_list(mock_file)

    assert chromosomes is not None
    assert chromosomes.index("chr1") < chromosomes.index("chr2")
    assert chromosomes.index("chr2") < chromosomes.index("chr10")
    assert chromosomes.index("chr10") < chromosomes.index("chrX")


@pytest.mark.unit
def test_get_chromosome_list_exception_handling():
    mock_file = MagicMock()

    mock_file.keys.side_effect = Exception("Test exception")

    chromosome_lister = ChromosomeLister()
    result = chromosome_lister.get_chromosome_list(mock_file)
    assert result is None


@pytest.mark.unit
def test_h5utils_initialization_real_methylation(methyl_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(methyl_h5_file)

    assert h5utils.h5_file == methyl_h5_file
    assert hasattr(h5utils, "chromosome_reader")
    assert hasattr(h5utils, "sample_indexer")
    assert hasattr(h5utils, "marker_indexer")
    assert hasattr(h5utils, "chromosome_lister")


def test_h5utils_initialization_real_genotype(genotype_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(genotype_h5_file)

    assert h5utils.h5_file == genotype_h5_file


@pytest.mark.unit
def test_context_manager_support():
    mock_file = MagicMock()

    with CachedH5Utils(mock_file) as h5utils:
        assert h5utils.h5_file == mock_file
        mock_file.close.assert_not_called()

    if hasattr(mock_file, "close"):
        mock_file.close.assert_called_once()


@pytest.mark.unit
def test_validate_file_structure_exception_handling():
    mock_file = MagicMock()

    h5utils = H5UtilsFactory.create_utils_from_file(mock_file)

    with patch.object(h5utils, "get_data_info") as mock_get_info:
        mock_get_info.side_effect = Exception("Test exception")

        is_valid = h5utils.validate_file_structure()
        assert is_valid is False


@pytest.mark.unit
def test_get_data_info_exception_handling():
    mock_file = MagicMock()

    mock_file.keys.return_value = ["chr1", "metadata"]

    mock_metadata = MagicMock()
    mock_metadata.keys.return_value = ["sampleList"]

    def getitem_func(key):
        if key == "metadata":
            return mock_metadata
        else:
            raise KeyError(f"Mock key not found: {key}")

    mock_file.__getitem__.side_effect = getitem_func

    h5utils = H5UtilsFactory.create_utils_from_file(mock_file)

    with patch.object(h5utils, "get_chromosomes") as mock_get_chromosomes:
        mock_get_chromosomes.side_effect = Exception("Test exception")

        result = h5utils.get_data_info()
        assert result is None


@pytest.mark.unit
def test_read_chromosome_with_specific_samples_and_markers():
    mock_file = MagicMock()
    mock_file.keys.return_value = ["chr1", "metadata"]

    def contains_func(item):
        return item in [
            "chr1",
            "metadata",
            "/metadata/sampleList",
            "/chr1/probeList",
            "/chr1/betas",
        ]

    mock_file.__contains__.side_effect = contains_func

    mock_chr1 = MagicMock()
    mock_chr1.keys.return_value = ["probeList", "betas"]

    def chr1_contains_func(item):
        return item in ["probeList", "betas"]

    mock_chr1.__contains__.side_effect = chr1_contains_func

    mock_metadata = MagicMock()
    mock_metadata.keys.return_value = ["sampleList"]

    def metadata_contains_func(item):
        return item in ["sampleList"]

    mock_metadata.__contains__.side_effect = metadata_contains_func

    def getitem_func(key):
        if key == "chr1":
            return mock_chr1
        elif key == "metadata":
            return mock_metadata
        else:
            raise KeyError(f"Mock key not found: {key}")

    mock_file.__getitem__.side_effect = getitem_func

    h5utils = H5UtilsFactory.create_utils_from_file(mock_file)

    with patch.object(ChromosomeReader, "read_chromosome_data") as mock_read:
        mock_df = pd.DataFrame(
            {
                "CGID": ["probe1", "probe2", "probe3"],
                "sample1": [0.1, 0.2, 0.3],
                "sample2": [0.4, 0.5, 0.6],
                "sample3": [0.7, 0.8, 0.9],
            }
        )
        mock_read.return_value = mock_df

        result = h5utils.read_chromosome("chr1")

        assert result is not None
        assert len(result) == 3
        assert "CGID" in result.columns

        mock_read.assert_called_once()
        args, kwargs = mock_read.call_args
        assert len(args) == 5
        assert args[1] == "chr1"


@pytest.mark.unit
def test_h5utils_basic_functionality():
    mock_file = MagicMock()
    mock_file.keys.return_value = ["chr1", "metadata"]

    def contains_func(item):
        return item in [
            "chr1",
            "metadata",
            "/metadata/sampleList",
            "/chr1/probeList",
            "/chr1/betas",
        ]

    mock_file.__contains__.side_effect = contains_func

    h5utils = H5UtilsFactory.create_utils_from_file(mock_file)
    assert h5utils.h5_file == mock_file

    assert hasattr(h5utils, "chromosome_reader")
    assert hasattr(h5utils, "sample_indexer")
    assert hasattr(h5utils, "marker_indexer")
    assert hasattr(h5utils, "chromosome_lister")

    chromosomes = h5utils.get_chromosomes()
    assert chromosomes is not None
    assert "chr1" in chromosomes


@pytest.mark.unit
def test_sample_indexer_alternative_path():
    mock_file = MagicMock()

    mock_file.keys.return_value = ["Metadata", "chr1"]
    mock_file.__contains__.side_effect = lambda item: item in [
        "/Metadata/IID",
        "Metadata",
    ]

    mock_metadata = MagicMock()
    mock_metadata.keys.return_value = ["IID"]

    def metadata_contains_func(item):
        return item in ["IID"]

    mock_metadata.__contains__.side_effect = metadata_contains_func

    mock_iid = MagicMock()
    mock_iid.__getitem__.return_value = np.array([b"sample1", b"sample2"])

    def getitem_func(key):
        if key == "/Metadata/IID":
            return mock_iid
        elif key == "Metadata":
            return mock_metadata
        else:
            raise KeyError(f"Mock key not found: {key}")

    mock_file.__getitem__.side_effect = getitem_func

    sample_indexer = SampleIndexer()
    indices = sample_indexer.get_sample_indices(
        mock_file,
        ["sample1"],
        "/Metadata/SampleList",
    )

    assert indices is not None
    assert len(indices) == 1
    assert indices[0] == 0


@pytest.mark.unit
def test_h5config():
    config = H5Config()
    assert config.cache_enabled is True
    assert config.chunk_size == 10000
    assert config.default_sample_prefix == "sample_"

    custom_config = H5Config(
        cache_enabled=False, chunk_size=5000, default_sample_prefix="custom_"
    )
    assert custom_config.cache_enabled is False
    assert custom_config.chunk_size == 5000
    assert custom_config.default_sample_prefix == "custom_"


@pytest.mark.integration
def test_debug_real_file_structure(methyl_h5_file):
    print(f"Root keys: {list(methyl_h5_file.keys())}")

    if "Metadata" in methyl_h5_file:
        metadata = methyl_h5_file["Metadata"]
        print(f"Metadata keys: {list(metadata.keys())}")

        samplelist_aliases = AliasUtils.get_aliases("SampleList")
        print(f"SampleList aliases: {samplelist_aliases[:10]}...")

        for key in metadata.keys():
            if key.lower() in [alias.lower() for alias in samplelist_aliases]:
                print(f"Found matching key: {key}")

    chr_keys = [k for k in methyl_h5_file.keys() if k.upper().startswith("CHR")]
    if chr_keys:
        chr_key = chr_keys[0]
        chr_group = methyl_h5_file[chr_key]
        print(f"{chr_key} keys: {list(chr_group.keys())}")


@pytest.mark.integration
def test_read_chromosome(methyl_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(methyl_h5_file)

    df = h5utils.read_chromosome("chr1")
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert "CGID" in df.columns
    assert len(df) == 5000
    assert len([col for col in df.columns if col != "CGID"]) == 11


@pytest.mark.unit
def test_get_marker_indices_all_markers(methyl_h5_file):
    marker_indexer = MarkerIndexer()
    chromosome_mapper = ChromosomeMapper(methyl_h5_file)
    data_type_detector = DataTypeDetector()
    indices = marker_indexer.get_marker_indices(
        methyl_h5_file, "chr1", None, None, chromosome_mapper, data_type_detector
    )

    assert indices is not None
    assert len(indices) == 5000
    assert indices[0] == 0
    assert indices[-1] == 4999


@pytest.mark.integration
def test_get_chromosomes(methyl_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(methyl_h5_file)
    chromosomes = h5utils.get_chromosomes()

    assert chromosomes is not None
    assert len(chromosomes) == 1


@pytest.mark.integration
def test_read_chromosome_h5utils(methyl_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(methyl_h5_file)
    df = h5utils.read_chromosome("chr1")

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert "CGID" in df.columns


@pytest.mark.integration
def test_get_sample_indices_h5utils(genotype_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(genotype_h5_file)

    df = h5utils.read_chromosome("chr1")
    assert df is not None

    sample_columns = [
        col for col in df.columns if col not in ["RSID", "BP", "A1", "A2"]
    ]
    sample_ids = sample_columns[:3]

    indices = h5utils.get_sample_indices(sample_ids)

    assert indices is not None
    assert len(indices) == 3


@pytest.mark.integration
def test_get_marker_indices_h5utils(methyl_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(methyl_h5_file)

    df = h5utils.read_chromosome("chr1")
    assert df is not None

    probe_ids = df["CGID"][:3].tolist()

    indices = h5utils.get_marker_indices("chr1", probe_ids)

    assert indices is not None
    assert len(indices) == 3
    assert indices == [0, 1, 2]


@pytest.mark.integration
def test_get_data_info_methylation(methyl_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(methyl_h5_file)
    info = h5utils.get_data_info()

    assert info is not None
    assert info["data_type"] == "Methylation"
    assert info["n_chromosomes"] == 1
    assert info["n_samples"] == 10
    assert info["sample_path"] == "/Metadata/SampleList"


@pytest.mark.integration
def test_get_data_info_genotype(genotype_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(genotype_h5_file)
    info = h5utils.get_data_info()

    assert info is not None
    assert info["data_type"] == "Genotype"
    assert info["n_chromosomes"] == 1
    assert info["n_samples"] == 10
    assert info["sample_path"] == "/Metadata/IID"


@pytest.mark.integration
def test_validate_file_structure_methylation(methyl_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(methyl_h5_file)
    is_valid = h5utils.validate_file_structure()

    assert is_valid is True


@pytest.mark.integration
def test_validate_file_structure_genotype(genotype_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(genotype_h5_file)
    is_valid = h5utils.validate_file_structure()

    assert is_valid is True


@pytest.mark.integration
def test_validate_file_structure_invalid():
    mock_file = MagicMock()
    mock_file.keys.return_value = []

    h5utils = H5UtilsFactory.create_utils_from_file(mock_file)
    is_valid = h5utils.validate_file_structure()

    assert is_valid is False


@pytest.mark.integration
def test_end_to_end_methylation_workflow(methyl_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(methyl_h5_file)

    chromosomes = h5utils.get_chromosomes()
    assert len(chromosomes) == 1
    assert "CHR1" in chromosomes

    df = h5utils.read_chromosome("CHR1")
    assert df is not None
    assert len(df) == 5000

    info = h5utils.get_data_info()
    assert info["data_type"] == "Methylation"
    assert info["n_samples"] == 10

    is_valid = h5utils.validate_file_structure()
    assert is_valid is True


@pytest.mark.integration
def test_end_to_end_genotype_workflow(genotype_h5_file):
    h5utils = H5UtilsFactory.create_utils_from_file(genotype_h5_file)

    chromosomes = h5utils.get_chromosomes()
    assert len(chromosomes) == 1
    assert "CHR1" in chromosomes

    df = h5utils.read_chromosome("CHR1")
    assert df is not None
    assert len(df) == 10000

    info = h5utils.get_data_info()
    assert info["data_type"] == "Genotype"
    assert info["n_samples"] == 10

    is_valid = h5utils.validate_file_structure()
    assert is_valid is True


@pytest.mark.integration
def test_error_handling(mock_h5_methylation):
    h5utils = H5UtilsFactory.create_utils_from_file(mock_h5_methylation)

    result = h5utils.read_chromosome("chr999")
    assert result is None

    indices = h5utils.get_sample_indices(["nonexistent_sample"])
    assert indices is not None
    assert len(indices) == 0


@pytest.mark.integration
def test_data_orientation_handling(methyl_h5_file, genotype_h5_file):
    h5utils_methyl = H5UtilsFactory.create_utils_from_file(methyl_h5_file)
    df_methyl = h5utils_methyl.read_chromosome("chr1")
    assert df_methyl is not None
    assert len(df_methyl) == 5000
    assert len(df_methyl.columns) == 12

    h5utils_geno = H5UtilsFactory.create_utils_from_file(genotype_h5_file)
    df_geno = h5utils_geno.read_chromosome("chr1")
    assert df_geno is not None
    assert len(df_geno) == 10000

    assert "RSID" in df_geno.columns

    sample_columns = [
        col for col in df_geno.columns if col not in ["RSID", "BP", "A1", "A2"]
    ]
    assert len(sample_columns) == 10

    assert len(df_geno.columns) == 14

    assert "BP" in df_geno.columns
    assert "A1" in df_geno.columns
    assert "A2" in df_geno.columns


if __name__ == "__main__":
    pytest.main([__file__])
