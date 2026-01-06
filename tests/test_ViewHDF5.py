#!/usr/bin/env python
import h5py
import numpy as np
import os
import pytest
import tempfile
from utils.LoggingUtils import log
from ViewHDF5 import ViewHDF5

log.setup(level="DEBUG")


@pytest.mark.integration
def test_genotype_analysis(data_dir, capsys):
    input_file = os.path.join(data_dir, "gen_data.h5")
    analyzer = ViewHDF5(input_file, missing_analysis=True)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Data Type: Genotype" in captured
    assert "Number of RSIDs" in captured
    assert "Number of samples" in captured
    assert "Total data points" in captured
    assert "missing genotype calls" in captured
    assert "Overall missing rate" in captured


@pytest.mark.integration
def test_methylation_analysis(data_dir, capsys):
    input_file = os.path.join(data_dir, "mcseq.h5")
    analyzer = ViewHDF5(input_file, missing_analysis=True)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Data Type: Methylation" in captured
    assert "Number of probes" in captured
    assert "Number of samples" in captured
    assert "Total data points" in captured
    assert "missing beta values" in captured
    assert "Overall missing rate" in captured


@pytest.mark.integration
def test_unknown_format(capsys):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as h5f:
            test_grp = h5f.create_group("test")
            test_grp.create_dataset("data", data=np.random.rand(10, 10))

    analyzer = ViewHDF5(tmp.name, missing_analysis=True)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Cannot analyze data: Unknown or unsupported file format" in captured
    assert "No chromosome data found for missing value analysis" in captured
    os.remove(tmp.name)


@pytest.mark.integration
def test_empty_file(capsys, caplog):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w"):
            pass

    analyzer = ViewHDF5(tmp.name)
    with caplog.at_level("WARNING"):
        success = analyzer.analyze_file()
    assert not success
    assert any(
        "HDF5 file appears to be empty" in record.message for record in caplog.records
    )
    assert any(
        "Failed to detect data type" in record.message for record in caplog.records
    )
    os.remove(tmp.name)


@pytest.mark.integration
def test_nonexistent_file(data_dir):
    input_file = os.path.join(data_dir, "nonexistent.h5")
    analyzer = ViewHDF5(input_file)
    success = analyzer.analyze_file()
    assert not success


@pytest.mark.integration
def test_corrupt_file(capsys):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(b"not a valid hdf5 file")
        tmp.flush()

    analyzer = ViewHDF5(tmp.name)
    with pytest.raises(SystemExit):
        analyzer.analyze_file()
    os.remove(tmp.name)


@pytest.mark.integration
def test_file_permissions(data_dir, capsys):
    import stat

    input_file = os.path.join(data_dir, "gen_data.h5")
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with open(input_file, "rb") as src:
            tmp.write(src.read())
        temp_file = tmp.name

    os.chmod(temp_file, stat.S_IWRITE)

    analyzer = ViewHDF5(temp_file)
    with pytest.raises(SystemExit):
        analyzer.analyze_file()

    os.chmod(temp_file, stat.S_IREAD | stat.S_IWRITE)
    os.remove(temp_file)


@pytest.mark.integration
def test_partial_genotype_data(capsys):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as h5f:
            chr_grp = h5f.create_group("chr1")
            chr_grp.create_dataset(
                "Genotype", data=np.random.choice([0, 1, 2], size=(10, 5))
            )

            meta_grp = h5f.create_group("Metadata")
            meta_grp.create_dataset(
                "IID", data=np.array([f"s{i}" for i in range(5)], dtype="S")
            )

    analyzer = ViewHDF5(tmp.name)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Data Type: Genotype" in captured
    os.remove(tmp.name)


@pytest.mark.integration
def test_partial_methylation_data(capsys):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as h5f:
            chr_grp = h5f.create_group("chr1")
            chr_grp.create_dataset("Methylation", data=np.random.rand(10, 5))

            meta_grp = h5f.create_group("Metadata")
            meta_grp.create_dataset(
                "SampleList", data=np.array([f"s{i}" for i in range(5)], dtype="S")
            )

    analyzer = ViewHDF5(tmp.name)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Data Type: Methylation" in captured
    os.remove(tmp.name)


@pytest.mark.integration
def test_multiple_chromosomes(data_dir, capsys):
    input_file = os.path.join(data_dir, "gen_data.h5")
    analyzer = ViewHDF5(input_file, missing_analysis=True)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Data Type: Genotype" in captured
    assert "chr" in captured.lower() or "chromosome" in captured.lower()


@pytest.mark.integration
def test_mixed_data_types(capsys):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as h5f:
            chr1_grp = h5f.create_group("chr1")
            chr1_grp.create_dataset(
                "Genotype", data=np.random.choice([0, 1, 2], size=(10, 5))
            )
            chr1_grp.create_dataset(
                "RSID", data=np.array([f"rs{i}" for i in range(10)], dtype="S")
            )

            chr2_grp = h5f.create_group("chr2")
            chr2_grp.create_dataset("Methylation", data=np.random.rand(8, 5))
            chr2_grp.create_dataset(
                "ProbeList", data=np.array([f"cg{i}" for i in range(8)], dtype="S")
            )

            meta_grp = h5f.create_group("Metadata")
            meta_grp.create_dataset(
                "IID", data=np.array([f"s{i}" for i in range(5)], dtype="S")
            )
            meta_grp.create_dataset(
                "SampleList", data=np.array([f"s{i}" for i in range(5)], dtype="S")
            )

    analyzer = ViewHDF5(tmp.name)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Data Type: Genotype" in captured
    os.remove(tmp.name)


@pytest.mark.integration
def test_large_missing_data(capsys):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as h5f:
            chr_grp = h5f.create_group("chr1")
            genotypes = np.full((20, 10), -1, dtype=float)
            valid_mask = np.random.random(genotypes.shape) < 0.1
            genotypes[valid_mask] = np.random.choice([0, 1, 2], size=np.sum(valid_mask))

            chr_grp.create_dataset("Genotype", data=genotypes)
            chr_grp.create_dataset(
                "RSID", data=np.array([f"rs{i}" for i in range(20)], dtype="S")
            )

            meta_grp = h5f.create_group("Metadata")
            meta_grp.create_dataset(
                "IID", data=np.array([f"s{i}" for i in range(10)], dtype="S")
            )

    analyzer = ViewHDF5(tmp.name, missing_analysis=True)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Data Type: Genotype" in captured
    assert "Overall missing rate" in captured

    import re

    missing_rate_match = re.search(r"Overall missing rate: (\d+(?:\.\d+)?)%", captured)
    assert (
        missing_rate_match is not None
    ), "Could not find overall missing rate in output"

    missing_rate = float(missing_rate_match.group(1))
    assert (
        missing_rate >= 80.0
    ), f"Expected high missing rate (>=80%), got {missing_rate}%"

    os.remove(tmp.name)


@pytest.mark.integration
def test_zero_missing_data(capsys):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as h5f:
            chr_grp = h5f.create_group("chr1")
            genotypes = np.random.choice([0, 1, 2], size=(15, 8))

            chr_grp.create_dataset("Genotype", data=genotypes)
            chr_grp.create_dataset(
                "RSID", data=np.array([f"rs{i}" for i in range(15)], dtype="S")
            )

            meta_grp = h5f.create_group("Metadata")
            meta_grp.create_dataset(
                "IID", data=np.array([f"s{i}" for i in range(8)], dtype="S")
            )

    analyzer = ViewHDF5(tmp.name, missing_analysis=True)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Data Type: Genotype" in captured
    assert "Overall missing rate" in captured
    assert "0.0%" in captured or "0%" in captured
    os.remove(tmp.name)


@pytest.mark.integration
def test_alternative_naming_conventions(capsys):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as h5f:
            chr_grp = h5f.create_group("CHROMOSOME1")
            chr_grp.create_dataset(
                "data", data=np.random.choice([0, 1, 2], size=(12, 6))
            )
            chr_grp.create_dataset(
                "snp", data=np.array([f"rs{i}" for i in range(12)], dtype="S")
            )

            meta_grp = h5f.create_group("META")
            meta_grp.create_dataset(
                "sample_id", data=np.array([f"s{i}" for i in range(6)], dtype="S")
            )

    analyzer = ViewHDF5(tmp.name)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Data Type: Genotype" in captured
    os.remove(tmp.name)


@pytest.mark.integration
def test_very_large_dimensions_simulation(capsys, monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as h5f:
            chr_grp = h5f.create_group("chr1")
            chr_grp.create_dataset(
                "Genotype", data=np.random.choice([0, 1, 2], size=(5, 3))
            )
            chr_grp.create_dataset(
                "RSID", data=np.array([f"rs{i}" for i in range(5)], dtype="S")
            )
            chr_grp.create_dataset(
                "A1", data=np.array([f"A{i}" for i in range(5)], dtype="S")
            )
            chr_grp.create_dataset(
                "A2", data=np.array([f"T{i}" for i in range(5)], dtype="S")
            )
            chr_grp.create_dataset("BP", data=np.array(range(5), dtype=np.int32))

            meta_grp = h5f.create_group("Metadata")
            meta_grp.create_dataset(
                "IID", data=np.array([f"s{i}" for i in range(3)], dtype="S")
            )

    analyzer = ViewHDF5(tmp.name)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Data Type: Genotype" in captured
    assert "5" in captured and "3" in captured
    os.remove(tmp.name)


@pytest.mark.integration
def test_detailed_output_verification(data_dir, capsys):
    input_file = os.path.join(data_dir, "gen_data.h5")
    analyzer = ViewHDF5(input_file, missing_analysis=True)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out

    expected_sections = [
        "HDF5 FILE STRUCTURE",
        "DATA SUMMARY",
        "Data Type:",
        "MISSING VALUES ANALYSIS",
    ]

    for section in expected_sections:
        assert section in captured, f"Missing section: {section}"

    lines = captured.split("\n")
    numeric_lines = [line for line in lines if any(char.isdigit() for char in line)]
    assert len(numeric_lines) > 0, "No numeric data found in output"


@pytest.mark.integration
def test_error_recovery(capsys):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as h5f:
            chr_grp = h5f.create_group("chr1")
            chr_grp.create_dataset("Genotype", data=np.array([[1.5, 2.7], [3.9, 4.1]]))
            chr_grp.create_dataset("RSID", data=np.array(["rs1", "rs2"], dtype="S"))

            meta_grp = h5f.create_group("Metadata")
            meta_grp.create_dataset("IID", data=np.array(["s1", "s2"], dtype="S"))

    analyzer = ViewHDF5(tmp.name)
    success = analyzer.analyze_file()
    assert success is not False
    captured = capsys.readouterr().out
    assert "Data Type: Genotype" in captured
    os.remove(tmp.name)
