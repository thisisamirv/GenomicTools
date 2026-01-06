#!/usr/bin/env python
import h5py
import multiprocessing
import os
import pandas as pd
import pytest
import sys
import tempfile
from ProcessHDF5 import ProcessHDF5
from utils.AliasUtils import AliasUtils
from utils.LoggingUtils import log
from utils.SystemUtils import SystemUtils

if sys.platform != "win32":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

log.setup(level="DEBUG")


def get_metadata_group_key(h5_file):
    key = AliasUtils.find_keys(h5_file, "Metadata")
    if key is None:
        raise KeyError("No metadata group found")
    return key


def get_sample_key(metadata_group, data_type):
    if data_type.lower() == "methylation":
        key = AliasUtils.find_keys(metadata_group, "SampleList")
    else:
        key = AliasUtils.find_keys(metadata_group, "IID")

    if key is None:
        raise KeyError("No sample list found")
    return key


def get_marker_key(chr_group, data_type):
    if data_type.lower() == "methylation":
        key = AliasUtils.find_keys(chr_group, "ProbeList")
    else:
        key = AliasUtils.find_keys(chr_group, "RSID")

    if key is None:
        raise KeyError("No marker list found")
    return key


def get_first_chromosome_key(h5_file):
    for key in h5_file.keys():
        base_key = AliasUtils.strip_numeric_suffix(key)
        if AliasUtils.get_field(base_key) == "CHR":
            return key
    raise KeyError("No chromosome groups found")


@pytest.mark.unit
def test_invalid_operation(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "invalid_operation_output.h5")
    with pytest.raises(ValueError):
        ProcessHDF5(
            input_file=input_file,
            output_file=output_file,
            operation="invalid_op",
        )


@pytest.mark.unit
def test_missing_input_file(data_dir, output_dir):
    input_file = os.path.join(data_dir, "nonexistent_file.h5")
    output_file = os.path.join(output_dir, "missing_input_output.txt")
    with pytest.raises(FileNotFoundError):
        ProcessHDF5(
            input_file=input_file,
            output_file=output_file,
            operation="names",
            names="samples",
        )


@pytest.mark.unit
def test_names_invalid_type(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "invalid_names_output.txt")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="names",
        names="not_a_type",
    )
    result = processor.run()
    assert result is False


@pytest.mark.unit
def test_add_metadata_missing_column(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_with_metadata_missing.h5")
    metadata_file = os.path.join(data_dir, "mcseq_metadata.csv")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="add_metadata",
        metadata=metadata_file,
        names="nonexistent_column",
    )
    result = processor.run()
    assert result is False


@pytest.mark.unit
def test_subset_no_samples_or_markers(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset_empty.h5")
    processor = ProcessHDF5(
        input_file=input_file, output_file=output_file, operation="subset"
    )
    result = processor.run()
    assert result is False


@pytest.mark.unit
def test_remove_nothing(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_remove_nothing.h5")
    processor = ProcessHDF5(
        input_file=input_file, output_file=output_file, operation="remove"
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)
    assert os.path.getsize(input_file) == os.path.getsize(output_file)


@pytest.mark.unit
def test_subset_invalid_samples(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset_invalid.h5")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        samples=["invalid_sample"],
    )
    result = processor.run()
    assert result is False


@pytest.mark.unit
def test_decode_array_empty(data_dir, output_dir):
    processor = ProcessHDF5(
        input_file=os.path.join(data_dir, "gen_data.h5"),
        output_file=os.path.join(output_dir, "decode_array_output.txt"),
        operation="names",
        data_type="genotype",
    )
    result = processor._decode_array([])
    assert result == []


@pytest.mark.unit
def test_process_input_str_comma(data_dir, output_dir):
    processor = ProcessHDF5(
        input_file=os.path.join(data_dir, "gen_data.h5"),
        output_file=os.path.join(output_dir, "process_input_comma.txt"),
        operation="names",
        samples="A,B,C",
        data_type="genotype",
    )
    ids, file = processor._process_input("A,B,C", "sample")
    assert ids == ["A", "B", "C"]
    assert file is None


@pytest.mark.unit
def test_process_input_str_single(data_dir, output_dir):
    processor = ProcessHDF5(
        input_file=os.path.join(data_dir, "gen_data.h5"),
        output_file=os.path.join(output_dir, "process_input_single.txt"),
        operation="names",
        samples="A",
        data_type="genotype",
    )
    ids, file = processor._process_input("A", "sample")
    assert ids == ["A"]
    assert file is None


@pytest.mark.unit
def test_process_data_in_chunks_missing_data(data_dir, output_dir):
    import h5py

    input_file = os.path.join(data_dir, "gen_data.h5")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=os.path.join(output_dir, "chunks_missing_output.csv"),
        operation="read",
    )
    with h5py.File(input_file, "r") as h5f:
        missing_chr = "chr_missing"
        result = processor._process_data_in_chunks(h5f, missing_chr)
        assert result is None


@pytest.mark.unit
def test_process_data_in_chunks_no_filtered_markers(data_dir, output_dir):
    import h5py

    input_file = os.path.join(data_dir, "gen_data.h5")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=os.path.join(output_dir, "chunks_no_marker_output.csv"),
        operation="read",
        markers=["nonexistent_marker"],
    )
    processor.data_type = "Genotype"
    with h5py.File(input_file, "r") as h5f:
        chrom = get_first_chromosome_key(h5f)
        chunks = processor._process_data_in_chunks(h5f, chrom)
        assert chunks == []


@pytest.mark.integration
def test_remove_chromosomes_genotype(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_removed_chromosomes.h5")

    with h5py.File(input_file, "r") as h5f:
        all_chromosomes = [key for key in h5f.keys() if key.lower().startswith("chr")]
        log.info(f"Available chromosomes in input: {all_chromosomes}")

        if len(all_chromosomes) >= 2:
            chromosomes_to_remove = all_chromosomes[:2]
        elif len(all_chromosomes) == 1:
            chromosomes_to_remove = all_chromosomes[:1]
        else:
            pytest.skip("No chromosomes found in test data")

    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="remove",
        chromosomes=",".join(chromosomes_to_remove),
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as h5f:
        remaining_chromosomes = [
            key for key in h5f.keys() if key.lower().startswith("chr")
        ]
        log.info(f"Remaining chromosomes after removal: {remaining_chromosomes}")

        for removed_chr in chromosomes_to_remove:
            assert (
                removed_chr not in remaining_chromosomes
            ), f"Chromosome {removed_chr} should have been removed"

        expected_remaining = [
            chr for chr in all_chromosomes if chr not in chromosomes_to_remove
        ]
        if expected_remaining:
            assert len(remaining_chromosomes) == len(
                expected_remaining
            ), f"Expected {len(expected_remaining)} chromosomes, got {len(remaining_chromosomes)}"
            for expected_chr in expected_remaining:
                assert (
                    expected_chr in remaining_chromosomes
                ), f"Chromosome {expected_chr} should still be present"

        assert "Metadata" in h5f or "metadata" in h5f


@pytest.mark.integration
def test_remove_chromosomes_methylation(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_removed_chromosomes.h5")

    with h5py.File(input_file, "r") as h5f:
        all_chromosomes = [key for key in h5f.keys() if key.lower().startswith("chr")]
        log.info(f"Available chromosomes in methylation data: {all_chromosomes}")

        if len(all_chromosomes) >= 2:
            chromosomes_to_remove = all_chromosomes[:1]
        elif len(all_chromosomes) == 1:
            chromosomes_to_remove = all_chromosomes[:1]
        else:
            pytest.skip("No chromosomes found in methylation test data")

    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="remove",
        chromosomes=",".join(chromosomes_to_remove),
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as h5f:
        remaining_chromosomes = [
            key for key in h5f.keys() if key.lower().startswith("chr")
        ]
        log.info(f"Remaining chromosomes after removal: {remaining_chromosomes}")

        for removed_chr in chromosomes_to_remove:
            assert (
                removed_chr not in remaining_chromosomes
            ), f"Chromosome {removed_chr} should have been removed"

        assert "Metadata" in h5f or "metadata" in h5f


@pytest.mark.integration
def test_remove_nonexistent_chromosomes(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_removed_nonexistent.h5")

    nonexistent_chromosomes = ["chrX", "chrY", "chr99"]

    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="remove",
        chromosomes=",".join(nonexistent_chromosomes),
    )
    result = processor.run()

    assert result is True
    assert os.path.exists(output_file)

    with h5py.File(input_file, "r") as input_h5f, h5py.File(
        output_file, "r"
    ) as output_h5f:
        input_chromosomes = [
            key for key in input_h5f.keys() if key.lower().startswith("chr")
        ]
        output_chromosomes = [
            key for key in output_h5f.keys() if key.lower().startswith("chr")
        ]

        assert len(input_chromosomes) == len(
            output_chromosomes
        ), "All chromosomes should be preserved when removing nonexistent ones"

        for chr_name in input_chromosomes:
            assert (
                chr_name in output_chromosomes
            ), f"Chromosome {chr_name} should still be present"


@pytest.mark.integration
def test_remove_all_chromosomes_error(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_removed_all.h5")

    with h5py.File(input_file, "r") as h5f:
        all_chromosomes = [key for key in h5f.keys() if key.lower().startswith("chr")]

        if not all_chromosomes:
            pytest.skip("No chromosomes found in test data")

    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="remove",
        chromosomes=",".join(all_chromosomes),
    )

    result = processor.run()

    if result is False:
        assert not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    else:
        assert os.path.exists(output_file)
        with h5py.File(output_file, "r") as h5f:
            remaining_chromosomes = [
                key for key in h5f.keys() if key.lower().startswith("chr")
            ]
            assert len(remaining_chromosomes) == 0, "No chromosomes should remain"


@pytest.mark.integration
def test_samples_extraction_methylation(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_sample_names.txt")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="names",
        names="samples",
    )
    result = processor.run()
    assert result is True
    df = pd.read_csv(output_file, header=None)
    assert not df.empty
    assert df.shape[1] == 1


@pytest.mark.integration
def test_markers_extraction_methylation(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_marker_names.txt")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="names",
        names="markers",
    )
    result = processor.run()
    assert result is True
    df = pd.read_csv(output_file, header=None)
    assert not df.empty
    assert df.shape[1] == 1


@pytest.mark.integration
def test_samples_extraction_genotype(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_sample_names.txt")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="names",
        names="samples",
    )
    result = processor.run()
    assert result is True
    df = pd.read_csv(output_file, header=None)
    assert not df.empty
    assert df.shape[1] == 1


@pytest.mark.integration
def test_markers_extraction_genotype(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_marker_names.txt")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="names",
        names="markers",
    )
    result = processor.run()
    assert result is True
    df = pd.read_csv(output_file, header=None)
    assert not df.empty
    assert df.shape[1] == 1


@pytest.mark.integration
def test_read_subset_genotype(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset.csv")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]

        chr_key = get_first_chromosome_key(h5f)
        marker_key = get_marker_key(h5f[chr_key], "genotype")
        markers = [
            m.decode("utf-8") if isinstance(m, bytes) else str(m)
            for m in h5f[f"{chr_key}/{marker_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="read",
        samples=samples,
        markers=markers,
    )
    result = processor.run()
    assert result is True
    df = pd.read_csv(output_file)
    assert not df.empty
    assert set(samples).intersection(df.columns)
    assert set(markers).intersection(df["RSID"].astype(str).unique())


@pytest.mark.integration
def test_read_subset_methylation(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_subset.csv")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "methylation")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]

        chr_key = get_first_chromosome_key(h5f)
        marker_key = get_marker_key(h5f[chr_key], "methylation")
        probes = [
            p.decode("utf-8") if isinstance(p, bytes) else str(p)
            for p in h5f[f"{chr_key}/{marker_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="read",
        samples=samples,
        markers=probes,
    )
    result = processor.run()
    assert result is True
    df = pd.read_csv(output_file)
    assert not df.empty
    assert set(samples).intersection(df.columns)
    assert set(probes).intersection(df["CGID"].astype(str).unique())


@pytest.mark.integration
def test_subset_genotype_samples_only(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset_samples.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:3]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        samples=samples,
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as h5f:
        assert "chr1" in h5f or "CHR1" in h5f
        assert "Metadata" in h5f or "metadata" in h5f


@pytest.mark.integration
def test_subset_genotype_markers_only(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset_markers.h5")
    with h5py.File(input_file, "r") as h5f:
        chr_key = get_first_chromosome_key(h5f)
        marker_key = get_marker_key(h5f[chr_key], "genotype")
        markers = [
            m.decode("utf-8") if isinstance(m, bytes) else str(m)
            for m in h5f[f"{chr_key}/{marker_key}"][:3]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        markers=markers,
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as h5f:
        assert "chr1" in h5f or "CHR1" in h5f
        assert "Metadata" in h5f or "metadata" in h5f


@pytest.mark.integration
def test_subset_methylation_samples_only(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_subset_samples.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "methylation")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:3]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        samples=samples,
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as h5f:
        assert "chr1" in h5f or "CHR1" in h5f
        assert "Metadata" in h5f or "metadata" in h5f


@pytest.mark.integration
def test_subset_methylation_markers_only(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_subset_markers.h5")
    with h5py.File(input_file, "r") as h5f:
        chr_key = get_first_chromosome_key(h5f)
        marker_key = get_marker_key(h5f[chr_key], "methylation")
        markers = [
            p.decode("utf-8") if isinstance(p, bytes) else str(p)
            for p in h5f[f"{chr_key}/{marker_key}"][:3]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        markers=markers,
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as h5f:
        assert "chr1" in h5f or "CHR1" in h5f
        assert "Metadata" in h5f or "metadata" in h5f


@pytest.mark.integration
def test_remove_genotype_samples(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_removed_samples.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        sample = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:1]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="remove",
        samples=sample,
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as h5f:
        assert "chr1" in h5f or "CHR1" in h5f
        assert "Metadata" in h5f or "metadata" in h5f


@pytest.mark.integration
def test_remove_genotype_markers(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_removed_markers.h5")
    with h5py.File(input_file, "r") as h5f:
        chr_key = get_first_chromosome_key(h5f)
        marker_key = get_marker_key(h5f[chr_key], "genotype")
        marker = [
            m.decode("utf-8") if isinstance(m, bytes) else str(m)
            for m in h5f[f"{chr_key}/{marker_key}"][:1]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="remove",
        markers=marker,
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as h5f:
        assert "chr1" in h5f or "CHR1" in h5f
        assert "Metadata" in h5f or "metadata" in h5f


@pytest.mark.integration
def test_remove_genotype_samples_and_markers(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_removed_both.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        sample = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:1]
        ]
        chr_key = get_first_chromosome_key(h5f)
        marker_key = get_marker_key(h5f[chr_key], "genotype")
        marker = [
            m.decode("utf-8") if isinstance(m, bytes) else str(m)
            for m in h5f[f"{chr_key}/{marker_key}"][:1]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="remove",
        samples=sample,
        markers=marker,
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as h5f:
        assert "chr1" in h5f or "CHR1" in h5f
        assert "Metadata" in h5f or "metadata" in h5f


@pytest.mark.integration
def test_names_extraction_snps(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_snp_names.txt")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="names",
        names="snps",
    )
    result = processor.run()
    assert result is True
    df = pd.read_csv(output_file, header=None)
    assert not df.empty
    assert df.shape[1] == 1


@pytest.mark.integration
def test_add_metadata_to_methylation(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_with_metadata.h5")
    metadata_file = os.path.join(data_dir, "mcseq_metadata.csv")
    cols = pd.read_csv(metadata_file).columns
    sample_id_names = {"sample_id", "iid", "mvp004_id"}
    non_id_cols = [c for c in cols if c.lower() not in sample_id_names]
    assert non_id_cols, "No non-sample-id columns found in metadata"
    valid_col = non_id_cols[0]

    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="add_metadata",
        metadata=metadata_file,
        names=valid_col,
    )
    result = processor.run()
    assert result is True
    with h5py.File(output_file, "r") as h5f:
        assert "Metadata" in h5f or "metadata" in h5f
        metadata_key = get_metadata_group_key(h5f)
        assert valid_col in h5f[metadata_key]


@pytest.mark.integration
def test_subset_with_chromosomes_genotype(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset_chroms.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        samples=samples,
        chromosomes="1",
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as h5f:
        chroms = [key for key in h5f.keys() if key.lower().startswith("chr")]
        assert any("1" in chrom for chrom in chroms)


@pytest.mark.integration
def test_names_requires_output(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_sample_names_required.txt")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="names",
        names="samples",
    )
    result = processor.run()
    assert result is True
    df = pd.read_csv(output_file, header=None)
    assert not df.empty
    assert df.shape[1] == 1


@pytest.mark.integration
def test_read_requires_output(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_read_required.csv")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
        chr_key = get_first_chromosome_key(h5f)
        marker_key = get_marker_key(h5f[chr_key], "genotype")
        markers = [
            m.decode("utf-8") if isinstance(m, bytes) else str(m)
            for m in h5f[f"{chr_key}/{marker_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="read",
        samples=samples,
        markers=markers,
    )
    result = processor.run()
    assert result is True
    df = pd.read_csv(output_file)
    assert not df.empty


@pytest.mark.integration
def test_process_input_with_file(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset_file_samples.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        samples_list = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        for s in samples_list:
            tmp.write(f"{s}\n")
        tmp.flush()
        samples_file = tmp.name
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        samples=samples_file,
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)
    os.unlink(samples_file)


@pytest.mark.integration
def test_names_extraction_probes(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_probe_names.txt")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="names",
        names="probes",
    )
    result = processor.run()
    assert result is True
    df = pd.read_csv(output_file, header=None)
    assert not df.empty
    assert df.shape[1] == 1


@pytest.mark.integration
def test_add_metadata_all_columns(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_with_all_metadata.h5")
    metadata_file = os.path.join(data_dir, "mcseq_metadata.csv")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="add_metadata",
        metadata=metadata_file,
        names=None,
    )
    result = processor.run()
    assert result is True
    with h5py.File(output_file, "r") as h5f:
        assert "Metadata" in h5f or "metadata" in h5f
        metadata_key = get_metadata_group_key(h5f)
        assert len(h5f[metadata_key].keys()) > 1


@pytest.mark.integration
def test_add_metadata_multiple_columns_string(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_with_multiple_metadata.h5")
    metadata_file = os.path.join(data_dir, "mcseq_metadata.csv")

    metadata_df = pd.read_csv(metadata_file)
    available_columns = list(metadata_df.columns)
    sample_id_columns = ["sample_id", "iid", "mvp004_id"]
    test_columns = [
        col for col in available_columns if col.lower() not in sample_id_columns
    ][:2]

    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="add_metadata",
        metadata=metadata_file,
        names=",".join(test_columns),
    )
    result = processor.run()
    assert result is True


@pytest.mark.integration
def test_add_metadata_multiple_columns_list(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_with_multiple_metadata_list.h5")
    metadata_file = os.path.join(data_dir, "mcseq_metadata.csv")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="add_metadata",
        metadata=metadata_file,
        names=["Age", "Sex"],
    )
    result = processor.run()
    assert result is True
    with h5py.File(output_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        assert "Age" in h5f[metadata_key]
        assert "Sex" in h5f[metadata_key]


@pytest.mark.integration
def test_read_with_chromosomes(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_read_chroms.csv")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "methylation")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="read",
        samples=samples,
        chromosomes="1",
    )
    result = processor.run()
    assert result is True
    df = pd.read_csv(output_file)
    assert not df.empty
    assert set(df["CHR"].unique()) == {1}


@pytest.mark.integration
def test_subset_with_chromosomes_list(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_subset_chroms_list.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "methylation")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        samples=samples,
        chromosomes=["1"],
    )
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as h5f:
        chroms = [key for key in h5f.keys() if key.lower().startswith("chr")]
        assert any("1" in chrom for chrom in chroms)


@pytest.mark.integration
def test_remove_genotype_with_additional_datasets(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_remove_additional.h5")
    with h5py.File(input_file, "r") as h5f:
        chr_key = get_first_chromosome_key(h5f)
        marker_key = get_marker_key(h5f[chr_key], "genotype")
        markers = [
            m.decode("utf-8") if isinstance(m, bytes) else str(m)
            for m in h5f[f"{chr_key}/{marker_key}"][:1]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="remove",
        markers=markers,
    )
    result = processor.run()
    assert result is True
    with h5py.File(output_file, "r") as h5f:
        chr_group = h5f[chr_key]
        has_additional = any(
            key in chr_group for key in ["A1", "a1", "BP", "bp", "pos"]
        )
        assert has_additional


@pytest.mark.integration
def test_add_metadata_numeric_column(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_with_numeric_metadata.h5")
    metadata_file = os.path.join(data_dir, "mcseq_metadata.csv")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="add_metadata",
        metadata=metadata_file,
        names="Age",
    )
    result = processor.run()
    assert result is True
    with h5py.File(output_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        assert "Age" in h5f[metadata_key]
        data = h5f[f"{metadata_key}/Age"][:]
        assert data.dtype.kind in "fi"


@pytest.mark.integration
def test_no_chromosomes(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset_no_chroms.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        samples=samples,
        chromosomes=[],
    )
    result = processor.run()
    assert result is True
    with h5py.File(output_file, "r") as h5f:
        chroms = [key for key in h5f.keys() if key.lower().startswith("chr")]
        assert len(chroms) > 0


@pytest.mark.integration
def test_process_input_list(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset_list_samples.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        samples=samples,
    )
    result = processor.run()
    assert result is True


@pytest.mark.integration
def test_process_input_tuple(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset_tuple_samples.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        samples = tuple(
            [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
            ]
        )
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        samples=samples,
    )
    result = processor.run()
    assert result is True


@pytest.mark.integration
def test_process_chromosomes_string(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset_chrom_string.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        samples=samples,
        chromosomes="1",
    )
    result = processor.run()
    assert result is True
    with h5py.File(output_file, "r") as h5f:
        chroms = [key for key in h5f.keys() if key.lower().startswith("chr")]
        assert any("1" in chrom for chrom in chroms)


@pytest.mark.integration
def test_process_chromosomes_list(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_subset_chrom_list.h5")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="subset",
        samples=samples,
        chromosomes=["1", "2"],
    )
    result = processor.run()
    assert result is True


@pytest.mark.integration
def test_run_read_operation_chunked_genotype(data_dir, output_dir, monkeypatch):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_read_chunked.csv")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "genotype")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="read",
        samples=samples,
    )
    monkeypatch.setattr(os.path, "getsize", lambda x: 11 * 1024**3)
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)


@pytest.mark.integration
def test_run_read_operation_chunked_genotype_with_marker_ids(
    data_dir, output_dir, monkeypatch
):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_read_chunked_markers.csv")
    with h5py.File(input_file, "r") as h5f:
        chr_key = get_first_chromosome_key(h5f)
        marker_key = get_marker_key(h5f[chr_key], "genotype")
        all_markers = [
            m.decode("utf-8") if isinstance(m, bytes) else str(m)
            for m in h5f[f"{chr_key}/{marker_key}"][:]
        ]
        markers = [all_markers[0], "nonexistent_marker"]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="read",
        markers=markers,
    )
    monkeypatch.setattr(os.path, "getsize", lambda x: 11 * 1024**3)
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)


@pytest.mark.integration
def test_run_read_operation_chunked_methylation(data_dir, output_dir, monkeypatch):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_read_chunked.csv")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "methylation")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
        chr_key = get_first_chromosome_key(h5f)
        marker_key = get_marker_key(h5f[chr_key], "methylation")
        markers = [
            p.decode("utf-8") if isinstance(p, bytes) else str(p)
            for p in h5f[f"{chr_key}/{marker_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="read",
        samples=samples,
        markers=markers,
    )
    processor.data_type = "Methylation"
    monkeypatch.setattr(os.path, "getsize", lambda x: 11 * 1024**3)
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)


@pytest.mark.integration
def test_run_read_operation_chunked_methylation_with_marker_ids(
    data_dir, output_dir, monkeypatch
):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_read_chunked_markers.csv")
    with h5py.File(input_file, "r") as h5f:
        metadata_key = get_metadata_group_key(h5f)
        sample_key = get_sample_key(h5f[metadata_key], "methylation")
        samples = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in h5f[f"/{metadata_key}/{sample_key}"][:2]
        ]
        chr_key = get_first_chromosome_key(h5f)
        marker_key = get_marker_key(h5f[chr_key], "methylation")
        markers = [
            p.decode("utf-8") if isinstance(p, bytes) else str(p)
            for p in h5f[f"{chr_key}/{marker_key}"][:2]
        ]
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=output_file,
        operation="read",
        samples=samples,
        markers=markers,
    )
    processor.data_type = "Methylation"
    monkeypatch.setattr(os.path, "getsize", lambda x: 11 * 1024**3)
    result = processor.run()
    assert result is True
    assert os.path.exists(output_file)


@pytest.mark.integration
def test_adjust_resources_low_memory(data_dir, monkeypatch):
    input_file = os.path.join(data_dir, "gen_data.h5")

    if not os.path.exists(input_file):
        pytest.skip(f"Test data file not found: {input_file}")

    def mock_get_memory_info():
        return {"available_gb": 4.0, "total_gb": 8.0}

    def mock_get_optimal_cores(reserve_cores=1):
        return max(1, 2 - reserve_cores)

    monkeypatch.setattr(
        SystemUtils, "get_memory_info", staticmethod(mock_get_memory_info)
    )
    monkeypatch.setattr(
        SystemUtils, "get_optimal_cores", staticmethod(mock_get_optimal_cores)
    )

    processor = ProcessHDF5(
        input_file=input_file,
        operation="read",
        output_file="test_low_memory_output.h5",
    )

    assert (
        processor.max_workers <= 4
    ), f"Should use fewer workers for low memory, got {processor.max_workers}"
    log.info(f"Low memory test: using {processor.max_workers} workers")

    assert (
        processor.max_workers == 1
    ), f"Expected 1 worker for low memory, got {processor.max_workers}"


@pytest.mark.integration
def test_adjust_resources_high_memory(data_dir, monkeypatch):
    input_file = os.path.join(data_dir, "gen_data.h5")

    if not os.path.exists(input_file):
        pytest.skip(f"Test data file not found: {input_file}")

    def mock_get_memory_info():
        return {"available_gb": 128.0, "total_gb": 256.0}

    def mock_get_optimal_cores(reserve_cores=1):
        return max(1, 16 - reserve_cores)

    monkeypatch.setattr(
        SystemUtils, "get_memory_info", staticmethod(mock_get_memory_info)
    )
    monkeypatch.setattr(
        SystemUtils, "get_optimal_cores", staticmethod(mock_get_optimal_cores)
    )

    processor = ProcessHDF5(
        input_file=input_file,
        operation="read",
        output_file="test_high_memory_output.h5",
    )

    assert (
        processor.max_workers >= 8
    ), f"Should use more workers for high memory, got {processor.max_workers}"
    log.info(f"High memory test: using {processor.max_workers} workers")

    assert (
        processor.max_workers == 15
    ), f"Expected 15 workers for high memory, got {processor.max_workers}"


@pytest.mark.integration
def test_adjust_resources_large_file(data_dir, output_dir, monkeypatch):
    input_file = os.path.join(data_dir, "gen_data.h5")
    monkeypatch.setattr(os.path, "getsize", lambda x: 51 * 1024**3)
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=os.path.join(output_dir, "large_file_adjust.h5"),
        operation="subset",
    )
    processor.max_workers = 10
    processor._adjust_resources()
    assert processor.max_workers == 6


@pytest.mark.integration
def test_check_memory_usage_warn(data_dir, output_dir, monkeypatch, caplog):
    import psutil

    class DummyMemInfo:
        rss = 10 * 1024**3

    class DummyVirtMem:
        available = 16 * 1024**3

    monkeypatch.setattr(psutil.Process, "memory_info", lambda self: DummyMemInfo())
    monkeypatch.setattr(psutil, "virtual_memory", lambda: DummyVirtMem())

    input_file = os.path.join(data_dir, "gen_data.h5")
    processor = ProcessHDF5(
        input_file=input_file,
        output_file=os.path.join(output_dir, "memory_warn_output.h5"),
        operation="subset",
    )
    with caplog.at_level("WARNING"):
        processor._check_memory_usage("test")
    assert any("High memory usage" in record.message for record in caplog.records)
