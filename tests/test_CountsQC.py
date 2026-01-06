#!/usr/bin/env python
import h5py
import numpy as np
import os
import pandas as pd
import pytest
import tempfile
from CountsQC import CountsQC
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.mark.unit
def test_invalid_metric(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "invalid_metric.txt")
    with pytest.raises(ValueError):
        CountsQC(
            input_file=input_file,
            output_file=output_file,
            metric="invalid_metric",
        )


@pytest.mark.unit
def test_invalid_file(data_dir, output_dir):
    input_file = os.path.join(data_dir, "nonexistent.h5")
    output_file = os.path.join(output_dir, "invalid_file.txt")
    with pytest.raises(FileNotFoundError):
        CountsQC(
            input_file=input_file,
            output_file=output_file,
            metric="marker_call_rate",
        )


@pytest.mark.unit
def test_auto_detect_data_type_genotype(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "auto_detect_genotype.txt")
    calculator = CountsQC(
        input_file=input_file,
        output_file=output_file,
        metric="marker_call_rate",
        data_type=None,
    )
    assert calculator.data_type == "Genotype"


@pytest.mark.unit
def test_invalid_probe_variance_genotype(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "invalid_probe_variance.txt")
    with pytest.raises(
        ValueError,
        match="Probe variance calculation is only applicable to methylation data",
    ):
        CountsQC(
            input_file=input_file,
            output_file=output_file,
            metric="probe_variance",
            data_type="Genotype",
        )


@pytest.mark.unit
def test_adjust_workers_large_file(data_dir, output_dir, monkeypatch):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "adjust_workers.txt")

    def mock_getsize(_):
        return 60 * 1024**3

    monkeypatch.setattr(os.path, "getsize", mock_getsize)
    calculator = CountsQC(
        input_file=input_file,
        output_file=output_file,
        metric="marker_call_rate",
    )
    assert calculator.max_workers <= 6


@pytest.mark.unit
def test_process_chromosome_marker_call_rate_empty(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "marker_call_rate_empty.txt")
    calculator = CountsQC(
        input_file=input_file,
        output_file=output_file,
        metric="marker_call_rate",
    )
    result = calculator.process_chromosome_marker_call_rate("invalid_chr")
    assert result is None


@pytest.mark.unit
def test_process_chromosome_sample_call_rate_empty(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "sample_call_rate_empty.txt")
    calculator = CountsQC(
        input_file=input_file,
        output_file=output_file,
        metric="sample_call_rate",
    )
    result = calculator.process_chromosome_sample_call_rate("invalid_chr")
    assert result is None


@pytest.mark.unit
def test_process_chromosome_probe_variance_empty(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    calculator = CountsQC(
        input_file=input_file,
        output_file=os.path.join(output_dir, "probe_variance_empty.txt"),
        metric="probe_variance",
        data_type="Methylation",
    )
    result = calculator.process_chromosome_probe_variance("invalid_chr")
    assert result is None


@pytest.mark.integration
def test_marker_call_rate_genotype(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_marker_call_rate.txt")
    calculator = CountsQC(
        input_file=input_file,
        metric="marker_call_rate",
        output_file=output_file,
        threshold=0.98,
        data_type="Genotype",
    )
    success = calculator.run()
    assert success
    assert os.path.exists(output_file)


@pytest.mark.integration
def test_sample_call_rate_genotype(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_sample_call_rate.txt")
    calculator = CountsQC(
        input_file=input_file,
        metric="sample_call_rate",
        output_file=output_file,
        threshold=0.90,
        data_type="Genotype",
    )
    success = calculator.run()
    assert success
    assert os.path.exists(output_file)
    if os.path.getsize(output_file) > 0:
        filtered = pd.read_csv(output_file, header=None)
        assert filtered.shape[1] == 1


@pytest.mark.integration
def test_marker_call_rate_methylation(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_marker_call_rate.txt")
    calculator = CountsQC(
        input_file=input_file,
        metric="marker_call_rate",
        output_file=output_file,
        threshold=0.98,
        data_type="Methylation",
    )
    success = calculator.run()
    assert success
    assert os.path.exists(output_file)


@pytest.mark.integration
def test_sample_call_rate_methylation(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_sample_call_rate.txt")
    calculator = CountsQC(
        input_file=input_file,
        metric="sample_call_rate",
        output_file=output_file,
        threshold=0.90,
        data_type="Methylation",
    )
    success = calculator.run()
    assert success
    assert os.path.exists(output_file)
    if os.path.getsize(output_file) > 0:
        filtered = pd.read_csv(output_file, header=None)
        assert filtered.shape[1] == 1


@pytest.mark.integration
def test_probe_variance_methylation(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_probe_variance.txt")
    calculator = CountsQC(
        input_file=input_file,
        metric="probe_variance",
        output_file=output_file,
        threshold=0.05,
        data_type="Methylation",
    )
    success = calculator.run()
    assert success
    assert os.path.exists(output_file)


@pytest.mark.integration
def test_sample_call_rate_with_filtered(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_file = os.path.join(output_dir, "gen_sample_call_rate_filtered.txt")
    calculator = CountsQC(
        input_file=input_file,
        metric="sample_call_rate",
        output_file=output_file,
        threshold=1.0,
        data_type="Genotype",
    )
    success = calculator.run()
    assert success
    assert os.path.exists(output_file)
    if os.path.getsize(output_file) > 0:
        filtered = pd.read_csv(output_file, header=None)
        assert filtered.shape[1] == 1


@pytest.mark.integration
def test_probe_variance_no_filtered(output_dir):
    tmp_filename = None
    output_file = os.path.join(output_dir, "methyl_probe_variance_no_filtered.txt")

    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_filename = tmp.name
            with h5py.File(tmp_filename, "w") as h5f:
                chr_grp = h5f.create_group("chr1")

                data = np.full((10, 5), 0.5, dtype=np.float32)
                chr_grp.create_dataset("Methylation", data=data)

                probe_ids = [f"cg{i:08d}" for i in range(10)]
                chr_grp.create_dataset("ProbeList", data=np.array(probe_ids, dtype="S"))

                meta_grp = h5f.create_group("Metadata")
                meta_grp.create_dataset(
                    "SampleList",
                    data=np.array([f"sample{i}" for i in range(5)], dtype="S"),
                )

        calculator = CountsQC(
            input_file=tmp_filename,
            metric="probe_variance",
            output_file=output_file,
            threshold=0.001,
            data_type="Methylation",
        )
        success = calculator.run()
        assert success
        assert os.path.exists(output_file)

        assert os.path.getsize(output_file) == 0

    finally:
        if tmp_filename and os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        if os.path.exists(output_file):
            os.remove(output_file)


@pytest.mark.integration
def test_auto_detect_methylation_data_type(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_meth:
        with h5py.File(tmp_meth.name, "w") as h5f:
            chr_grp = h5f.create_group("chr1")
            betas = np.array([[0.1, 0.2], [0.4, 0.5]], dtype=np.float32)
            chr_grp.create_dataset("Methylation", data=betas)
            chr_grp.create_dataset("CGID", data=np.array(["cg1", "cg2"], dtype="S"))

            meta_grp = h5f.create_group("Metadata")
            meta_grp.create_dataset(
                "SampleList", data=np.array(["s1", "s2"], dtype="S")
            )

    output_file = os.path.join(output_dir, "auto_detect_methylation.txt")
    calculator = CountsQC(
        input_file=tmp_meth.name,
        output_file=output_file,
        metric="marker_call_rate",
        data_type=None,
    )
    assert calculator.data_type == "Methylation"

    os.remove(tmp_meth.name)
    if os.path.exists(output_file):
        os.remove(output_file)


@pytest.mark.integration
def test_marker_call_rate_with_alias_variations(output_dir):
    tmp_filename = None
    output_file = os.path.join(output_dir, "gen_marker_call_rate_alias.txt")

    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_filename = tmp.name
            with h5py.File(tmp_filename, "w") as h5f:
                chr_grp = h5f.create_group("CHROMOSOME1")

                data = np.array([[0, 1, 2], [2, 0, 1], [-1, -1, -1]], dtype=np.int8)
                chr_grp.create_dataset("GT", data=data)

                chr_grp.create_dataset(
                    "SNP", data=np.array(["rs1", "rs2", "rs3"], dtype="S")
                )

                meta_grp = h5f.create_group("META")
                meta_grp.create_dataset(
                    "SAMPLE_ID", data=np.array(["s1", "s2", "s3"], dtype="S")
                )

        calculator = CountsQC(
            input_file=tmp_filename,
            metric="marker_call_rate",
            output_file=output_file,
            threshold=0.9,
            data_type="Genotype",
        )
        success = calculator.run()
        assert success
        assert os.path.exists(output_file)

        if os.path.getsize(output_file) > 0:
            filtered = pd.read_csv(output_file, header=None)
            assert "rs3" in filtered[0].values

    finally:
        if tmp_filename and os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        if os.path.exists(output_file):
            os.remove(output_file)


@pytest.mark.integration
def test_data_type_detection_fallback(output_dir):
    tmp_filename = None
    output_file = os.path.join(output_dir, "detected_data_type.txt")

    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_filename = tmp.name
            with h5py.File(tmp_filename, "w") as h5f:
                chr_grp = h5f.create_group("chr1")

                chr_grp.create_dataset("Genotype", data=np.zeros((5, 3), dtype=np.int8))
                chr_grp.create_dataset(
                    "RSID",
                    data=np.array(["rs1", "rs2", "rs3", "rs4", "rs5"], dtype="S"),
                )

                meta_grp = h5f.create_group("Metadata")
                meta_grp.create_dataset(
                    "IID", data=np.array(["s1", "s2", "s3"], dtype="S")
                )

        calculator = CountsQC(
            input_file=tmp_filename,
            metric="marker_call_rate",
            output_file=output_file,
            threshold=0.9,
        )
        assert calculator.data_type in ["Genotype", "Methylation"]
        success = calculator.run()
        assert success

    finally:
        if tmp_filename and os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        if os.path.exists(output_file):
            os.remove(output_file)


@pytest.mark.integration
def test_chromosome_detection_fallback(output_dir):
    tmp_filename = None
    output_file = os.path.join(output_dir, "chromosome_fallback.txt")

    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_filename = tmp.name
            with h5py.File(tmp_filename, "w") as h5f:
                chr1_grp = h5f.create_group("CHROM_1")
                chr2_grp = h5f.create_group("CHROM_2")

                chr1_grp.create_dataset(
                    "Genotype", data=np.zeros((3, 2), dtype=np.int8)
                )
                chr1_grp.create_dataset(
                    "RSID", data=np.array(["rs1", "rs2", "rs3"], dtype="S")
                )

                chr2_grp.create_dataset(
                    "Genotype", data=np.zeros((2, 2), dtype=np.int8)
                )
                chr2_grp.create_dataset(
                    "RSID", data=np.array(["rs4", "rs5"], dtype="S")
                )

                meta_grp = h5f.create_group("Metadata")
                meta_grp.create_dataset("IID", data=np.array(["s1", "s2"], dtype="S"))

        calculator = CountsQC(
            input_file=tmp_filename,
            metric="marker_call_rate",
            output_file=output_file,
            threshold=0.9,
            data_type="Genotype",
        )
        success = calculator.run()
        assert success

    finally:
        if tmp_filename and os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        if os.path.exists(output_file):
            os.remove(output_file)
