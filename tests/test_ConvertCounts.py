#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import pytest
import re
import shutil
import tempfile
import h5py
from ConvertCounts import ConvertCounts
from utils.AliasUtils import AliasUtils
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.mark.integration
def test_hdf5_to_csv_methylation(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_converted.csv")
    converter = ConvertCounts(
        input_file=input_file, output_file=output_file, chip="EPIC"
    )
    success = converter.convert()
    assert success
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert not df.empty
    assert "CGID" in df.columns
    assert df.shape[1] == 11


@pytest.mark.integration
def test_plink_to_hdf5(data_dir, output_dir):
    input_prefix = os.path.join(data_dir, "gen_data")
    output_file = os.path.join(output_dir, "plink_converted.h5")
    converter = ConvertCounts(input_file=input_prefix, output_file=output_file)
    success = converter.convert()
    assert success
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as hf:
        assert "CHR1" in hf
        genotype_key = AliasUtils.find_keys(hf["CHR1"], "Genotype")
        rsid_key = AliasUtils.find_keys(hf["CHR1"], "RSID")
        assert genotype_key is not None
        assert rsid_key is not None


@pytest.mark.integration
def test_hdf5_to_plink(data_dir, output_dir):
    input_file = os.path.join(data_dir, "gen_data.h5")
    output_prefix = os.path.join(output_dir, "hdf5_to_plink")
    converter = ConvertCounts(input_file=input_file, output_file=output_prefix)
    success = converter.convert()
    assert success
    assert os.path.exists(f"{output_prefix}.bed")
    assert os.path.exists(f"{output_prefix}.bim")
    assert os.path.exists(f"{output_prefix}.fam")

    bim_df = pd.read_csv(
        f"{output_prefix}.bim",
        sep="\t",
        header=None,
        names=["CHR", "RSID", "cm", "BP", "A1", "A2"],
    )
    assert len(bim_df) == 10000
    assert all(bim_df["CHR"] == 1)

    fam_df = pd.read_csv(
        f"{output_prefix}.fam",
        sep="\t",
        header=None,
        names=["fid", "iid", "father", "mother", "sex", "phenotype"],
    )
    assert len(fam_df) == 10

    assert os.path.getsize(f"{output_prefix}.bed") > 3


@pytest.mark.integration
def test_csv_to_hdf5_methylation(output_dir, monkeypatch):
    def mock_download_and_process_manifest(self, tmpdirname):
        csv_file = os.path.join(tmpdirname, "EPIC_array_v1.B5.csv")
        pd.DataFrame(
            {"IlmnID": ["cg00000029", "cg00000165"], "CHR_hg38": ["16", "3"]}
        ).to_csv(csv_file, index=False)
        manifest = pd.read_csv(csv_file)
        manifest = manifest.rename(columns={"IlmnID": "CGID", "CHR_hg38": "chromosome"})
        manifest["chromosome"] = manifest["chromosome"].apply(
            lambda x: re.sub(r"^CHR", "", str(x)) if isinstance(x, str) else x
        )
        autosomal_chromosomes = [str(i) for i in range(1, 23)]
        manifest = manifest[manifest["chromosome"].isin(autosomal_chromosomes)]
        manifest = manifest[manifest["chromosome"].apply(lambda x: str(x).isdigit())]
        manifest["chromosome"] = manifest["chromosome"].astype(int)
        manifest = manifest[manifest["chromosome"] > 0]
        return manifest.drop_duplicates().reset_index(drop=True)

    monkeypatch.setattr(
        ConvertCounts,
        "_download_and_process_manifest",
        mock_download_and_process_manifest,
    )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        df = pd.DataFrame(
            {
                "CGID": ["cg00000029", "cg00000165"],
                "chromosome": [16, 3],
                "sample1": [0.1, 0.4],
                "sample2": [0.2, 0.5],
                "sample3": [0.3, 0.6],
            }
        )
        df.to_csv(tmp_csv.name, index=False)
    output_file = os.path.join(output_dir, "csv_to_hdf5_methyl.h5")
    converter = ConvertCounts(
        input_file=tmp_csv.name, output_file=output_file, chip="EPIC"
    )
    success = converter.convert()
    assert success
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as hf:
        assert "CHR16" in hf or "CHR3" in hf
        chr_groups = [key for key in hf.keys() if key.startswith("CHR")]
        assert len(chr_groups) > 0, "No chromosome groups found"
        chr_group = hf[chr_groups[0]]
        methylation_key = AliasUtils.find_keys(chr_group, "Methylation")
        probelist_key = AliasUtils.find_keys(chr_group, "ProbeList")
        assert methylation_key is not None
        assert probelist_key is not None
    os.remove(tmp_csv.name)


@pytest.mark.integration
def test_transpose_hdf5_to_csv(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "transposed.csv")
    converter = ConvertCounts(
        input_file=input_file, output_file=output_file, transpose=True
    )
    success = converter.convert()
    assert success
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert df.shape[0] == 10
    assert df.shape[1] == 5001
    assert "sample_id" in df.columns


@pytest.mark.integration
def test_invalid_plink_input_missing_files(data_dir, output_dir):
    input_prefix = os.path.join(data_dir, "nonexistent_plink")
    output_file = os.path.join(output_dir, "invalid_plink.h5")
    with pytest.raises(ValueError, match="Cannot determine input format"):
        converter = ConvertCounts(input_file=input_prefix, output_file=output_file)
        converter.convert()


@pytest.mark.integration
def test_invalid_file_extension(data_dir, output_dir):
    input_file = os.path.join(data_dir, "invalid.xyz")
    output_file = os.path.join(output_dir, "invalid.h5")
    with pytest.raises(ValueError, match="Unsupported file format: .xyz"):
        converter = ConvertCounts(input_file=input_file, output_file=output_file)
        converter.convert()


@pytest.mark.integration
def test_same_format_conversion(data_dir, output_dir):
    input_file = os.path.join(data_dir, "mcseq.h5")
    output_file = os.path.join(output_dir, "methyl_output.h5")
    converter = ConvertCounts(input_file=input_file, output_file=output_file)
    assert not converter.convert()


@pytest.mark.integration
def test_unsupported_conversion(data_dir, output_dir):
    input_file = os.path.join(data_dir, "unsupported.txt")
    output_file = os.path.join(output_dir, "invalid.txt")
    converter = ConvertCounts(input_file=input_file, output_file=output_file)
    assert not converter.convert()


@pytest.mark.integration
def test_empty_csv(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        open(tmp_csv.name, "w").close()
    output_file = os.path.join(output_dir, "empty_csv.h5")
    converter = ConvertCounts(
        input_file=tmp_csv.name, output_file=output_file, chip="EPIC"
    )
    assert not converter.convert()
    os.remove(tmp_csv.name)


@pytest.mark.integration
def test_csv_no_chip_methylation(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        df = pd.DataFrame(
            {
                "CGID": ["cg00000029", "cg00000165"],
                "sample1": [0.1, 0.4],
                "sample2": [0.2, 0.5],
            }
        )
        df.to_csv(tmp_csv.name, index=False)
    output_file = os.path.join(output_dir, "no_chip.h5")
    converter = ConvertCounts(input_file=tmp_csv.name, output_file=output_file)
    assert not converter.convert()
    os.remove(tmp_csv.name)


@pytest.mark.integration
def test_csv_invalid_data_type(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        df = pd.DataFrame(
            {"invalid_id": ["id1", "id2"], "sample1": [0.1, 0.4], "sample2": [0.2, 0.5]}
        )
        df.to_csv(tmp_csv.name, index=False)
    output_file = os.path.join(output_dir, "invalid_type.h5")
    converter = ConvertCounts(input_file=tmp_csv.name, output_file=output_file)
    assert not converter.convert()
    os.remove(tmp_csv.name)


@pytest.mark.integration
def test_csv_to_hdf5_genotype(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        df = pd.DataFrame(
            {
                "RSID": ["rs1", "rs2"],
                "chromosome": [1, 2],
                "sample1": [0, 1],
                "sample2": [1, 2],
            }
        )
        df.to_csv(tmp_csv.name, index=False)
    output_file = os.path.join(output_dir, "gen_to_hdf5.h5")
    converter = ConvertCounts(input_file=tmp_csv.name, output_file=output_file)
    success = converter.convert()
    assert success
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as hf:
        chr_groups = [
            key
            for key in hf.keys()
            if AliasUtils.strip_numeric_suffix(key).upper() in ["CHR", "CHROMOSOME"]
        ]
        assert len(chr_groups) >= 2
        genotype_key_chr1 = AliasUtils.find_keys(hf["CHR1"], "Genotype")
        rsid_key_chr1 = AliasUtils.find_keys(hf["CHR1"], "RSID")
        genotype_key_chr2 = AliasUtils.find_keys(hf["CHR2"], "Genotype")
        assert genotype_key_chr1 and rsid_key_chr1 and genotype_key_chr2
        assert hf["CHR1"][genotype_key_chr1].shape == (1, 2)
        assert hf["CHR2"][genotype_key_chr2].shape == (1, 2)
    os.remove(tmp_csv.name)


@pytest.mark.integration
def test_csv_to_hdf5_methylation_no_chrom(output_dir, monkeypatch):
    def mock_download_and_process_manifest(self, tmpdirname):
        csv_file = os.path.join(tmpdirname, "EPIC_array_v1.B5.csv")
        pd.DataFrame(
            {"IlmnID": ["cg00000029", "cg00000165"], "CHR_hg38": ["16", "0"]}
        ).to_csv(csv_file, index=False)
        manifest = pd.read_csv(csv_file)
        manifest = manifest.rename(columns={"IlmnID": "CGID", "CHR_hg38": "chromosome"})
        manifest["chromosome"] = manifest["chromosome"].apply(
            lambda x: re.sub(r"^CHR", "", str(x)) if isinstance(x, str) else x
        )
        return manifest.drop_duplicates().reset_index(drop=True)

    monkeypatch.setattr(
        ConvertCounts,
        "_download_and_process_manifest",
        mock_download_and_process_manifest,
    )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        df = pd.DataFrame(
            {
                "CGID": ["cg00000029", "cg00000165"],
                "sample1": [0.1, 0.4],
                "sample2": [0.2, 0.5],
                "sample3": [0.3, 0.6],
            }
        )
        df.to_csv(tmp_csv.name, index=False)
    output_file = os.path.join(output_dir, "csv_to_hdf5_methyl_no_chrom.h5")
    converter = ConvertCounts(
        input_file=tmp_csv.name, output_file=output_file, chip="EPIC"
    )
    success = converter.convert()
    assert success
    assert os.path.exists(output_file)
    with h5py.File(output_file, "r") as hf:
        assert "CHR16" in hf
        methylation_key = AliasUtils.find_keys(hf["CHR16"], "Methylation")
        probelist_key = AliasUtils.find_keys(hf["CHR16"], "ProbeList")
        assert methylation_key is not None
        assert probelist_key is not None
    os.remove(tmp_csv.name)


@pytest.mark.integration
def test_empty_hdf5_methylation(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w"):
            pass
    output_file = os.path.join(output_dir, "empty_hdf5.csv")
    converter = ConvertCounts(
        input_file=tmp_hdf5.name, output_file=output_file, chip="EPIC"
    )
    assert not converter.convert()
    os.remove(tmp_hdf5.name)


@pytest.mark.integration
def test_invalid_hdf5_genotype(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            hf.create_group("CHR1")
    output_prefix = os.path.join(output_dir, "invalid_hdf5_to_plink")
    converter = ConvertCounts(input_file=tmp_hdf5.name, output_file=output_prefix)
    assert not converter.convert()
    os.remove(tmp_hdf5.name)


@pytest.mark.integration
def test_plink_missing_bed(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".bim", delete=False) as tmp_bim:
        with tempfile.NamedTemporaryFile(suffix=".fam", delete=False) as tmp_fam:
            pd.DataFrame(
                {
                    "chrom": [1],
                    "snp": ["rs1"],
                    "cm": [0],
                    "pos": [100],
                    "A1": ["A"],
                    "A2": ["G"],
                }
            ).to_csv(tmp_bim.name, sep="\t", header=False, index=False)
            pd.DataFrame(
                {
                    "fid": ["s1"],
                    "iid": ["s1"],
                    "father": ["0"],
                    "mother": ["0"],
                    "sex": [0],
                    "Phenotype": [0],
                }
            ).to_csv(tmp_fam.name, sep="\t", header=False, index=False)
            prefix = os.path.join(output_dir, "plink_no_bed")
            shutil.copy2(tmp_bim.name, f"{prefix}.bim")
            shutil.copy2(tmp_fam.name, f"{prefix}.fam")
            output_file = os.path.join(output_dir, "plink_no_bed.h5")
            converter = ConvertCounts(input_file=prefix, output_file=output_file)
            assert not converter.convert()
            os.remove(f"{prefix}.bim")
            os.remove(f"{prefix}.fam")
            os.remove(tmp_bim.name)
            os.remove(tmp_fam.name)


@pytest.mark.integration
def test_plink_invalid_header(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".bed", delete=False) as tmp_bed:
        with tempfile.NamedTemporaryFile(suffix=".bim", delete=False) as tmp_bim:
            with tempfile.NamedTemporaryFile(suffix=".fam", delete=False) as tmp_fam:
                with open(tmp_bed.name, "wb") as f:
                    f.write(b"\x00\x00\x00")
                pd.DataFrame(
                    {
                        "chrom": [1],
                        "snp": ["rs1"],
                        "cm": [0],
                        "pos": [100],
                        "A1": ["A"],
                        "A2": ["G"],
                    }
                ).to_csv(tmp_bim.name, sep="\t", header=False, index=False)
                pd.DataFrame(
                    {
                        "fid": ["s1"],
                        "iid": ["s1"],
                        "father": ["0"],
                        "mother": ["0"],
                        "sex": [0],
                        "Phenotype": [0],
                    }
                ).to_csv(tmp_fam.name, sep="\t", header=False, index=False)
                prefix = os.path.join(output_dir, "plink_invalid_header")
                shutil.copy2(tmp_bed.name, f"{prefix}.bed")
                shutil.copy2(tmp_bim.name, f"{prefix}.bim")
                shutil.copy2(tmp_fam.name, f"{prefix}.fam")
                output_file = os.path.join(output_dir, "plink_invalid_header.h5")
                converter = ConvertCounts(input_file=prefix, output_file=output_file)
                assert not converter.convert()
                os.remove(f"{prefix}.bed")
                os.remove(f"{prefix}.bim")
                os.remove(f"{prefix}.fam")
                os.remove(tmp_bed.name)
                os.remove(tmp_bim.name)
                os.remove(tmp_fam.name)


@pytest.mark.integration
def test_plink_with_nan(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".bed", delete=False) as tmp_bed:
        with tempfile.NamedTemporaryFile(suffix=".bim", delete=False) as tmp_bim:
            with tempfile.NamedTemporaryFile(suffix=".fam", delete=False) as tmp_fam:
                with open(tmp_bed.name, "wb") as f:
                    f.write(b"\x6c\x1b\x01")
                    f.write(bytearray([0x00]))
                pd.DataFrame(
                    {
                        "chrom": [1],
                        "snp": ["rs1"],
                        "cm": [0],
                        "pos": [100],
                        "A1": ["A"],
                        "A2": ["G"],
                    }
                ).to_csv(tmp_bim.name, sep="\t", header=False, index=False)
                pd.DataFrame(
                    {
                        "fid": [np.nan],
                        "iid": ["s1"],
                        "father": ["0"],
                        "mother": ["0"],
                        "sex": [np.nan],
                        "Phenotype": [np.nan],
                    }
                ).to_csv(tmp_fam.name, sep="\t", header=False, index=False)
                prefix = os.path.join(output_dir, "plink_with_nan")
                shutil.copy2(tmp_bed.name, f"{prefix}.bed")
                shutil.copy2(tmp_bim.name, f"{prefix}.bim")
                shutil.copy2(tmp_fam.name, f"{prefix}.fam")
                output_file = os.path.join(output_dir, "plink_with_nan.h5")
                converter = ConvertCounts(input_file=prefix, output_file=output_file)
                success = converter.convert()
                assert success
                assert os.path.exists(output_file)
                with h5py.File(output_file, "r") as hf:
                    metadata_key = AliasUtils.find_keys(hf, "Metadata")
                    metadata = hf[metadata_key]
                    fid_key = AliasUtils.find_keys(metadata, "FID")
                    sex_key = AliasUtils.find_keys(metadata, "Sex")
                    phenotype_key = AliasUtils.find_keys(metadata, "Phenotype")
                    assert fid_key and sex_key and phenotype_key
                    assert metadata[fid_key][0].decode() == "missing"
                    assert metadata[sex_key][0] == -9
                    assert metadata[phenotype_key][0] == -9
                os.remove(f"{prefix}.bed")
                os.remove(f"{prefix}.bim")
                os.remove(f"{prefix}.fam")
                os.remove(tmp_bed.name)
                os.remove(tmp_bim.name)
                os.remove(tmp_fam.name)


@pytest.mark.integration
def test_invalid_chip_methylation(output_dir, monkeypatch):
    def mock_download_and_process_manifest(self, tmpdirname):
        csv_file = os.path.join(tmpdirname, "invalid_manifest.csv")
        pd.DataFrame(
            {"IlmnID": ["cg00000029", "cg00000165"], "CHR_hg38": ["16", "3"]}
        ).to_csv(csv_file, index=False)
        manifest = pd.read_csv(csv_file)
        manifest = manifest.rename(columns={"IlmnID": "CGID", "CHR_hg38": "chromosome"})
        return manifest.drop_duplicates().reset_index(drop=True)

    monkeypatch.setattr(
        ConvertCounts,
        "_download_and_process_manifest",
        mock_download_and_process_manifest,
    )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        df = pd.DataFrame(
            {
                "CGID": ["cg00000029", "cg00000165"],
                "sample1": [0.1, 0.4],
                "sample2": [0.2, 0.5],
            }
        )
        df.to_csv(tmp_csv.name, index=False)
    output_file = os.path.join(output_dir, "invalid_chip.h5")
    converter = ConvertCounts(
        input_file=tmp_csv.name, output_file=output_file, chip="INVALID"
    )
    assert not converter.convert()
    os.remove(tmp_csv.name)


@pytest.mark.integration
def test_csv_empty_chromosomes(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        df = pd.DataFrame(
            {
                "CGID": ["cg00000029", "cg00000165"],
                "chromosome": ["X", "Y"],
                "sample1": [0.1, 0.4],
                "sample2": [0.2, 0.5],
            }
        )
        df.to_csv(tmp_csv.name, index=False)
    output_file = os.path.join(output_dir, "empty_chrom.h5")
    converter = ConvertCounts(
        input_file=tmp_csv.name, output_file=output_file, chip="EPIC"
    )
    assert not converter.convert()
    os.remove(tmp_csv.name)


@pytest.mark.integration
def test_hdf5_no_valid_data(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            metadata = hf.create_group("metadata")
            metadata.create_dataset(
                AliasUtils.get_aliases("SampleList")[0], data=["s1", "s2"], dtype="S50"
            )
            hf.create_group("CHR1")
    output_file = os.path.join(output_dir, "no_valid_data.csv")
    converter = ConvertCounts(
        input_file=tmp_hdf5.name, output_file=output_file, chip="EPIC"
    )
    assert not converter.convert()
    os.remove(tmp_hdf5.name)


@pytest.mark.integration
def test_hdf5_missing_snp_metadata(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            metadata = hf.create_group("metadata")
            metadata.create_dataset(
                AliasUtils.get_aliases("IID")[0], data=["s1", "s2"], dtype="S50"
            )
            chr1 = hf.create_group("CHR1")
            chr1.create_dataset(
                AliasUtils.get_aliases("Genotype")[0],
                data=np.zeros((2, 2), dtype=np.int8),
            )
            chr1.create_dataset("genotype", data=np.zeros((2, 2), dtype=np.int8))
    output_prefix = os.path.join(output_dir, "missing_snp_metadata")
    converter = ConvertCounts(input_file=tmp_hdf5.name, output_file=output_prefix)
    assert not converter.convert()
    os.remove(tmp_hdf5.name)


@pytest.mark.integration
def test_plink_empty_chromosome(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".bed", delete=False) as tmp_bed:
        with tempfile.NamedTemporaryFile(suffix=".bim", delete=False) as tmp_bim:
            with tempfile.NamedTemporaryFile(suffix=".fam", delete=False) as tmp_fam:
                with open(tmp_bed.name, "wb") as f:
                    f.write(b"\x6c\x1b\x01")
                    f.write(bytearray([0x00]))
                pd.DataFrame(
                    {
                        "chrom": [2],
                        "snp": ["rs2"],
                        "cm": [0],
                        "pos": [200],
                        "A1": ["A"],
                        "A2": ["G"],
                    }
                ).to_csv(tmp_bim.name, sep="\t", header=False, index=False)
                pd.DataFrame(
                    {
                        "fid": ["s1"],
                        "iid": ["s1"],
                        "father": ["0"],
                        "mother": ["0"],
                        "sex": [0],
                        "Phenotype": [0],
                    }
                ).to_csv(tmp_fam.name, sep="\t", header=False, index=False)
                prefix = os.path.join(output_dir, "plink_empty_chrom")
                shutil.copy2(tmp_bed.name, f"{prefix}.bed")
                shutil.copy2(tmp_bim.name, f"{prefix}.bim")
                shutil.copy2(tmp_fam.name, f"{prefix}.fam")
                output_file = os.path.join(output_dir, "plink_empty_chrom.h5")
                converter = ConvertCounts(input_file=prefix, output_file=output_file)
                success = converter.convert()
                assert success
                os.remove(f"{prefix}.bed")
                os.remove(f"{prefix}.bim")
                os.remove(f"{prefix}.fam")
                os.remove(tmp_bed.name)
                os.remove(tmp_bim.name)
                os.remove(tmp_fam.name)


@pytest.mark.integration
def test_hdf5_variant_mismatch(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            metadata = hf.create_group("metadata")
            metadata.create_dataset("iid", data=["s1", "s2"], dtype="S50")
            chr1 = hf.create_group("CHR1")
            chr1.create_dataset("snp", data=["rs1", "rs2"], dtype="S50")
            chr1.create_dataset("pos", data=[100, 200], dtype=np.int32)
            chr1.create_dataset("A1", data=["A", "A"], dtype="S10")
            chr1.create_dataset("A2", data=["G", "G"], dtype="S10")
            chr1.create_dataset("genotype", data=np.zeros((1, 2), dtype=np.int8))
    output_prefix = os.path.join(output_dir, "variant_mismatch")
    converter = ConvertCounts(input_file=tmp_hdf5.name, output_file=output_prefix)
    assert not converter.convert()
    os.remove(tmp_hdf5.name)


@pytest.mark.integration
def test_hdf5_sample_mismatch(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            metadata = hf.create_group("metadata")
            metadata.create_dataset("iid", data=["s1", "s2"], dtype="S50")
            chr1 = hf.create_group("CHR1")
            chr1.create_dataset("snp", data=["rs1"], dtype="S50")
            chr1.create_dataset("pos", data=[100], dtype=np.int32)
            chr1.create_dataset("A1", data=["A"], dtype="S10")
            chr1.create_dataset("A2", data=["G"], dtype="S10")
            chr1.create_dataset("genotype", data=np.zeros((3, 1), dtype=np.int8))
    output_prefix = os.path.join(output_dir, "sample_mismatch")
    converter = ConvertCounts(input_file=tmp_hdf5.name, output_file=output_prefix)
    assert not converter.convert()
    os.remove(tmp_hdf5.name)


@pytest.mark.integration
def test_transpose_invalid_columns(output_dir):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_hdf5:
        with h5py.File(tmp_hdf5.name, "w") as hf:
            metadata = hf.create_group("metadata")
            metadata.create_dataset(
                AliasUtils.get_aliases("SampleList")[0], data=["s1", "s2"], dtype="S50"
            )
            chr1 = hf.create_group("CHR1")
            chr1.create_dataset(
                AliasUtils.get_aliases("Methylation")[0],
                data=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            )
            chr1.create_dataset(
                AliasUtils.get_aliases("ProbeList")[0], data=["cg1", "cg2"], dtype="S50"
            )
    output_file = os.path.join(output_dir, "transpose_invalid.csv")
    converter = ConvertCounts(
        input_file=tmp_hdf5.name, output_file=output_file, transpose=True
    )
    success = converter.convert()
    assert success
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert "sample_id" in df.columns
    assert "cg1" in df.columns
    assert "cg2" in df.columns
    os.remove(tmp_hdf5.name)
