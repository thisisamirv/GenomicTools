#!/usr/bin/env python
import h5py
import numpy as np
import os
import pytest
import shutil
from unittest.mock import patch
from utils.LoggingUtils import log
from VariantQC import VariantQC

log.setup(level="DEBUG")


def find_dataset_in_group(group, expected_names):
    for key in group.keys():
        if key.lower() in [name.lower() for name in expected_names]:
            return key
    return None


def assert_dataset_exists(group, expected_names, error_msg):
    found_key = find_dataset_in_group(group, expected_names)
    assert found_key is not None, f"{error_msg}. Available keys: {list(group.keys())}"
    return found_key


def create_test_genotype_file(
    filepath, n_variants=100, n_samples=50, use_canonical_names=True
):
    with h5py.File(filepath, "w") as h5f:
        genotype_name = "Genotype" if use_canonical_names else "data"
        rsid_name = "RSID" if use_canonical_names else "snp"

        chr_grp = h5f.create_group("chr1")

        genotypes = np.random.choice(
            [0, 1, 2], size=(n_variants, n_samples), p=[0.6, 0.3, 0.1]
        )
        missing_mask = np.random.random((n_variants, n_samples)) < 0.2
        genotypes = genotypes.astype(float)
        genotypes[missing_mask] = -1

        rsids = [f"rs{i + 1000000}" for i in range(n_variants)]
        positions = np.sort(np.random.randint(1000, 100000000, n_variants))

        chr_grp.create_dataset(genotype_name, data=genotypes)
        chr_grp.create_dataset(rsid_name, data=np.array(rsids, dtype="S"))
        chr_grp.create_dataset("A1", data=np.array(["A"] * n_variants, dtype="S"))
        chr_grp.create_dataset("A2", data=np.array(["G"] * n_variants, dtype="S"))
        chr_grp.create_dataset("BP", data=positions)

        meta_grp = h5f.create_group("metadata")
        meta_grp.create_dataset(
            "iid", data=np.array([f"sample_{i}" for i in range(n_samples)], dtype="S")
        )

        populations = np.random.choice([1, 2, 3], size=n_samples, p=[0.5, 0.3, 0.2])
        meta_grp.create_dataset("population", data=populations)

    return filepath


def create_test_file_with_maf(filepath, n_variants=100, n_samples=50):
    create_test_genotype_file(filepath, n_variants, n_samples)

    with h5py.File(filepath, "r+") as h5f:
        maf_values = np.random.beta(0.5, 2, n_variants)
        h5f["chr1"].create_dataset("MAF", data=maf_values)

    return filepath


def create_ld_test_file(filepath):
    with h5py.File(filepath, "w") as h5f:
        chr_grp = h5f.create_group("chr1")

        genotypes = np.array(
            [
                [0, 1, 2, 0, 1, 0, 1, 2, 0, 1],
                [0, 1, 2, 0, 1, 0, 1, 2, 0, 1],
                [2, 1, 0, 2, 1, 2, 1, 0, 2, 1],
            ],
            dtype=float,
        )

        chr_grp.create_dataset("Genotype", data=genotypes)
        chr_grp.create_dataset("RSID", data=np.array(["rs1", "rs2", "rs3"], dtype="S"))
        chr_grp.create_dataset("BP", data=[1000, 2000, 3000])
        chr_grp.create_dataset("A1", data=np.array(["A", "A", "A"], dtype="S"))
        chr_grp.create_dataset("A2", data=np.array(["G", "G", "G"], dtype="S"))

        meta_grp = h5f.create_group("metadata")
        meta_grp.create_dataset(
            "iid", data=np.array([f"s{i}" for i in range(10)], dtype="S")
        )
        meta_grp.create_dataset(
            "population", data=np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1])
        )

    return filepath


@pytest.fixture(autouse=True)
def mock_alias_utils():
    with patch("VariantQC.AliasUtils.get_aliases") as mock_get_aliases:
        mock_get_aliases.side_effect = lambda field: {
            "Genotype": ["Genotype", "genotype", "data", "gt"],
            "RSID": ["RSID", "rsid", "snp", "snp_id"],
            "MAF": ["MAF", "maf", "minor_allele_freq"],
            "HWE": ["HWE", "hwe", "hardy_weinberg"],
            "Metadata": ["metadata", "meta_data", "Metadata"],
            "Population": ["population", "pop", "Population"],
            "IID": ["IID", "iid", "sample_id"],
            "A1": ["A1", "a1", "allele1"],
            "A2": ["A2", "a2", "allele2"],
            "BP": ["BP", "bp", "position", "pos"],
        }.get(field, [field])
        yield mock_get_aliases


@pytest.fixture(scope="session")
def test_data_files(output_dir):
    test_files = {}

    basic_file = os.path.join(output_dir, "test_gen_data.h5")
    test_files["basic"] = create_test_genotype_file(basic_file)

    maf_file = os.path.join(output_dir, "test_gen_with_maf.h5")
    test_files["with_maf"] = create_test_file_with_maf(maf_file)

    ld_file = os.path.join(output_dir, "test_ld_data.h5")
    test_files["ld_test"] = create_ld_test_file(ld_file)

    yield test_files

    for filepath in test_files.values():
        if os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except OSError:
                pass


@pytest.mark.unit
def test_invalid_analysis_type(output_dir):
    with pytest.raises(ValueError):
        VariantQC(
            input_file="dummy.h5",
            output_file=os.path.join(output_dir, "invalid_output.h5"),
            analysis_type="invalid",
        )


@pytest.mark.unit
def test_missing_input_file(output_dir):
    with pytest.raises(FileNotFoundError):
        VariantQC(
            input_file="nonexistent.h5",
            output_file=os.path.join(output_dir, "missing_output.h5"),
            analysis_type="maf",
        )


@pytest.mark.integration
def test_maf_calculation(test_data_files, output_dir):
    input_file = test_data_files["basic"]
    output_file = os.path.join(output_dir, "gen_with_maf.h5")

    analyzer = VariantQC(
        input_file=input_file,
        analysis_type="maf",
        output_file=output_file,
        threshold=0.01,
    )
    result = analyzer.run()

    assert result is not None
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as h5f:
        assert "chr1" in h5f
        chr_grp = h5f["chr1"]
        maf_key = assert_dataset_exists(
            chr_grp, ["MAF", "maf", "minor_allele_freq"], "MAF dataset not found"
        )

        maf_values = chr_grp[maf_key][:]
        assert len(maf_values) > 0
        assert np.all(maf_values >= 0)
        assert np.all(maf_values <= 0.5)


@pytest.mark.integration
def test_hwe_calculation(test_data_files, output_dir):
    input_file = test_data_files["basic"]
    output_file = os.path.join(output_dir, "gen_with_hwe.h5")

    analyzer = VariantQC(
        input_file=input_file,
        analysis_type="hwe",
        output_file=output_file,
        threshold=1e-6,
        pop_code=None,
    )
    result = analyzer.run()

    assert result is not None
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as h5f:
        assert "chr1" in h5f
        chr_grp = h5f["chr1"]
        hwe_key = assert_dataset_exists(
            chr_grp, ["HWE", "hwe", "hardy_weinberg"], "HWE dataset not found"
        )

        hwe_values = chr_grp[hwe_key][:]
        assert len(hwe_values) > 0
        assert np.all(hwe_values >= 0)
        assert np.all(hwe_values <= 1)


@pytest.mark.integration
def test_hwe_with_pop_code(test_data_files, output_dir):
    input_file = test_data_files["basic"]
    output_file = os.path.join(output_dir, "gen_with_hwe_pop.h5")

    analyzer = VariantQC(
        input_file=input_file,
        analysis_type="hwe",
        output_file=output_file,
        threshold=1e-6,
        pop_code=1,
    )
    result = analyzer.run()

    assert result is not None
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as h5f:
        assert "chr1" in h5f
        chr_grp = h5f["chr1"]
        hwe_found = False
        for key in chr_grp.keys():
            if "hwe" in key.lower() or "hardy_weinberg" in key.lower():
                hwe_found = True
                break
        assert hwe_found, f"HWE dataset not found in: {list(chr_grp.keys())}"


@pytest.mark.integration
def test_maf_filter(test_data_files, output_dir):
    input_file = test_data_files["basic"]
    output_file = os.path.join(output_dir, "gen_maf_filtered.h5")

    analyzer = VariantQC(
        input_file=input_file,
        analysis_type="maf_filter",
        output_file=output_file,
        threshold=0.01,
    )
    result = analyzer.run()

    assert result is not None
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as h5f:
        assert "filter_info" in h5f
        assert h5f["filter_info"].attrs["filter_type"] == "maf_filter"

        if "chr1" in h5f:
            chr_grp = h5f["chr1"]
            genotype_key = find_dataset_in_group(
                chr_grp, ["Genotype", "genotype", "data"]
            )
            if genotype_key:
                with h5py.File(input_file, "r") as orig_h5f:
                    orig_genotype_key = find_dataset_in_group(
                        orig_h5f["chr1"], ["Genotype", "genotype", "data"]
                    )
                    if orig_genotype_key:
                        filtered_shape = chr_grp[genotype_key].shape[0]
                        original_shape = orig_h5f["chr1"][orig_genotype_key].shape[0]
                        assert filtered_shape <= original_shape


@pytest.mark.integration
def test_ld_prune(test_data_files, output_dir):
    input_file = test_data_files["ld_test"]
    output_file = os.path.join(output_dir, "gen_ld_pruned.h5")

    analyzer = VariantQC(
        input_file=input_file,
        analysis_type="ld_prune",
        output_file=output_file,
        window_size=3,
        step_size=1,
        r2_threshold=0.5,
        maf_threshold=0.0,
    )
    result = analyzer.run()

    assert result is not None
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as h5f:
        assert "filter_info" in h5f
        assert h5f["filter_info"].attrs["filter_type"] == "ld_prune"

        if "chr1" in h5f:
            chr_grp = h5f["chr1"]
            rsid_key = find_dataset_in_group(chr_grp, ["RSID", "rsid", "snp"])
            if rsid_key:
                assert len(chr_grp[rsid_key]) < 3


@pytest.mark.integration
def test_maf_existing(test_data_files, output_dir):
    input_file = test_data_files["with_maf"]
    output_file = os.path.join(output_dir, "gen_maf_existing.h5")

    analyzer = VariantQC(
        input_file=input_file,
        analysis_type="maf_filter",
        output_file=output_file,
        threshold=0.01,
    )

    assert analyzer.has_maf_values
    result = analyzer.run()
    assert result is not None


@pytest.mark.integration
def test_maf_in_place(test_data_files, output_dir):
    input_file = test_data_files["basic"]
    temp_file = os.path.join(output_dir, "temp_inplace.h5")
    shutil.copy(input_file, temp_file)

    analyzer = VariantQC(
        input_file=temp_file, analysis_type="maf", output_file=temp_file
    )
    result = analyzer.run()

    assert result == temp_file

    with h5py.File(temp_file, "r") as h5f:
        chr_grp = h5f["chr1"]
        maf_key = find_dataset_in_group(chr_grp, ["MAF", "maf", "minor_allele_freq"])
        assert maf_key is not None

    os.unlink(temp_file)


@pytest.mark.integration
def test_population_mask_creation(output_dir):
    temp_file = os.path.join(output_dir, "test_pop_mask.h5")

    with h5py.File(temp_file, "w") as h5f:
        chr_grp = h5f.create_group("chr1")
        chr_grp.create_dataset(
            "Genotype", data=np.random.choice([0, 1, 2], size=(10, 5))
        )
        chr_grp.create_dataset(
            "RSID", data=np.array([f"rs{i}" for i in range(10)], dtype="S")
        )

        meta_grp = h5f.create_group("metadata")
        meta_grp.create_dataset("population", data=np.array([1, 1, 2, 2, 1]))
        meta_grp.create_dataset(
            "iid", data=np.array([f"s{i}" for i in range(5)], dtype="S")
        )

    analyzer = VariantQC(
        input_file=temp_file,
        output_file=os.path.join(output_dir, "hwe_pop1.csv"),
        analysis_type="hwe",
        pop_code=1,
    )
    success = analyzer.create_population_mask()

    assert success
    assert np.sum(analyzer.sample_mask) == 3

    os.unlink(temp_file)


@pytest.mark.integration
def test_population_mask_string_codes(output_dir):
    temp_file = os.path.join(output_dir, "test_pop_string.h5")

    with h5py.File(temp_file, "w") as h5f:
        chr_grp = h5f.create_group("chr1")
        chr_grp.create_dataset(
            "Genotype", data=np.random.choice([0, 1, 2], size=(10, 3))
        )
        chr_grp.create_dataset(
            "RSID", data=np.array([f"rs{i}" for i in range(10)], dtype="S")
        )

        meta_grp = h5f.create_group("metadata")
        pops = np.array(["EUR", "AFR", "EUR"], dtype="S")
        meta_grp.create_dataset("population", data=pops)
        meta_grp.create_dataset(
            "iid", data=np.array([f"s{i}" for i in range(3)], dtype="S")
        )

    analyzer = VariantQC(
        input_file=temp_file,
        output_file=os.path.join(output_dir, "hwe_pop_eur.csv"),
        analysis_type="hwe",
        pop_code="EUR",
    )
    success = analyzer.create_population_mask()

    assert success
    assert np.sum(analyzer.sample_mask) == 2

    os.unlink(temp_file)


@pytest.mark.integration
def test_no_chromosomes(output_dir):
    temp_file = os.path.join(output_dir, "test_no_chr.h5")

    with h5py.File(temp_file, "w") as h5f:
        h5f.create_group("metadata")

    analyzer = VariantQC(
        input_file=temp_file,
        output_file=os.path.join(output_dir, "maf_temp_output.h5"),
        analysis_type="maf",
    )
    result = analyzer.run()

    assert result is None
    os.unlink(temp_file)


@pytest.mark.integration
def test_no_variants_kept(output_dir):
    temp_file = os.path.join(output_dir, "test_no_kept.h5")

    with h5py.File(temp_file, "w") as h5f:
        chr_grp = h5f.create_group("chr1")
        data = np.zeros((10, 10), dtype=float)
        chr_grp.create_dataset("Genotype", data=data)
        chr_grp.create_dataset(
            "RSID", data=np.array([f"rs{i}" for i in range(10)], dtype="S")
        )

        meta_grp = h5f.create_group("metadata")
        meta_grp.create_dataset(
            "iid", data=np.array([f"s{i}" for i in range(10)], dtype="S")
        )

    output_file = os.path.join(output_dir, "no_kept_output.h5")
    analyzer = VariantQC(
        input_file=temp_file,
        analysis_type="maf_filter",
        output_file=output_file,
        threshold=0.01,
    )
    result = analyzer.run()

    assert result is None

    os.unlink(temp_file)


@pytest.mark.integration
def test_adjust_workers_large_file(test_data_files, monkeypatch, output_dir):
    input_file = test_data_files["basic"]

    def mock_getsize(filename):
        return 60 * 1024**3

    monkeypatch.setattr(os.path, "getsize", mock_getsize)
    output_file = os.path.join(output_dir, "adjust_workers_output.h5")
    analyzer = VariantQC(
        input_file=input_file, analysis_type="maf", output_file=output_file
    )
    assert analyzer.max_workers <= 6


@pytest.mark.integration
def test_hwe_calculation_detailed(test_data_files, output_dir):
    temp_file = os.path.join(output_dir, "test_hwe_detailed.h5")

    with h5py.File(temp_file, "w") as h5f:
        chr_grp = h5f.create_group("chr1")

        n_samples = 100
        genotypes = np.array(
            [
                [0] * 36 + [1] * 48 + [2] * 16,
                [0] * 50 + [1] * 10 + [2] * 40,
            ],
            dtype=float,
        )

        chr_grp.create_dataset("Genotype", data=genotypes)
        chr_grp.create_dataset(
            "RSID", data=np.array(["rs_perfect", "rs_deviant"], dtype="S")
        )
        chr_grp.create_dataset("A1", data=np.array(["A", "A"], dtype="S"))
        chr_grp.create_dataset("A2", data=np.array(["G", "G"], dtype="S"))
        chr_grp.create_dataset("BP", data=[1000, 2000])

        meta_grp = h5f.create_group("metadata")
        meta_grp.create_dataset(
            "iid", data=np.array([f"s{i}" for i in range(n_samples)], dtype="S")
        )

    output_file = os.path.join(output_dir, "hwe_detailed_output.h5")
    analyzer = VariantQC(
        input_file=temp_file,
        analysis_type="hwe",
        output_file=output_file,
        threshold=1e-6,
    )
    result = analyzer.run()

    assert result is not None
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as h5f:
        chr_grp = h5f["chr1"]
        hwe_key = assert_dataset_exists(
            chr_grp, ["HWE", "hwe", "hardy_weinberg"], "HWE dataset not found"
        )
        hwe_values = chr_grp[hwe_key][:]

        assert hwe_values[0] > 0.05
        assert hwe_values[1] < 0.05

    os.unlink(temp_file)


@pytest.mark.integration
def test_ld_prune_window_handling(output_dir):
    temp_file = os.path.join(output_dir, "test_ld_window.h5")

    with h5py.File(temp_file, "w") as h5f:
        chr_grp = h5f.create_group("chr1")

        positions = [1000, 2000, 3000, 8000, 9000]
        genotypes = np.random.choice([0, 1, 2], size=(5, 20))

        genotypes[1] = genotypes[0]

        chr_grp.create_dataset("Genotype", data=genotypes.astype(float))
        chr_grp.create_dataset(
            "RSID", data=np.array([f"rs{i}" for i in range(5)], dtype="S")
        )
        chr_grp.create_dataset("BP", data=positions)
        chr_grp.create_dataset("A1", data=np.array(["A"] * 5, dtype="S"))
        chr_grp.create_dataset("A2", data=np.array(["G"] * 5, dtype="S"))

        meta_grp = h5f.create_group("metadata")
        meta_grp.create_dataset(
            "iid", data=np.array([f"s{i}" for i in range(20)], dtype="S")
        )

    output_file = os.path.join(output_dir, "ld_window_output.h5")
    analyzer = VariantQC(
        input_file=temp_file,
        analysis_type="ld_prune",
        output_file=output_file,
        window_size=5000,
        step_size=1000,
        r2_threshold=0.8,
        maf_threshold=0.0,
    )
    result = analyzer.run()

    assert result is not None
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as h5f:
        assert "filter_info" in h5f
        chr_grp = h5f["chr1"]
        rsid_key = find_dataset_in_group(chr_grp, ["RSID", "rsid", "snp"])
        if rsid_key:
            assert len(chr_grp[rsid_key]) < 5

    os.unlink(temp_file)


@pytest.mark.integration
def test_maf_filter_edge_cases(output_dir):
    temp_file = os.path.join(output_dir, "test_maf_edge.h5")

    with h5py.File(temp_file, "w") as h5f:
        chr_grp = h5f.create_group("chr1")

        genotypes = np.array(
            [
                [0] * 20,
                [0] * 19 + [1],
                [0] * 18 + [1] * 2,
            ],
            dtype=float,
        )

        chr_grp.create_dataset("Genotype", data=genotypes)
        chr_grp.create_dataset(
            "RSID", data=np.array(["rs_zero", "rs_threshold", "rs_above"], dtype="S")
        )
        chr_grp.create_dataset("A1", data=np.array(["A"] * 3, dtype="S"))
        chr_grp.create_dataset("A2", data=np.array(["G"] * 3, dtype="S"))
        chr_grp.create_dataset("BP", data=[1000, 2000, 3000])

        meta_grp = h5f.create_group("metadata")
        meta_grp.create_dataset(
            "iid", data=np.array([f"s{i}" for i in range(20)], dtype="S")
        )

    output_file = os.path.join(output_dir, "maf_edge_output.h5")
    analyzer = VariantQC(
        input_file=temp_file,
        analysis_type="maf_filter",
        output_file=output_file,
        threshold=0.01,
    )
    result = analyzer.run()

    assert result is not None
    assert os.path.exists(output_file)

    with h5py.File(output_file, "r") as h5f:
        chr_grp = h5f["chr1"]
        rsid_key = find_dataset_in_group(chr_grp, ["RSID", "rsid", "snp"])
        if rsid_key:
            remaining_rsids = [
                r.decode() if isinstance(r, bytes) else str(r)
                for r in chr_grp[rsid_key][:]
            ]
            assert "rs_above" in remaining_rsids
            assert "rs_zero" not in remaining_rsids

    os.unlink(temp_file)
