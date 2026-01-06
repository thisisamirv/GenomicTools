#!/usr/bin/env python
import os
import pandas as pd
import pytest
import tempfile
from Annotator import Annotator
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.mark.integration
def test_annotate_ewas_methylseq(data_dir, output_dir):
    ewas_input = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    output_file = os.path.join(output_dir, "annotated_ewas.csv")
    annotator = Annotator(
        input_file=ewas_input,
        output=output_file,
        chip="MethylSeq",
        genome_version="hg38",
        analysis_type="EWAS",
    )
    annotator.annotate()
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert set(df.columns).issuperset(
        {
            "CGID",
            "CHR",
            "CPG_REGION",
            "GENE",
            "GENE_ID",
            "TSS",
            "TSS_DIST",
            "NEAREST_GENE_DIST",
            "STRAND",
            "BIOTYPE",
        }
    )


@pytest.mark.integration
def test_annotate_ewas_450k(data_dir, output_dir):
    ewas_input = os.path.join(data_dir, "ewas_450k_linear.csv")

    output_file = os.path.join(output_dir, "annotated_ewas_450k.csv")
    annotator = Annotator(
        input_file=ewas_input,
        output=output_file,
        chip="450k",
        genome_version="hg38",
        analysis_type="EWAS",
    )
    annotator.annotate()
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert "CGID" in df.columns
    assert "CHR" in df.columns
    assert "BP" in df.columns


@pytest.mark.integration
def test_annotate_ewas_epic(data_dir, output_dir):
    ewas_input = os.path.join(data_dir, "ewas_450k_linear.csv")

    output_file = os.path.join(output_dir, "annotated_ewas_epic.csv")
    annotator = Annotator(
        input_file=ewas_input,
        output=output_file,
        chip="EPIC",
        genome_version="hg38",
        analysis_type="EWAS",
    )
    annotator.annotate()
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert "CGID" in df.columns
    assert "CHR" in df.columns
    assert "BP" in df.columns


@pytest.mark.integration
def test_annotate_gwas_auto_detection(data_dir, output_dir):
    gwas_input = os.path.join(data_dir, "gwas_linear.csv")

    output_file = os.path.join(output_dir, "annotated_gwas_auto.csv")
    annotator = Annotator(
        input_file=gwas_input,
        output=output_file,
        genome_version="hg38",
        analysis_type="auto",
    )
    annotator.annotate()
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)

    assert set(df.columns).issuperset(
        {"RSID", "CHR", "BP", "GENE", "GENE_ID", "TSS", "TSS_DIST", "NEAREST_GENE_DIST"}
    )

    assert set(df.columns).issuperset({"REGULATORY_REGION", "REG_GENE", "REG_DISTANCE"})

    assert "CPG_REGION" not in df.columns
    assert "BIOTYPE" not in df.columns

    assert annotator.chip == "Genotype"


@pytest.mark.integration
def test_annotate_gwas_genotype(data_dir, output_dir):
    gwas_input = os.path.join(data_dir, "gwas_linear.csv")

    output_file = os.path.join(output_dir, "annotated_gwas_genotype.csv")
    annotator = Annotator(
        input_file=gwas_input,
        output=output_file,
        chip="Genotype",
        genome_version="hg38",
        analysis_type="GWAS",
    )
    annotator.annotate()
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)

    assert set(df.columns).issuperset(
        {"RSID", "CHR", "BP", "GENE", "GENE_ID", "TSS", "TSS_DIST", "NEAREST_GENE_DIST"}
    )

    assert set(df.columns).issuperset({"REGULATORY_REGION", "REG_GENE", "REG_DISTANCE"})

    assert "CPG_REGION" not in df.columns
    assert "BIOTYPE" not in df.columns

    assert annotator.chip == "Genotype"


@pytest.mark.integration
def test_gwas_regulatory_annotation_functionality(data_dir, output_dir):
    gwas_input = os.path.join(data_dir, "gwas_linear.csv")

    output_file = os.path.join(output_dir, "annotated_gwas_regulatory.csv")
    annotator = Annotator(
        input_file=gwas_input,
        output=output_file,
        chip="Genotype",
        genome_version="hg38",
        analysis_type="GWAS",
    )
    annotator.annotate()

    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)

    assert "REGULATORY_REGION" in df.columns
    assert "REG_GENE" in df.columns
    assert "REG_DISTANCE" in df.columns

    regulatory_annotated = df["REGULATORY_REGION"].notna().sum()

    if len(df) > 0:
        print(
            f"Total variants: {len(df)}, Regulatory annotated: {regulatory_annotated}"
        )

        reg_types = df["REGULATORY_REGION"].dropna().unique()
        expected_types = ["promoter", "enhancer", "gene_body"]

        if len(reg_types) > 0:
            assert any(
                reg_type in expected_types for reg_type in reg_types
            ), f"Expected regulatory types {expected_types}, but got {reg_types}"


@pytest.mark.integration
def test_gwas_gene_distance_calculation(data_dir, output_dir):
    gwas_input = os.path.join(data_dir, "gwas_linear.csv")

    output_file = os.path.join(output_dir, "annotated_gwas_distances.csv")
    annotator = Annotator(
        input_file=gwas_input,
        output=output_file,
        chip="Genotype",
        genome_version="hg38",
        analysis_type="GWAS",
    )
    annotator.annotate()

    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)

    assert "TSS_DIST" in df.columns
    assert "NEAREST_GENE_DIST" in df.columns

    tss_distances = df["TSS_DIST"].dropna()
    if len(tss_distances) > 0:
        assert all(tss_distances >= 0), "TSS distances should be non-negative"

    nearest_distances = df["NEAREST_GENE_DIST"].dropna()
    if len(nearest_distances) > 0:
        assert all(
            nearest_distances >= 0
        ), "Nearest gene distances should be non-negative"


@pytest.mark.integration
def test_gwas_protein_coding_filter(data_dir, output_dir):
    gwas_input = os.path.join(data_dir, "gwas_linear.csv")

    output_file1 = os.path.join(output_dir, "annotated_gwas_protein_only.csv")
    annotator1 = Annotator(
        input_file=gwas_input,
        output=output_file1,
        chip="Genotype",
        genome_version="hg38",
        analysis_type="GWAS",
        protein_coding=True,
    )
    annotator1.annotate()

    output_file2 = os.path.join(output_dir, "annotated_gwas_all_genes.csv")
    annotator2 = Annotator(
        input_file=gwas_input,
        output=output_file2,
        chip="Genotype",
        genome_version="hg38",
        analysis_type="GWAS",
        protein_coding=False,
    )
    annotator2.annotate()

    assert os.path.exists(output_file1)
    assert os.path.exists(output_file2)

    df1 = pd.read_csv(output_file1)
    df2 = pd.read_csv(output_file2)

    assert len(df1) == len(df2)

    genes1 = df1["GENE"].notna().sum()
    genes2 = df2["GENE"].notna().sum()

    assert genes2 >= genes1


@pytest.mark.integration
def test_gwas_column_ordering(data_dir, output_dir):
    gwas_input = os.path.join(data_dir, "gwas_linear.csv")

    output_file = os.path.join(output_dir, "annotated_gwas_columns.csv")
    annotator = Annotator(
        input_file=gwas_input,
        output=output_file,
        chip="Genotype",
        genome_version="hg38",
        analysis_type="GWAS",
    )
    annotator.annotate()

    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)

    columns = list(df.columns)

    assert "RSID" in columns
    assert "CHR" in columns
    assert "BP" in columns

    gene_cols = ["GENE", "GENE_ID", "TSS", "TSS_DIST", "NEAREST_GENE_DIST", "STRAND"]
    for col in gene_cols:
        assert col in columns

    reg_cols = ["REGULATORY_REGION", "REG_GENE", "REG_DISTANCE"]
    for col in reg_cols:
        assert col in columns

    rsid_idx = columns.index("RSID")
    gene_idx = columns.index("GENE") if "GENE" in columns else len(columns)
    assert rsid_idx < gene_idx


@pytest.mark.integration
def test_gwas_vs_ewas_annotation_differences(data_dir, output_dir):
    gwas_input = os.path.join(data_dir, "gwas_linear.csv")
    ewas_input = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    gwas_output = os.path.join(output_dir, "compare_gwas.csv")
    gwas_annotator = Annotator(
        input_file=gwas_input,
        output=gwas_output,
        chip="Genotype",
        genome_version="hg38",
        analysis_type="GWAS",
    )
    gwas_annotator.annotate()

    ewas_output = os.path.join(output_dir, "compare_ewas.csv")
    ewas_annotator = Annotator(
        input_file=ewas_input,
        output=ewas_output,
        chip="MethylSeq",
        genome_version="hg38",
        analysis_type="EWAS",
    )
    ewas_annotator.annotate()

    assert os.path.exists(gwas_output)
    assert os.path.exists(ewas_output)

    gwas_df = pd.read_csv(gwas_output)
    ewas_df = pd.read_csv(ewas_output)

    assert "RSID" in gwas_df.columns
    assert "CGID" in ewas_df.columns
    assert "RSID" not in ewas_df.columns
    assert "CGID" not in gwas_df.columns

    assert "REGULATORY_REGION" in gwas_df.columns
    assert "CPG_REGION" in ewas_df.columns
    assert "REGULATORY_REGION" not in ewas_df.columns
    assert "CPG_REGION" not in gwas_df.columns

    common_gene_cols = ["GENE", "GENE_ID", "TSS", "TSS_DIST", "NEAREST_GENE_DIST"]
    for col in common_gene_cols:
        assert col in gwas_df.columns
        assert col in ewas_df.columns


@pytest.mark.integration
def test_gwas_error_handling_missing_columns(data_dir, output_dir):
    incomplete_data = pd.DataFrame(
        {
            "VARIANT_ID": ["var1", "var2", "var3"],
            "CHR": ["chr1", "chr1", "chr2"],
            "COEF": [0.1, 0.2, 0.3],
            "P": [0.01, 0.02, 0.03],
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        incomplete_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        output_file = os.path.join(output_dir, "gwas_error_test.csv")
        annotator = Annotator(
            input_file=temp_input_path,
            output=output_file,
            chip="Genotype",
            genome_version="hg38",
            analysis_type="GWAS",
        )

        result = annotator.annotate()

        if result is not None and os.path.exists(output_file):
            df = pd.read_csv(output_file)
            assert len(df) == len(incomplete_data)

    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)


@pytest.mark.integration
def test_gwas_small_dataset_performance(data_dir, output_dir):
    gwas_input = os.path.join(data_dir, "gwas_linear.csv")

    original_data = pd.read_csv(gwas_input)
    small_data = original_data.head(50)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        small_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        output_file = os.path.join(output_dir, "annotated_gwas_small.csv")

        import time

        start_time = time.time()

        annotator = Annotator(
            input_file=temp_input_path,
            output=output_file,
            chip="Genotype",
            genome_version="hg38",
            analysis_type="GWAS",
        )
        annotator.annotate()

        end_time = time.time()
        annotation_time = end_time - start_time

        assert os.path.exists(output_file)
        df = pd.read_csv(output_file)

        assert len(df) == 50
        assert set(df.columns).issuperset(
            {"RSID", "CHR", "BP", "GENE", "REGULATORY_REGION"}
        )

        print(f"Annotation of 50 variants took {annotation_time:.2f} seconds")
        assert annotation_time < 300

    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)


@pytest.mark.integration
def test_annotate_hg19(data_dir, output_dir):
    ewas_input = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    output_file = os.path.join(output_dir, "annotated_ewas_hg19.csv")
    annotator = Annotator(
        input_file=ewas_input,
        output=output_file,
        chip="MethylSeq",
        genome_version="hg19",
        analysis_type="EWAS",
    )
    annotator.annotate()
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert set(df.columns).issuperset(
        {"CGID", "CHR", "GENE", "GENE_ID", "TSS", "TSS_DIST", "NEAREST_GENE_DIST"}
    )


@pytest.mark.integration
def test_annotate_protein_coding_false(data_dir, output_dir):
    ewas_input = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    output_file = os.path.join(output_dir, "annotated_ewas_all_genes.csv")
    annotator = Annotator(
        input_file=ewas_input,
        output=output_file,
        chip="MethylSeq",
        genome_version="hg38",
        analysis_type="EWAS",
        protein_coding=False,
    )
    annotator.annotate()
    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)
    assert set(df.columns).issuperset(
        {"CGID", "CHR", "GENE", "GENE_ID", "TSS", "TSS_DIST", "NEAREST_GENE_DIST"}
    )


@pytest.mark.integration
def test_annotate_with_custom_output_path(data_dir):
    ewas_input = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "custom_annotated_ewas.csv")
        annotator = Annotator(
            input_file=ewas_input,
            output=output_file,
            chip="MethylSeq",
            genome_version="hg38",
            analysis_type="EWAS",
        )
        annotator.annotate()
        assert os.path.exists(output_file)
        df = pd.read_csv(output_file)
        assert len(df) > 0


@pytest.mark.integration
def test_annotate_small_dataset(data_dir, output_dir):
    ewas_input = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    original_data = pd.read_csv(ewas_input)
    small_data = original_data.head(100)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        small_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        output_file = os.path.join(output_dir, "annotated_ewas_small.csv")
        annotator = Annotator(
            input_file=temp_input_path,
            output=output_file,
            chip="MethylSeq",
            genome_version="hg38",
            analysis_type="EWAS",
        )
        annotator.annotate()
        assert os.path.exists(output_file)
        df = pd.read_csv(output_file)
        assert len(df) == 100
        assert set(df.columns).issuperset(
            {"CGID", "CHR", "GENE", "GENE_ID", "TSS", "TSS_DIST", "NEAREST_GENE_DIST"}
        )
    finally:
        os.unlink(temp_input_path)


@pytest.mark.integration
def test_annotate_single_chromosome(data_dir, output_dir):
    ewas_input = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    original_data = pd.read_csv(ewas_input)
    single_chr_data = original_data[original_data["CHR"] == "CHR1"].head(50)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        single_chr_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        output_file = os.path.join(output_dir, "annotated_ewas_chr1.csv")
        annotator = Annotator(
            input_file=temp_input_path,
            output=output_file,
            chip="MethylSeq",
            genome_version="hg38",
            analysis_type="EWAS",
        )
        annotator.annotate()
        assert os.path.exists(output_file)
        df = pd.read_csv(output_file)
        assert len(df) == len(single_chr_data)
        assert (df["CHR"] == "CHR1").all()
        assert set(df.columns).issuperset(
            {"CGID", "CHR", "CPG_REGION", "GENE", "BIOTYPE"}
        )
    finally:
        os.unlink(temp_input_path)


@pytest.mark.integration
def test_error_handling_invalid_chip(data_dir, output_dir):
    ewas_input = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    output_file = os.path.join(output_dir, "annotated_ewas_invalid.csv")
    annotator = Annotator(
        input_file=ewas_input,
        output=output_file,
        chip="InvalidChip",
        genome_version="hg38",
        analysis_type="EWAS",
    )
    result = annotator.annotate()
    assert result is None
    assert not os.path.exists(output_file)

    valid_chips = ["450k", "EPIC", "MethylSeq", "Genotype"]
    for chip in valid_chips:
        test_annotator = Annotator(
            input_file=ewas_input,
            output=output_file,
            chip=chip,
            genome_version="hg38",
            analysis_type="auto",
        )
        assert chip in test_annotator.annotation_methods


@pytest.mark.integration
def test_error_handling_invalid_genome(data_dir, output_dir):
    ewas_input = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    output_file = os.path.join(output_dir, "annotated_ewas_invalid_genome.csv")
    annotator = Annotator(
        input_file=ewas_input,
        output=output_file,
        chip="MethylSeq",
        genome_version="hg20",
        analysis_type="EWAS",
    )
    result = annotator.annotate()
    assert result is None
    assert not os.path.exists(output_file)


@pytest.mark.integration
def test_error_handling_missing_input(output_dir):
    output_file = os.path.join(output_dir, "annotated_ewas_missing_input.csv")
    annotator = Annotator(
        input_file="nonexistent_file.csv",
        output=output_file,
        chip="MethylSeq",
        genome_version="hg38",
        analysis_type="EWAS",
    )
    result = annotator.annotate()
    assert result is None
    assert not os.path.exists(output_file)


@pytest.mark.integration
def test_annotation_consistency(data_dir, output_dir):
    ewas_input = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    output_file1 = os.path.join(output_dir, "annotated_ewas_consistency1.csv")
    output_file2 = os.path.join(output_dir, "annotated_ewas_consistency2.csv")

    annotator1 = Annotator(
        input_file=ewas_input,
        output=output_file1,
        chip="MethylSeq",
        genome_version="hg38",
        analysis_type="EWAS",
    )
    annotator1.annotate()

    annotator2 = Annotator(
        input_file=ewas_input,
        output=output_file2,
        chip="MethylSeq",
        genome_version="hg38",
        analysis_type="EWAS",
    )
    annotator2.annotate()

    assert os.path.exists(output_file1)
    assert os.path.exists(output_file2)

    df1 = pd.read_csv(output_file1)
    df2 = pd.read_csv(output_file2)

    assert df1.shape == df2.shape
    assert list(df1.columns) == list(df2.columns)

    key_cols = [
        "CGID",
        "CHR",
        "GENE",
        "GENE_ID",
        "TSS",
        "TSS_DIST",
        "NEAREST_GENE_DIST",
    ]
    for col in key_cols:
        if col in df1.columns:
            pd.testing.assert_series_equal(
                df1[col].fillna(""), df2[col].fillna(""), check_names=False
            )


@pytest.mark.integration
def test_memory_efficiency_large_dataset(data_dir, output_dir):
    ewas_input = os.path.join(data_dir, "ewas_mcseq_linear.csv")

    original_data = pd.read_csv(ewas_input)
    large_data = pd.concat([original_data] * 3, ignore_index=True)

    for i in range(len(large_data)):
        if i >= len(original_data):
            replica_num = i // len(original_data)
            large_data.iloc[i, large_data.columns.get_loc("CGID")] = (
                f"{large_data.iloc[i]['CGID']}_rep{replica_num}"
            )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        large_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        output_file = os.path.join(output_dir, "annotated_ewas_large.csv")
        annotator = Annotator(
            input_file=temp_input_path,
            output=output_file,
            chip="MethylSeq",
            genome_version="hg38",
            analysis_type="EWAS",
        )
        annotator.annotate()
        assert os.path.exists(output_file)
        df = pd.read_csv(output_file)

        print(f"Input rows: {len(large_data)}, Output rows: {len(df)}")
        assert len(df) == len(large_data)

        assert set(df.columns).issuperset(
            {"CGID", "CHR", "GENE", "GENE_ID", "TSS", "TSS_DIST", "NEAREST_GENE_DIST"}
        )
    finally:
        os.unlink(temp_input_path)


@pytest.mark.integration
def test_annotate_with_custom_reference_file(output_dir):
    mock_reference_data = pd.DataFrame(
        {
            "IlmnID": ["cg00000001", "cg00000002", "cg00000003"],
            "CHR": ["CHR1", "CHR1", "CHR2"],
            "MAPINFO": [1000, 2000, 3000],
            "STRAND": ["+", "-", "+"],
            "Gene": ["GENE1", "GENE2", "GENE3"],
            "CPG_REGION": ["Island", "Shore", "Open Sea"],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_ref:
        mock_reference_data.to_csv(temp_ref.name, index=False)
        temp_ref_path = temp_ref.name

    try:
        output_file = os.path.join(output_dir, "annotated_ewas_custom_ref.csv")

        test_data = pd.DataFrame(
            {
                "CGID": ["cg00000001", "cg00000002"],
                "CHR": ["CHR1", "CHR1"],
                "Methylation_COEF": [0.1, 0.2],
                "Methylation_P": [0.01, 0.02],
            }
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as temp_input:
            test_data.to_csv(temp_input.name, index=False)
            temp_input_path = temp_input.name

        try:
            annotator = Annotator(
                input_file=temp_input_path,
                output=output_file,
                chip="450k",
                genome_version="hg38",
                analysis_type="EWAS",
                reference=temp_ref_path,
            )
            annotator.annotate()

            assert os.path.exists(output_file)
            df = pd.read_csv(output_file)
            assert len(df) > 0
            assert "GENE" in df.columns or "CPG_REGION" in df.columns

        finally:
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)

    finally:
        if os.path.exists(temp_ref_path):
            os.unlink(temp_ref_path)


@pytest.mark.integration
def test_error_handling_invalid_reference(data_dir, output_dir):
    test_data = pd.DataFrame(
        {
            "CGID": ["chr1_1000", "chr1_2000", "chr2_3000"],
            "CHR": ["CHR1", "CHR1", "CHR2"],
            "Methylation_COEF": [0.1, 0.2, 0.3],
            "Methylation_P": [0.01, 0.02, 0.03],
        }
    )

    input_file = os.path.join(data_dir, "test_input_invalid_ref.csv")
    test_data.to_csv(input_file, index=False)

    try:
        output_file = os.path.join(output_dir, "annotated_ewas_invalid_ref.csv")
        annotator = Annotator(
            input_file=input_file,
            output=output_file,
            chip="450k",
            genome_version="hg38",
            analysis_type="EWAS",
            reference="nonexistent_reference.csv",
        )
        annotator.annotate()
        assert os.path.exists(output_file)

    finally:
        if os.path.exists(input_file):
            os.unlink(input_file)
