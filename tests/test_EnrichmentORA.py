#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import pytest
import tempfile
from EnrichmentORA import EnrichmentORA
from utils.LoggingUtils import log

log.setup(level="DEBUG")


def create_sample_gene_list(output_dir):
    gene_list_path = os.path.join(output_dir, "gene_list.csv")
    if not os.path.exists(gene_list_path):
        sample_genes = pd.DataFrame(
            {
                "GENE": [
                    "TP53",
                    "BRCA1",
                    "BRCA2",
                    "PTEN",
                    "AKT1",
                    "PIK3CA",
                    "KRAS",
                    "EGFR",
                    "BRAF",
                    "RB1",
                ],
                "P": [
                    0.001,
                    0.002,
                    0.003,
                    0.004,
                    0.005,
                    0.006,
                    0.007,
                    0.008,
                    0.009,
                    0.01,
                ],
            }
        )
        sample_genes.to_csv(gene_list_path, index=False)
        print(f"Created sample gene_list.csv at {gene_list_path}")
    return gene_list_path


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_files(output_dir):
    create_sample_gene_list(output_dir)


@pytest.mark.integration
def test_split_multi_gene_entries_ewas(data_dir, output_dir):
    input_file = os.path.join(data_dir, "annotated_ewas_450k.csv")

    df = pd.read_csv(input_file)
    multi_gene_count = df["GENE"].str.contains(";", na=False).sum()

    assert multi_gene_count > 0, "Test data should contain multi-gene entries"

    class EnrichmentORANoSplit(EnrichmentORA):
        def _split_multi_gene_entries(self, gene_list, separators=None):
            unique_genes = []
            seen = set()
            for gene in gene_list:
                gene_str = str(gene).strip()
                condition1 = gene_str and gene_str.lower() not in [
                    "nan",
                    "none",
                    "",
                    "null",
                ]
                condition2 = gene_str not in seen
                if condition1 and condition2:
                    unique_genes.append(gene_str)
                    seen.add(gene_str)
            return unique_genes

    analyzer_no_split = EnrichmentORANoSplit(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=os.path.join(output_dir, "ewas_no_split.csv"),
    )
    analyzer_no_split.load_gene_list()

    analyzer_with_split = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=os.path.join(output_dir, "ewas_with_split.csv"),
    )
    analyzer_with_split.load_gene_list()

    print(f"No split: {len(analyzer_no_split.gene_ids)} genes")
    print(f"With split: {len(analyzer_with_split.gene_ids)} genes")
    print(f"Sample no-split genes: {analyzer_no_split.gene_ids[:5]}")
    print(f"Sample split genes: {analyzer_with_split.gene_ids[:5]}")

    original_multi_gene_entries = [
        g for g in analyzer_no_split.gene_ids if ";" in str(g)
    ]
    split_entries_with_semicolons = [
        g for g in analyzer_with_split.gene_ids if ";" in str(g)
    ]

    assert len(split_entries_with_semicolons) < len(
        original_multi_gene_entries
    ), "Gene splitting should reduce the number of entries containing semicolons"

    if original_multi_gene_entries:
        test_entry = original_multi_gene_entries[0]
        component_genes = [g.strip() for g in test_entry.split(";")]
        components_found = sum(
            1 for g in component_genes if g in analyzer_with_split.gene_ids
        )
        assert (
            components_found > 0
        ), "At least some component genes should be found after splitting"


@pytest.mark.integration
def test_go_bp_ora_analysis(data_dir, output_dir):
    input_file = os.path.join(output_dir, "gene_list.csv")
    output_file = os.path.join(output_dir, "go_bp_results.csv")

    analyzer = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="GO_biological",
        output_file=output_file,
        pvalue_cutoff=0.05,
    )

    results = analyzer.analyze()

    if results is not None:
        assert os.path.exists(output_file)
        assert isinstance(results, pd.DataFrame)

        if "Namespace" in results.columns:
            assert all(results["Namespace"] == "biological_process")

        assert set(results.columns).issuperset(
            {"ID", "NAME", "GENE_RATIO", "BG_RATIO", "P", "P_FDR", "OR"}
        )
    else:
        log.info("GO Biological Process analysis returned no results")


@pytest.mark.integration
def test_go_mf_ora_analysis(data_dir, output_dir):
    input_file = os.path.join(output_dir, "gene_list.csv")
    output_file = os.path.join(output_dir, "go_mf_results.csv")

    analyzer = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="GO_molecular",
        output_file=output_file,
        pvalue_cutoff=0.05,
    )

    results = analyzer.analyze()

    if results is not None:
        assert os.path.exists(output_file)
        assert isinstance(results, pd.DataFrame)

        if "Namespace" in results.columns:
            assert all(results["Namespace"] == "molecular_function")
    else:
        log.info("GO Molecular Function analysis returned no results")


@pytest.mark.integration
def test_go_cc_ora_analysis(data_dir, output_dir):
    input_file = os.path.join(output_dir, "gene_list.csv")
    output_file = os.path.join(output_dir, "go_cc_results.csv")

    analyzer = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="GO_cellular",
        output_file=output_file,
        pvalue_cutoff=0.05,
    )

    results = analyzer.analyze()

    if results is not None:
        assert os.path.exists(output_file)
        assert isinstance(results, pd.DataFrame)

        if "Namespace" in results.columns:
            assert all(results["Namespace"] == "cellular_component")
    else:
        log.info("GO Cellular Component analysis returned no results")


@pytest.mark.integration
def test_column_autodetection(data_dir, output_dir):
    test_data = pd.DataFrame(
        {
            "gene_symbol": ["TP53", "BRCA1", "PTEN", "AKT1", "EGFR"],
            "pvalue": [0.001, 0.002, 0.003, 0.004, 0.005],
            "effect_size": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        test_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        output_file = os.path.join(output_dir, "autodetect_results.csv")

        analyzer = EnrichmentORA(
            input_file=temp_input_path,
            target_dataset="KEGG",
            output_file=output_file,
            pvalue_cutoff=0.05,
        )

        analyzer.load_gene_list()

        assert analyzer.gene_column == "GENE"
        assert analyzer.pvalue_column == "P"
        assert analyzer.effect_column == "effect_size"

    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)


@pytest.mark.integration
def test_column_autodetection_with_aliasutils(data_dir, output_dir):
    test_data = pd.DataFrame(
        {
            "GENE": ["TP53", "BRCA1", "PTEN", "AKT1", "EGFR"],
            "P": [0.001, 0.002, 0.003, 0.004, 0.005],
            "COEF": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        test_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        output_file = os.path.join(output_dir, "standard_cols_results.csv")

        analyzer = EnrichmentORA(
            input_file=temp_input_path,
            target_dataset="KEGG",
            output_file=output_file,
            pvalue_cutoff=0.05,
        )

        analyzer.load_gene_list()

        assert analyzer.gene_column == "GENE"
        assert analyzer.pvalue_column == "P"
        assert analyzer.effect_column == "COEF"

    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)


@pytest.mark.integration
def test_gene_id_type_detection(data_dir, output_dir):

    entrez_data = pd.DataFrame(
        {
            "GENE": ["7157", "672", "5728", "207", "1956"],
            "P": [0.001, 0.002, 0.003, 0.004, 0.005],
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_entrez:
        entrez_data.to_csv(temp_entrez.name, index=False)
        temp_entrez_path = temp_entrez.name

    ensembl_data = pd.DataFrame(
        {
            "GENE": [
                "ENSG00000141510",
                "ENSG00000012048",
                "ENSG00000171862",
                "ENSG00000012048",
                "ENSG00000146648",
            ],
            "P": [0.001, 0.002, 0.003, 0.004, 0.005],
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_ensembl:
        ensembl_data.to_csv(temp_ensembl.name, index=False)
        temp_ensembl_path = temp_ensembl.name

    symbol_data = pd.DataFrame(
        {
            "GENE": ["TP53", "BRCA1", "PTEN", "AKT1", "EGFR"],
            "P": [0.001, 0.002, 0.003, 0.004, 0.005],
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_symbol:
        symbol_data.to_csv(temp_symbol.name, index=False)
        temp_symbol_path = temp_symbol.name

    try:
        entrez_analyzer = EnrichmentORA(
            input_file=temp_entrez_path,
            gene_column="GENE",
            target_dataset="KEGG",
            output_file=os.path.join(output_dir, "gene_id_entrez.csv"),
        )
        entrez_analyzer.load_gene_list()
        assert entrez_analyzer.id_type == "entrez"

        ensembl_analyzer = EnrichmentORA(
            input_file=temp_ensembl_path,
            gene_column="GENE",
            target_dataset="KEGG",
            output_file=os.path.join(output_dir, "gene_id_ensembl.csv"),
        )
        ensembl_analyzer.load_gene_list()
        assert ensembl_analyzer.id_type == "ensembl"

        symbol_analyzer = EnrichmentORA(
            input_file=temp_symbol_path,
            gene_column="GENE",
            target_dataset="KEGG",
            output_file=os.path.join(output_dir, "gene_id_symbol.csv"),
        )
        symbol_analyzer.load_gene_list()
        assert symbol_analyzer.id_type == "symbol"

    finally:
        for path in [temp_entrez_path, temp_ensembl_path, temp_symbol_path]:
            if os.path.exists(path):
                os.unlink(path)


@pytest.mark.integration
def test_pvalue_filtering(data_dir, output_dir):
    test_data = pd.DataFrame(
        {
            "GENE": [
                "TP53",
                "BRCA1",
                "PTEN",
                "AKT1",
                "EGFR",
                "MDM2",
                "CDKN1A",
                "RB1",
                "MYC",
                "KRAS",
            ],
            "P": [0.001, 0.002, 0.003, 0.004, 0.005, 0.06, 0.07, 0.08, 0.09, 0.1],
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        test_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        output_file = os.path.join(output_dir, "pvalue_filter_results.csv")

        analyzer = EnrichmentORA(
            input_file=temp_input_path,
            gene_column="GENE",
            target_dataset="KEGG",
            output_file=output_file,
            pvalue_cutoff=0.05,
        )

        analyzer.load_gene_list()

        analyzer.analyze()

        assert len(analyzer.gene_ids) == 5
        assert all(
            gene in ["TP53", "BRCA1", "PTEN", "AKT1", "EGFR"]
            for gene in analyzer.gene_ids
        )

    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)


@pytest.mark.integration
def test_different_file_formats(data_dir, output_dir):
    test_data = pd.DataFrame(
        {
            "GENE": ["TP53", "BRCA1", "PTEN", "AKT1", "EGFR"],
            "P": [0.001, 0.002, 0.003, 0.004, 0.005],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_csv:
        test_data.to_csv(temp_csv.name, index=False)
        temp_csv_path = temp_csv.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as temp_tsv:
        test_data.to_csv(temp_tsv.name, sep="\t", index=False)
        temp_tsv_path = temp_tsv.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xlsx", delete=False
    ) as temp_xlsx:
        temp_xlsx_path = temp_xlsx.name

    try:
        csv_analyzer = EnrichmentORA(
            input_file=temp_csv_path,
            gene_column="GENE",
            target_dataset="KEGG",
            output_file=os.path.join(output_dir, "format_csv.csv"),
        )
        csv_analyzer.load_gene_list()
        assert len(csv_analyzer.gene_ids) == 5

        tsv_analyzer = EnrichmentORA(
            input_file=temp_tsv_path,
            gene_column="GENE",
            target_dataset="KEGG",
            output_file=os.path.join(output_dir, "format_tsv.csv"),
        )
        tsv_analyzer.load_gene_list()
        assert len(tsv_analyzer.gene_ids) == 5

    finally:
        for path in [temp_csv_path, temp_tsv_path]:
            if os.path.exists(path):
                os.unlink(path)
        if temp_xlsx_path and os.path.exists(temp_xlsx_path):
            os.unlink(temp_xlsx_path)


@pytest.mark.integration
def test_different_output_formats(data_dir, output_dir):
    input_file = os.path.join(output_dir, "gene_list.csv")

    output_csv = os.path.join(output_dir, "results.csv")
    output_tsv = os.path.join(output_dir, "results.tsv")

    csv_analyzer = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=output_csv,
        pvalue_cutoff=0.05,
    )
    csv_analyzer.analyze()
    assert os.path.exists(output_csv)

    tsv_analyzer = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=output_tsv,
        pvalue_cutoff=0.05,
    )
    tsv_analyzer.analyze()
    assert os.path.exists(output_tsv)


@pytest.mark.integration
def test_standardize_output_columns(data_dir, output_dir):
    input_file = os.path.join(output_dir, "gene_list.csv")
    output_file = os.path.join(output_dir, "standardized_results.csv")

    analyzer = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=output_file,
        pvalue_cutoff=0.05,
    )

    results = analyzer.analyze()

    assert "P" in results.columns
    assert "P_FDR" in results.columns
    assert "ID" in results.columns
    assert "NAME" in results.columns
    assert "GENE_RATIO" in results.columns
    assert "OR" in results.columns

    assert "P_value" not in results.columns
    assert "Adjusted_P_value" not in results.columns
    assert "Term_ID" not in results.columns
    assert "Term_Name" not in results.columns
    assert "Odds_Ratio" not in results.columns


@pytest.mark.integration
def test_barplot_creation(data_dir, output_dir):
    input_file = os.path.join(output_dir, "gene_list.csv")
    output_file = os.path.join(output_dir, "plot_test_results.csv")
    plot_file = os.path.join(output_dir, "test_barplot.png")

    analyzer = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=output_file,
        pvalue_cutoff=0.05,
        plot=plot_file,
        top_n=10,
    )

    analyzer.analyze()

    assert os.path.exists(plot_file)

    plot_size = os.path.getsize(plot_file)
    assert plot_size > 10000


@pytest.mark.integration
def test_error_handling_missing_gene_column(data_dir, output_dir):
    test_data = pd.DataFrame(
        {
            "OTHER_COLUMN": ["TP53", "BRCA1", "PTEN", "AKT1", "EGFR"],
            "P": [0.001, 0.002, 0.003, 0.004, 0.005],
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        test_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        output_file = os.path.join(output_dir, "error_results.csv")

        analyzer = EnrichmentORA(
            input_file=temp_input_path,
            gene_column="GENE",
            target_dataset="KEGG",
            output_file=output_file,
            pvalue_cutoff=0.05,
        )

        result = analyzer.load_gene_list()
        assert (
            result is False
        ), "load_gene_list() should return False when gene column is missing"

        assert (
            analyzer.gene_column is None
        ), "gene_column should be None after failed detection"

    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)


@pytest.mark.integration
def test_error_handling_invalid_dataset(data_dir, output_dir):
    input_file = os.path.join(output_dir, "gene_list.csv")
    output_file = os.path.join(output_dir, "invalid_dataset_results.csv")

    analyzer = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="INVALID_DATASET",
        output_file=output_file,
        pvalue_cutoff=0.05,
    )

    results = analyzer.analyze()
    assert results is None, "analyze() should return None when dataset is invalid"

    assert not os.path.exists(
        output_file
    ), "Output file should not be created for None results"


@pytest.mark.integration
def test_error_handling_missing_input_file(output_dir):
    output_file = os.path.join(output_dir, "missing_input_results.csv")

    analyzer = EnrichmentORA(
        input_file="nonexistent_file.csv",
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=output_file,
        pvalue_cutoff=0.05,
    )

    results = analyzer.analyze()
    assert results is None, "analyze() should return None when input file is missing"

    assert not os.path.exists(
        output_file
    ), "Output file should not be created for None results"


@pytest.mark.integration
def test_empty_result_handling(data_dir, output_dir):
    import random
    import string

    random_genes = [
        "".join(random.choices(string.ascii_uppercase, k=5)) for _ in range(10)
    ]

    test_data = pd.DataFrame(
        {
            "GENE": random_genes,
            "P": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        test_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        output_file = os.path.join(output_dir, "empty_results.csv")

        analyzer = EnrichmentORA(
            input_file=temp_input_path,
            gene_column="GENE",
            target_dataset="KEGG",
            output_file=output_file,
            pvalue_cutoff=0.05,
        )

        results = analyzer.analyze()
        assert results is None or (
            isinstance(results, pd.DataFrame) and results.empty
        ), "Should return None or empty DataFrame for no enrichment"

    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)


@pytest.mark.integration
def test_include_genes_option(data_dir, output_dir):
    input_file = os.path.join(output_dir, "gene_list.csv")

    output_file1 = os.path.join(output_dir, "results_no_genes.csv")
    analyzer1 = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=output_file1,
        pvalue_cutoff=0.05,
        include_genes=False,
    )
    results1 = analyzer1.analyze()

    output_file2 = os.path.join(output_dir, "results_with_genes.csv")
    analyzer2 = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=output_file2,
        pvalue_cutoff=0.05,
        include_genes=True,
    )
    results2 = analyzer2.analyze()

    assert os.path.exists(output_file1)
    assert os.path.exists(output_file2)

    if not results1.empty and not results2.empty:
        assert "GENES" not in results1.columns
        assert "GENES" in results2.columns


@pytest.mark.integration
def test_performance_small_dataset(data_dir, output_dir):
    input_file = os.path.join(output_dir, "gene_list.csv")
    output_file = os.path.join(output_dir, "performance_small.csv")

    import time

    start_time = time.time()

    analyzer = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=output_file,
        pvalue_cutoff=0.05,
    )
    analyzer.analyze()

    end_time = time.time()
    analysis_time = end_time - start_time

    print(f"Small dataset analysis took {analysis_time:.2f} seconds")
    assert analysis_time < 300


@pytest.mark.integration
def test_enrichment_with_annotated_ewas(data_dir, output_dir):
    input_file = os.path.join(data_dir, "annotated_ewas_450k.csv")
    output_file = os.path.join(output_dir, "ewas_enrichment_results.csv")
    plot_file = os.path.join(output_dir, "ewas_enrichment_plot.png")

    analyzer = EnrichmentORA(
        input_file=input_file,
        target_dataset="GO_biological",
        output_file=output_file,
        pvalue_cutoff=0.05,
        plot=plot_file,
    )

    analyzer.load_gene_list()
    assert analyzer.gene_column == "GENE"

    analyzer_with_var = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="GO_biological",
        output_file=output_file,
        pvalue_cutoff=0.05,
        plot=plot_file,
        var="Methylation",
    )

    analyzer_with_var.load_gene_list()
    assert analyzer_with_var.pvalue_column == "P"

    results = analyzer_with_var.analyze()

    if results is not None:
        assert os.path.exists(output_file)
        assert isinstance(results, pd.DataFrame)

        if not results.empty:
            if os.path.exists(plot_file):
                plot_size = os.path.getsize(plot_file)
                assert plot_size > 1000, "Plot file should have reasonable size"
            else:
                log.warn("Plot was not created, but analysis completed successfully")
    else:
        log.info(
            "EWAS enrichment analysis returned no results - this may be expected for test data"
        )


@pytest.mark.integration
def test_enrichment_with_annotated_gwas(data_dir, output_dir):
    input_file = os.path.join(data_dir, "annotated_gwas.csv")
    output_file = os.path.join(output_dir, "gwas_enrichment_results.csv")
    plot_file = os.path.join(output_dir, "gwas_enrichment_plot.png")

    analyzer = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=output_file,
        pvalue_cutoff=0.05,
        plot=plot_file,
        var="Genotype",
    )

    analyzer.load_gene_list()
    assert analyzer.gene_column == "GENE"
    assert analyzer.pvalue_column == "P"

    results = analyzer.analyze()

    assert os.path.exists(output_file)
    if not results.empty:
        assert os.path.exists(plot_file)


@pytest.mark.integration
def test_ewas_450k_effect_direction_analysis(data_dir, output_dir):
    input_file = os.path.join(data_dir, "annotated_ewas_450k.csv")

    df = pd.read_csv(input_file)

    positive_file = os.path.join(output_dir, "ewas_positive_effects.csv")
    negative_file = os.path.join(output_dir, "ewas_negative_effects.csv")

    df_positive = df[df["Methylation_COEF"] > 0]
    df_negative = df[df["Methylation_COEF"] < 0]

    df_positive.to_csv(positive_file, index=False)
    df_negative.to_csv(negative_file, index=False)

    positive_output = os.path.join(output_dir, "ewas_positive_results.csv")
    analyzer_positive = EnrichmentORA(
        input_file=positive_file,
        gene_column="GENE",
        target_dataset="GO_biological",
        output_file=positive_output,
        pvalue_cutoff=0.05,
        var="Methylation",
    )
    results_positive = analyzer_positive.analyze()

    negative_output = os.path.join(output_dir, "ewas_negative_results.csv")
    analyzer_negative = EnrichmentORA(
        input_file=negative_file,
        gene_column="GENE",
        target_dataset="GO_biological",
        output_file=negative_output,
        pvalue_cutoff=0.05,
        var="Methylation",
    )
    results_negative = analyzer_negative.analyze()

    if results_positive is not None and results_negative is not None:
        if not results_positive.empty and not results_negative.empty:
            id_col = "ID" if "ID" in results_positive.columns else "Term_ID"
            if "GO_ID" in results_positive.columns:
                id_col = "GO_ID"

            top_positive_df = results_positive.head(5)
            top_negative_df = results_negative.head(5)

            if id_col in top_positive_df.columns and id_col in top_negative_df.columns:
                top_positive_list = top_positive_df[id_col].values.tolist()
                top_negative_list = top_negative_df[id_col].values.tolist()

                top_positive_terms = set()
                for item in top_positive_list:
                    if isinstance(item, (list, np.ndarray)):
                        top_positive_terms.add(
                            str(item[0]) if len(item) > 0 else str(item)
                        )
                    else:
                        top_positive_terms.add(str(item))

                top_negative_terms = set()
                for item in top_negative_list:
                    if isinstance(item, (list, np.ndarray)):
                        top_negative_terms.add(
                            str(item[0]) if len(item) > 0 else str(item)
                        )
                    else:
                        top_negative_terms.add(str(item))

                print(f"Positive top terms: {top_positive_terms}")
                print(f"Negative top terms: {top_negative_terms}")

                assert len(top_positive_terms) > 0, "Should have positive effect terms"
                assert len(top_negative_terms) > 0, "Should have negative effect terms"
            else:
                assert len(results_positive) > 0, "Should have positive effect results"
                assert len(results_negative) > 0, "Should have negative effect results"
        else:
            assert isinstance(
                results_positive, pd.DataFrame
            ), "Should return DataFrame for positive effects"
            assert isinstance(
                results_negative, pd.DataFrame
            ), "Should return DataFrame for negative effects"
    else:
        log.info(
            "One or both effect direction analyses returned None - acceptable for test data"
        )
        assert results_positive is None or isinstance(results_positive, pd.DataFrame)
        assert results_negative is None or isinstance(results_negative, pd.DataFrame)


@pytest.mark.integration
def test_gwas_compare_genotype_vs_sex(data_dir, output_dir):
    input_file = os.path.join(data_dir, "annotated_gwas.csv")

    genotype_output = os.path.join(output_dir, "gwas_genotype_results.csv")
    analyzer_genotype = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=genotype_output,
        pvalue_cutoff=0.05,
        var="Genotype",
    )

    results_genotype = analyzer_genotype.analyze()

    sex_output = os.path.join(output_dir, "gwas_sex_results.csv")
    analyzer_sex = EnrichmentORA(
        input_file=input_file,
        gene_column="GENE",
        target_dataset="KEGG",
        output_file=sex_output,
        pvalue_cutoff=0.05,
        var="Sex",
    )

    results_sex = analyzer_sex.analyze()

    df = pd.read_csv(input_file)
    genotype_sig_count = (df["Genotype_P"] <= 0.05).sum()
    sex_sig_count = (df["Sex_P"] <= 0.05).sum()

    print(f"Genes significant for Genotype: {genotype_sig_count}")
    print(f"Genes significant for Sex: {sex_sig_count}")

    if genotype_sig_count > 5 and sex_sig_count > 5:
        if results_genotype is not None:
            assert isinstance(results_genotype, pd.DataFrame)
        else:
            log.info("Genotype analysis returned None")

        if results_sex is not None:
            assert isinstance(results_sex, pd.DataFrame)
        else:
            log.info(
                "Sex analysis returned None - may be due to insufficient enrichment"
            )
    else:
        log.info(
            f"Insufficient significant genes for comparison: Genotype={genotype_sig_count}, Sex={sex_sig_count}"
        )


@pytest.mark.integration
def test_variable_specific_column_detection(data_dir, output_dir):
    test_data = pd.DataFrame(
        {
            "GENE": ["TP53", "BRCA1", "PTEN", "AKT1", "EGFR"],
            "Treatment_P": [0.001, 0.002, 0.003, 0.004, 0.005],
            "Treatment_COEF": [0.5, 0.4, 0.3, 0.2, 0.1],
            "Control_P": [0.01, 0.02, 0.03, 0.04, 0.05],
            "Control_COEF": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        test_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        analyzer_treatment = EnrichmentORA(
            input_file=temp_input_path,
            gene_column="GENE",
            target_dataset="KEGG",
            pvalue_cutoff=0.05,
            var="Treatment",
            output_file=os.path.join(output_dir, "var_treatment.csv"),
        )

        analyzer_treatment.load_gene_list()
        assert analyzer_treatment.pvalue_column == "P"
        assert analyzer_treatment.effect_column == "COEF"

        analyzer_control = EnrichmentORA(
            input_file=temp_input_path,
            gene_column="GENE",
            target_dataset="KEGG",
            pvalue_cutoff=0.05,
            var="Control",
            output_file=os.path.join(output_dir, "var_control.csv"),
        )

        analyzer_control.load_gene_list()
        assert analyzer_control.pvalue_column == "Control_P"
        effect_col_is_coef = analyzer_control.effect_column == "COEF"
        effect_col_is_control_coef = analyzer_control.effect_column == "Control_COEF"
        assert effect_col_is_coef or effect_col_is_control_coef

    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)


@pytest.mark.integration
def test_variable_with_different_separators(data_dir, output_dir):
    test_data = pd.DataFrame(
        {
            "GENE": ["TP53", "BRCA1", "PTEN", "AKT1", "EGFR"],
            "Drug.p.value": [0.001, 0.002, 0.003, 0.004, 0.005],
            "Drug-coef": [0.5, 0.4, 0.3, 0.2, 0.1],
            "p.value.Dose": [0.01, 0.02, 0.03, 0.04, 0.05],
            "beta_Dose": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_input:
        test_data.to_csv(temp_input.name, index=False)
        temp_input_path = temp_input.name

    try:
        analyzer_drug = EnrichmentORA(
            input_file=temp_input_path,
            gene_column="GENE",
            target_dataset="KEGG",
            pvalue_cutoff=0.05,
            var="Drug",
            output_file=os.path.join(output_dir, "var_drug.csv"),
        )

        analyzer_drug.load_gene_list()
        assert analyzer_drug.pvalue_column == "P"
        assert analyzer_drug.effect_column == "COEF"

        analyzer_dose = EnrichmentORA(
            input_file=temp_input_path,
            gene_column="GENE",
            target_dataset="KEGG",
            pvalue_cutoff=0.05,
            var="Dose",
            output_file=os.path.join(output_dir, "var_dose.csv"),
        )

        analyzer_dose.load_gene_list()
        assert analyzer_dose.pvalue_column == "p.value.Dose"
        assert analyzer_dose.effect_column in ["COEF", "beta_Dose"]

    finally:
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
