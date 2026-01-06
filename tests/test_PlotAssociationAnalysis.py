#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import pytest
from PlotAssociationAnalysis import PlotAssociationAnalysis
from utils.LoggingUtils import log

log.setup(level="DEBUG")


@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def ewas_data(data_dir):
    ewas_file = os.path.join(data_dir, "annotated_ewas_450k.csv")
    if not os.path.exists(ewas_file):
        pytest.skip(f"Test data file not found: {ewas_file}")
    return pd.read_csv(ewas_file)


@pytest.fixture
def gwas_data(data_dir):
    gwas_file = os.path.join(data_dir, "annotated_gwas.csv")
    if not os.path.exists(gwas_file):
        pytest.skip(f"Test data file not found: {gwas_file}")
    return pd.read_csv(gwas_file)


@pytest.fixture
def basic_plotter():
    return PlotAssociationAnalysis(
        input_file=None,
        output="test_plot",
        width=10,
        height=6,
        threshold=5e-8,
        var=None,
        n_annot=20,
        plot_type="manhattan",
    )


@pytest.mark.unit
def test_initialization():
    plotter = PlotAssociationAnalysis(input_file="example.csv", output="test_plot")
    assert plotter.input_file == "example.csv"
    assert plotter.output == "test_plot"

    plotter = PlotAssociationAnalysis(
        input_file="example.csv",
        output="test_plot",
        width=12,
        height=8,
        colors="red,blue,green",
        threshold=5e-7,
        var="Methylation",
        n_annot=15,
    )
    assert plotter.width == 12
    assert plotter.height == 8
    assert "red" in plotter.colors
    assert "blue" in plotter.colors
    assert "green" in plotter.colors
    assert plotter.threshold == 5e-7
    assert plotter.var == "Methylation"
    assert plotter.n_annot == 15

    plotter = PlotAssociationAnalysis(
        input_file="example.csv", output="test_plot", colors=["purple", "orange"]
    )
    assert "purple" in plotter.colors
    assert "orange" in plotter.colors


@pytest.mark.unit
def test_standardize_columns(ewas_data, basic_plotter):
    test_data = ewas_data.copy()
    test_data = test_data.rename(
        columns={
            "CGID": "Probe_ID",
            "CHR": "chromosome",
            "BP": "position",
            "Methylation_COEF": "Beta",
            "Methylation_P": "p.value",
        }
    )

    standardized = basic_plotter.standardize_columns(test_data)

    assert "CGID" in standardized.columns
    assert "CHR" in standardized.columns
    assert "BP" in standardized.columns

    assert basic_plotter.data_type == "EWAS"
    assert basic_plotter.id_col == "CGID"


@pytest.mark.unit
def test_determine_variable_columns(ewas_data, gwas_data, basic_plotter):
    basic_plotter.var = None
    basic_plotter.var_p = None
    basic_plotter.var_coef = None

    basic_plotter.determine_variable_columns(ewas_data)
    assert basic_plotter.var_p == "Methylation_P"
    assert basic_plotter.var_coef == "Methylation_COEF"

    gwas_plotter = PlotAssociationAnalysis(input_file=None, output="test_plot")

    gwas_plotter.determine_variable_columns(gwas_data)
    assert gwas_plotter.var_p == "Genotype_P"
    assert gwas_plotter.var_coef == "Genotype_COEF"


@pytest.mark.unit
def test_detect_analysis_type(ewas_data, gwas_data, basic_plotter):
    analysis_type = basic_plotter._detect_analysis_type(ewas_data)
    assert analysis_type == "EWAS"

    analysis_type = basic_plotter._detect_analysis_type(gwas_data)
    assert analysis_type == "GWAS"

    ambiguous_data = pd.DataFrame(
        {"CHR": [1, 2, 3], "BP": [100, 200, 300], "P": [0.05, 0.01, 0.001]}
    )
    analysis_type = basic_plotter._detect_analysis_type(ambiguous_data)
    assert analysis_type in ["EWAS", "GWAS"]


@pytest.mark.unit
def test_identify_p_value_column(ewas_data, gwas_data, basic_plotter):
    p_col = basic_plotter._identify_p_value_column(ewas_data)
    assert p_col == "Methylation_P"

    p_col = basic_plotter._identify_p_value_column(gwas_data)
    assert p_col == "Genotype_P"

    generic_data = ewas_data.copy()
    generic_data["P"] = generic_data["Methylation_P"]
    generic_data = generic_data.drop(columns=["Methylation_P"])
    p_col = basic_plotter._identify_p_value_column(generic_data)
    assert p_col == "P"


@pytest.mark.unit
def test_identify_effect_column(ewas_data, gwas_data, basic_plotter):
    effect_col = basic_plotter._identify_effect_column(ewas_data)
    assert effect_col == "Methylation_COEF"

    effect_col = basic_plotter._identify_effect_column(gwas_data)
    assert effect_col == "Genotype_COEF"

    generic_data = ewas_data.copy()
    generic_data["COEF"] = generic_data["Methylation_COEF"]
    generic_data = generic_data.drop(columns=["Methylation_COEF"])
    effect_col = basic_plotter._identify_effect_column(generic_data)
    assert effect_col == "COEF"


@pytest.mark.unit
def test_standardize_columns_edge_cases():
    bad_data = pd.DataFrame({"Col1": [1, 2, 3], "Col2": [4, 5, 6], "Col3": [7, 8, 9]})

    plotter = PlotAssociationAnalysis(input_file=None, output="test.png")

    result = plotter.standardize_columns(bad_data)
    assert result is None

    snp_data = pd.DataFrame(
        {"SNP": ["rs1", "rs2", "rs3"], "Col2": [4, 5, 6], "Col3": [7, 8, 9]}
    )

    result = plotter.standardize_columns(snp_data)
    assert result is not None
    assert plotter.id_col == "RSID"
    assert plotter.data_type == "GWAS"


@pytest.mark.integration
def test_manhattan_plot_ewas(ewas_data, output_dir, basic_plotter):
    output_file = os.path.join(output_dir, "ewas_manhattan.png")

    plotter = PlotAssociationAnalysis(
        input_file=None, output=output_file, width=10, height=6, annotate_genes=True
    )

    plotter.data = ewas_data
    plotter.determine_variable_columns(ewas_data)
    plotter.manhattan_plot()

    assert os.path.exists(output_file)

    assert os.path.getsize(output_file) > 1000


@pytest.mark.integration
def test_manhattan_plot_gwas(gwas_data, output_dir, basic_plotter):
    output_file = os.path.join(output_dir, "gwas_manhattan.png")

    plotter = PlotAssociationAnalysis(input_file=None, output=output_file)

    plotter.data = gwas_data
    plotter.manhattan_plot(
        p_col="Genotype_P",
        id_col="RSID",
        chr_col="CHR",
        pos_col="BP",
    )

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 1000


@pytest.mark.integration
def test_qq_plot_ewas(ewas_data, output_dir, basic_plotter):
    output_file = os.path.join(output_dir, "ewas_qq.png")

    plotter = PlotAssociationAnalysis(
        input_file=None, output=output_file, width=10, height=6, annotate_genes=True
    )

    plotter.data = ewas_data
    plotter.determine_variable_columns(ewas_data)
    plotter.qq_plot()

    assert os.path.exists(output_file)

    assert os.path.getsize(output_file) > 1000


@pytest.mark.integration
def test_qq_plot_gwas(gwas_data, output_dir, basic_plotter):
    output_file = os.path.join(output_dir, "gwas_qq.png")

    plotter = PlotAssociationAnalysis(input_file=None, output=output_file)

    plotter.data = gwas_data
    plotter.qq_plot(p_col="Genotype_P")

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 1000


@pytest.mark.integration
def test_volcano_plot_ewas(ewas_data, output_dir, basic_plotter):
    output_file = os.path.join(output_dir, "ewas_volcano.png")

    plotter = PlotAssociationAnalysis(
        input_file=None,
        output=output_file,
        width=10,
        height=6,
        annotate_genes=True,
        n_annot=5,
    )

    plotter.data = ewas_data
    plotter.determine_variable_columns(ewas_data)
    plotter.volcano_plot()

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 1000


@pytest.mark.integration
def test_volcano_plot_gwas(gwas_data, output_dir, basic_plotter):
    output_file = os.path.join(output_dir, "gwas_volcano.png")

    plotter = PlotAssociationAnalysis(input_file=None, output=output_file)

    plotter.data = gwas_data
    plotter.volcano_plot(
        p_col="Genotype_P",
        effect_col="Genotype_COEF",
        id_col="RSID",
    )

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 1000


@pytest.mark.integration
def test_region_plot_ewas(ewas_data, output_dir, basic_plotter):
    output_file = os.path.join(output_dir, "ewas_region.png")

    plotter = PlotAssociationAnalysis(input_file=None, output=output_file)

    top_cpg = ewas_data.sort_values("Methylation_P").iloc[0]["CGID"]

    plotter.data = ewas_data
    plotter.region_plot(
        id_col="CGID",
        p_col="Methylation_P",
        chr_col="CHR",
        pos_col="BP",
        gene_col="GENE",
        focus_id=top_cpg,
        window_size=500000,
    )

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 1000


@pytest.mark.integration
def test_region_plot_gwas(gwas_data, output_dir, basic_plotter):
    output_file = os.path.join(output_dir, "gwas_region.png")

    plotter = PlotAssociationAnalysis(input_file=None, output=output_file)

    top_snp = gwas_data.sort_values("Genotype_P").iloc[0]["RSID"]

    plotter.data = gwas_data
    plotter.region_plot(
        id_col="RSID",
        p_col="Genotype_P",
        chr_col="CHR",
        pos_col="BP",
        gene_col="GENE",
        focus_id=top_snp,
        window_size=500000,
    )

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 1000


@pytest.mark.integration
def test_with_cli_arguments():
    from argparse import Namespace

    args = Namespace(
        input="example.csv",
        output="test_output",
        width=12,
        height=8,
        var="Methylation",
        threshold=1e-8,
        plot_type="manhattan",
        n_annot=20,
    )

    plotter = PlotAssociationAnalysis.from_args(args)

    assert plotter.input_file == "example.csv"
    assert plotter.output == "test_output"
    assert plotter.width == 12
    assert plotter.height == 8
    assert plotter.var == "Methylation"
    assert plotter.threshold == 1e-8
    assert plotter.plot_type == "manhattan"


@pytest.mark.integration
def test_end_to_end_workflow(ewas_data, output_dir):
    output_prefix = os.path.join(output_dir, "end_to_end")

    plotter = PlotAssociationAnalysis(input_file=None, output=output_prefix)
    plotter.data = ewas_data
    plot_files = []
    manhattan_file = f"{output_prefix}_manhattan.png"
    plotter.output = manhattan_file
    plotter.manhattan_plot(
        p_col="Methylation_P",
        id_col="CGID",
        chr_col="CHR",
        pos_col="BP",
    )
    plot_files.append(manhattan_file)

    qq_file = f"{output_prefix}_qq.png"
    plotter.output = qq_file
    plotter.qq_plot(p_col="Methylation_P")
    plot_files.append(qq_file)

    volcano_file = f"{output_prefix}_volcano.png"
    plotter.output = volcano_file
    plotter.volcano_plot(
        p_col="Methylation_P",
        effect_col="Methylation_COEF",
        id_col="CGID",
    )
    plot_files.append(volcano_file)

    for plot_file in plot_files:
        assert os.path.exists(plot_file)
        assert os.path.getsize(plot_file) > 1000


@pytest.mark.integration
def test_handling_large_datasets(output_dir):
    n_samples = 1000000
    chrs = np.random.choice(range(1, 23), n_samples)
    positions = np.random.randint(1, 250000000, n_samples)
    p_values = np.random.random(n_samples) * 0.1

    sig_indices = np.random.choice(range(n_samples), 100, replace=False)
    p_values[sig_indices] = np.random.random(100) * 1e-8

    large_data = pd.DataFrame(
        {
            "CHR": chrs,
            "BP": positions,
            "P": p_values,
            "RSID": [f"rs{i}" for i in range(n_samples)],
        }
    )

    output_file = os.path.join(output_dir, "large_dataset.png")

    plotter = PlotAssociationAnalysis(
        input_file=None, output=output_file, downsample=True, max_points=10000
    )

    plotter.data = large_data
    plotter.manhattan_plot(p_col="P", id_col="RSID", chr_col="CHR", pos_col="BP")

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 1000


@pytest.mark.unit
def test_missing_required_columns(output_dir):
    incomplete_data = pd.DataFrame(
        {
            "RSID": ["rs1", "rs2", "rs3"],
            "BP": [100, 200, 300],
            "P": [0.05, 0.01, 0.001],
        }
    )

    output_file = os.path.join(output_dir, "missing_columns.png")

    plotter = PlotAssociationAnalysis(input_file=None, output=output_file)

    plotter.data = incomplete_data

    result = plotter.manhattan_plot(
        p_col="P", id_col="RSID", chr_col="CHR", pos_col="BP"
    )

    assert result is False


@pytest.mark.unit
def test_invalid_p_values(output_dir):
    invalid_data = pd.DataFrame(
        {
            "RSID": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "CHR": [1, 1, 2, 2, 3],
            "BP": [100, 200, 300, 400, 500],
            "P": [0.05, -0.01, 1.5, np.nan, 0.0],
        }
    )

    output_file = os.path.join(output_dir, "invalid_pvalues.png")

    plotter = PlotAssociationAnalysis(input_file=None, output=output_file)

    plotter.data = invalid_data

    result = plotter.qq_plot(p_col="P")

    assert result
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 1000


@pytest.mark.integration
def test_variable_specific_workflow(ewas_data, gwas_data, output_dir):
    ewas_output = os.path.join(output_dir, "ewas_methylation_manhattan.png")
    ewas_plotter = PlotAssociationAnalysis(
        input_file=None,
        output=ewas_output,
        var="Methylation",
        width=10,
        height=6,
    )
    ewas_plotter.data = ewas_data
    ewas_plotter.manhattan_plot()
    assert os.path.exists(ewas_output)
    assert os.path.getsize(ewas_output) > 1000

    gwas_output = os.path.join(output_dir, "gwas_sex_manhattan.png")
    gwas_plotter = PlotAssociationAnalysis(
        input_file=None,
        output=gwas_output,
        var="Sex",
        width=10,
        height=6,
        annotate_genes=True,
    )
    gwas_plotter.data = gwas_data
    gwas_plotter.manhattan_plot()
    assert os.path.exists(gwas_output)
    assert os.path.getsize(gwas_output) > 1000


@pytest.mark.integration
def test_chunk_processing(output_dir):
    large_file = os.path.join(output_dir, "large_test.csv")

    n_rows = 1000000
    np.random.seed(42)
    cgids = [f"cg{i:08d}" for i in range(n_rows)]
    chrs = np.random.choice(["CHR1", "CHR2", "CHR3", "CHR4", "CHR5"], n_rows)
    bps = np.random.randint(1, 250000000, n_rows)
    methylation_coefs = np.random.normal(0, 1, n_rows)
    methylation_ses = np.random.uniform(0.1, 0.5, n_rows)
    methylation_ps = np.random.uniform(0, 1, n_rows) ** 2

    large_data = pd.DataFrame(
        {
            "CGID": cgids,
            "CHR": chrs,
            "BP": bps,
            "Methylation_COEF": methylation_coefs,
            "Methylation_SE": methylation_ses,
            "Methylation_P": methylation_ps,
        }
    )

    significant_idx = np.random.choice(n_rows, 1000, replace=False)
    large_data.loc[significant_idx, "Methylation_P"] = np.random.uniform(0, 1e-8, 1000)

    sample_size = min(100000, n_rows)
    large_data_sample = large_data.sample(sample_size, random_state=42)
    large_data_sample.to_csv(large_file, index=False)

    output_plot = os.path.join(output_dir, "large_manhattan.png")
    plotter = PlotAssociationAnalysis(
        input_file=large_file,
        output=output_plot,
        var="Methylation",
        downsample=True,
        max_points=10000,
        n_annot=20,
    )

    orig_getsize = os.path.getsize
    try:
        os.path.getsize = lambda x: 2 * 1024 * 1024 * 1024

        assert plotter.read_data()
        assert plotter.manhattan_plot()

        assert os.path.exists(output_plot)
        assert os.path.getsize(output_plot) > 1000
    finally:
        os.path.getsize = orig_getsize


@pytest.mark.integration
def test_miami_plot_gwas(gwas_data, output_dir):
    output_file = os.path.join(output_dir, "gwas_miami.png")
    plotter = PlotAssociationAnalysis(
        input_file=None,
        output=output_file,
        plot_type="miami",
        annotate_genes=True,
        width=12,
        height=8,
        n_annot=10,
    )
    plotter.data = gwas_data

    plotter.id_col = "RSID"

    plotter.determine_variable_columns(gwas_data)
    result = plotter.create_plot()
    assert result is True
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 1000


@pytest.mark.integration
def test_miami_plot_large_dataset(output_dir):
    n_samples = 500000
    np.random.seed(42)

    chrs = np.random.choice(range(1, 23), n_samples)
    positions = np.random.randint(1, 250000000, n_samples)

    p_values = np.random.random(n_samples) ** 2

    coefficients = np.random.normal(0, 0.5, n_samples)

    sig_indices = np.random.choice(range(n_samples), 1000, replace=False)
    p_values[sig_indices] = np.random.random(1000) * 1e-8
    coefficients[sig_indices[:500]] = np.random.normal(2, 0.5, 500)
    coefficients[sig_indices[500:]] = np.random.normal(-2, 0.5, 500)

    genes = [f"GENE{i // 100}" for i in range(n_samples)]

    large_data = pd.DataFrame(
        {
            "CHR": chrs,
            "BP": positions,
            "Genotype_P": p_values,
            "Genotype_COEF": coefficients,
            "RSID": [f"rs{i}" for i in range(n_samples)],
            "GENE": genes,
        }
    )

    pos_count = (large_data["Genotype_COEF"] > 0).sum()
    neg_count = (large_data["Genotype_COEF"] < 0).sum()
    log.info(
        f"Mock data: {pos_count} positive coefficients, {neg_count} negative coefficients"
    )

    output_file = os.path.join(output_dir, "large_miami.png")

    plotter = PlotAssociationAnalysis(
        input_file=None,
        output=output_file,
        plot_type="miami",
        downsample=True,
        max_points=50000,
        annotate_genes=True,
        n_annot=5,
        colors="red,blue,green,purple,orange",
        width=14,
        height=10,
    )

    plotter.data = large_data
    plotter.id_col = "RSID"
    plotter.determine_variable_columns(large_data)
    result = plotter.create_plot()

    assert result is True
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 1000

    assert "red" in plotter.colors
    assert "blue" in plotter.colors
    assert len(plotter.colors) == 5


@pytest.mark.integration
def test_miami_plot_with_string_colors(output_dir):
    np.random.seed(123)
    n_samples = 10000

    test_data = pd.DataFrame(
        {
            "CHR": np.random.choice([1, 2, 3], n_samples),
            "BP": np.random.randint(1, 100000000, n_samples),
            "Genotype_P": np.random.random(n_samples) * 0.1,
            "Genotype_COEF": np.random.normal(0, 1, n_samples),
            "RSID": [f"rs{i}" for i in range(n_samples)],
            "GENE": [f"GENE{i // 1000}" for i in range(n_samples)],
        }
    )

    output_file = os.path.join(output_dir, "miami_string_colors.png")

    plotter = PlotAssociationAnalysis(
        input_file=None,
        output=output_file,
        plot_type="miami",
        colors="#FF5733,#33FF57,#3357FF,#FF33F1,#F1FF33",
        annotate_genes=True,
        n_annot=3,
    )

    plotter.data = test_data
    plotter.id_col = "RSID"
    plotter.determine_variable_columns(test_data)
    result = plotter.create_plot()

    assert result is True
    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 1000

    assert "#FF5733" in plotter.colors
    assert "#33FF57" in plotter.colors
    assert len(plotter.colors) == 5


@pytest.mark.integration
def test_with_all_cli_arguments():
    from argparse import Namespace

    args = Namespace(
        input="example.csv",
        output="test_output",
        width=12,
        height=8,
        var="Methylation",
        threshold=0.05,
        n_annot=20,
        plot_type="manhattan",
        downsample=True,
        max_points=50000,
        colors="red,blue,green",
        annotate_genes=True,
    )

    plotter = PlotAssociationAnalysis.from_args(args)

    assert plotter.input_file == "example.csv"
    assert plotter.output == "test_output"
    assert plotter.width == 12
    assert plotter.height == 8
    assert plotter.var == "Methylation"
    assert plotter.threshold == 0.05
    assert plotter.n_annot == 20
    assert plotter.plot_type == "manhattan"
    assert plotter.downsample
    assert plotter.max_points == 50000
    assert "red" in plotter.colors
    assert plotter.annotate_genes


if __name__ == "__main__":
    pytest.main()
