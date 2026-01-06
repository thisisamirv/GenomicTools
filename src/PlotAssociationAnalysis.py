#!/usr/bin/env python
# Import required modules
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy import stats
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Tuple, Union
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.LoggingUtils import log
from utils.ParsingUtils import ParseToList, ParseToKeyValueDict


class PlotAssociationAnalysis:
    def __init__(
        self,
        output: str,
        input_file: Optional[str] = None,
        width: int = 12,
        height: int = 8,
        colors: Optional[Union[str, List[str]]] = None,
        threshold: float = None,
        var: Optional[str] = None,
        n_annot: int = 20,
        plot_type: str = "manhattan",
        max_points: int = 100000,
        annotate_genes: bool = False,
        sample_sizes: Optional[Union[str, Dict[str, Any]]] = None,
        skip: Optional[str] = None,
    ) -> None:
        self.input_file = input_file
        self.output = output
        self.width = width
        self.height = height
        if colors is not None:
            self.colors = ParseToList(colors)
        else:
            self.colors = [
                "#4477AA",
                "#66CCEE",
                "#228833",
                "#CCBB44",
                "#EE6677",
                "#AA3377",
                "#BBBBBB",
                "#000000",
                "#44AA99",
                "#999933",
            ]
        self.threshold = threshold
        self.var = var
        self.n_annot = n_annot
        self.plot_type = plot_type
        self.max_points = max_points
        self.annotate_genes = annotate_genes
        self.plot_type = plot_type.lower()
        self.skip_range = skip
        self.sample_sizes = ParseToKeyValueDict(sample_sizes)
        for k, v in list(self.sample_sizes.items()):
            if v is None:
                self.sample_sizes[k] = None
                continue
            s = str(v).strip()
            if s == "" or s.lower() in ("none", "null"):
                self.sample_sizes[k] = None
                continue
            try:
                self.sample_sizes[k] = int(s)
            except Exception:
                self.sample_sizes[k] = None
        n = self.sample_sizes.get("N")
        n_cases = self.sample_sizes.get("N_cases")
        n_controls = self.sample_sizes.get("N_controls")
        n_effective: Optional[float] = None
        if n_cases is not None and n_controls is not None:
            try:
                if (n_cases + n_controls) > 0:
                    nominator = 4.0 * float(n_cases) * float(n_controls)
                    deominator = float(n_cases) + float(n_controls)
                    n_effective = float(nominator) / float(deominator)
                else:
                    n_effective = None
            except Exception:
                n_effective = None
        elif n is not None and (n_cases is None and n_controls is None):
            try:
                n_effective = float(n)
            except Exception:
                n_effective = None
        elif (n_cases is None or n_controls is None) and n is None:
            if self.plot_type in ("qq", "lambda", "calibration"):
                log.warn(
                    "Sample size information incomplete: one of N_cases/N_controls missing and N not provided."
                )
            n_effective = None
        elif (n_cases is None or n_controls is None) and n is not None:
            try:
                n_effective = float(n)
            except Exception:
                n_effective = None

        self.sample_sizes["N_effective"] = n_effective
        self.data = None
        self.data_type = None
        self.id_col = None
        self.var_p = None
        self.var_fdr = None
        self.var_holm = None
        self.var_coef = None
        self.fig = None
        self.MIN_P_VALUE = 1e-300

    @classmethod
    def from_args(cls, args: Any) -> "PlotAssociationAnalysis":
        return cls(
            input_file=args.input,
            output=args.output,
            width=args.width,
            height=args.height,
            threshold=args.threshold if hasattr(args, "threshold") else None,
            var=args.var,
            n_annot=args.n_annot,
            plot_type=args.plot_type,
            colors=args.colors if hasattr(args, "colors") else None,
            annotate_genes=(
                args.annotate_genes if hasattr(args, "annotate_genes") else False
            ),
            max_points=args.max_points if hasattr(args, "max_points") else 100000,
            sample_sizes=args.sample_sizes if hasattr(args, "sample_sizes") else None,
            skip=args.skip if hasattr(args, "skip") else None,
        )

    def _apply_jittering_for_ties(
        self, data: pd.DataFrame, p_col: str, id_col: str, max_ties: int = 50
    ) -> pd.DataFrame:
        if len(data) == 0:
            return data
        min_p = data[p_col].min()
        ties_mask = data[p_col] == min_p
        n_ties = ties_mask.sum()
        if n_ties > max_ties:
            log.info(
                f"Found {n_ties} p-values tied at minimum value {min_p}. Applying jittering."
            )
            data_copy = data.copy()
            if min_p > 0:
                jitter_factors = np.linspace(0.3, 0.95, n_ties)
                np.random.seed(42)
                np.random.shuffle(jitter_factors)
                tied_indices = data_copy.index[ties_mask]
                for i, idx in enumerate(tied_indices):
                    original_p = data_copy.loc[idx, p_col]
                    jittered_p = original_p * jitter_factors[i]
                    jittered_p = max(jittered_p, self.MIN_P_VALUE)
                    data_copy.loc[idx, p_col] = jittered_p
                jittered_values = data_copy.loc[tied_indices, p_col]
                min_jittered = jittered_values.min()
                max_jittered = jittered_values.max()
                log.info(
                    f"Applied jittering to {n_ties} tied p-values. "
                    f"Original: {min_p:.2e}, Jittered range: {min_jittered:.2e} to {max_jittered:.2e}"
                )
                log.info(
                    f"In -log10 scale - Original: {-np.log10(min_p):.2f}, "
                    f"Jittered range: {-np.log10(max_jittered):.2f} to {-np.log10(min_jittered):.2f}"
                )
            return data_copy
        return data

    def _detect_analysis_type(self, data: pd.DataFrame) -> str:
        ewas_indicators = ["CGID", "Methylation"]
        for indicator in ewas_indicators:
            if AliasUtils.find_keys(dict.fromkeys(data.columns), indicator):
                return "EWAS"
        gwas_indicators = ["RSID", "A1", "A2", "REF", "ALT", "MAF", "INFO"]
        for indicator in gwas_indicators:
            if AliasUtils.find_keys(dict.fromkeys(data.columns), indicator):
                return "GWAS"
        chr_col = AliasUtils.find_keys(dict.fromkeys(data.columns), "CHR")
        bp_col = AliasUtils.find_keys(dict.fromkeys(data.columns), "BP")
        if chr_col and bp_col:
            for potential_id_col in data.columns:
                if data[potential_id_col].dtype == "object":
                    sample_values = (
                        data[potential_id_col].dropna().astype(str).head(100)
                    )
                    rs_count = sample_values.str.contains("rs", case=False).sum()
                    if rs_count > len(sample_values) * 0.1:
                        return "GWAS"
                    cg_count = sample_values.str.contains("cg", case=False).sum()
                    if cg_count > len(sample_values) * 0.1:
                        return "EWAS"
        log.warn("Could not definitively determine analysis type. Defaulting to EWAS.")
        return "EWAS"

    def _get_gene_labels(self, df: pd.DataFrame, id_col: str) -> Dict[str, str]:
        gene_col = AliasUtils.find_keys(dict.fromkeys(df.columns), "GENE")
        if gene_col:
            log.info(f"Using gene column: {gene_col}")
            gene_labels: Dict[str, str] = {}
            for _, row in df.iterrows():
                marker_id = row[id_col]
                gene_value = row[gene_col]
                if pd.notna(gene_value) and gene_value:
                    first_gene = str(gene_value).split(";")[0]
                    gene_labels[marker_id] = first_gene
                else:
                    gene_labels[marker_id] = marker_id
            return gene_labels
        gene_id_col = AliasUtils.find_keys(dict.fromkeys(df.columns), "GENE_ID")
        if gene_id_col:
            log.info(f"Using gene ID column: {gene_id_col}")
            gene_labels: Dict[str, str] = {}
            for _, row in df.iterrows():
                marker_id = row[id_col]
                gene_id_value = row[gene_id_col]
                if pd.notna(gene_id_value) and gene_id_value:
                    first_gene_id = str(gene_id_value).split(";")[0]
                    gene_labels[marker_id] = first_gene_id
                else:
                    gene_labels[marker_id] = marker_id
            return gene_labels
        log.warn("No gene or gene ID column found. Using marker IDs for annotation.")
        return {str(id): str(id) for id in df[id_col].values}

    def _load_data(self) -> bool:
        if self.data is None and self.input_file:
            try:
                file_size = os.path.getsize(self.input_file) / (1024 * 1024)
                if file_size > 1000:
                    log.warn(
                        f"Large file detected ({file_size:.1f} MB). Using chunk processing."
                    )
                    reader = pd.read_csv(
                        self.input_file, sep=None, engine="python", chunksize=100000
                    )
                    chunks = []
                    first_chunk = next(reader)
                    first_chunk = self.standardize_columns(first_chunk)
                    if first_chunk is None:
                        return False
                    if not self.determine_variable_columns(first_chunk):
                        return False
                    n_chunks = int(file_size / 100)
                    sample_size = min(1.0, 10000000 / (n_chunks * 100000))
                    chunks.append(first_chunk.sample(frac=sample_size, random_state=42))
                    for chunk in tqdm(reader, total=n_chunks, desc="reading chunks", unit="chunk"):
                        chunks.append(chunk.sample(frac=sample_size, random_state=42))
                    self.data = pd.concat(chunks)
                    log.info(
                        f"Loaded and sampled {len(self.data)} rows from {self.input_file}"
                    )
                else:
                    log.info(f"Reading input data from {self.input_file}")
                    self.data = pd.read_csv(self.input_file, sep=None, engine="python")
                    log.info(f"Loaded {len(self.data)} rows from {self.input_file}")
                    self.data = self.standardize_columns(self.data)
                    if self.data is None:
                        return False
                    if not self.determine_variable_columns(self.data):
                        return False
            except Exception as e:
                log.error(f"Error loading data: {e}")
                return False
        return True if self.data is not None and not self.data.empty else False

    @staticmethod
    def safe_chr_sort(chr_val: Any) -> int:
        if pd.isna(chr_val):
            return 999
        try:
            chr_str = str(chr_val).upper()
            for prefix in ["CHR", "CHROMOSOME", "CH"]:
                if chr_str.startswith(prefix):
                    prefix_len = len(prefix)
                    chr_str = chr_str[prefix_len:]
                    break
            if chr_str in ["X", "CHRX"]:
                return 23
            elif chr_str in ["Y", "CHRY"]:
                return 24
            elif chr_str in ["M", "MT", "CHRM", "CHRMT"]:
                return 25
            elif chr_str.startswith("UN") or chr_str == "U":
                return 100
            return int(chr_str)
        except (ValueError, TypeError):
            return 200

    def _position_annotations_smartly(
        self,
        annotations: List[Dict[str, Any]],
        ax: Any,
        plot_bounds: Dict[str, float],
        min_distance: float = 0.5,
    ) -> List[Dict[str, Any]]:
        if not annotations:
            return annotations
        annotations = sorted(annotations, key=lambda a: a["x"])
        x_margin = 0.05 * (plot_bounds["xmax"] - plot_bounds["xmin"])
        y_margin = 0.1 * (plot_bounds["ymax"] - plot_bounds["ymin"])
        for i, annotation in enumerate(annotations):
            x, y, label = annotation["x"], annotation["y"], annotation["label"]
            text_width = (
                len(label) * 0.008 * (plot_bounds["xmax"] - plot_bounds["xmin"])
            )
            text_height = 0.03 * (plot_bounds["ymax"] - plot_bounds["ymin"])
            if x - text_width / 2 < plot_bounds["xmin"] + x_margin:
                annotation["x"] = plot_bounds["xmin"] + x_margin + text_width / 2
            elif x + text_width / 2 > plot_bounds["xmax"] - x_margin:
                annotation["x"] = plot_bounds["xmax"] - x_margin - text_width / 2
            if y > 0:
                max_y = plot_bounds["ymax"] - y_margin - text_height
                if y > max_y:
                    annotation["y"] = max_y
                for j in range(i):
                    other = annotations[j]
                    x_overlap = abs(annotation["x"] - other["x"]) < text_width
                    y_overlap = abs(annotation["y"] - other["y"]) < min_distance
                    if x_overlap and y_overlap:
                        annotation["y"] = other["y"] + min_distance
                        if annotation["y"] > max_y:
                            annotation["y"] = max_y
            else:
                min_y = plot_bounds["ymin"] + y_margin + text_height
                if y < min_y:
                    annotation["y"] = min_y
                for j in range(i):
                    other = annotations[j]
                    x_overlap = abs(annotation["x"] - other["x"]) < text_width
                    y_overlap = abs(annotation["y"] - other["y"]) < min_distance
                    if x_overlap and y_overlap:
                        annotation["y"] = other["y"] - min_distance
                        if annotation["y"] < min_y:
                            annotation["y"] = min_y
        return annotations

    def manhattan_plot(
        self,
        p_col: Optional[str] = None,
        id_col: Optional[str] = None,
        chr_col: Optional[str] = None,
        pos_col: Optional[str] = None,
    ) -> bool:
        if not self._load_data():
            return False
        if p_col is None:
            if not self.var_p:
                if not self.determine_variable_columns(self.data):
                    log.error("Could not identify p-value column")
                    return False
            p_col = self.var_p
        if id_col is None:
            for id_type in ["RSID", "SNP", "CGID", "PROBE_ID"]:
                id_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), id_type)
                if id_col:
                    break
            if not id_col:
                id_col = self.id_col if self.id_col else self.data.columns[0]
        if chr_col is None:
            chr_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), "CHR")
            if not chr_col:
                log.error("Could not identify chromosome column")
                return False
        if pos_col is None:
            pos_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), "BP")
            if not pos_col:
                log.error("Could not identify position column")
                return False
        required_cols = [p_col, chr_col, pos_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            msg = f"Missing required columns: {', '.join(missing_cols)}"
            log.error(msg)
            return False
        plot_data = self.data.copy()
        plot_data[chr_col] = plot_data[chr_col].astype(str)
        plot_data[chr_col] = (
            plot_data[chr_col]
            .str.replace("chr", "", case=False)
            .str.replace("CHR", "", case=False)
        )
        try:
            plot_data[chr_col] = pd.to_numeric(plot_data[chr_col], errors="coerce")
            plot_data = plot_data.dropna(subset=[chr_col])
            plot_data = plot_data.sort_values([chr_col, pos_col])
        except Exception:
            plot_data = plot_data.sort_values([chr_col, pos_col])
        plot_data[p_col] = pd.to_numeric(plot_data[p_col], errors="coerce")
        plot_data.loc[plot_data[p_col] == 0, p_col] = self.MIN_P_VALUE
        plot_data.loc[plot_data[p_col] < self.MIN_P_VALUE, p_col] = self.MIN_P_VALUE
        plot_data = plot_data[(plot_data[p_col] > 0) & (plot_data[p_col] <= 1)]
        plot_data = self._apply_jittering_for_ties(
            plot_data, p_col, id_col, max_ties=50
        )
        if len(plot_data) == 0:
            msg = "No valid data remaining after filtering"
            log.error(msg)
            return False
        plt.figure(figsize=(self.width, self.height))
        chromosomes = plot_data[chr_col].unique()
        clean_chromosomes = []
        for chrom in chromosomes:
            chr_str = str(chrom)
            for prefix in ["chr", "CHR", "chromosome", "Chromosome"]:
                if chr_str.lower().startswith(prefix.lower()):
                    prefix_len = len(prefix)
                    chr_str = chr_str[prefix_len:]
                    break
            clean_chromosomes.append(chr_str)
        chrom_mapping = dict(zip(chromosomes, clean_chromosomes))
        chromosomes = sorted(
            chromosomes, key=lambda x: self.safe_chr_sort(chrom_mapping[x])
        )
        chrom_data_list = []
        start_pos = 0
        for i, chrom in enumerate(tqdm(chromosomes, desc="prepare chroms", unit="chrom")):
            chrom_data = plot_data[plot_data[chr_col] == chrom]
            end_pos = start_pos + chrom_data[pos_col].max()
            chrom_data_list.append(
                {
                    "chrom": chrom,
                    "start": start_pos,
                    "end": end_pos,
                    "center": (start_pos + end_pos) / 2,
                }
            )
            start_pos = end_pos + 1000000
        chrom_df = pd.DataFrame(chrom_data_list)
        for i, chrom in enumerate(tqdm(chromosomes, desc="plot chroms", unit="chrom")):
            chrom_data = plot_data[plot_data[chr_col] == chrom]
            if len(chrom_data) == 0:
                continue
            chrom_pos = chrom_data[pos_col]
            chrom_start = chrom_df.loc[chrom_df["chrom"] == chrom, "start"].values[0]
            adj_pos = chrom_pos + chrom_start
            log_p = -np.log10(chrom_data[p_col])
            color = self.colors[i % len(self.colors)]
            plt.scatter(adj_pos, log_p, s=18, c=color, alpha=0.8, edgecolors="none")
        try:
            all_log_p = -np.log10(plot_data[p_col])
            finite_mask = np.isfinite(all_log_p)
            if finite_mask.any():
                max_logp = float(all_log_p[finite_mask].max())
                if np.isfinite(max_logp) and max_logp > 0:
                    plt.ylim(0, max_logp * 1.05)
        except Exception:
            pass
        if self.threshold is not None and self.threshold > 0:
            plt.axhline(
                y=-np.log10(self.threshold), color="red", linestyle="--", alpha=0.7
            )
            plt.text(
                plt.xlim()[1] * 0.99,
                -np.log10(self.threshold) + 0.1,
                f"p = {self.threshold}",
                ha="right",
                va="bottom",
                color="red",
                fontsize=12,
            )

        try:
            adj_map = dict(zip(chrom_df["chrom"], chrom_df["start"]))
            all_adj = plot_data[pos_col] + plot_data[chr_col].map(adj_map)
            min_adj = float(all_adj.min())
            max_adj = float(all_adj.max())
            if np.isfinite(min_adj) and np.isfinite(max_adj) and max_adj > min_adj:
                range_width = max_adj - min_adj
                left = min_adj
                right = max_adj + max(range_width * 0.005, 1e-6)
                plt.xlim(left, right)
        except Exception:
            pass
        if id_col and id_col in plot_data.columns:
            top_n = min(self.n_annot, len(plot_data))
            top_hits = plot_data.nsmallest(top_n, p_col)
            if self.annotate_genes:
                gene_labels = self._get_gene_labels(top_hits, id_col)
            xlim = plt.xlim()
            ylim = plt.ylim()
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            annotations = []
            for i, (_, hit) in enumerate(top_hits.iterrows()):
                chrom = hit[chr_col]
                if chrom not in chrom_df["chrom"].values:
                    continue
                hit_pos = hit[pos_col]
                chrom_start = chrom_df.loc[chrom_df["chrom"] == chrom, "start"].values[
                    0
                ]
                point_x = hit_pos + chrom_start
                point_y = -np.log10(hit[p_col])
                if self.annotate_genes:
                    label = gene_labels.get(hit[id_col], hit[id_col])
                else:
                    label = str(hit[id_col])
                x_center = (xlim[0] + xlim[1]) / 2
                if point_x < x_center:
                    offset_x = -0.027 * x_range - (i % 4) * 0.01 * x_range
                    offset_y = 0.017 * y_range + (i % 6) * 0.013 * y_range
                else:
                    offset_x = 0.027 * x_range + (i % 4) * 0.01 * x_range
                    offset_y = 0.017 * y_range + (i % 6) * 0.013 * y_range
                annotation_x = point_x + offset_x
                annotation_y = point_y + offset_y
                annotation_x = max(
                    xlim[0] + 0.02 * x_range,
                    min(xlim[1] - 0.02 * x_range, annotation_x),
                )
                annotation_y = max(
                    ylim[0] + 0.02 * y_range, min(ylim[1] + 0.1 * y_range, annotation_y)
                )
                annotations.append(
                    {
                        "x": annotation_x,
                        "y": annotation_y,
                        "label": label,
                        "original_x": point_x,
                        "original_y": point_y,
                        "hit_data": hit,
                    }
                )
            annotations = sorted(annotations, key=lambda a: a["y"], reverse=True)
            min_distance = 0.04 * y_range
            for i in range(1, len(annotations)):
                current = annotations[i]
                for j in range(i):
                    other = annotations[j]
                    x_dist = abs(current["x"] - other["x"])
                    y_dist = abs(current["y"] - other["y"])
                    if x_dist < 0.1 * x_range and y_dist < min_distance:
                        if current["y"] > other["y"]:
                            current["y"] = other["y"] + min_distance
                        else:
                            current["y"] = other["y"] - min_distance
            max_annotation_y = (
                max([a["y"] for a in annotations]) if annotations else ylim[1]
            )
            if max_annotation_y > ylim[1]:
                new_ylim = (ylim[0], max_annotation_y + 0.1 * y_range)
                plt.ylim(new_ylim)
            for annotation in annotations:
                plt.scatter(
                    annotation["original_x"],
                    annotation["original_y"],
                    s=40,
                    c="red",
                    alpha=1.0,
                    edgecolors="black",
                    zorder=10,
                )
                plt.plot(
                    [annotation["original_x"], annotation["x"]],
                    [annotation["original_y"], annotation["y"]],
                    "k-",
                    alpha=0.6,
                    linewidth=0.8,
                    zorder=9,
                )
                plt.text(
                    annotation["x"],
                    annotation["y"],
                    annotation["label"],
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=0,
                    zorder=11,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="gray",
                    ),
                )

        def safe_int_convert(x: Any) -> str:
            try:
                if float(x) == int(float(x)):
                    return str(int(float(x)))
                else:
                    return str(x)
            except (ValueError, OverflowError):
                return str(x)

        chrom_labels = [safe_int_convert(x) for x in chrom_df["chrom"]]
        plt.xticks(chrom_df["center"], chrom_labels, rotation=90, fontsize=10)
        plt.xlabel("Chromosome", fontsize=16, labelpad=12)
        plt.ylabel("-log10(p-value)", fontsize=16)
        plt.title("Manhattan Plot", fontsize=20)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.savefig(self.output, dpi=600)
        plt.close()
        log.info(f"Manhattan plot saved to {self.output}")
        return True

    def qq_plot(self, p_col: Optional[str] = None) -> bool:
        if not self._load_data():
            return False
        if p_col is None:
            if not self.var_p:
                if not self.determine_variable_columns(self.data):
                    log.error("Could not identify p-value column")
                    return False
            p_col = self.var_p
        if p_col not in self.data.columns:
            msg = f"Missing required column: {p_col}"
            log.error(msg)
            return False
        plot_data = self.data.copy()
        plot_data[p_col] = pd.to_numeric(plot_data[p_col], errors="coerce")
        valid_p = plot_data[(plot_data[p_col] > 0) & (plot_data[p_col] <= 1)][p_col]
        if len(valid_p) == 0:
            msg = f"No valid p-values found in column: {p_col}"
            log.error(msg)
            return False

        n_total = len(valid_p)
        sorted_p = np.sort(valid_p)
        cutoff_index = int(0.01 * n_total)
        null_p = sorted_p[cutoff_index:] if n_total > 100 else sorted_p
        n = len(null_p)
        if n == 0:
            log.error(
                "No p-values remain after excluding extremes; cannot compute QQ plot"
            )
            return False

        chi_medians = np.median(stats.chi2.ppf(1 - null_p, 1))
        lambda_raw = chi_medians / stats.chi2.ppf(0.5, 1)
        n_eff = self.sample_sizes.get("N_effective")
        if n_eff is not None and n_eff > 1000:
            log.info(f"Calculating adjusted λ using N_effective = {n_eff}")
            lambda_gc = 1.0 + (lambda_raw - 1.0) * (1000.0 / float(n_eff))
        else:
            lambda_gc = lambda_raw
        lambda_gc = round(lambda_gc, 2)

        plt.figure(figsize=(self.width, self.height))
        observed = -np.log10(null_p)
        expected_p = (np.arange(1, n + 1) - 0.5) / n
        expected = -np.log10(expected_p)
        log.info("QQ plot focusing on null distribution:")
        log.info(f"Using {n}/{n_total} p-values (excluding most extreme {n_total - n})")
        log.info(
            f"Observed -log10(p) range: {observed.min():.2f} to {observed.max():.2f}"
        )
        log.info(
            f"Expected -log10(p) range: {expected.min():.2f} to {expected.max():.2f}"
        )
        log.info(f"λ = {lambda_gc}")
        max_val = max(observed.max(), expected.max()) * 1.05
        plt.plot([0, max_val], [0, max_val], "r--", label="Expected", alpha=0.7)
        plt.scatter(expected, observed, s=30, alpha=0.8)
        info_text = f"λ = {lambda_gc}\n{n:,}/{n_total:,} SNPs\n(excluding top 1%)"
        plt.text(
            0.95,
            0.05,
            info_text,
            transform=plt.gca().transAxes,
            ha="right",
            va="bottom",
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )
        plt.xlabel("Expected -log10(p)", fontsize=18)
        plt.ylabel("Observed -log10(p)", fontsize=18)
        plt.title("QQ Plot (null distribution)", fontsize=24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output, dpi=600)
        plt.close()
        log.info(f"QQ plot saved to {self.output}")
        return True

    def calibration_plot(self, p_col: Optional[str] = None) -> bool:
        if not self._load_data():
            return False
        if p_col is None:
            if not self.var_p:
                if not self.determine_variable_columns(self.data):
                    log.error("Could not identify p-value column")
                    return False
            p_col = self.var_p
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        ax1.hist(
            self.data[p_col][self.data[p_col] >= 0.1],
            bins=50,
            alpha=0.7,
            density=True,
            color="skyblue",
        )
        ax1.axhline(y=1.0, color="red", linestyle="--", label="Expected (uniform)")
        ax1.set_xlabel("P-value (≥ 0.1)")
        ax1.set_ylabel("Density")
        ax1.set_title("Null P-value Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        thresholds = [1.0, 0.5, 0.3, 0.1, 0.05]
        lambdas = []
        for thresh in tqdm(thresholds, desc="calib λ thresholds", unit="thresh"):
            null_p = self.data[p_col][self.data[p_col] >= thresh]
            if len(null_p) > 100:
                chi_med = np.median(stats.chi2.ppf(1 - null_p, 1))
                lam_raw = chi_med / stats.chi2.ppf(0.5, 1)
                n_eff = self.sample_sizes.get("N_effective")
                if n_eff is not None and n_eff > 1000:
                    lam = 1.0 + (lam_raw - 1.0) * (1000.0 / float(n_eff))
                else:
                    lam = lam_raw
                lambdas.append(lam)
            else:
                lambdas.append(np.nan)
        valid_mask = ~np.isnan(lambdas)
        ax2.plot(np.array(thresholds)[valid_mask], np.array(lambdas)[valid_mask], "bo-")
        ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.7)
        ax2.set_xlabel("P-value threshold")
        ax2.set_ylabel("λ")
        ax2.set_title("Genomic Inflation by Threshold")
        ax2.grid(True, alpha=0.3)
        null_p = self.data[p_col][self.data[p_col] >= 0.001]
        n_null = len(null_p)
        observed_null = -np.log10(np.sort(null_p))
        expected_null = -np.log10((np.arange(1, n_null + 1) - 0.5) / n_null)
        ax3.scatter(expected_null, observed_null, alpha=0.6, s=10)
        max_val = max(observed_null.max(), expected_null.max())
        ax3.plot([0, max_val], [0, max_val], "r--", alpha=0.7)
        ax3.set_xlabel("Expected -log10(p)")
        ax3.set_ylabel("Observed -log10(p)")
        ax3.set_title("QQ Plot (null region, p ≥ 0.001)")
        ax3.grid(True, alpha=0.3)
        thresholds_enrich = [0.05, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        enrichments = []
        for thresh in tqdm(thresholds_enrich, desc="enrichment thresholds", unit="thresh"):
            observed = np.sum(self.data[p_col] < thresh)
            expected = len(self.data[p_col]) * thresh
            enrichments.append(observed / expected if expected > 0 else 0)
        ax4.semilogx(thresholds_enrich, enrichments, "go-")
        ax4.axhline(y=1.0, color="red", linestyle="--", alpha=0.7)
        ax4.set_xlabel("P-value threshold")
        ax4.set_ylabel("Observed/Expected ratio")
        ax4.set_title("Signal Enrichment")
        ax4.grid(True, alpha=0.3)
        chi_med_all = np.median(stats.chi2.ppf(1 - self.data[p_col], 1))
        lambda_raw_all = chi_med_all / stats.chi2.ppf(0.5, 1)
        n_eff_all = self.sample_sizes.get("N_effective")
        if n_eff_all is not None and n_eff_all > 1000:
            log.info(f"Calculating adjusted lambda with N_effective={n_eff_all}")
            lambda_display = 1.0 + (lambda_raw_all - 1.0) * (1000.0 / float(n_eff_all))
        else:
            lambda_display = lambda_raw_all
        fig.suptitle(
            f"Calibration Assessment (λ = {lambda_display:.2f}, n = {len(self.data[p_col]):,})",
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(self.output, dpi=600, bbox_inches="tight")
        plt.close()
        return True

    def lambda_distribution_plot(self, p_col: Optional[str] = None) -> bool:
        if not self._load_data():
            return False
        if p_col is None:
            if not self.var_p:
                if not self.determine_variable_columns(self.data):
                    log.error("Could not identify p-value column")
                    return False
            p_col = self.var_p
        thresholds = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]
        lambdas = []
        n_snps = []
        n_eff = self.sample_sizes.get("N_effective")
        if n_eff is not None and n_eff > 1000:
            log.info(f"Calculating adjusted lambda using N_effective = {n_eff}")
        for thresh in tqdm(thresholds, desc="λ thresholds", unit="thresh"):
            null_p = self.data[p_col][self.data[p_col] >= thresh]
            if len(null_p) > 100:
                chi_med = np.median(stats.chi2.ppf(1 - null_p, 1))
                lambda_raw = chi_med / stats.chi2.ppf(0.5, 1)
                if n_eff is not None and n_eff > 1000:
                    lambda_val = 1.0 + (lambda_raw - 1.0) * (1000.0 / float(n_eff))
                else:
                    lambda_val = lambda_raw
                lambdas.append(lambda_val)
                n_snps.append(len(null_p))
            else:
                lambdas.append(np.nan)
                n_snps.append(0)
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        valid_idx = ~np.isnan(lambdas)
        plt.plot(
            np.array(thresholds)[valid_idx],
            np.array(lambdas)[valid_idx],
            "bo-",
            linewidth=2,
            markersize=8,
        )
        plt.axhline(
            y=1.0, color="blue", linestyle="--", alpha=0.7, label="λ = 1 (no inflation)"
        )
        plt.axhline(
            y=1.05,
            color="yellow",
            linestyle="--",
            alpha=0.7,
            label="λ = 1.05 (mild inflation)",
        )
        plt.axhline(
            y=1.10,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label="λ = 1.10 (moderate inflation)",
        )
        plt.axhline(
            y=1.20,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="λ = 1.20 (severe inflation)",
        )
        plt.xlabel("P-value threshold", fontsize=14)
        plt.ylabel("Genomic inflation factor (λ)", fontsize=14)
        plt.title("Genomic Inflation Across P-value Thresholds", fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig = plt.gcf()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placed_bboxes = []
        for i, (thresh, lam, n) in enumerate(
            tqdm(
                zip(thresholds, lambdas, n_snps),
                total=len(thresholds),
                desc="λ annotations",
                unit="ann",
            )
        ):
            if not np.isnan(lam):
                x_pos = thresh
                y_pos = lam
                annotation_text = f"λ={lam:.2f}\nn={n:,}"
                text_disp = ax.transData.transform((x_pos, y_pos)) + np.array(
                    [0.0, 8.0]
                )
                text_data = ax.transData.inverted().transform(text_disp)
                annot = ax.annotate(
                    annotation_text,
                    xy=(x_pos, y_pos),
                    xytext=(text_data[0], text_data[1]),
                    textcoords="data",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="gray",
                    ),
                    arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.7, color="black"),
                    zorder=10,
                )
                fig.canvas.draw()
                bbox = annot.get_window_extent(renderer=renderer)
                max_iters = 20
                iters = 0
                while iters < max_iters:
                    overlap = any(bbox.overlaps(existing) for existing in placed_bboxes)
                    if not overlap:
                        break
                    text_disp = text_disp + np.array([0.0, 12.0])
                    text_data = ax.transData.inverted().transform(text_disp)
                    annot.set_position((text_data[0], text_data[1]))
                    fig.canvas.draw()
                    bbox = annot.get_window_extent(renderer=renderer)
                    iters += 1
                ax_bb = ax.get_window_extent(renderer=renderer)
                if bbox.y1 > ax_bb.y1:
                    shift_down = bbox.y1 - ax_bb.y1 + 6.0
                    text_disp = text_disp - np.array([0.0, shift_down])
                    text_data = ax.transData.inverted().transform(text_disp)
                    annot.set_position((text_data[0], text_data[1]))
                    fig.canvas.draw()
                    bbox = annot.get_window_extent(renderer=renderer)
                placed_bboxes.append(bbox)
        plt.savefig(self.output, dpi=600, bbox_inches="tight")
        plt.close()
        log.info("Lambda distribution plot shows calibration across p-value ranges")
        return True

    def volcano_plot(
        self,
        p_col: Optional[str] = None,
        effect_col: Optional[str] = None,
        id_col: Optional[str] = None,
    ) -> bool:
        if not self._load_data():
            return False
        if p_col is None:
            if not self.var_p:
                if not self.determine_variable_columns(self.data):
                    log.error("Could not identify p-value column")
                    return False
            p_col = self.var_p
        if effect_col is None:
            if not self.var_coef:
                if not self.determine_variable_columns(self.data):
                    log.error("Could not identify effect size column")
                    return False
            effect_col = self.var_coef
        if id_col is None:
            for id_type in ["RSID", "SNP", "CGID", "PROBE_ID"]:
                id_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), id_type)
                if id_col:
                    break
            if not id_col:
                id_col = self.id_col if self.id_col else self.data.columns[0]
        required_cols = [p_col, effect_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            msg = f"Missing required columns: {', '.join(missing_cols)}"
            log.error(msg)
            return False
        plot_data = self.data.copy()
        plot_data[p_col] = pd.to_numeric(plot_data[p_col], errors="coerce")
        plot_data.loc[plot_data[p_col] == 0, p_col] = self.MIN_P_VALUE
        plot_data.loc[plot_data[p_col] < self.MIN_P_VALUE, p_col] = self.MIN_P_VALUE
        plot_data = plot_data[(plot_data[p_col] > 0) & (plot_data[p_col] <= 1)]
        plot_data[effect_col] = pd.to_numeric(plot_data[effect_col], errors="coerce")
        plot_data = plot_data[~plot_data[effect_col].isna()]
        plot_data = self._apply_jittering_for_ties(
            plot_data, p_col, effect_col, max_ties=50
        )
        plt.figure(figsize=(self.width, self.height))
        plot_data["neg_log_p"] = -np.log10(plot_data[p_col])
        plot_data[effect_col] = pd.to_numeric(plot_data[effect_col], errors="coerce")
        plot_data["neg_log_p"] = -np.log10(plot_data[p_col])
        if self.threshold is None:
            threshold_val = 5e-8
        elif isinstance(self.threshold, (int, float)) and self.threshold > 0:
            threshold_val = self.threshold
        else:
            threshold_val = None
        eff = plot_data[effect_col].astype(float)
        sig_mask = (threshold_val is not None) & (plot_data[p_col] < threshold_val)
        colors = np.full(len(plot_data), "gray", dtype=object)
        pos_mask = sig_mask & (eff > 0)
        neg_mask = sig_mask & (eff <= 0)
        colors[pos_mask.values] = "red"
        colors[neg_mask.values] = "blue"
        plt.scatter(
            plot_data[effect_col], plot_data["neg_log_p"], c=colors, s=30, alpha=0.7
        )
        if threshold_val is not None:
            plt.axhline(
                y=-np.log10(threshold_val), color="black", linestyle="--", alpha=0.3
            )
        plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
        xlim = plt.xlim()
        ylim = plt.ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        if id_col and id_col in plot_data.columns:
            top_n = min(self.n_annot, len(plot_data))
            top_hits = plot_data.nsmallest(top_n, p_col)
            if self.annotate_genes:
                gene_labels = self._get_gene_labels(top_hits, id_col)
            annotations = []
            for i, (_, hit) in enumerate(tqdm(
                top_hits.iterrows(),
                total=len(top_hits),
                desc="volcano annotations",
                unit="annot"
            )):
                point_x = hit[effect_col]
                point_y = hit["neg_log_p"]
                if self.annotate_genes:
                    label = gene_labels.get(hit[id_col], hit[id_col])
                else:
                    label = str(hit[id_col])
                x_center = (xlim[0] + xlim[1]) / 2
                if point_x < x_center:
                    offset_x = -0.08 * x_range - (i % 4) * 0.03 * x_range
                    offset_y = 0.05 * y_range + (i % 6) * 0.04 * y_range
                else:
                    offset_x = 0.08 * x_range + (i % 4) * 0.03 * x_range
                    offset_y = 0.05 * y_range + (i % 6) * 0.04 * y_range
                annotation_x = point_x + offset_x
                annotation_y = point_y + offset_y
                annotation_x = max(
                    xlim[0] + 0.02 * x_range,
                    min(xlim[1] - 0.02 * x_range, annotation_x),
                )
                annotation_y = max(
                    ylim[0] + 0.02 * y_range, min(ylim[1] + 0.3 * y_range, annotation_y)
                )
                annotations.append(
                    {
                        "x": annotation_x,
                        "y": annotation_y,
                        "label": label,
                        "original_x": point_x,
                        "original_y": point_y,
                        "hit_data": hit,
                    }
                )
            annotations = sorted(annotations, key=lambda a: a["y"], reverse=True)
            min_distance = 0.04 * y_range
            for i in range(1, len(annotations)):
                current = annotations[i]
                for j in range(i):
                    other = annotations[j]
                    x_dist = abs(current["x"] - other["x"])
                    y_dist = abs(current["y"] - other["y"])
                    if x_dist < 0.1 * x_range and y_dist < min_distance:
                        if current["y"] > other["y"]:
                            current["y"] = other["y"] + min_distance
                        else:
                            current["y"] = other["y"] - min_distance
            max_annotation_y = (
                max([a["y"] for a in annotations]) if annotations else ylim[1]
            )
            if max_annotation_y > ylim[1]:
                new_ylim = (ylim[0], max_annotation_y + 0.1 * y_range)
                plt.ylim(new_ylim)
                legend_position = "upper left"
            else:
                legend_position = "upper right"
            for annotation in annotations:
                plt.scatter(
                    annotation["original_x"],
                    annotation["original_y"],
                    s=40,
                    c="red",
                    alpha=1.0,
                    edgecolors="black",
                    zorder=10,
                )
                plt.plot(
                    [annotation["original_x"], annotation["x"]],
                    [annotation["original_y"], annotation["y"]],
                    "k-",
                    alpha=0.6,
                    linewidth=0.8,
                    zorder=9,
                )
                plt.text(
                    annotation["x"],
                    annotation["y"],
                    annotation["label"],
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0,
                    zorder=11,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="gray",
                    ),
                )
        else:
            legend_position = "upper right"
        plt.xlabel("Effect Size", fontsize=20)
        plt.ylabel("-log10(p-value)", fontsize=20)
        plt.title("Volcano Plot", fontsize=24)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="Significant Positive",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=8,
                label="Significant Negative",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label="Not Significant",
            ),
        ]

        if threshold_val is None:
            threshold_label = "Threshold: None"
            threshold_handle = Line2D([0], [0], color="black", linestyle="--", label=threshold_label)
        else:
            try:
                neglog = -np.log10(float(threshold_val))
                threshold_label = f"Threshold: p = {threshold_val:.0e} (-log10={neglog:.2f})"
            except Exception:
                threshold_label = f"Threshold: p = {threshold_val}"
            threshold_handle = Line2D([0], [0], color="black", linestyle="--", label=threshold_label)
        legend_elements.append(threshold_handle)
        
        try:
            axes = plt.gca()
            fig = axes.figure
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            annotation_bboxes = []
            for artist in axes.get_children():
                try:
                    if hasattr(artist, "get_bbox_patch") and artist.get_bbox_patch() is not None:
                        abox = artist.get_window_extent(renderer=renderer)
                        annotation_bboxes.append(abox)
                except Exception:
                    continue

            try:
                pts_x = plot_data[effect_col].astype(float).values
                pts_y = plot_data["neg_log_p"].astype(float).values
                if pts_x.size > 0 and pts_y.size > 0:
                    pts = np.column_stack((pts_x, pts_y))
                    pts_disp = axes.transData.transform(pts)
                else:
                    pts_disp = np.zeros((0, 2))
            except Exception:
                pts_disp = np.zeros((0, 2))

            candidates = [
                "upper right",
                "upper left",
                "lower right",
                "lower left",
                "center right",
                "center left",
                "upper center",
                "lower center",
            ]
            placed_legend = None
            padding_px = 6.0

            def _bbox_contains_any_points(lbbox, points_disp, pad_px=0.0):
                if points_disp.size == 0:
                    return False
                expanded = lbbox.expanded(1.0 + pad_px / max(lbbox.width, 1e-6), 1.0 + pad_px / max(lbbox.height, 1e-6))
                x0, y0, x1, y1 = expanded.x0, expanded.y0, expanded.x1, expanded.y1
                xs = points_disp[:, 0]
                ys = points_disp[:, 1]
                inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
                return bool(np.any(inside))

            for loc in candidates:
                try:
                    legend = axes.legend(handles=legend_elements, loc=loc, frameon=True, fontsize=12)
                    fig.canvas.draw()
                    lbbox = legend.get_window_extent(renderer=renderer)
                    overlap_annotations = any(lbbox.overlaps(ab) for ab in annotation_bboxes)
                    overlap_points = _bbox_contains_any_points(lbbox, pts_disp, pad_px=padding_px)
                    if not overlap_annotations and not overlap_points:
                        placed_legend = legend
                        break
                    legend.remove()
                except Exception:
                    try:
                        legend.remove()
                    except Exception:
                        pass
                    continue

            if placed_legend is None:
                try:
                    placed_legend = axes.legend(
                        handles=legend_elements,
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0,
                        frameon=True,
                        fontsize=12,
                    )
                except Exception:
                    placed_legend = None

            if placed_legend is None:
                plt.legend(handles=legend_elements, loc=legend_position, frameon=True, fontsize=12)
        except Exception:
            plt.legend(handles=legend_elements, loc=legend_position, frameon=True, fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output, dpi=600)
        plt.close()
        log.info(f"Volcano plot saved to {self.output}")
        return True

    def region_plot(
        self,
        id_col: Optional[str] = None,
        p_col: Optional[str] = None,
        chr_col: Optional[str] = None,
        pos_col: Optional[str] = None,
        gene_col: Optional[str] = None,
        focus_id: Optional[str] = None,
        window_size: int = 500000,
    ) -> bool:
        if not self._load_data():
            return False
        if p_col is None:
            if not self.var_p:
                if not self.determine_variable_columns(self.data):
                    log.error("Could not identify p-value column")
                    return False
            p_col = self.var_p
        if chr_col is None:
            chr_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), "CHR")
            if chr_col is None:
                log.error("Could not identify chromosome column")
                return False
        if pos_col is None:
            pos_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), "BP")
            if pos_col is None:
                log.error("Could not identify position column")
                return False
        if id_col is None:
            for id_type in ["RSID", "SNP", "CGID", "PROBE_ID"]:
                id_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), id_type)
                if id_col:
                    break
            if not id_col:
                id_col = self.id_col if self.id_col else self.data.columns[0]
        if gene_col is None:
            for field in ["NEAREST_GENE", "GENE"]:
                gene_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), field)
                if gene_col:
                    break
        required_cols = [id_col, p_col, chr_col, pos_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            msg = f"Missing required columns: {', '.join(missing_cols)}"
            log.error(msg)
            return False
        if focus_id is None:
            focus_row = self.data.loc[self.data[p_col].idxmin()]
            focus_id = focus_row[id_col]
        else:
            focus_rows = self.data[self.data[id_col] == focus_id]
            if len(focus_rows) == 0:
                msg = f"Focus variant {focus_id} not found in data"
                log.error(msg)
                return False
            focus_row = focus_rows.iloc[0]
        focus_chr = focus_row[chr_col]
        focus_pos = focus_row[pos_col]
        focus_p = focus_row[p_col]
        region_start = focus_pos - window_size / 2
        region_end = focus_pos + window_size / 2
        chr_match = self.data[chr_col] == focus_chr
        pos_ge_start = self.data[pos_col] >= region_start
        pos_le_end = self.data[pos_col] <= region_end
        region_data = self.data[chr_match & pos_ge_start & pos_le_end].copy()
        if len(region_data) == 0:
            msg = f"No variants found in region around {focus_id}"
            log.error(msg)
            return False
        has_ld = False
        if "R2" in region_data.columns or "r2" in region_data.columns:
            ld_col = "R2" if "R2" in region_data.columns else "r2"
            has_ld = True
            region_data[ld_col] = pd.to_numeric(region_data[ld_col], errors="coerce")
            region_data[ld_col] = region_data[ld_col].fillna(0)
        else:
            region_data["R2"] = 0
            region_data.loc[region_data[id_col] == focus_id, "R2"] = 1
            ld_col = "R2"
        plt.figure(figsize=(self.width, self.height))
        cmap = cm.plasma
        norm = Normalize(vmin=0, vmax=1)
        scatter = plt.scatter(
            region_data[pos_col],
            -np.log10(region_data[p_col]),
            c=region_data[ld_col],
            cmap=cmap,
            norm=norm,
            s=30,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )
        focus_marker = plt.scatter(
            focus_pos,
            -np.log10(focus_p),
            s=100,
            c="purple",
            marker="*",
            edgecolors="black",
            linewidths=1,
            zorder=10,
        )
        if gene_col and gene_col in region_data.columns:
            genes = set()
            for gene in region_data[gene_col].dropna().unique():
                for g in str(gene).split(";"):
                    g = g.strip()
                    if g:
                        genes.add(g)
            if genes:
                xlim = plt.xlim()
                ylim = plt.ylim()
                y_pos = ylim[0] + 0.05 * (ylim[1] - ylim[0])
                x_positions = np.linspace(
                    xlim[0] + 0.1 * (xlim[1] - xlim[0]),
                    xlim[1] - 0.1 * (xlim[1] - xlim[0]),
                    len(genes),
                )
                for i, gene in enumerate(sorted(genes)):
                    if i < len(x_positions):
                        plt.text(
                            x_positions[i],
                            y_pos,
                            gene,
                            ha="center",
                            va="center",
                            fontsize=16,
                            rotation=45,
                            bbox=dict(
                                boxstyle="round,pad=0.1",
                                facecolor="lightyellow",
                                alpha=0.8,
                            ),
                        )
        if has_ld:
            cbar = plt.colorbar(scatter, label="r²")
            cbar.set_label("r²", fontsize=20)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_alpha(1)
            cbar.draw_all()
        condition1 = self.threshold is not None
        condition2 = isinstance(self.threshold, (int, float))
        condition3 = self.threshold > 0
        if condition1 and condition2 and condition3:
            plt.axhline(
                y=-np.log10(self.threshold), color="red", linestyle="--", alpha=0.7
            )
        plt.xlabel(f"Chromosome {focus_chr} Position (Mb)", fontsize=18)
        plt.ylabel("-log10(p-value)", fontsize=18)
        if gene_col and pd.notna(focus_row.get(gene_col, None)) and focus_row[gene_col]:
            gene_name = focus_row[gene_col]
            plt.title(f"Regional Association Plot - {gene_name} Region", fontsize=24)
        else:
            plt.title(f"Regional Association Plot - {focus_id}", fontsize=24)
        plt.legend([focus_marker], [focus_id], loc="upper right", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        def format_mb(x: float, pos: Any) -> str:
            return f"{x / 1000000:.1f}"

        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_mb))
        plt.xlabel(f"Chromosome {focus_chr} Position (Mb)")
        plt.tight_layout()
        plt.savefig(self.output, dpi=600)
        plt.close()
        log.info(f"Region plot saved to {self.output}")
        return True

    def standardize_columns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        column_mappings: Dict[str, str] = {}
        cgid_col = AliasUtils.find_keys(dict.fromkeys(df.columns), "CGID")
        rsid_col = AliasUtils.find_keys(dict.fromkeys(df.columns), "RSID")
        if cgid_col:
            self.data_type = "EWAS"
            self.id_col = "CGID"
            required_fields = [
                "CGID",
                "CHR",
                "BP",
                "COEF",
                "SE",
                "P",
                "P_FDR",
                "P_HOLM",
                "P_BONF",
            ]
        elif rsid_col:
            self.data_type = "GWAS"
            self.id_col = "RSID"
            required_fields = [
                "RSID",
                "CHR",
                "BP",
                "COEF",
                "SE",
                "P",
                "P_FDR",
                "P_HOLM",
                "P_BONF",
            ]
        else:
            for potential_id in ["SNP", "PROBE_ID", "ID", "NAME"]:
                potential_col = AliasUtils.find_keys(
                    dict.fromkeys(df.columns), potential_id
                )
                if potential_col:
                    self.data_type = "UNKNOWN"
                    self.id_col = potential_id
                    log.warn(
                        f"Using {potential_id} as identifier column. Data type is unknown."
                    )
                    required_fields = [potential_id, "CHR", "BP", "COEF", "SE", "P"]
                    break
            else:
                log.error("No suitable identifier column found")
                return None
        optional_fields = ["GENE", "NEAREST_GENE", "T-STAT"]
        all_fields = required_fields + optional_fields
        for field in all_fields:
            found_col = AliasUtils.find_keys(dict.fromkeys(df.columns), field)
            if found_col and found_col not in column_mappings:
                fc_lower = found_col.lower()
                field_lower = field.lower()
                ends_with_prefixed = any(
                    fc_lower.endswith(sep + field_lower)
                    for sep in AliasUtils.STANDARD_SEPARATORS
                    if sep != ""
                )
                if ends_with_prefixed and fc_lower != field_lower:
                    continue
                column_mappings[found_col] = field
        if column_mappings:
            df = df.rename(columns=column_mappings)
            log.debug(f"Applied column mappings: {column_mappings}")
        return df

    def determine_variable_columns(self, df: pd.DataFrame) -> bool:
        self.var_p = None
        self.var_fdr = None
        self.var_holm = None
        self.var_coef = None
        if self.var:
            log.info(f"Using specified variable: {self.var}")
            existing_mappings: Dict[str, str] = {}
            variable_columns = AliasUtils.get_all_variable_columns(
                df, self.var, existing_mappings
            )
            self.var_p = variable_columns.get("P")
            self.var_coef = variable_columns.get("COEF")
            self.var_fdr = variable_columns.get("P_FDR")
            self.var_holm = variable_columns.get("P_HOLM")
            if self.var_p:
                log.info(f"Successfully configured for variable: {self.var}")
                log.info(f"P-value column: {self.var_p}")
                if self.var_coef:
                    log.info(f"Coefficient column: {self.var_coef}")
                return True
            else:
                log.error(
                    f"Could not find p-value column for specified variable: {self.var}"
                )
                return False
        detected_variable = AliasUtils.auto_detect_variable(df)
        if detected_variable:
            self.var = detected_variable
            log.info(f"Auto-detected variable: {detected_variable}")
            existing_mappings: Dict[str, str] = {}
            variable_columns = AliasUtils.get_all_variable_columns(
                df, detected_variable, existing_mappings
            )
            self.var_p = variable_columns.get("P")
            self.var_coef = variable_columns.get("COEF")
            self.var_fdr = variable_columns.get("P_FDR")
            self.var_holm = variable_columns.get("P_HOLM")
            if self.var_p:
                log.info(
                    f"Successfully configured for auto-detected variable: {detected_variable}"
                )
                return True
        log.info("Attempting generic column detection using AliasUtils")
        existing_mappings: Dict[str, str] = {}
        self.var_p = AliasUtils.find_p_column_comprehensive(df, existing_mappings)
        if self.var_p:
            existing_mappings[self.var_p] = "P"
        self.var_coef = AliasUtils.find_coef_column_comprehensive(df, existing_mappings)
        if self.var_coef:
            existing_mappings[self.var_coef] = "COEF"
        all_p_columns = AliasUtils.get_all_p_value_columns(df, existing_mappings)
        self.var_fdr = all_p_columns.get("P_FDR")
        self.var_holm = all_p_columns.get("P_HOLM")
        if self.var_p:
            self.var = "Generic"
            log.info("Using generic column configuration")
            log.info(f"P-value column: {self.var_p}")
            if self.var_coef:
                log.info(f"Coefficient column: {self.var_coef}")
            return True
        log.info("Attempting to calculate missing columns from available statistics")
        if not self.var_p:
            calculated_p, p_source = AliasUtils.calculate_p_from_other_stats(
                df, existing_mappings
            )
            if calculated_p is not None:
                df["CALCULATED_P"] = calculated_p
                self.var_p = "CALCULATED_P"
                log.info(f"Calculated p-values from {p_source}")
        if self.var_coef and not AliasUtils.find_se_column_comprehensive(
            df, existing_mappings
        ):
            calculated_se, se_source = AliasUtils.calculate_se_from_other_stats(
                df, existing_mappings
            )
            if calculated_se is not None:
                df["CALCULATED_SE"] = calculated_se
                log.info(f"Calculated SE from {se_source}")
        if self.var_p:
            self.var = "Generic"
            log.info("Successfully configured using calculated columns")
            return True
        log.error("No suitable p-value columns found for plotting")
        return False

    def read_data(self) -> bool:
        return self._load_data()

    def create_plot(self) -> bool:
        if self.plot_type == "miami":
            return self.create_miami_plot()
        elif self.plot_type == "manhattan":
            return self.manhattan_plot()
        elif self.plot_type == "qq":
            return self.qq_plot()
        elif self.plot_type == "volcano":
            return self.volcano_plot()
        elif self.plot_type == "calibration":
            return self.calibration_plot()
        elif self.plot_type == "lambda":
            return self.lambda_distribution_plot()
        elif self.plot_type == "region":
            return self.region_plot()
        else:
            log.error(f"Unknown plot type: {self.plot_type}")
            return False

    def _get_chromosome_boundaries_miami(self) -> Optional[Dict[Any, Dict[str, float]]]:
        if not self._load_data():
            return None
        if not self.var_coef:
            log.error("Miami plot requires coefficient column")
            return None
        if self.var_coef not in self.data.columns:
            log.error(f"Coefficient column {self.var_coef} not found in data")
            return None
        chr_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), "CHR")
        bp_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), "BP")
        if not chr_col or not bp_col:
            log.error("Could not find CHR or BP columns for Miami plot")
            return None
        pos_data = self.data[self.data[self.var_coef] > 0].copy()
        neg_data = self.data[self.data[self.var_coef] < 0].copy()
        if len(pos_data) == 0 or len(neg_data) == 0:
            log.error("Miami plot requires both positive and negative coefficients")
            return None
        self.gwas1 = pd.DataFrame(
            {
                "RSID": pos_data[self.id_col],
                "CHR": pos_data[chr_col],
                "BP": pos_data[bp_col],
                "p": pos_data[self.var_p],
            }
        )
        self.gwas2 = pd.DataFrame(
            {
                "RSID": neg_data[self.id_col],
                "CHR": neg_data[chr_col],
                "BP": neg_data[bp_col],
                "p": neg_data[self.var_p],
            }
        )
        self.top_gwas1 = self.gwas1.sort_values(by="p").head(self.n_annot)
        self.top_gwas2 = self.gwas2.sort_values(by="p").head(self.n_annot)
        combined_data = pd.concat([self.gwas1, self.gwas2])
        chromosomes = combined_data["CHR"].unique()
        chromosomes = sorted(chromosomes, key=self.safe_chr_sort)
        chr_boundaries: Dict[Any, Dict[str, float]] = {}
        pos = 0
        for chrom in chromosomes:
            chrom_data = combined_data[combined_data["CHR"] == chrom]
            if len(chrom_data) > 0:
                chr_len = chrom_data["BP"].max()
                chr_boundaries[chrom] = {
                    "start": pos,
                    "end": pos + chr_len,
                    "center": pos + chr_len / 2,
                }
                pos += chr_len + 10000000
        return chr_boundaries

    def _ensure_coefficient_column_for_miami(self) -> bool:
        if self.var_coef and self.var_coef in self.data.columns:
            return True
        log.info(
            "Miami plot requires coefficient column - attempting comprehensive detection"
        )
        existing_mappings = {
            col: "used" for col in [self.var_p, self.var_fdr, self.var_holm] if col
        }
        self.var_coef = AliasUtils.find_coef_column_comprehensive(
            self.data,
            existing_mappings,
            target_variable=self.var if self.var != "Generic" else None,
        )
        if self.var_coef:
            log.info(f"Found coefficient column: {self.var_coef}")
            return True
        log.error("Could not find or identify a coefficient column for Miami plot")
        return False

    def create_miami_plot(self) -> bool:
        try:
            log.info("Creating the Miami plot.")
            skip_ranges = []
            if self.skip_range:
                try:
                    skip_dict = ParseToKeyValueDict(self.skip_range)
                    skip_from = skip_dict.get("from", None)
                    skip_to = skip_dict.get("to", None)
                    skip_side = skip_dict.get("side", "unilateral").lower()
                    if skip_from == "None" or skip_from is None:
                        skip_from = None
                    if skip_to == "None" or skip_to is None:
                        skip_to = None
                    if skip_from is None or skip_to is None:
                        log.info(
                            "No skip ranges specified, creating regular Miami plot"
                        )
                        skip_ranges = []
                    else:
                        try:
                            skip_from = float(skip_from)
                            skip_to = float(skip_to)
                        except (ValueError, TypeError):
                            log.warning(
                                "Invalid skip range values, creating regular Miami plot"
                            )
                            skip_ranges = []
                        else:
                            if skip_from < 0 and skip_to < 0:
                                abs_from = abs(skip_from)
                                abs_to = abs(skip_to)
                                if abs_from >= abs_to:
                                    raise ValueError(
                                        "For negative skip ranges, 'from' should be closer to 0 than 'to'"
                                    )
                            elif skip_from >= 0 and skip_to >= 0:
                                if skip_from >= skip_to:
                                    raise ValueError(
                                        "'from' value must be less than 'to' value"
                                    )
                            else:
                                raise ValueError(
                                    "'from' and 'to' must have the same sign for unilateral skipping"
                                )
                            if skip_side not in ["unilateral", "bilateral"]:
                                raise ValueError(
                                    "'side' must be either 'unilateral' or 'bilateral'"
                                )
                            if skip_side == "unilateral":
                                if skip_from >= 0 and skip_to >= 0:
                                    skip_ranges = [(skip_from, skip_to)]
                                    log.info(
                                        "Using unilateral skip range in POSITIVE effects plot:"
                                    )
                                    log.info(f"  {skip_from} to {skip_to}")
                                elif skip_from <= 0 and skip_to <= 0:
                                    abs_from = abs(skip_from)
                                    abs_to = abs(skip_to)
                                    skip_ranges = [(abs_from, abs_to)]
                                    log.info(
                                        f"Using unilateral skip range in NEGATIVE effects plot: {abs_from} to {abs_to}"
                                    )
                                else:
                                    raise ValueError(
                                        "For unilateral skipping, both 'from' and 'to' must have the same sign"
                                    )
                            else:
                                abs_from = abs(skip_from)
                                abs_to = abs(skip_to)
                                skip_ranges = [(abs_from, abs_to), (-abs_to, -abs_from)]
                                log.info(
                                    f"Using bilateral skip ranges: {abs_from} to {abs_to} and {-abs_to} to {-abs_from}"
                                )
                except Exception as e:
                    log.warning(
                        f"Error parsing skip range: {e}. Creating regular Miami plot without skipping."
                    )
                    skip_ranges = []
            if not self.var_coef or self.var_coef not in self.data.columns:
                if not self.determine_variable_columns(self.data):
                    log.error(
                        "Could not find or identify a coefficient column for Miami plot"
                    )
                    return False
            chr_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), "CHR")
            bp_col = AliasUtils.find_keys(dict.fromkeys(self.data.columns), "BP")
            id_col = (
                self.id_col
                if self.id_col
                else AliasUtils.find_keys(dict.fromkeys(self.data.columns), "RSID")
            )
            p_col = self.var_p
            if not p_col:
                log.error("Could not find p-value column for Miami plot")
                return False
            if not all([chr_col, bp_col, id_col, p_col]):
                log.error("Could not find required columns for Miami plot")
                return False
            plot_data = self.data.copy()
            plot_data[p_col] = pd.to_numeric(plot_data[p_col], errors="coerce")
            val1 = plot_data[p_col] > 0
            val2 = plot_data[p_col] <= 1
            val3 = ~plot_data[p_col].isna()
            valid_p_mask = val1 & val2 & val3
            plot_data = plot_data[valid_p_mask]
            if len(plot_data) == 0:
                log.error("No valid p-values found after cleaning")
                return False
            plot_data.loc[plot_data[p_col] == 0, p_col] = self.MIN_P_VALUE
            plot_data.loc[plot_data[p_col] < self.MIN_P_VALUE, p_col] = self.MIN_P_VALUE
            plot_data[self.var_coef] = pd.to_numeric(
                plot_data[self.var_coef], errors="coerce"
            )
            valid_coef_mask = ~plot_data[self.var_coef].isna()
            plot_data = plot_data[valid_coef_mask]
            if len(plot_data) == 0:
                log.error("No valid coefficients found after cleaning")
                return False
            pos_data = plot_data[plot_data[self.var_coef] > 0].copy()
            neg_data = plot_data[plot_data[self.var_coef] < 0].copy()
            if len(pos_data) == 0:
                log.error("No positive coefficients found for Miami plot")
                return False
            if len(neg_data) == 0:
                log.error("No negative coefficients found for Miami plot")
                return False
            log.info(
                f"Miami plot data: {len(pos_data)} positive, {len(neg_data)} negative coefficients"
            )
            pos_data = self._apply_jittering_for_ties(
                pos_data, p_col, id_col, max_ties=50
            )
            neg_data = self._apply_jittering_for_ties(
                neg_data, p_col, id_col, max_ties=50
            )
            for df in [pos_data, neg_data]:
                df[chr_col] = (
                    df[chr_col]
                    .astype(str)
                    .str.replace("chr", "", case=False)
                    .str.replace("CHR", "", case=False)
                )
            combined_data = pd.concat([pos_data, neg_data])
            chromosomes = sorted(
                combined_data[chr_col].unique(), key=self.safe_chr_sort
            )
            chrom_data_list = []
            start_pos = 0
            for chrom in tqdm(chromosomes, desc="miami chroms", unit="chrom"):
                chrom_data = combined_data[combined_data[chr_col] == chrom]
                if len(chrom_data) > 0:
                    end_pos = start_pos + chrom_data[bp_col].max()
                    chrom_data_list.append(
                        {
                            "chrom": chrom,
                            "start": start_pos,
                            "end": end_pos,
                            "center": (start_pos + end_pos) / 2,
                        }
                    )
                    start_pos = end_pos + 1000000
            if len(chrom_data_list) == 0:
                log.error("No valid chromosome data found")
                return False
            chrom_df = pd.DataFrame(chrom_data_list)
            pos_data["adj_pos"] = pos_data[bp_col] + pos_data[chr_col].map(
                dict(zip(chrom_df["chrom"], chrom_df["start"]))
            )
            neg_data["adj_pos"] = neg_data[bp_col] + neg_data[chr_col].map(
                dict(zip(chrom_df["chrom"], chrom_df["start"]))
            )
            try:
                pos_data["log_p"] = -np.log10(pos_data[p_col])
                neg_data["log_p"] = -np.log10(neg_data[p_col])
                pos_valid = ~(np.isinf(pos_data["log_p"]) | np.isnan(pos_data["log_p"]))
                neg_valid = ~(np.isinf(neg_data["log_p"]) | np.isnan(neg_data["log_p"]))
                pos_data = pos_data[pos_valid]
                neg_data = neg_data[neg_valid]
                if len(pos_data) == 0 or len(neg_data) == 0:
                    log.error("No valid -log10(p) values after calculation")
                    return False
                pos_max = pos_data["log_p"].max()
                neg_max = neg_data["log_p"].max()
                condition1 = np.isnan(pos_max)
                condition2 = np.isinf(pos_max)
                condition3 = np.isnan(neg_max)
                condition4 = np.isinf(neg_max)
                if condition1 or condition2 or condition3 or condition4:
                    log.error("Invalid maximum -log10(p) values calculated")
                    return False
                log.info(f"Positive data -log10(p) range: 0 to {pos_max:.2f}")
                log.info(f"Negative data -log10(p) range: 0 to {neg_max:.2f}")
            except Exception as e:
                log.error(f"Error calculating -log10(p) values: {e}")
                return False
            pos_data, neg_data = self._validate_miami_data(pos_data, neg_data, p_col)
            if pos_data is None or neg_data is None:
                return False
            if skip_ranges and len(skip_ranges) > 0:
                has_valid_skip = False
                for skip_start, skip_end in skip_ranges:
                    condition1 = skip_start is not None
                    condition2 = skip_end is not None
                    condition3 = skip_start != skip_end
                    if condition1 and condition2 and condition3:
                        has_valid_skip = True
                        break
                if has_valid_skip:
                    if self.threshold is None or self.threshold <= 0:
                        threshold_log = None
                    else:
                        threshold_log = -np.log10(max(self.threshold, 1e-300))
                    skip_side_info = None
                    if skip_side == "unilateral":
                        if skip_from >= 0 and skip_to >= 0:
                            skip_side_info = "positive"
                        elif skip_from <= 0 and skip_to <= 0:
                            skip_side_info = "negative"
                    log.info("Creating Miami plot with axis breaks")
                    broken_miami = MiamiBrokenAxes(
                        skip_ranges=skip_ranges,
                        pos_data=pos_data,
                        neg_data=neg_data,
                        chromosomes=chromosomes,
                        chrom_df=chrom_df,
                        chr_col=chr_col,
                        id_col=id_col,
                        p_col=p_col,
                        threshold=threshold_log,
                        n_annot=self.n_annot,
                        annotate_genes=self.annotate_genes,
                        width=self.width,
                        height=self.height,
                        parent_instance=self,
                        skip_side_info=skip_side_info,
                    )
                    broken_miami.create_plot()
                    broken_miami.save_plot(self.output)
                else:
                    log.info("Creating regular Miami plot (no valid skip ranges)")
                    self._create_regular_miami_plot(
                        pos_data,
                        neg_data,
                        chromosomes,
                        chrom_df,
                        chr_col,
                        id_col,
                        p_col,
                    )
            else:
                log.info("Creating regular Miami plot (no skip ranges specified)")
                self._create_regular_miami_plot(
                    pos_data, neg_data, chromosomes, chrom_df, chr_col, id_col, p_col
                )
            log.info(f"Miami plot saved to {self.output}")
            return True
        except Exception as e:
            log.error(f"Error creating Miami plot: {e}")
            sys.exit(1)

    def _add_annotations_to_regular_miami(
        self,
        pos_ax: Any,
        neg_ax: Any,
        pos_data: pd.DataFrame,
        neg_data: pd.DataFrame,
        id_col: str,
        p_col: str,
    ) -> None:
        top_pos = pos_data.nsmallest(min(self.n_annot // 2, 5), p_col)
        top_neg = neg_data.nsmallest(min(self.n_annot // 2, 5), p_col)
        gene_labels_pos: Dict[str, str] = {}
        gene_labels_neg: Dict[str, str] = {}
        if self.annotate_genes:
            gene_labels_pos = self._get_gene_labels(top_pos, id_col)
            gene_labels_neg = self._get_gene_labels(top_neg, id_col)
        for i, (_, row) in enumerate(tqdm(
            top_pos.iterrows(),
            total=len(top_pos),
            desc="miami pos annot",
            unit="annot"
        )):
            pos_ax.scatter(
                row["adj_pos"],
                row["log_p"],
                s=40,
                c="red",
                alpha=1.0,
                edgecolors="black",
                zorder=10,
            )
            label = (
                gene_labels_pos.get(row[id_col], str(row[id_col]))
                if self.annotate_genes
                else str(row[id_col])
            )
            x_offset = 10 + (i % 3) * 15
            y_offset = 15 + (i % 2) * 10
            pos_ax.annotate(
                label,
                xy=(row["adj_pos"], row["log_p"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=10,
                ha="left",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
                arrowprops=dict(arrowstyle="->", lw=0.5, alpha=0.7),
                zorder=11,
            )
        for i, (_, row) in enumerate(tqdm(
            top_neg.iterrows(),
            total=len(top_neg),
            desc="miami neg annot",
            unit="annot"
        )):
            neg_ax.scatter(
                row["adj_pos"],
                row["log_p"],
                s=40,
                c="red",
                alpha=1.0,
                edgecolors="black",
                zorder=10,
            )
            label = (
                gene_labels_neg.get(row[id_col], str(row[id_col]))
                if self.annotate_genes
                else str(row[id_col])
            )
            x_offset = 10 + (i % 3) * 15
            y_offset = -(20 + (i % 2) * 10)
            neg_ax.annotate(
                label,
                xy=(row["adj_pos"], row["log_p"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=10,
                ha="left",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
                arrowprops=dict(arrowstyle="->", lw=0.5, alpha=0.7),
                zorder=11,
            )

    def _create_regular_miami_plot(
        self,
        pos_data: pd.DataFrame,
        neg_data: pd.DataFrame,
        chromosomes: List[Any],
        chrom_df: pd.DataFrame,
        chr_col: str,
        id_col: str,
        p_col: str,
    ) -> None:
        fig = plt.figure(figsize=(self.width, self.height))
        gs = gridspec.GridSpec(
            nrows=3,
            ncols=1,
            height_ratios=[1, 0.045, 1],
            hspace=0.015,
            left=0.15,
            right=0.95,
            top=0.92,
            bottom=0.08,
        )
        pos_ax = plt.Subplot(fig, gs[0])
        fig.add_subplot(pos_ax)
        neg_ax = plt.Subplot(fig, gs[2])
        fig.add_subplot(neg_ax)
        pos_colors = ["#4683B5", "#EFAD0E"]
        neg_colors = ["#4683B5", "#EFAD0E"]
        for i, chrom in enumerate(chromosomes):
            chrom_pos = pos_data[pos_data[chr_col] == chrom]
            if len(chrom_pos) > 0:
                color = pos_colors[i % 2]
                pos_ax.scatter(
                    chrom_pos["adj_pos"],
                    chrom_pos["log_p"],
                    s=18,
                    c=color,
                    alpha=0.8,
                    edgecolors="none",
                )
            chrom_neg = neg_data[neg_data[chr_col] == chrom]
            if len(chrom_neg) > 0:
                color = neg_colors[i % 2]
                neg_ax.scatter(
                    chrom_neg["adj_pos"],
                    chrom_neg["log_p"],
                    s=18,
                    c=color,
                    alpha=0.8,
                    edgecolors="none",
                )
        pos_max = pos_data["log_p"].max() if len(pos_data) > 0 else 10
        neg_max = neg_data["log_p"].max() if len(neg_data) > 0 else 10
        pos_ax.set_ylim(0, pos_max * 1.05)
        neg_ax.set_ylim(0, neg_max * 1.05)
        neg_ax.invert_yaxis()

        try:
            all_adj = pd.concat([pos_data["adj_pos"], neg_data["adj_pos"]])
            min_adj = float(all_adj.min())
            max_adj = float(all_adj.max())
            if np.isfinite(min_adj) and np.isfinite(max_adj) and max_adj > min_adj:
                range_width = max_adj - min_adj
                left = min_adj
                right = max_adj + max(range_width * 0.005, 1e-6)
                pos_ax.set_xlim(left, right)
                neg_ax.set_xlim(left, right)
        except Exception:
            pass

        if self.threshold is not None and self.threshold > 0:
            threshold_val = -np.log10(self.threshold)
            if threshold_val <= pos_max:
                pos_ax.axhline(y=threshold_val, color="red", linestyle="--", alpha=0.7)
            if threshold_val <= neg_max:
                neg_ax.axhline(y=threshold_val, color="red", linestyle="--", alpha=0.7)
        pos_ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=1.0)
        neg_ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=1.0)
        for ax in [pos_ax, neg_ax]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks([])
            ax.set_xticklabels([])
        pos_ax.spines["bottom"].set_visible(True)
        neg_ax.spines["top"].set_visible(True)
        neg_ax.spines["bottom"].set_visible(False)
        pos_ax.set_ylabel("-log₁₀(p)\nPositive Effects", fontsize=12)
        neg_ax.set_ylabel("-log₁₀(p)\nNegative Effects", fontsize=12)
        gap_ax = plt.Subplot(fig, gs[1])
        fig.add_subplot(gap_ax)
        gap_ax.set_xlim(pos_ax.get_xlim())
        gap_ax.set_ylim(0, 1)
        for spine in gap_ax.spines.values():
            spine.set_visible(False)
        gap_ax.set_xticks([])
        gap_ax.set_yticks([])

        def safe_int_convert(x: Any) -> str:
            try:
                if float(x) == int(float(x)):
                    return str(int(float(x)))
                else:
                    return str(x)
            except (ValueError, OverflowError):
                return str(x)

        for _, row in chrom_df.iterrows():
            gap_ax.text(
                row["center"],
                0.5,
                safe_int_convert(row["chrom"]),
                ha="center",
                va="center",
                fontsize=10,
                rotation=90,
                weight="normal",
                transform=gap_ax.transData,
            )
        if self.n_annot > 0:
            self._add_annotations_to_regular_miami(
                pos_ax, neg_ax, pos_data, neg_data, id_col, p_col
            )
        plt.savefig(self.output, dpi=600, bbox_inches="tight")
        plt.close()

    def _add_annotations_to_axis(
        self,
        ax: Any,
        pos_data: pd.DataFrame,
        neg_data: pd.DataFrame,
        id_col: str,
        p_col: str,
    ) -> None:
        top_pos = pos_data.nsmallest(min(self.n_annot // 2, 5), p_col)
        top_neg = neg_data.nsmallest(min(self.n_annot // 2, 5), p_col)
        gene_labels_pos: Dict[str, str] = {}
        gene_labels_neg: Dict[str, str] = {}
        if self.annotate_genes:
            gene_labels_pos = self._get_gene_labels(top_pos, id_col)
            gene_labels_neg = self._get_gene_labels(top_neg, id_col)
        for i, (_, row) in enumerate(top_pos.iterrows()):
            ax.scatter(
                row["adj_pos"],
                row["log_p"],
                s=40,
                c="red",
                alpha=1.0,
                edgecolors="black",
                zorder=10,
            )
            label = (
                gene_labels_pos.get(row[id_col], str(row[id_col]))
                if self.annotate_genes
                else str(row[id_col])
            )
            x_offset = 10 + (i % 3) * 15
            y_offset = 15 + (i % 2) * 10
            ax.annotate(
                label,
                xy=(row["adj_pos"], row["log_p"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=10,
                ha="left",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
                arrowprops=dict(arrowstyle="->", lw=0.5, alpha=0.7),
                zorder=11,
            )
        for i, (_, row) in enumerate(top_neg.iterrows()):
            point_y = -row["log_p"]
            ax.scatter(
                row["adj_pos"],
                point_y,
                s=40,
                c="red",
                alpha=1.0,
                edgecolors="black",
                zorder=10,
            )
            label = (
                gene_labels_neg.get(row[id_col], str(row[id_col]))
                if self.annotate_genes
                else str(row[id_col])
            )
            x_offset = 10 + (i % 3) * 15
            y_offset = -(20 + (i % 2) * 10)
            ax.annotate(
                label,
                xy=(row["adj_pos"], point_y),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=10,
                ha="left",
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
                arrowprops=dict(arrowstyle="->", lw=0.5, alpha=0.7),
                zorder=11,
            )

    def _validate_miami_data(
        self, pos_data: pd.DataFrame, neg_data: pd.DataFrame, p_col: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        required_cols = ["log_p", "adj_pos"]
        for col in required_cols:
            if col not in pos_data.columns:
                log.error(f"Missing required column '{col}' in positive data")
                return None, None
            if col not in neg_data.columns:
                log.error(f"Missing required column '{col}' in negative data")
                return None, None
        pos_clean = pos_data.copy()
        if p_col in pos_clean.columns:
            pos_clean.loc[pos_clean[p_col] == 0, p_col] = self.MIN_P_VALUE
            pos_clean.loc[pos_clean[p_col] < self.MIN_P_VALUE, p_col] = self.MIN_P_VALUE
            pos_clean["log_p"] = -np.log10(pos_clean[p_col])
        pos_valid = ~(np.isinf(pos_clean["log_p"]) | np.isnan(pos_clean["log_p"]))
        pos_clean = pos_clean[pos_valid]
        neg_clean = neg_data.copy()
        if p_col in neg_clean.columns:
            neg_clean.loc[neg_clean[p_col] == 0, p_col] = self.MIN_P_VALUE
            neg_clean.loc[neg_clean[p_col] < self.MIN_P_VALUE, p_col] = self.MIN_P_VALUE
            neg_clean["log_p"] = -np.log10(neg_clean[p_col])
        neg_valid = ~(np.isinf(neg_clean["log_p"]) | np.isnan(neg_clean["log_p"]))
        neg_clean = neg_clean[neg_valid]
        if len(pos_clean) == 0:
            log.error("No valid positive data after cleaning")
            return None, None
        if len(neg_clean) == 0:
            log.error("No valid negative data after cleaning")
            return None, None
        log.info(
            f"Data validation: {len(pos_clean)} positive, {len(neg_clean)} negative points"
        )
        return pos_clean, neg_clean

    def run(self) -> None:
        try:
            log.info(f"Starting {self.plot_type} plot generation.")
            if not self.read_data():
                raise ValueError("Failed to read or process the input data")
            if not self.create_plot():
                raise ValueError(f"Failed to create the {self.plot_type} plot")
            log.success(
                f"{self.plot_type.capitalize()} plot generation completed successfully."
            )
        except FileNotFoundError as e:
            log.error(f"Input file not found: {e}")
            sys.exit(1)
        except pd.errors.ParserError as e:
            log.error(f"Error parsing CSV file: {e}")
            sys.exit(1)
        except ValueError as e:
            log.error(f"Value error in plot generation: {e}")
            sys.exit(1)
        except Exception as e:
            log.error(f"Unexpected error in PlotAssociationAnalysis: {e}")
            sys.exit(1)


class MiamiBrokenAxes:
    def __init__(
        self,
        skip_ranges: List[Tuple[float, float]],
        pos_data: pd.DataFrame,
        neg_data: pd.DataFrame,
        chromosomes: List[Any],
        chrom_df: pd.DataFrame,
        chr_col: str,
        id_col: str,
        p_col: str,
        threshold: Optional[float],
        n_annot: int,
        annotate_genes: bool,
        width: int,
        height: int,
        parent_instance: PlotAssociationAnalysis,
        skip_side_info: Optional[str] = None,
    ) -> None:
        self.skip_ranges = skip_ranges
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.chromosomes = chromosomes
        self.chrom_df = chrom_df
        self.chr_col = chr_col
        self.id_col = id_col
        self.p_col = p_col
        self.threshold = threshold
        self.n_annot = n_annot
        self.annotate_genes = annotate_genes
        self.width = width
        self.height = height
        self.parent_instance = parent_instance
        self.pos_colors = ["#4683B5", "#EFAD0E"]
        self.neg_colors = ["#4683B5", "#EFAD0E"]
        self.skip_from = None
        self.skip_to = None
        self.pos_skip_from = None
        self.pos_skip_to = None
        self.neg_skip_from = None
        self.neg_skip_to = None
        if skip_side_info == "positive":
            for skip_start, skip_end in skip_ranges:
                if skip_start > 0:
                    self.pos_skip_from = skip_start
                    self.pos_skip_to = skip_end
                    self.skip_from = skip_start
                    self.skip_to = skip_end
                    break
        elif skip_side_info == "negative":
            for skip_start, skip_end in skip_ranges:
                if skip_start > 0:
                    self.neg_skip_from = skip_start
                    self.neg_skip_to = skip_end
                    self.skip_from = skip_start
                    self.skip_to = skip_end
                    break
        else:
            for skip_start, skip_end in skip_ranges:
                if skip_start > 0:
                    self.skip_from = skip_start
                    self.skip_to = skip_end
                    self.pos_skip_from = skip_start
                    self.pos_skip_to = skip_end
                    self.neg_skip_from = skip_start
                    self.neg_skip_to = skip_end
                    break
        if len(self.pos_data) > 0 and "log_p" in self.pos_data.columns:
            pos_max_raw = self.pos_data["log_p"].max()
            self.pos_max = (
                pos_max_raw
                if not (np.isnan(pos_max_raw) or np.isinf(pos_max_raw))
                else 10
            )
        else:
            self.pos_max = 10
        if len(self.neg_data) > 0 and "log_p" in self.neg_data.columns:
            neg_max_raw = self.neg_data["log_p"].max()
            self.neg_max = (
                neg_max_raw
                if not (np.isnan(neg_max_raw) or np.isinf(neg_max_raw))
                else 10
            )
        else:
            self.neg_max = 10
        if self.skip_from is None or self.skip_to is None:
            log.warning("Invalid skip ranges, using defaults")
            default_skip_from = 5
            default_skip_to = max(10, min(self.pos_max * 0.7, self.neg_max * 0.7))
            if self.pos_skip_from is None:
                self.pos_skip_from = default_skip_from
            if self.pos_skip_to is None:
                self.pos_skip_to = default_skip_to
            if self.neg_skip_from is None:
                self.neg_skip_from = default_skip_from
            if self.neg_skip_to is None:
                self.neg_skip_to = default_skip_to
            self.skip_from = default_skip_from
            self.skip_to = default_skip_to
        if self.pos_skip_from is not None and self.pos_skip_to is not None:
            self.pos_skip_to = min(self.pos_skip_to, self.pos_max * 0.9)
        if self.neg_skip_from is not None and self.neg_skip_to is not None:
            self.neg_skip_to = min(self.neg_skip_to, self.neg_max * 0.9)
        log.info(
            f"Miami plot ranges - Positive: 0 to {self.pos_max:.2f}, Negative: 0 to {self.neg_max:.2f}"
        )
        if skip_side_info == "positive":
            log.info(
                f"Positive skip ranges: {self.pos_skip_from} to {self.pos_skip_to}"
            )
        elif skip_side_info == "negative":
            log.info(
                f"Negative skip ranges: {self.neg_skip_from} to {self.neg_skip_to}"
            )
        else:
            log.info(f"Bilateral skip ranges: {self.skip_from} to {self.skip_to}")
        self.fig = None
        self.pos_low_ax = None
        self.pos_high_ax = None
        self.pos_full_ax = None
        self.neg_low_ax = None
        self.neg_high_ax = None
        self.neg_full_ax = None
        self.skip_side_info = skip_side_info

    def _round_to_nice_number(self, x: float) -> int:
        if x <= 1:
            return 1
        magnitude = 10 ** int(np.floor(np.log10(x)))
        normalized = x / magnitude
        if normalized <= 1:
            nice = 1
        elif normalized <= 2:
            nice = 2
        elif normalized <= 5:
            nice = 5
        else:
            nice = 10
        return int(nice * magnitude)

    def _create_all_four_plots(self, gs: Any) -> None:
        self.gs = gs
        self.pos_high_ax = plt.Subplot(self.fig, gs[0])
        self.fig.add_subplot(self.pos_high_ax)
        self.pos_high_ax.set_ylim(self.pos_skip_to, self.pos_max + self.pos_max * 0.05)
        self.pos_low_ax = plt.Subplot(self.fig, gs[2])
        self.fig.add_subplot(self.pos_low_ax)
        self.pos_low_ax.set_ylim(0, self.pos_skip_from)
        self.neg_low_ax = plt.Subplot(self.fig, gs[4])
        self.fig.add_subplot(self.neg_low_ax)
        self.neg_low_ax.set_ylim(0, self.neg_skip_from)
        self.neg_low_ax.invert_yaxis()
        self.neg_high_ax = plt.Subplot(self.fig, gs[6])
        self.fig.add_subplot(self.neg_high_ax)
        self.neg_high_ax.set_ylim(self.neg_skip_to, self.neg_max + self.neg_max * 0.05)
        self.neg_high_ax.invert_yaxis()
        self.chromosome_gap_gs = gs[3]

    def _create_pos_break_neg_full(self, gs: Any) -> None:
        self.gs = gs
        self.pos_high_ax = plt.Subplot(self.fig, gs[0])
        self.fig.add_subplot(self.pos_high_ax)
        self.pos_high_ax.set_ylim(self.pos_skip_to, self.pos_max + self.pos_max * 0.05)
        self.pos_low_ax = plt.Subplot(self.fig, gs[2])
        self.fig.add_subplot(self.pos_low_ax)
        self.pos_low_ax.set_ylim(0, self.pos_skip_from)
        self.neg_full_ax = plt.Subplot(self.fig, gs[4])
        self.fig.add_subplot(self.neg_full_ax)
        self.neg_full_ax.set_ylim(0, self.neg_max + self.neg_max * 0.05)
        self.neg_full_ax.invert_yaxis()
        self.neg_low_ax = None
        self.neg_high_ax = None
        self.chromosome_gap_gs = gs[3]

    def _create_pos_full_neg_break(self, gs: Any) -> None:
        self.gs = gs
        self.pos_full_ax = plt.Subplot(self.fig, gs[0])
        self.fig.add_subplot(self.pos_full_ax)
        self.pos_full_ax.set_ylim(0, self.pos_max + self.pos_max * 0.05)
        self.neg_low_ax = plt.Subplot(self.fig, gs[2])
        self.fig.add_subplot(self.neg_low_ax)
        self.neg_low_ax.set_ylim(0, self.neg_skip_from)
        self.neg_low_ax.invert_yaxis()
        self.neg_high_ax = plt.Subplot(self.fig, gs[4])
        self.fig.add_subplot(self.neg_high_ax)
        self.neg_high_ax.set_ylim(self.neg_skip_to, self.neg_max + self.neg_max * 0.05)
        self.neg_high_ax.invert_yaxis()
        self.pos_low_ax = None
        self.pos_high_ax = None
        self.chromosome_gap_gs = gs[1]

    def create_plot(self) -> None:
        self.fig = plt.figure(figsize=(self.width, self.height))
        pos_needs_break = self.skip_side_info in [None, "positive"]
        neg_needs_break = self.skip_side_info in [None, "negative"]
        if pos_needs_break and neg_needs_break:
            height_ratios = [
                max(1, self.pos_max - self.pos_skip_to),
                3.2,
                self.pos_skip_from,
                7.0,
                self.neg_skip_from,
                3.2,
                max(1, self.neg_max - self.neg_skip_to),
            ]
            gs = gridspec.GridSpec(
                nrows=7,
                ncols=1,
                height_ratios=height_ratios,
                hspace=0.02,
                left=0.15,
                right=0.95,
                top=0.92,
                bottom=0.08,
            )
            self._create_all_four_plots(gs)
        elif pos_needs_break:
            height_ratios = [
                max(1, self.pos_max - self.pos_skip_to),
                3.2,
                self.pos_skip_from,
                7.0,
                self.neg_max,
            ]
            gs = gridspec.GridSpec(
                nrows=5,
                ncols=1,
                height_ratios=height_ratios,
                hspace=0.02,
                left=0.15,
                right=0.95,
                top=0.92,
                bottom=0.08,
            )
            self._create_pos_break_neg_full(gs)
        elif neg_needs_break:
            height_ratios = [
                self.pos_max,
                7.0,
                self.neg_skip_from,
                3.2,
                max(1, self.neg_max - self.neg_skip_to),
            ]
            gs = gridspec.GridSpec(
                nrows=5,
                ncols=1,
                height_ratios=height_ratios,
                hspace=0.02,
                left=0.15,
                right=0.95,
                top=0.92,
                bottom=0.08,
            )
            self._create_pos_full_neg_break(gs)
        self._plot_data()
        self._format_axes()
        self._add_chromosome_labels_in_gap()
        self._add_y_labels()
        self.fig.suptitle("")
        if self.n_annot > 0:
            self._add_annotations()

    def _plot_data(self) -> None:
        if len(self.pos_data) > 0:
            if hasattr(self, "pos_full_ax") and self.pos_full_ax is not None:
                self._plot_data_on_axis(
                    self.pos_full_ax, self.pos_data, self.pos_colors
                )
                self._add_threshold_line(self.pos_full_ax)
            else:
                if hasattr(self, "pos_high_ax") and self.pos_high_ax is not None:
                    pos_high_data = self.pos_data[
                        self.pos_data["log_p"] >= self.pos_skip_to
                    ]
                    self._plot_data_on_axis(
                        self.pos_high_ax, pos_high_data, self.pos_colors
                    )
                    self._add_threshold_line(self.pos_high_ax)
                if hasattr(self, "pos_low_ax") and self.pos_low_ax is not None:
                    pos_low_data = self.pos_data[
                        self.pos_data["log_p"] <= self.pos_skip_from
                    ]
                    self._plot_data_on_axis(
                        self.pos_low_ax, pos_low_data, self.pos_colors
                    )
                    self._add_threshold_line(self.pos_low_ax)
        if len(self.neg_data) > 0:
            if hasattr(self, "neg_full_ax") and self.neg_full_ax is not None:
                self._plot_data_on_axis(
                    self.neg_full_ax, self.neg_data, self.neg_colors
                )
                self._add_threshold_line(self.neg_full_ax)
            else:
                if hasattr(self, "neg_low_ax") and self.neg_low_ax is not None:
                    neg_low_data = self.neg_data[
                        self.neg_data["log_p"] <= self.neg_skip_from
                    ]
                    self._plot_data_on_axis(
                        self.neg_low_ax, neg_low_data, self.neg_colors
                    )
                    self._add_threshold_line(self.neg_low_ax)
                if hasattr(self, "neg_high_ax") and self.neg_high_ax is not None:
                    neg_high_data = self.neg_data[
                        self.neg_data["log_p"] >= self.neg_skip_to
                    ]
                    self._plot_data_on_axis(
                        self.neg_high_ax, neg_high_data, self.neg_colors
                    )
                    self._add_threshold_line(self.neg_high_ax)

        try:
            all_adj = pd.concat([self.pos_data["adj_pos"], self.neg_data["adj_pos"]])
            min_adj = float(all_adj.min())
            max_adj = float(all_adj.max())
            if np.isfinite(min_adj) and np.isfinite(max_adj) and max_adj > min_adj:
                range_width = max_adj - min_adj
                left = min_adj
                right = max_adj + max(range_width * 0.005, 1e-6)
                for ax_name in [
                    "pos_full_ax",
                    "pos_high_ax",
                    "pos_low_ax",
                    "neg_full_ax",
                    "neg_high_ax",
                    "neg_low_ax",
                ]:
                    if hasattr(self, ax_name):
                        ax = getattr(self, ax_name)
                        if ax is not None:
                            ax.set_xlim(left, right)
                if hasattr(self, "gap_ax") and self.gap_ax is not None:
                    self.gap_ax.set_xlim(left, right)
        except Exception:
            pass

    def _add_threshold_line(self, ax: Any) -> None:
        if ax is None or self.threshold is None or self.threshold <= 0:
            return
        ylim = ax.get_ylim()
        y_min, y_max = min(ylim), max(ylim)
        if y_min <= self.threshold <= y_max:
            ax.axhline(y=self.threshold, color="red", linestyle="--", alpha=0.7)

    def _plot_data_on_axis(
        self, ax: Any, data: pd.DataFrame, colors: List[str]
    ) -> None:
        if ax is None or len(data) == 0:
            return
        for i, chrom in enumerate(self.chromosomes):
            chrom_data = data[data[self.chr_col] == chrom]
            if len(chrom_data) > 0:
                color = colors[i % 2]
                ax.scatter(
                    chrom_data["adj_pos"],
                    chrom_data["log_p"],
                    s=18,
                    c=color,
                    alpha=0.8,
                    edgecolors="none",
                )

    def _format_axes(self) -> None:
        all_axes = []
        if hasattr(self, "pos_high_ax") and self.pos_high_ax is not None:
            all_axes.append(self.pos_high_ax)
        if hasattr(self, "pos_low_ax") and self.pos_low_ax is not None:
            all_axes.append(self.pos_low_ax)
        if hasattr(self, "pos_full_ax") and self.pos_full_ax is not None:
            all_axes.append(self.pos_full_ax)
        if hasattr(self, "neg_low_ax") and self.neg_low_ax is not None:
            all_axes.append(self.neg_low_ax)
        if hasattr(self, "neg_high_ax") and self.neg_high_ax is not None:
            all_axes.append(self.neg_high_ax)
        if hasattr(self, "neg_full_ax") and self.neg_full_ax is not None:
            all_axes.append(self.neg_full_ax)
        for ax in all_axes:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.grid(False)
            ax.set_axisbelow(False)
            yticks = ax.get_yticks()
            non_zero_ticks = [tick for tick in yticks if abs(tick) > 0.1]
            ax.set_yticks(non_zero_ticks)
            ax.set_yticklabels(
                [
                    f"{abs(tick):.0f}" if abs(tick) >= 1 else f"{abs(tick):.1f}"
                    for tick in non_zero_ticks
                ],
                fontsize=8,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(True)
        self._format_specific_axes()

    def _format_specific_axes(self) -> None:
        if hasattr(self, "pos_low_ax") and self.pos_low_ax is not None:
            self.pos_low_ax.spines["bottom"].set_visible(True)
            self.pos_low_ax.axhline(
                y=0, color="black", linestyle="-", linewidth=1.5, alpha=1.0, zorder=5
            )
        if hasattr(self, "pos_full_ax") and self.pos_full_ax is not None:
            self.pos_full_ax.spines["bottom"].set_visible(True)
            self.pos_full_ax.axhline(
                y=0, color="black", linestyle="-", linewidth=1.5, alpha=1.0, zorder=5
            )
        if hasattr(self, "neg_full_ax") and self.neg_full_ax is not None:
            self.neg_full_ax.axhline(
                y=0, color="black", linestyle="-", linewidth=1.5, alpha=1.0, zorder=5
            )
        elif hasattr(self, "neg_low_ax") and self.neg_low_ax is not None:
            self.neg_low_ax.axhline(
                y=0, color="black", linestyle="-", linewidth=1.5, alpha=1.0, zorder=5
            )
        if hasattr(self, "pos_high_ax") and self.pos_high_ax is not None:
            self.pos_high_ax.spines["bottom"].set_visible(False)
        if hasattr(self, "neg_high_ax") and self.neg_high_ax is not None:
            self.neg_high_ax.spines["bottom"].set_visible(False)
        self._add_all_caps()

    def _add_all_caps(self) -> None:
        if hasattr(self, "pos_high_ax") and self.pos_high_ax is not None:
            self._add_horizontal_cap(self.pos_high_ax, at_bottom=True)
        if hasattr(self, "pos_low_ax") and self.pos_low_ax is not None:
            self._add_horizontal_cap(self.pos_low_ax, at_bottom=False)
        if hasattr(self, "neg_low_ax") and self.neg_low_ax is not None:
            self._add_horizontal_cap(self.neg_low_ax, at_bottom=True)
        if hasattr(self, "neg_high_ax") and self.neg_high_ax is not None:
            self._add_horizontal_cap(self.neg_high_ax, at_bottom=False)

    def _add_horizontal_cap(self, ax: Any, at_bottom: bool = True) -> None:
        if ax is None:
            return
        pos = ax.get_position()
        cap_length = 0.015
        if at_bottom:
            y_pos = pos.y0
        else:
            y_pos = pos.y1
        x_pos = pos.x0
        line = plt.Line2D(
            [x_pos - cap_length / 2, x_pos + cap_length / 2],
            [y_pos, y_pos],
            transform=self.fig.transFigure,
            color="k",
            linewidth=plt.rcParams["axes.linewidth"] * 2,
            clip_on=False,
            zorder=10,
        )
        self.fig.add_artist(line)

    def _add_chromosome_labels_in_gap(self) -> None:
        if hasattr(self, "chromosome_gap_gs"):
            gap_gs = self.chromosome_gap_gs
        else:
            if self.skip_side_info == "positive":
                gap_gs = self.gs[3] if hasattr(self, "gs") else None
            elif self.skip_side_info == "negative":
                gap_gs = self.gs[1] if hasattr(self, "gs") else None
            else:
                gap_gs = self.gs[3] if hasattr(self, "gs") else None
        if gap_gs is None:
            return
        gap_ax = plt.Subplot(self.fig, gap_gs)
        self.fig.add_subplot(gap_ax)
        xlim = None
        if hasattr(self, "pos_low_ax") and self.pos_low_ax is not None:
            xlim = self.pos_low_ax.get_xlim()
        elif hasattr(self, "pos_full_ax") and self.pos_full_ax is not None:
            xlim = self.pos_full_ax.get_xlim()
        elif hasattr(self, "neg_low_ax") and self.neg_low_ax is not None:
            xlim = self.neg_low_ax.get_xlim()
        elif hasattr(self, "neg_full_ax") and self.neg_full_ax is not None:
            xlim = self.neg_full_ax.get_xlim()
        if xlim is not None:
            gap_ax.set_xlim(xlim)
        gap_ax.set_ylim(0, 1)
        gap_ax.spines["top"].set_visible(False)
        gap_ax.spines["bottom"].set_visible(False)
        gap_ax.spines["left"].set_visible(False)
        gap_ax.spines["right"].set_visible(False)
        gap_ax.set_xticks([])
        gap_ax.set_yticks([])

        def safe_int_convert(x: Any) -> str:
            try:
                if float(x) == int(float(x)):
                    return str(int(float(x)))
                else:
                    return str(x)
            except (ValueError, OverflowError):
                return str(x)

        for _, row in self.chrom_df.iterrows():
            gap_ax.text(
                row["center"],
                0.5,
                safe_int_convert(row["chrom"]),
                ha="center",
                va="center",
                fontsize=8,
                rotation=90,
                weight="normal",
                transform=gap_ax.transData,
            )
        self.gap_ax = gap_ax

    def _add_y_labels(self) -> None:
        condition1 = hasattr(self, "pos_full_ax")
        condition2 = self.pos_full_ax is not None
        condition3 = hasattr(self, "pos_high_ax")
        condition4 = self.pos_high_ax is not None
        condition5 = hasattr(self, "pos_low_ax")
        condition6 = self.pos_low_ax is not None
        if condition1 and condition2:
            pos_pos = self.pos_full_ax.get_position()
            pos_middle_y = (pos_pos.y0 + pos_pos.y1) / 2
        elif condition3 and condition4 and condition5 and condition6:
            pos_high_pos = self.pos_high_ax.get_position()
            pos_low_pos = self.pos_low_ax.get_position()
            pos_middle_y = (pos_high_pos.y0 + pos_low_pos.y1) / 2
        else:
            pos_middle_y = None
        if pos_middle_y is not None:
            self.fig.text(
                0.10,
                pos_middle_y,
                "-log₁₀(p)\nPositive Effects",
                fontsize=10,
                ha="center",
                va="center",
                rotation=90,
                transform=self.fig.transFigure,
            )

        condition1 = hasattr(self, "neg_full_ax")
        condition2 = self.neg_full_ax is not None
        condition3 = hasattr(self, "neg_high_ax")
        condition4 = self.neg_high_ax is not None
        condition5 = hasattr(self, "neg_low_ax")
        condition6 = self.neg_low_ax is not None
        if condition1 and condition2:
            neg_pos = self.neg_full_ax.get_position()
            neg_middle_y = (neg_pos.y0 + neg_pos.y1) / 2
        elif condition3 and condition4 and condition5 and condition6:
            neg_low_pos = self.neg_low_ax.get_position()
            neg_high_pos = self.neg_high_ax.get_position()
            neg_middle_y = (neg_low_pos.y1 + neg_high_pos.y0) / 2
        else:
            neg_middle_y = None
        if neg_middle_y is not None:
            self.fig.text(
                0.10,
                neg_middle_y,
                "-log₁₀(p)\nNegative Effects",
                fontsize=10,
                ha="center",
                va="center",
                rotation=90,
                transform=self.fig.transFigure,
            )

    def _add_annotations(self) -> None:
        top_pos = self.pos_data.nsmallest(min(self.n_annot // 2, 5), self.p_col)
        top_neg = self.neg_data.nsmallest(min(self.n_annot // 2, 5), self.p_col)
        pos_axes: List[Any] = []
        if hasattr(self, "pos_full_ax") and self.pos_full_ax is not None:
            pos_axes.append(self.pos_full_ax)
        else:
            if hasattr(self, "pos_high_ax") and self.pos_high_ax is not None:
                pos_axes.append(self.pos_high_ax)
            if hasattr(self, "pos_low_ax") and self.pos_low_ax is not None:
                pos_axes.append(self.pos_low_ax)
        if pos_axes:
            self._add_annotations_to_plots(pos_axes, top_pos, is_negative=False)
        neg_axes: List[Any] = []
        if hasattr(self, "neg_full_ax") and self.neg_full_ax is not None:
            neg_axes.append(self.neg_full_ax)
        else:
            if hasattr(self, "neg_low_ax") and self.neg_low_ax is not None:
                neg_axes.append(self.neg_low_ax)
            if hasattr(self, "neg_high_ax") and self.neg_high_ax is not None:
                neg_axes.append(self.neg_high_ax)
        if neg_axes:
            self._add_annotations_to_plots(neg_axes, top_neg, is_negative=True)

    def _add_annotations_to_plots(
        self, axes: List[Any], top_hits: pd.DataFrame, is_negative: bool = False
    ) -> None:
        if top_hits.empty:
            return
        gene_labels: Dict[str, str] = {}
        if self.annotate_genes:
            gene_labels = self.parent_instance._get_gene_labels(top_hits, self.id_col)
        for ax in axes:
            ylim = ax.get_ylim()
            y_min, y_max = min(ylim), max(ylim)
            hits_in_range = top_hits[
                (top_hits["log_p"] >= y_min - 0.1) & (top_hits["log_p"] <= y_max + 0.1)
            ]
            for i, (_, hit) in enumerate(hits_in_range.iterrows()):
                label = (
                    gene_labels.get(hit[self.id_col], str(hit[self.id_col]))
                    if self.annotate_genes
                    else str(hit[self.id_col])
                )
                if is_negative:
                    xytext = (10, -15)
                    va = "top"
                else:
                    xytext = (10, 10)
                    va = "bottom"
                ax.annotate(
                    label,
                    xy=(hit["adj_pos"], hit["log_p"]),
                    xytext=xytext,
                    textcoords="offset points",
                    fontsize=9,
                    ha="left",
                    va=va,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", lw=0.5),
                )

    def save_plot(self, filename: str) -> None:
        plt.savefig(filename, dpi=600, bbox_inches="tight")
        plt.close()


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(flags=["-a", "--var"], type=str, default=None, required=True),
    OptionConfig(flags=["-o", "--output"], type=str, default=None, required=True),
    OptionConfig(flags=["-t", "--threshold"], type=float, default=None, required=False),
    OptionConfig(flags=["-n", "--n_annot"], type=int, default=20, required=False),
    OptionConfig(
        flags=["-p", "--plot_type"],
        type=str,
        default="manhattan",
        required=False,
        choices=[
            "manhattan",
            "miami",
            "qq",
            "volcano",
            "calibration",
            "lambda",
            "region",
        ],
    ),
    OptionConfig(flags=["-w", "--width"], type=int, default=12, required=False),
    OptionConfig(flags=["-e", "--height"], type=int, default=8, required=False),
    OptionConfig(flags=["-m", "--max_points"], type=int, default=10000, required=False),
    OptionConfig(flags=["-c", "--colors"], type=str, default=None, required=False),
    OptionConfig(
        flags=["-g", "--annotate_genes"], type=bool, default=False, required=False
    ),
    OptionConfig(
        flags=["-N", "--sample_sizes"],
        type=str,
        default="N=None,N_cases=None,N_controls=None",
        required=False,
    ),
    OptionConfig(
        flags=["-s", "--skip"],
        type=str,
        default="from=None,to=None,side=unilateral",
        required=False,
    ),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="PlotAssociationAnalysis")
    opt = framework.run()

    plotter = PlotAssociationAnalysis(
        input_file=opt.input,
        var=opt.var,
        output=opt.output,
        threshold=opt.threshold,
        n_annot=opt.n_annot,
        plot_type=opt.plot_type,
        width=opt.width,
        height=opt.height,
        max_points=opt.max_points,
        colors=opt.colors,
        annotate_genes=opt.annotate_genes,
        sample_sizes=opt.sample_sizes,
        skip=opt.skip,
    )
    plotter.run()
