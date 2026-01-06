#!/usr/bin/env python
# Import required modules
import gzip
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import requests
import statsmodels.stats.multitest as smm
import sys
import textwrap
from scipy import stats
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.ConvertGeneID import ConvertGeneID
from utils.LoggingUtils import log
from utils.SystemUtils import SystemUtils


class EnrichmentORA:
    def __init__(
        self,
        output_file: str,
        input_file: Optional[str] = None,
        gene_column: Optional[str] = None,
        target_dataset: str = "KEGG",
        pvalue_cutoff: float = 0.05,
        include_genes: bool = False,
        plot: Optional[str] = None,
        top_n: int = 20,
        var: Optional[str] = None,
        plot_option: int = 1,
    ) -> None:
        self.input_file: Optional[str] = input_file
        self.gene_column: Optional[str] = gene_column
        self.id_type: Optional[str] = None
        self.target_dataset: str = target_dataset
        self.output_file: str = output_file
        self.pvalue_cutoff: float = pvalue_cutoff
        self.include_genes: bool = include_genes
        self.plot: Optional[str] = plot
        self.top_n: int = top_n
        self.species: str = "human"
        self.analysis_type: str = "ORA"
        self.var: Optional[str] = var
        self.plot_option: int = plot_option
        self.pvalue_column: Optional[str] = None
        self.effect_column: Optional[str] = None
        self.gene_ids: List[str] = []
        self.input_data: Optional[pd.DataFrame] = None
        self.results: Optional[pd.DataFrame] = None
        self._kegg_data: Optional[Dict[str, Any]] = None
        self._go_data: Optional[Dict[str, Any]] = None
        self.go_terms: Dict[str, Any] = {}
        self.namespace_terms: Dict[str, Dict[str, Any]] = {
            "biological_process": {},
            "molecular_function": {},
            "cellular_component": {},
        }
        self.namespace_gene_maps: Dict[str, Dict[str, Dict[str, List[str]]]] = {
            "biological_process": {"term_to_gene": {}, "gene_to_term": {}},
            "molecular_function": {"term_to_gene": {}, "gene_to_term": {}},
            "cellular_component": {"term_to_gene": {}, "gene_to_term": {}},
        }
        self.kegg_gene_dict: Dict[str, Any] = {}
        self.pathway_to_genes: Dict[str, List[str]] = {}
        self.gene_to_pathways: Dict[str, List[str]] = {}
        self.pathway_info: Dict[str, str] = {}
        self.max_workers: int = SystemUtils.get_optimal_cores(reserve_cores=1)
        log.info(f"Using {self.max_workers} cores for parallel processing")
        self._suppress_split_logging: bool = False

    def standardize_input_columns(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        df = df.copy()
        column_mappings: Dict[str, str] = {}
        relevant_fields = [
            "GENE",
            "GENE_ID",
            "P",
            "P_FDR",
            "COEF",
            "SE",
            "T-STAT",
            "N",
            "NEAREST_GENE",
            "NEAREST_GENE_DIST",
        ]
        for col in df.columns:
            if col in column_mappings:
                continue
            for logical_field in relevant_fields:
                found_field = AliasUtils.find_keys({col: True}, logical_field)
                if found_field:
                    if logical_field not in column_mappings.values():
                        column_mappings[col] = logical_field
                        log.debug(
                            f"Mapped column '{col}' to logical field '{logical_field}'"
                        )
                        break
        if column_mappings:
            df = df.rename(columns=column_mappings)
            log.debug(f"Applied column mappings: {column_mappings}")
        return df, column_mappings

    def _detect_gene_column(self, df: pd.DataFrame) -> Optional[str]:
        if self.gene_column is not None and self.gene_column in df.columns:
            return self.gene_column
        gene_fields = ["GENE", "GENE_ID"]
        for field in gene_fields:
            column = AliasUtils.find_keys(dict.fromkeys(df.columns), field)
            if column:
                log.info(f"Auto-detected gene column: {column}")
                return column
        gene_patterns = ["gene", "symbol", "id", "ensembl", "entrez"]
        for col in df.columns:
            if any(pattern in col.lower() for pattern in gene_patterns):
                log.info(f"Auto-detected gene column: {col}")
                return col
        return None

    def _detect_pvalue_column(self) -> Optional[str]:
        if self.input_data is None:
            return None
        if self.var:
            log.info(f"Looking for p-value column for variable: {self.var}")
            var_p_col = AliasUtils.find_complex_keys(
                self.input_data.columns, self.var, "P"
            )
            if var_p_col:
                log.info(
                    f"Found variable-specific p-value column via AliasUtils: {var_p_col}"
                )
                return var_p_col
            p_aliases = AliasUtils.get_aliases("P")
            var_patterns = [self.var.lower(), self.var.upper(), self.var.title()]
            separators = ["_", ".", "-", ""]
            for var in var_patterns:
                for sep in separators:
                    for alias in p_aliases:
                        combo1 = f"{var}{sep}{alias}"
                        combo2 = f"{alias}{sep}{var}"
                        for col in self.input_data.columns:
                            condition1 = col.lower() == combo1.lower()
                            condition2 = col.lower() == combo2.lower()
                            if condition1 or condition2:
                                log.info(
                                    f"Auto-detected p-value column for variable '{self.var}': {col}"
                                )
                                return col
        p_col = AliasUtils.find_keys(dict.fromkeys(self.input_data.columns), "P")
        if p_col:
            log.info(f"Auto-detected p-value column via AliasUtils: {p_col}")
            return p_col
        for correction in ["P_FDR", "P_BONFERRONI", "P_HOLM", "P_BACON"]:
            corrected_p = AliasUtils.find_keys(
                dict.fromkeys(self.input_data.columns), correction
            )
            if corrected_p:
                log.info(
                    f"Auto-detected corrected p-value column ({correction}): {corrected_p}"
                )
                return corrected_p
        p_aliases = AliasUtils.get_aliases("P")
        for col in self.input_data.columns:
            col_lower = col.lower()
            for alias in p_aliases:
                if alias.lower() in col_lower:
                    log.info(
                        f"Auto-detected p-value column via pattern matching: {col}"
                    )
                    return col
        return None

    def _detect_effect_column(self) -> Optional[str]:
        if self.input_data is None:
            return None
        if self.var:
            log.info(f"Looking for effect size column for variable: {self.var}")
            var_effect_col = AliasUtils.find_complex_keys(
                self.input_data.columns, self.var, "COEF"
            )
            if var_effect_col:
                log.info(
                    f"Found variable-specific effect column via AliasUtils: {var_effect_col}"
                )
                return var_effect_col
            for effect_field in ["T-STAT", "Z"]:
                var_effect_col = AliasUtils.find_complex_keys(
                    self.input_data.columns, self.var, effect_field
                )
                if var_effect_col:
                    log.info(
                        f"Found variable-specific {effect_field} column via AliasUtils: {var_effect_col}"
                    )
                    return var_effect_col
        for field in ["COEF", "T-STAT", "Z"]:
            effect_col = AliasUtils.find_keys(
                dict.fromkeys(self.input_data.columns), field
            )
            if effect_col:
                log.info(f"Auto-detected {field} column via AliasUtils: {effect_col}")
                return effect_col
        coef_aliases = AliasUtils.get_aliases("COEF")
        tstat_aliases = AliasUtils.get_aliases("T-STAT")
        z_aliases = AliasUtils.get_aliases("Z")
        all_effect_aliases = coef_aliases + tstat_aliases + z_aliases
        for col in self.input_data.columns:
            col_lower = col.lower()
            for alias in all_effect_aliases:
                if alias.lower() in col_lower:
                    log.info(
                        f"Auto-detected effect size column via pattern matching: {col}"
                    )
                    return col
        return None

    def _detect_id_type(self) -> str:
        if not self.gene_ids:
            log.warn("No gene IDs available for type detection")
            return "symbol"
        sample_size = min(10, len(self.gene_ids))
        sample_genes = random.sample(self.gene_ids, sample_size)
        log.info(f"Auto-detecting ID type from {sample_size} gene samples...")
        valid_genes: List[str] = []
        for gene in sample_genes:
            gene_str = str(gene).strip()
            if gene_str and gene_str.lower() not in ["nan", "none", "", "null"]:
                valid_genes.append(gene_str)
        if not valid_genes:
            log.warn(
                "No valid gene IDs found for type detection, defaulting to 'symbol'"
            )
            return "symbol"
        entrez_count = 0
        ensembl_count = 0
        symbol_count = 0
        for gene in valid_genes:
            if gene.isdigit():
                entrez_count += 1
            elif gene.upper().startswith("ENSG"):
                ensembl_count += 1
            else:
                symbol_count += 1
        if entrez_count == len(valid_genes):
            detected_type = "entrez"
        elif ensembl_count == len(valid_genes):
            detected_type = "ensembl"
        else:
            detected_type = "symbol"
        log.info(f"Auto-detected ID type: {detected_type}")
        log.debug(
            f"Sample analysis - Entrez: {entrez_count}, Ensembl: {ensembl_count}, Symbol: {symbol_count}"
        )
        return detected_type

    def load_gene_list(self) -> bool:
        try:
            if not self.input_file:
                log.error("No input file specified")
                return False
            log.info(f"Loading data from {self.input_file}")
            if not os.path.exists(self.input_file):
                log.error(f"Input file not found: {self.input_file}")
                return False
            file_ext = os.path.splitext(self.input_file)[1].lower()
            if file_ext == ".csv":
                df = pd.read_csv(self.input_file)
            elif file_ext in [".tsv", ".txt"]:
                df = pd.read_csv(self.input_file, sep="\t")
            else:
                log.error(f"Unsupported file format: {self.input_file}")
                return False
            df, column_mappings = self.standardize_input_columns(df)
            self.input_data = df
            if self.gene_column is None or self.gene_column not in df.columns:
                self.gene_column = self._detect_gene_column(df)
                if self.gene_column is None:
                    log.error("Could not auto-detect gene column in the input file")
                    return False
            if self.gene_column not in df.columns:
                log.error(f"Column '{self.gene_column}' not found in the input file")
                return False
            self.pvalue_column = self._detect_pvalue_column()
            self.effect_column = self._detect_effect_column()
            if self.pvalue_column and self.pvalue_cutoff < 1.0:
                log.info(f"Filtering data by p-value cutoff: {self.pvalue_cutoff}")
                df = df[df[self.pvalue_column] <= self.pvalue_cutoff]
                log.info(f"Kept {len(df)} rows after p-value filtering")
            initial_gene_ids: List[str] = (
                df[self.gene_column].dropna().unique().tolist()
            )
            log.info(
                f"Extracted {len(initial_gene_ids)} initial gene entries from input file"
            )
            split_genes = self._split_multi_gene_entries(initial_gene_ids)
            log.info(
                f"After splitting multi-gene entries: {len(split_genes)} unique genes"
            )
            self.gene_ids = self._clean_individual_genes(split_genes)
            self.id_type = self._detect_id_type()
            return True
        except Exception as e:
            log.error(f"Error loading input file: {e}")
            return False

    def _split_multi_gene_entries_core(
        self, gene_list: List[str], separators: Optional[List[str]] = None
    ) -> Tuple[List[str], int, List[str]]:
        if separators is None:
            separators = [";", ",", "|"]
        all_genes: List[str] = []
        multi_gene_count = 0
        for gene_entry in gene_list:
            gene_str = str(gene_entry).strip()
            if not gene_str or gene_str.lower() in ["nan", "none", "", "null"]:
                continue
            has_separator = any(sep in gene_str for sep in separators)
            if has_separator:
                multi_gene_count += 1
                current_parts = [gene_str]
                for separator in separators:
                    new_parts: List[str] = []
                    for part in current_parts:
                        if separator in part:
                            pieces = [
                                p.strip() for p in part.split(separator) if p.strip()
                            ]
                            new_parts.extend(pieces)
                        else:
                            new_parts.append(part)
                    current_parts = new_parts
                for gene in current_parts:
                    gene_clean = gene.strip()
                    if gene_clean and gene_clean.lower() not in [
                        "nan",
                        "none",
                        "",
                        "null",
                    ]:
                        all_genes.append(gene_clean)
            else:
                all_genes.append(gene_str)
        unique_genes: List[str] = []
        seen: set = set()
        for gene in all_genes:
            if gene not in seen:
                unique_genes.append(gene)
                seen.add(gene)
        semicolon_remaining: List[str] = [g for g in unique_genes if ";" in g]
        return unique_genes, multi_gene_count, semicolon_remaining

    def _split_multi_gene_entries(
        self, gene_list: List[str], separators: Optional[List[str]] = None
    ) -> List[str]:
        unique_genes, multi_gene_count, semicolon_remaining = (
            self._split_multi_gene_entries_core(gene_list, separators)
        )
        suppress_logs = getattr(self, "_suppress_split_logging", False)
        if multi_gene_count > 0 and not suppress_logs:
            log.info(
                f"Split {multi_gene_count} multi-gene entries using separators: {separators or [';', ',', '|']}"
            )
            log.debug(
                f"Gene count change: {len(gene_list)} entries -> {len(unique_genes)} unique genes"
            )
        if semicolon_remaining and not suppress_logs:
            log.warning(
                f"WARNING: {len(semicolon_remaining)} genes still contain semicolons after splitting!"
            )
            for gene in semicolon_remaining[:3]:
                log.debug(f"  Example unsplit gene: '{gene}'")
        return unique_genes

    def _clean_individual_genes(self, gene_list: List[str]) -> List[str]:
        cleaned_genes: List[str] = []
        excluded_count = 0
        seen: set = set()
        separators = [";", ",", "|"]

        def process_gene(gene_value: str) -> None:
            nonlocal excluded_count
            gene_str = str(gene_value).strip()
            if not gene_str or gene_str.lower() in ["nan", "none", "", "null", "na"]:
                excluded_count += 1
                return
            if any(sep in gene_str for sep in separators):
                previous_flag = getattr(self, "_suppress_split_logging", False)
                self._suppress_split_logging = True
                try:
                    split_parts = self._split_multi_gene_entries([gene_str], separators)
                finally:
                    self._suppress_split_logging = previous_flag
                if len(split_parts) != 1 or split_parts[0] != gene_str:
                    for part in split_parts:
                        process_gene(part)
                    return
                log.debug(
                    f"Unable to split multi-gene entry, keeping original: {gene_str[:30]}"
                )
            problematic_patterns = ["LINC0", "MIR", "SNORD", "SNORA", "FLJ"]
            for pattern in problematic_patterns:
                if gene_str.startswith(pattern):
                    excluded_count += 1
                    return
            if len(gene_str) > 20:
                excluded_count += 1
                log.debug(f"Excluding overly long individual gene: {gene_str[:30]}")
                return
            if gene_str not in seen:
                seen.add(gene_str)
                cleaned_genes.append(gene_str)

        for gene in gene_list:
            process_gene(gene)
        if excluded_count > 0:
            log.info(
                f"Excluded {excluded_count} problematic gene entries during cleaning"
            )
        log.info(f"Final cleaned gene list: {len(cleaned_genes)} genes")
        return cleaned_genes

    def convert_gene_ids(self) -> List[str]:
        try:
            target_id_type = "entrez" if self.target_dataset == "KEGG" else "symbol"
            if self.id_type != target_id_type:
                log.info(
                    f"Converting gene IDs from {self.id_type} to {target_id_type}..."
                )
                converted_ids, conversion_stats = ConvertGeneID(
                    self.gene_ids,
                    id_from=self.id_type,
                    id_to=target_id_type,
                    species=self.species,
                    batch_size=500,
                    show_progress=True,
                    return_stats=True,
                )
                result_ids: List[str] = []
                for original, converted in zip(self.gene_ids, converted_ids):
                    if converted:
                        if isinstance(converted, list):
                            result_ids.extend([str(x) for x in converted])
                        else:
                            result_ids.append(str(converted))
                    else:
                        result_ids.append(original)
                log.info(
                    f"Gene ID conversion completed: {conversion_stats['success_rate']:.1f}% success rate"
                )
                success_count = conversion_stats["success_count"]
                total_count = conversion_stats["total_count"]
                log.info(f"Successfully converted: {success_count}/{total_count} genes")
                if conversion_stats.get("failure_count", 0) > 0:
                    log.info(
                        f"{conversion_stats['failure_count']} genes failed conversion - keeping original IDs"
                    )
                log.info(f"Final gene count: {len(result_ids)} gene IDs")
                return result_ids
            else:
                log.info(
                    f"No conversion needed, keeping {len(self.gene_ids)} {self.id_type} IDs"
                )
                return self.gene_ids
        except Exception as e:
            log.error(f"Error converting gene IDs: {e}")
            log.info("Falling back to original gene IDs")
            return self.gene_ids

    def _download_api_data(self, url: str) -> str:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download data: {response.status_code}")
        return response.text

    def _download_kegg_data(self) -> Dict[str, Any]:
        if self._kegg_data is not None:
            return self._kegg_data
        log.info("Downloading KEGG pathway data...")
        gene_data = self._download_api_data("https://rest.kegg.jp/list/hsa")
        for line in gene_data.strip().split("\n"):
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 3:
                    gene_id = parts[0]
                    gene_type = parts[1]
                    description = parts[-1]
                    if ";" in description:
                        gene_names_part, main_desc = description.split(";", 1)
                        gene_names = [
                            name.strip() for name in gene_names_part.split(",")
                        ]
                        main_desc = main_desc.strip()
                    else:
                        gene_names = [name.strip() for name in description.split(",")]
                        main_desc = ""
                    self.kegg_gene_dict[gene_id] = {
                        "gene_type": gene_type,
                        "gene_names": gene_names,
                        "description": main_desc,
                    }
        pathway_data = self._download_api_data("https://rest.kegg.jp/link/hsa/pathway")
        for line in pathway_data.strip().split("\n"):
            if "\t" in line:
                pathway_id, gene_id = line.split("\t")
                if pathway_id not in self.pathway_to_genes:
                    self.pathway_to_genes[pathway_id] = []
                self.pathway_to_genes[pathway_id].append(gene_id)
                if gene_id not in self.gene_to_pathways:
                    self.gene_to_pathways[gene_id] = []
                self.gene_to_pathways[gene_id].append(pathway_id)
        pathway_info_data = self._download_api_data(
            "https://rest.kegg.jp/list/pathway/hsa"
        )
        for line in pathway_info_data.strip().split("\n"):
            if "\t" in line:
                pathway_id, pathway_name = line.split("\t")
                self.pathway_info[pathway_id] = pathway_name.replace(
                    " - Homo sapiens (human)", ""
                )
        self._kegg_data = {
            "gene_dict": self.kegg_gene_dict,
            "pathway_to_genes": self.pathway_to_genes,
            "gene_to_pathways": self.gene_to_pathways,
            "pathway_info": self.pathway_info,
        }
        return self._kegg_data

    def _download_go_data(self) -> Dict[str, Any]:
        if self._go_data is not None:
            return self._go_data
        log.info("Downloading and parsing Gene Ontology data...")
        url = "https://purl.obolibrary.org/obo/go.obo"
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download GO OBO file: {response.status_code}")
        content = response.text
        current_term: Optional[Dict[str, Any]] = None
        excluded_fields = ["synonym", "def", "is_a"]
        for line in content.split("\n"):
            line = line.strip()
            if line == "[Term]":
                current_term = {}
                continue
            if line == "" and current_term is not None:
                if "id" in current_term and "namespace" in current_term:
                    go_id = current_term["id"]
                    namespace = current_term["namespace"]
                    self.go_terms[go_id] = current_term
                    if namespace in self.namespace_terms:
                        self.namespace_terms[namespace][go_id] = current_term
                current_term = None
                continue
            if current_term is not None and ":" in line:
                key, value = line.split(":", 1)
                value = value.strip()
                if key in excluded_fields:
                    continue
                if key in current_term:
                    if isinstance(current_term[key], list):
                        current_term[key].append(value)
                    else:
                        current_term[key] = [current_term[key], value]
                else:
                    current_term[key] = value
        url = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(
                f"Failed to download GO Annotation file: {response.status_code}"
            )
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            content = f.read().decode("utf-8")
        lines = [
            line for line in content.split("\n") if line and not line.startswith("!")
        ]
        gene_to_go: Dict[str, List[str]] = {}
        go_to_gene: Dict[str, List[str]] = {}
        for line in lines:
            fields = line.split("\t")
            if len(fields) >= 7:
                gene_symbol = fields[2]
                qualifier = fields[3]
                go_id = fields[4]
                if "NOT" in qualifier:
                    continue
                if gene_symbol not in gene_to_go:
                    gene_to_go[gene_symbol] = []
                if go_id not in gene_to_go[gene_symbol]:
                    gene_to_go[gene_symbol].append(go_id)
                if go_id not in go_to_gene:
                    go_to_gene[go_id] = []
                if gene_symbol not in go_to_gene[go_id]:
                    go_to_gene[go_id].append(gene_symbol)
                for namespace in self.namespace_terms:
                    if go_id in self.namespace_terms[namespace]:
                        if (
                            go_id
                            not in self.namespace_gene_maps[namespace]["term_to_gene"]
                        ):
                            self.namespace_gene_maps[namespace]["term_to_gene"][
                                go_id
                            ] = []
                        if (
                            gene_symbol
                            not in self.namespace_gene_maps[namespace]["term_to_gene"][
                                go_id
                            ]
                        ):
                            self.namespace_gene_maps[namespace]["term_to_gene"][
                                go_id
                            ].append(gene_symbol)
                        if (
                            gene_symbol
                            not in self.namespace_gene_maps[namespace]["gene_to_term"]
                        ):
                            self.namespace_gene_maps[namespace]["gene_to_term"][
                                gene_symbol
                            ] = []
                        if (
                            go_id
                            not in self.namespace_gene_maps[namespace]["gene_to_term"][
                                gene_symbol
                            ]
                        ):
                            self.namespace_gene_maps[namespace]["gene_to_term"][
                                gene_symbol
                            ].append(go_id)
        log.info(f"Parsed {len(self.go_terms)} total GO terms")
        for namespace in self.namespace_terms:
            log.info(
                f"  {namespace.replace('_', ' ').title()}: {len(self.namespace_terms[namespace])} terms"
            )
        log.info(f"Parsed {len(gene_to_go)} genes with GO annotations")
        log.info(f"Parsed {len(go_to_gene)} GO terms with gene annotations")
        self._go_data = {
            "all_terms": self.go_terms,
            "namespaces": self.namespace_terms,
            "gene_maps": self.namespace_gene_maps,
        }
        return self._go_data

    def _calculate_enrichment(
        self,
        term_id: str,
        term_name: str,
        term_genes: List[str],
        all_genes: Union[List[str], set],
        query_genes: List[str],
        extras: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        term_gene_count = len(term_genes)
        if term_gene_count < 3:
            return None
        term_genes_in_query = set(term_genes).intersection(set(query_genes))
        n_term_genes_in_query = len(term_genes_in_query)
        if n_term_genes_in_query == 0:
            return None
        a = n_term_genes_in_query
        b = term_gene_count - n_term_genes_in_query
        c = len(query_genes) - n_term_genes_in_query
        total_genes = len(all_genes)
        genes_in_term_not_in_query = term_gene_count - n_term_genes_in_query
        d = total_genes - len(query_genes) - genes_in_term_not_in_query
        contingency_table = np.array([[a, b], [c, d]])
        odds_ratio, p_value = stats.fisher_exact(
            contingency_table, alternative="greater"
        )
        gene_ratio = f"{n_term_genes_in_query}/{len(query_genes)}"
        bg_ratio = f"{term_gene_count}/{total_genes}"
        result: Dict[str, Any] = {
            "Term_ID": term_id,
            "Term_Name": term_name,
            "Gene_Ratio": gene_ratio,
            "BG_Ratio": bg_ratio,
            "P_value": p_value,
            "Odds_Ratio": odds_ratio,
        }
        if self.include_genes:
            result["GENES"] = ",".join(term_genes_in_query)
        if extras:
            result.update(extras)
        return result

    def run_kegg_analysis(self) -> Optional[pd.DataFrame]:
        try:
            log.info("Starting KEGG pathway enrichment analysis")
            gene_ids = self.convert_gene_ids()
            if not gene_ids:
                log.error("No valid gene IDs after conversion")
                return None
            gene_ids = [str(g) for g in gene_ids if g and str(g).strip()]
            log.info(f"Using {len(gene_ids)} cleaned gene IDs for KEGG analysis")
            kegg_data = self._download_kegg_data()
            all_kegg_genes: set = set()
            for genes in kegg_data["pathway_to_genes"].values():
                all_kegg_genes.update(genes)
            all_kegg_genes = {g.replace("hsa:", "") for g in all_kegg_genes}
            query_genes = ["hsa:" + g for g in gene_ids if g]
            query_genes_in_kegg = [
                g for g in query_genes if g in kegg_data["gene_to_pathways"]
            ]
            if not query_genes_in_kegg:
                log.warn("No query genes found in KEGG pathways")
                return None
            log.info(
                f"Using {len(query_genes_in_kegg)} of {len(gene_ids)} genes found in KEGG"
            )
            log.info(f"Using {self.max_workers} cores for pathway analysis")
            log.info("Calculating pathway enrichment...")
            results: List[Dict[str, Any]] = []
            for pathway_id in tqdm(
                kegg_data["pathway_to_genes"], desc="Analyzing pathways"
            ):
                pathway_genes = kegg_data["pathway_to_genes"][pathway_id]
                lookup_id = (
                    pathway_id.replace("path:", "")
                    if "path:" in pathway_id
                    else pathway_id
                )
                pathway_name = kegg_data["pathway_info"].get(lookup_id, pathway_id)
                pathway_genes_in_query = set(pathway_genes).intersection(
                    set(query_genes_in_kegg)
                )
                pathway_gene_ids = [
                    g.replace("hsa:", "") for g in pathway_genes_in_query
                ]
                pathway_gene_symbols: List[str] = []
                for gene_id in pathway_gene_ids:
                    if f"hsa:{gene_id}" in kegg_data["gene_dict"]:
                        gene_info = kegg_data["gene_dict"][f"hsa:{gene_id}"]
                        if gene_info["gene_names"]:
                            pathway_gene_symbols.append(gene_info["gene_names"][0])
                extras: Dict[str, Any] = {}
                if self.include_genes:
                    extras["GENES"] = (
                        ",".join(pathway_gene_symbols)
                        if pathway_gene_symbols
                        else ",".join(pathway_gene_ids)
                    )
                enrichment = self._calculate_enrichment(
                    pathway_id,
                    pathway_name,
                    pathway_genes,
                    list(all_kegg_genes),
                    query_genes_in_kegg,
                    extras=extras,
                )
                if enrichment:
                    results.append(enrichment)
            if not results:
                log.warn("No enriched pathways found")
                return None
            result_df = pd.DataFrame(results).sort_values("P_value")
            if len(result_df) > 0:
                log.info("Applying multiple testing correction...")
                _, result_df["Adjusted_P_value"], _, _ = smm.multipletests(
                    result_df["P_value"], method="fdr_bh"
                )
                filter_column = "Adjusted_P_value"
                result_df = result_df[result_df[filter_column] <= self.pvalue_cutoff]
                log.info(
                    f"Found {len(result_df)} significant pathways at {filter_column} <= {self.pvalue_cutoff}"
                )
                result_df["Term_ID"] = result_df["Term_ID"].str.replace("path:", "")
            self.results = result_df
            return result_df
        except Exception as e:
            log.error(f"Error in KEGG ORA analysis: {e}")
            return None

    def run_go_analysis(self) -> Optional[pd.DataFrame]:
        try:
            namespace_map = {
                "biological": "biological_process",
                "molecular": "molecular_function",
                "cellular": "cellular_component",
            }
            namespace = self.target_dataset.split("_")[1].lower()
            namespace_full = namespace_map.get(namespace)
            if not namespace_full:
                log.error(f"Invalid GO namespace: {namespace}")
            log.info(f"Starting GO {namespace.upper()} enrichment analysis")
            gene_ids = self.convert_gene_ids()
            if not gene_ids:
                log.error("No valid gene IDs after conversion")
                return None
            gene_ids = [str(g) for g in gene_ids if g and str(g).strip()]
            log.info(f"Using {len(gene_ids)} cleaned gene IDs for GO analysis")
            go_data = self._download_go_data()
            term_dict = go_data["namespaces"][namespace_full]
            go_to_gene = go_data["gene_maps"][namespace_full]["term_to_gene"]
            all_go_genes: set = set()
            for genes in go_to_gene.values():
                all_go_genes.update(genes)
            query_genes_in_go = [g for g in gene_ids if g in all_go_genes]
            if not query_genes_in_go:
                log.warn(f"No query genes found in GO {namespace_full}")
                return None
            log.info(
                f"Using {len(query_genes_in_go)} of {len(gene_ids)} genes found in GO {namespace_full}"
            )
            log.info("Calculating GO term enrichment...")
            results: List[Dict[str, Any]] = []
            for go_id in tqdm(go_to_gene, desc="Analyzing GO terms"):
                go_genes = go_to_gene[go_id]
                term_name = (
                    term_dict[go_id].get("name", go_id) if go_id in term_dict else go_id
                )
                enrichment = self._calculate_enrichment(
                    go_id,
                    term_name,
                    go_genes,
                    list(all_go_genes),
                    query_genes_in_go,
                    extras={
                        "GO_ID": go_id,
                        "GO_Term": term_name,
                        "Namespace": namespace_full,
                    },
                )
                if enrichment:
                    results.append(enrichment)
            if not results:
                log.warn(f"No enriched GO {namespace_full} terms found")
                return None
            result_df = pd.DataFrame(results).sort_values("P_value")
            if len(result_df) > 0:
                log.info("Applying multiple testing correction...")
                _, result_df["Adjusted_P_value"], _, _ = smm.multipletests(
                    result_df["P_value"], method="fdr_bh"
                )
                filter_column = "Adjusted_P_value"
                result_df = result_df[result_df[filter_column] <= self.pvalue_cutoff]
                log.info(
                    f"Found {len(result_df)} significant GO terms at {filter_column} <= {self.pvalue_cutoff}"
                )
            self.results = result_df
            return result_df
        except Exception as e:
            log.error(f"Error in GO ORA analysis: {e}")
            return None

    def analyze(self) -> Optional[pd.DataFrame]:
        try:
            log.info("Starting ORA enrichment analysis")
            if self.var:
                log.info(f"Analyzing variable: {self.var}")
            if not self.load_gene_list():
                log.error("Failed to load gene list. Exiting.")
                return None
            valid_datasets = ["KEGG", "GO_biological", "GO_molecular", "GO_cellular"]
            if self.target_dataset not in valid_datasets:
                log.error(f"Unsupported target dataset for ORA: {self.target_dataset}")
                log.error(f"Valid datasets are: {valid_datasets}")
                return None
            if self.target_dataset == "KEGG":
                self.results = self.run_kegg_analysis()
            elif self.target_dataset.startswith("GO_"):
                self.results = self.run_go_analysis()
            else:
                log.error(f"Unsupported target dataset for ORA: {self.target_dataset}")
                return None
            if self.results is not None:
                self.results = self.standardize_output_columns(self.results)
                self.save_results()
                if self.plot:
                    if self.results.empty:
                        log.warn(
                            "Skipping barplot generation - no enriched terms to plot"
                        )
                    else:
                        self.create_barplot(output_file=self.plot, top_n=self.top_n)
                if self.results.empty:
                    log.info("No significant enrichment found")
                else:
                    log.info(f"Found {len(self.results)} enriched terms/pathways")
                return self.results
            log.info("No enrichment results were produced")
            return pd.DataFrame()
        except Exception as e:
            log.error(f"Error in enrichment analysis: {e}")
            return None

    def create_barplot(
        self, output_file: Optional[str] = None, top_n: Optional[int] = None
    ) -> Optional[plt.Figure]:
        try:
            if self.results is None or self.results.empty:
                log.warn("No results to plot")
                return None
            data = self.results.copy()
            log.debug(f"Initial data shape: {data.shape}")
            if len(data.columns) != len(set(data.columns)):
                log.debug("Removing duplicate columns before plotting")
                data = data.loc[:, ~data.columns.duplicated()]
            p_col = "P" if "P" in data.columns else "P_value"
            adj_p_col = "P_FDR" if "P_FDR" in data.columns else "Adjusted_P_value"
            for col in data.columns:
                if col in [p_col, adj_p_col, "OR"] or any(
                    term in col.lower() for term in ["p_", "fdr", "p.value", "odds"]
                ):
                    try:
                        if data[col].dtype == "object":
                            data[col] = data[col].apply(
                                lambda x: float(x) if isinstance(x, str) else x
                            )
                    except Exception:
                        pass
            if adj_p_col in data.columns:
                data = data.sort_values(adj_p_col, ascending=True)
            if top_n is not None and len(data) > top_n:
                data = data.head(top_n)
            term_col = "NAME" if "NAME" in data.columns else "Term_Name"
            if term_col not in data.columns:
                term_col = next(
                    (
                        col
                        for col in [
                            "Pathway_Name",
                            "GO_Term",
                            "Term_Name",
                            "Description",
                            "Term",
                        ]
                        if col in data.columns
                    ),
                    data.columns[1] if len(data.columns) > 1 else data.columns[0],
                )
            term_series = data[term_col]
            if isinstance(term_series, pd.DataFrame):
                term_series = term_series.iloc[:, 0]
            term_labels: List[str] = []
            for i, term in enumerate(term_series.values):
                if pd.isna(term):
                    term_labels.append(f"Term_{i}")
                else:
                    term_str = str(term).strip()
                    if len(term_str) > 60:
                        wrapped = textwrap.fill(term_str, width=60)
                        term_labels.append(wrapped)
                    else:
                        term_labels.append(term_str)
            if self.plot_option == 1:
                return self._create_barplot_option1(
                    data, term_labels, adj_p_col, output_file
                )
            elif self.plot_option == 2:
                return self._create_barplot_option2(
                    data, term_labels, adj_p_col, output_file
                )
            else:
                log.error(f"Invalid plot option: {self.plot_option}. Must be 1 or 2.")
                return None
        except Exception as e:
            log.error(f"Error creating barplot: {e}")
            return None

    def _create_barplot_option1(
        self,
        data: pd.DataFrame,
        term_labels: List[str],
        adj_p_col: str,
        output_file: Optional[str],
    ) -> Optional[plt.Figure]:
        try:
            actual_data_length = len(data)
            y_positions = np.arange(actual_data_length)
            ratio_col = "GENE_RATIO" if "GENE_RATIO" in data.columns else "Gene_Ratio"
            if ratio_col in data.columns:
                data["Count"] = data[ratio_col].apply(
                    lambda x: int(x.split("/")[0]) if "/" in str(x) else 0
                )
            elif not any(col in data.columns for col in ["Count", "Gene_Count"]):
                data["Count"] = 5
            count_col = next(
                (col for col in ["Count", "Gene_Count"] if col in data.columns), "Count"
            )
            count_values = data[count_col].values
            fig, ax = plt.subplots(figsize=(10, max(6, actual_data_length * 0.4)))
            if actual_data_length > 0 and adj_p_col in data.columns:
                neg_log_p = -np.log10(data[adj_p_col] + 1e-300)
                colors = plt.cm.viridis(
                    (neg_log_p - neg_log_p.min()) / (neg_log_p.max() - neg_log_p.min())
                )
            else:
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, actual_data_length))
            ax.barh(y=y_positions, width=count_values, color=colors)
            ax.set_xlabel("Gene Count", fontsize=12)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(term_labels, fontsize=10)
            ax.set_ylabel("")
            ax.grid(axis="x", linestyle="--", alpha=0.3, color="gray")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if adj_p_col in data.columns and actual_data_length > 0:
                neg_log_p = -np.log10(data[adj_p_col] + 1e-300)
                norm = plt.Normalize(vmin=neg_log_p.min(), vmax=neg_log_p.max())
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
                sm.set_array(neg_log_p)
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label("-log₁₀(Adjusted p-value)", fontsize=10)
            title_text = (
                self.target_dataset.replace("_", " ") + " ORA Enrichment Results"
            )
            plt.title(title_text, fontsize=14)
            plt.tight_layout()
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                log.info(f"Barplot saved to {output_file}")
                plt.close()
            return fig
        except Exception as e:
            log.error(f"Error creating barplot option 1: {e}")
            return None

    def _create_barplot_option2(
        self,
        data: pd.DataFrame,
        term_labels: List[str],
        adj_p_col: str,
        output_file: Optional[str],
    ) -> Optional[plt.Figure]:
        try:
            actual_data_length = len(data)
            data = data.sort_values(adj_p_col, ascending=False)
            term_labels = term_labels[::-1]
            y_positions = np.arange(actual_data_length)
            if adj_p_col not in data.columns:
                log.error(f"Column {adj_p_col} not found for option 2 plot")
                return None
            neg_log_p_values = -np.log10(data[adj_p_col] + 1e-300)
            or_col = "OR" if "OR" in data.columns else "Odds_Ratio"
            if or_col not in data.columns:
                log.error(f"Column {or_col} not found for option 2 plot")
                return None
            or_values = data[or_col].values
            fig, ax = plt.subplots(figsize=(10, max(6, actual_data_length * 0.4)))
            if len(or_values) > 0 and or_values.max() > or_values.min():
                colors = plt.cm.viridis(
                    (or_values - or_values.min()) / (or_values.max() - or_values.min())
                )
            else:
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, actual_data_length))
            ax.barh(y=y_positions, width=neg_log_p_values, color=colors)
            ax.set_xlabel("-log₁₀(Adjusted p-value)", fontsize=12)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(term_labels, fontsize=10)
            ax.set_ylabel("")
            ax.grid(axis="x", linestyle="--", alpha=0.3, color="gray")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if len(or_values) > 0 and or_values.max() > or_values.min():
                norm = plt.Normalize(vmin=or_values.min(), vmax=or_values.max())
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
                sm.set_array(or_values)
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label("Odds Ratio", fontsize=10)
            title_text = (
                self.target_dataset.replace("_", " ") + " ORA Enrichment Results"
            )
            plt.title(title_text, fontsize=14)
            plt.tight_layout()
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches="tight")
                log.info(f"Barplot saved to {output_file}")
                plt.close()
            return fig
        except Exception as e:
            log.error(f"Error creating barplot option 2: {e}")
            return None

    def standardize_output_columns(
        self, df: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return df
        df = df.copy()
        standard_mappings = {
            "P_value": "P",
            "Adjusted_P_value": "P_FDR",
            "P_BONF": "P_BONFERRONI",
            "Term_ID": "ID",
            "Term_Name": "NAME",
            "Gene_Ratio": "GENE_RATIO",
            "BG_Ratio": "BG_RATIO",
            "Odds_Ratio": "OR",
            "GO_ID": "ID",
            "GO_Term": "NAME",
        }
        for logical_field in [
            "P",
            "P_FDR",
            "P_BONFERRONI",
            "P_HOLM",
            "P_BACON",
            "COEF",
            "SE",
        ]:
            if logical_field in df.columns:
                continue
            found_col = AliasUtils.find_keys(dict.fromkeys(df.columns), logical_field)
            if found_col and found_col != logical_field:
                log.debug(f"Standardizing column '{found_col}' to '{logical_field}'")
                df = df.rename(columns={found_col: logical_field})
        columns_to_rename: Dict[str, str] = {}
        for col, standard in standard_mappings.items():
            if col in df.columns and standard not in df.columns:
                columns_to_rename[col] = standard
        if columns_to_rename:
            df = df.rename(columns=columns_to_rename)
            log.debug(f"Applied custom column mappings: {columns_to_rename}")
        if len(df.columns) != len(set(df.columns)):
            log.debug("Removing duplicate columns")
            df = df.loc[:, ~df.columns.duplicated()]
        return df

    def save_results(self) -> None:
        try:
            if self.results is None:
                log.warn("No results to save")
                return
            results_to_save = self.results.copy()
            if results_to_save.empty:
                log.warn("Result set is empty; saving file with headers only")
            sort_col = "P_FDR" if "P_FDR" in results_to_save.columns else "P"
            if sort_col in results_to_save.columns:
                results_to_save = results_to_save.sort_values(sort_col)
            for col in results_to_save.columns:
                if "OR" in col or "Odds_Ratio" in col:
                    results_to_save[col] = results_to_save[col].apply(
                        lambda x: (
                            f"{float(x):.3f}" if isinstance(x, (float, int)) else x
                        )
                    )
                elif any(
                    p_val in col
                    for p_val in ["P_value", "P_FDR", "P", "Adjusted_P_value"]
                ):
                    results_to_save[col] = results_to_save[col].apply(
                        lambda x: (
                            f"{float(x):.2e}" if isinstance(x, (float, int)) else x
                        )
                    )
            directory = os.path.dirname(self.output_file)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                log.debug(f"Created output directory: {directory}")
            log.info(f"Saving results to {self.output_file}")
            ext = os.path.splitext(self.output_file)[1].lower()
            if ext == ".csv":
                results_to_save.to_csv(self.output_file, index=False)
            elif ext in [".tsv", ".txt"]:
                results_to_save.to_csv(self.output_file, sep="\t", index=False)
            else:
                results_to_save.to_csv(self.output_file, index=False)
            log.success(f"Results successfully saved to {self.output_file}")
        except Exception as e:
            log.error(f"Error saving results: {e}")
            sys.exit(1)


options = [
    OptionConfig(flags=["-i", "--input"], type=str, required=True),
    OptionConfig(flags=["-c", "--column"], type=str, required=False),
    OptionConfig(
        flags=["-d", "--dataset"],
        type=str,
        default="KEGG",
        required=False,
        choices=["KEGG", "GO_biological", "GO_molecular", "GO_cellular"],
    ),
    OptionConfig(flags=["-o", "--output"], type=str, required=True),
    OptionConfig(flags=["-pv", "--pvalue"], type=float, default=0.05, required=False),
    OptionConfig(
        flags=["-u", "--include_genes"], type=bool, default=False, required=False
    ),
    OptionConfig(flags=["-p", "--plot"], type=str, default=None, required=False),
    OptionConfig(flags=["-n", "--top_n"], type=int, default=20, required=False),
    OptionConfig(flags=["-a", "--var"], type=str, default=None, required=False),
    OptionConfig(
        flags=["-t", "--option"], type=int, default=1, required=False, choices=[1, 2]
    ),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="EnrichmentORA")
    opt = framework.run()
    analyzer = EnrichmentORA(
        input_file=opt.input,
        gene_column=opt.column,
        target_dataset=opt.dataset,
        output_file=opt.output,
        pvalue_cutoff=opt.pvalue,
        include_genes=opt.include_genes,
        plot=opt.plot,
        top_n=opt.top_n,
        var=opt.var,
        plot_option=opt.option,
    )
    results = analyzer.analyze()
