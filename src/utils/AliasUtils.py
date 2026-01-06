#!/usr/bin/env python
# Import required modules
import numpy as np
import pandas as pd
import re
from scipy import stats
from scipy.stats import norm
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Set
from .LoggingUtils import log


class AliasUtils:
    _alias_cache = {}
    STANDARD_SEPARATORS = ["", "_", ".", "-", " "]
    STANDARD_ID_SUFFIXES = ["ID", "id", "Id", "identifier", "Identifier", "IDENTIFIER"]
    ALIASES = {
        "3' UTR": None,
        "5' UTR": None,
        "A1": None,
        "A2": None,
        "ALT": None,
        "BIOTYPE": None,
        "BP": None,
        "CGID": None,
        "CHR": None,
        "CI": None,
        "CM": None,
        "COEF": None,
        "EAF": None,
        "END": None,
        "EWAS": None,
        "Exon": None,
        "Father": None,
        "FID": None,
        "GENE": None,
        "GENE_ID": None,
        "Genotype": None,
        "HWE": None,
        "IID": None,
        "INFO": None,
        "Intergenic": None,
        "Intron": None,
        "MAF": None,
        "Metadata": None,
        "Methylation": None,
        "Mother": None,
        "NEAREST_GENE": None,
        "N": None,
        "NEAREST_GENE_DIST": None,
        "P": None,
        "P_BACON": None,
        "P_BONF": None,
        "P_FDR": None,
        "P_HOLM": None,
        "Phenotype": None,
        "ProbeList": None,
        "Promoter": None,
        "Protein": None,
        "REF": None,
        "RSID": None,
        "SampleList": None,
        "SE": None,
        "Sex": None,
        "START": None,
        "STRAND": None,
        "T-STAT": None,
        "Transcript": None,
        "TSS": None,
        "TSS_DIST": None,
        "Z": None,
    }

    @staticmethod
    def _generate_utr_aliases(number: int) -> List[str]:
        prefixes = [str(number), {3: "three", 5: "five"}[number].title()]
        prefixes.extend([p.upper() for p in prefixes])
        primes = ["'", "prime", "Prime", "PRIME"]
        utrs = ["UTR", "utr"]
        return AliasUtils._generate_compound_aliases(
            prefixes, [f"{p}_{u}" for p in primes for u in utrs]
        )

    @staticmethod
    def generate_3utr_aliases() -> List[str]:
        return AliasUtils._generate_utr_aliases(3)

    @staticmethod
    def generate_5utr_aliases() -> List[str]:
        return AliasUtils._generate_utr_aliases(5)

    @staticmethod
    def generate_a1_aliases() -> List[str]:
        prefixes = [
            "A",
            "a",
            "Allele",
            "allele",
            "ALLELE",
            "Effect",
            "effect",
            "EFFECT",
            "Coded",
            "coded",
            "CODED",
        ]
        ones = ["1", "One", "one", "ONE"]
        suffixes = ["", "_allele", "_Allele", "_ALLELE", "Allele", "allele", "ALLELE"]
        aliases = set()
        aliases.add("EA")
        aliases.add("ea")
        for prefix in ["A", "a"]:
            for one in ["1"]:
                aliases.add(f"{prefix}{one}")
        for prefix in prefixes:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for one in ones:
                    aliases.add(f"{prefix}{sep}{one}")
        for prefix in ["Effect", "effect", "EFFECT", "Coded", "coded", "CODED"]:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for allele in ["Allele", "allele", "ALLELE"]:
                    for one in ones:
                        aliases.add(f"{prefix}{sep}{allele}{sep}{one}")
                        aliases.add(f"{prefix}{sep}{one}{sep}{allele}")
        for prefix in ["A", "a"]:
            for one in ["1"]:
                for suffix in suffixes:
                    if suffix:
                        aliases.add(f"{prefix}{one}{suffix}")
        return sorted(aliases)

    @staticmethod
    def generate_a2_aliases() -> List[str]:
        prefixes = [
            "A",
            "a",
            "Allele",
            "allele",
            "ALLELE",
            "Other",
            "other",
            "OTHER",
            "NonEffect",
            "noneffect",
            "NONEFFECT",
            "Non_Effect",
            "non_effect",
            "NON_EFFECT",
            "NonCoded",
            "noncoded",
            "NONCODED",
            "Non_Coded",
            "non_coded",
            "NON_CODED",
        ]
        twos = ["2", "Two", "two", "TWO"]
        suffixes = ["", "_allele", "_Allele", "_ALLELE", "Allele", "allele", "ALLELE"]
        aliases = set()
        for prefix in ["A", "a"]:
            for two in ["2"]:
                aliases.add(f"{prefix}{two}")
        for prefix in prefixes:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for two in twos:
                    aliases.add(f"{prefix}{sep}{two}")
        for prefix in [
            "Other",
            "other",
            "OTHER",
            "NonEffect",
            "noneffect",
            "NONEFFECT",
            "Non_Effect",
            "non_effect",
            "NON_EFFECT",
            "NonCoded",
            "noncoded",
            "NONCODED",
            "Non_Coded",
            "non_coded",
            "NON_CODED",
        ]:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for allele in ["Allele", "allele", "ALLELE"]:
                    for two in twos:
                        aliases.add(f"{prefix}{sep}{allele}{sep}{two}")
                        aliases.add(f"{prefix}{sep}{two}{sep}{allele}")
        for prefix in ["A", "a"]:
            for two in ["2"]:
                for suffix in suffixes:
                    if suffix:
                        aliases.add(f"{prefix}{two}{suffix}")
        return sorted(aliases)

    @staticmethod
    def generate_alt_aliases() -> List[str]:
        alt_forms = ["ALT", "alt", "Alt"]
        alternate_forms = ["Alternate", "alternate", "ALTERNATE"]
        allele_forms = ["Allele", "allele", "ALLELE"]
        aliases = set()
        aliases.update(alt_forms)
        for alt in alt_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for allele in allele_forms:
                    aliases.add(f"{alt}{sep}{allele}")
        for alternate in alternate_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for allele in allele_forms:
                    aliases.add(f"{alternate}{sep}{allele}")
        aliases.update(alternate_forms)
        return sorted(aliases)

    @staticmethod
    def generate_biotype_aliases() -> List[str]:
        gene_forms = ["Gene", "gene", "GENE"]
        type_forms = ["Type", "type", "TYPE"]
        biotype_forms = ["Biotype", "biotype", "BIOTYPE"]
        aliases = set()
        for biotype in biotype_forms:
            aliases.add(biotype)
            for gene in gene_forms:
                for sep in AliasUtils.STANDARD_SEPARATORS:
                    aliases.add(f"{gene}{sep}{biotype}")

        for type_ in type_forms:
            for gene in gene_forms:
                for sep in AliasUtils.STANDARD_SEPARATORS:
                    aliases.add(f"{gene}{sep}{type_}")
        return sorted(aliases)

    @staticmethod
    def generate_bp_aliases() -> List[str]:
        bp_forms = ["BP", "bp", "Bp", "bP"]
        pos_forms = ["POS", "pos", "Pos", "pOS"]
        position_forms = ["position", "Position", "POSITION"]
        mapinfo_forms = ["MAPINFO", "mapinfo", "MapInfo", "Mapinfo", "mapInfo"]
        aliases = set()
        aliases.update(bp_forms)
        aliases.update(pos_forms)
        aliases.update(position_forms)
        aliases.update(mapinfo_forms)
        for bp in bp_forms + pos_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for position in position_forms:
                    aliases.add(f"{bp}{sep}{position}")
        for mapinfo in mapinfo_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for bp in bp_forms:
                    aliases.add(f"{mapinfo}{sep}{bp}")
                    aliases.add(f"{bp}{sep}{mapinfo}")
        return sorted(aliases)

    @staticmethod
    def generate_cgid_aliases() -> List[str]:
        cgid_forms = ["CGID", "cgid", "Cgid", "cgiD"]
        probe_forms = ["probe", "Probe", "PROBE"]
        ilmn_forms = ["IlmnID", "ilmnID", "ILMNID", "ilmnid", "Ilmn_Id", "Ilmn_Id"]
        cpg_forms = [
            "CpG",
            "cpg",
            "CPG",
            "CpG_site",
            "cpg_site",
            "CPG_SITE",
            "CpGsite",
            "cpgsite",
            "CPGsite",
            "CPGSite",
            "CpG_Site",
            "cpg_Site",
            "CPG_Site",
        ]
        site_forms = ["site", "Site", "SITE"]
        aliases = set()
        aliases.update(cgid_forms)
        for cg in ["CG", "cg", "Cg"]:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{cg}{sep}{idf}")
        for probe in probe_forms:
            aliases.add(probe)
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{probe}{sep}{idf}")
        aliases.update(ilmn_forms)
        aliases.update(cpg_forms)
        for cpg in ["CpG", "cpg", "CPG"]:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for site in site_forms:
                    aliases.add(f"{cpg}{sep}{site}")
        return sorted(aliases)

    @staticmethod
    def generate_chr_aliases() -> List[str]:
        chr_forms = ["CHR", "chr", "Chr", "cHR"]
        chromosome_forms = ["chromosome", "Chromosome", "CHROMOSOME"]
        aliases = set()
        aliases.update(chr_forms)
        aliases.update(chromosome_forms)
        for chr_ in chr_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for chrom in chromosome_forms:
                    aliases.add(f"{chr_}{sep}{chrom}")
                    for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                        if idf:
                            aliases.add(f"{chr_}{sep}{chrom}{idf}")
        for chrom in chromosome_forms:
            for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                if idf:
                    aliases.add(f"{chrom}{idf}")
        return sorted(aliases)

    @staticmethod
    def generate_ci_aliases() -> List[str]:
        ci_forms = ["CI", "ci", "Ci"]
        confidence_forms = ["confidence", "Confidence", "CONFIDENCE"]
        interval_forms = ["interval", "Interval", "INTERVAL"]
        aliases = set()
        aliases.update(ci_forms)
        for ci in ci_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for conf in confidence_forms:
                    aliases.add(f"{ci}{sep}{conf}")
                for interval in interval_forms:
                    aliases.add(f"{ci}{sep}{interval}")
        return sorted(aliases)

    @staticmethod
    def generate_cm_aliases() -> List[str]:
        cm_forms = ["CM", "cm", "cM", "Cm", "cM", "c_m"]
        centimorgan_forms = ["centimorgan", "Centimorgan", "CENTIMORGAN"]
        genetic_distance_forms = [
            "genetic_distance",
            "Genetic_Distance",
            "GENETIC_DISTANCE",
            "geneticdistance",
            "GeneticDistance",
            "GENETICDISTANCE",
        ]
        aliases = set()
        aliases.update(cm_forms)
        aliases.update(centimorgan_forms)
        aliases.update(genetic_distance_forms)
        for cm in cm_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for centi in centimorgan_forms:
                    aliases.add(f"{cm}{sep}{centi}")
                for gd in genetic_distance_forms:
                    aliases.add(f"{cm}{sep}{gd}")
        for centi in centimorgan_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for gd in genetic_distance_forms:
                    aliases.add(f"{centi}{sep}{gd}")
        return sorted(aliases)

    @staticmethod
    def generate_coef_aliases() -> List[str]:
        coef_forms = [
            "COEF",
            "coef",
            "Coef",
            "COEFFICIENT",
            "coefficient",
            "Coefficient",
            "coefficient_value",
            "Coefficient_Value",
            "coefficientValue",
            "CoefficientValue",
            "coef_value",
            "Coef_Value",
            "coefValue",
            "CoefValue",
            "coefficientval",
            "coefficient_val",
            "coef_val",
            "coefval",
        ]
        beta_forms = ["Beta", "beta", "BETA"]
        aliases = set()
        aliases.update(coef_forms)
        aliases.update(beta_forms)
        for coef in coef_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for beta in beta_forms:
                    aliases.add(f"{coef}{sep}{beta}")
                    aliases.add(f"{beta}{sep}{coef}")
        for coef in coef_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for coeff in ["Coefficient", "coefficient", "COEFFICIENT"]:
                    aliases.add(f"{coef}{sep}{coeff}")
        return sorted(aliases)

    @staticmethod
    def generate_eaf_aliases() -> List[str]:
        eaf_forms = ["EAF", "eaf", "Eaf"]
        effect_allel_forms = [
            "Effect_Allele",
            "effect_allele",
            "EFFECT_ALLELE",
            "EffectAllele",
            "effectAllele",
            "EFFECTALLELE",
        ]
        freq_forms = ["Freq", "freq", "FREQ", "Frequency", "frequency", "FREQUENCY"]
        aliases = set()
        aliases.update(eaf_forms)
        for effect in effect_allel_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for freq in freq_forms:
                    aliases.add(f"{effect}{sep}{freq}")

        return sorted(aliases if False else aliases) if False else sorted(aliases)

    @staticmethod
    def generate_end_aliases() -> List[str]:
        end_forms = ["END", "end", "End"]
        builds = [
            "",
            "_hg38",
            "_HG38",
            "_hg19",
            "_HG19",
            "_grch38",
            "_GRCH38",
            "_grch37",
            "_GRCH37",
        ]
        aliases = set()
        for end in end_forms:
            for build in builds:
                aliases.add(f"{end}{build}")
        for end in end_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for build in [
                    "hg38",
                    "HG38",
                    "hg19",
                    "HG19",
                    "grch38",
                    "GRCH38",
                    "grch37",
                    "GRCH37",
                ]:
                    aliases.add(f"{end}{sep}{build}")
        return sorted(aliases)

    @staticmethod
    def generate_ewas_aliases() -> List[str]:
        ewas_forms = ["EWAS", "ewas", "Ewas", "EWas"]
        aliases = set()
        aliases.update(ewas_forms)
        for base in ["epigenome", "Epigenome", "EPIGENOME"]:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for assoc in ["wide", "Wide", "WIDE"]:
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for assoc2 in ["association", "Association", "ASSOCIATION"]:
                            for sep3 in AliasUtils.STANDARD_SEPARATORS:
                                for study in ["study", "Study", "STUDY"]:
                                    alias = f"{base}{sep1}{assoc}{sep2}{assoc2}{sep3}{study}"
                                    aliases.add(alias)
        return sorted(aliases)

    @staticmethod
    def generate_exon_aliases() -> List[str]:
        base_forms = ["EXON", "exon", "Exon"]
        return AliasUtils._generate_standard_aliases(
            base_forms, AliasUtils.STANDARD_ID_SUFFIXES
        )

    @staticmethod
    def generate_father_aliases() -> List[str]:
        father_forms = ["father", "Father", "FATHER"]
        aliases = set()
        aliases.update(father_forms)
        for paternal in ["paternal", "Paternal", "PATERNAL"]:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{paternal}{sep}{idf}")
        return sorted(aliases)

    @staticmethod
    def generate_fid_aliases() -> List[str]:
        fid_forms = ["FID", "fid", "Fid", "FAMILYID", "familyid", "FamilyID"]
        aliases = set()
        aliases.update(fid_forms)
        for family in ["Family", "family", "FAMILY"]:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{family}{sep}{idf}")
        return sorted(aliases)

    @staticmethod
    def generate_gene_aliases() -> List[str]:
        gene_forms = ["GENE", "gene", "Gene"]
        name_forms = ["name", "Name", "NAME"]
        symbol_forms = ["symbol", "Symbol", "SYMBOL"]
        db_aliases = [
            "gene_symbol",
            "Gene_Symbol",
            "GENE_SYMBOL",
            "genesymbol",
            "GeneSymbol",
            "GENESYMBOL",
            "approved_symbol",
            "Approved_Symbol",
            "APPROVED_SYMBOL",
            "hgnc_symbol",
            "HGNC_Symbol",
            "HGNC_SYMBOL",
            "hgncsymbol",
            "HGNCSymbol",
            "HGNCSYMBOL",
            "official_gene_symbol",
            "Official_Gene_Symbol",
            "OFFICIAL_GENE_SYMBOL",
            "official_symbol",
            "Official_Symbol",
            "OFFICIAL_SYMBOL",
        ]
        aliases = set()
        aliases.update(gene_forms)
        for gene in gene_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for name in name_forms:
                    aliases.add(f"{gene}{sep}{name}")
                for symbol in symbol_forms:
                    aliases.add(f"{gene}{sep}{symbol}")
        aliases.update(db_aliases)
        return sorted(aliases)

    @staticmethod
    def generate_gene_id_aliases() -> List[str]:
        gene_forms = ["GENE", "gene", "Gene"]
        db_prefixes = [
            "ensembl",
            "Ensembl",
            "ENSEMBL",
            "entrez_gene",
            "Entrez_Gene",
            "ENTREZ_GENE",
            "entrez",
            "Entrez",
            "ENTREZ",
            "ncbi_gene",
            "NCBI_GENE",
            "ncbiGene",
            "NCBIGene",
            "NCBIgene",
            "gene_stable",
            "Gene_Stable",
            "GENE_STABLE",
            "geneStable",
            "GeneStable",
            "GENESTABLE",
        ]
        aliases = set()
        for gene in gene_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{gene}{sep}{idf}")
        for gene in gene_forms:
            for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                aliases.add(f"{gene}{idf}")
        for prefix in db_prefixes:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{prefix}{sep}{idf}")
        return sorted(aliases)

    @staticmethod
    def generate_genotype_aliases() -> List[str]:
        genotype_forms = [
            "GENOTYPE",
            "genotype",
            "Genotype",
            "genotypes",
            "Genotypes",
            "GENOTYPES",
        ]
        gt_forms = ["GT", "gt", "Gt"]
        suffixes = [
            "",
            "array",
            "Array",
            "ARRAY",
            "matrix",
            "Matrix",
            "MATRIX",
            "call",
            "Call",
            "CALL",
            "calls",
            "Calls",
            "CALLS",
            "data",
            "Data",
            "DATA",
            "value",
            "Value",
            "VALUE",
            "values",
            "Values",
            "VALUES",
        ]
        aliases = set(genotype_forms)
        for base in genotype_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for suffix in suffixes:
                    if suffix:
                        aliases.add(f"{base}{sep}{suffix}")
        for base in gt_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for suffix in suffixes:
                    if suffix:
                        aliases.add(f"{base}{sep}{suffix}")
        return sorted(aliases)

    @staticmethod
    def generate_hwe_aliases() -> List[str]:
        hwe_forms = ["HWE", "hwe", "Hwe"]
        parts = [
            ["Hardy", "hardy", "HARDY"],
            ["Weinberg", "weinberg", "WEINBERG"],
            ["Equilibrium", "equilibrium", "EQUILIBRIUM"],
        ]
        aliases = set(hwe_forms)
        for hardy in parts[0]:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for weinberg in parts[1]:
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for equilibrium in parts[2]:
                            alias = f"{hardy}{sep1}{weinberg}{sep2}{equilibrium}"
                            aliases.add(alias)
                    aliases.add(f"{hardy}{sep1}{weinberg}")
        for hardy in parts[0]:
            for weinberg in parts[1]:
                for equilibrium in parts[2]:
                    aliases.add(f"{hardy}{weinberg}{equilibrium}")
                aliases.add(f"{hardy}{weinberg}")
        return sorted(aliases)

    @staticmethod
    def generate_iid_aliases() -> List[str]:
        iid_forms = ["IID", "iid", "Iid"]
        individual_forms = ["Individual", "individual", "INDIVIDUAL"]
        sample_forms = ["Sample", "sample", "SAMPLE"]
        aliases = set(iid_forms)
        aliases.update(individual_forms)
        aliases.update(sample_forms)
        for base in individual_forms + sample_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{base}{sep}{idf}")
        return sorted(aliases)

    @staticmethod
    def generate_info_aliases() -> List[str]:
        info_forms = ["INFO", "info", "Info"]
        score_forms = ["score", "Score", "SCORE"]
        aliases = set(info_forms)
        for info in info_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for score in score_forms:
                    aliases.add(f"{info}{sep}{score}")
        return sorted(aliases)

    @staticmethod
    def generate_intergenic_aliases() -> List[str]:
        intergenic_forms = ["INTERGENIC", "intergenic", "Intergenic"]
        region_forms = ["Region", "region", "REGION"]
        aliases = set(intergenic_forms)
        for intergenic in intergenic_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for region in region_forms:
                    aliases.add(f"{intergenic}{sep}{region}")
        return sorted(aliases)

    @staticmethod
    def generate_intron_aliases() -> List[str]:
        intron_forms = ["INTRON", "intron", "Intron"]
        aliases = set()
        for intron in intron_forms:
            aliases.add(intron)
            for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                if idf:
                    aliases.add(f"{intron}{idf}")
        for intron in intron_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{intron}{sep}{idf}")
        return sorted(aliases)

    @staticmethod
    def generate_maf_aliases() -> List[str]:
        maf_forms = ["MAF", "maf", "Maf"]
        minor_forms = ["Minor", "minor", "MINOR"]
        allele_forms = ["Allele", "allele", "ALLELE"]
        frequency_forms = ["Frequency", "frequency", "FREQUENCY"]
        aliases = set(maf_forms)
        for minor in minor_forms:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for allele in allele_forms:
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for freq in frequency_forms:
                            aliases.add(f"{minor}{sep1}{allele}{sep2}{freq}")
        return sorted(aliases)

    @staticmethod
    def generate_metadata_aliases() -> List[str]:
        meta_forms = ["METADATA", "metadata", "Metadata", "MetaData"]
        aliases = set(meta_forms)
        for meta in ["meta", "Meta", "META"]:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for data in ["data", "Data", "DATA"]:
                    aliases.add(f"{meta}{sep}{data}")
        return sorted(aliases)

    @staticmethod
    def generate_methylation_aliases() -> List[str]:
        methylation_forms = ["METHYLATION", "methylation", "Methylation"]
        meth_forms = ["meth", "Meth", "METH"]
        value_bases = ["beta", "Beta", "BETA", "M", "m"]
        suffixes = [
            "value",
            "Value",
            "VALUE",
            "values",
            "Values",
            "VALUES",
            "vals",
            "Vals",
            "VALS",
        ]
        aliases = set(methylation_forms)
        aliases.update(meth_forms)
        for base in value_bases:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for suffix in suffixes:
                    aliases.add(f"{base}{sep}{suffix}")
        for base in value_bases:
            for s in ["s", "S"]:
                aliases.add(f"{base}{s}")
        for methyl in ["Methylation", "methylation", "METHYLATION"]:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for suffix in ["values", "Values", "VALUES"]:
                    aliases.add(f"{methyl}{sep}{suffix}")
        return sorted(aliases)

    @staticmethod
    def generate_mother_aliases() -> List[str]:
        mother_forms = ["mother", "Mother", "MOTHER"]
        aliases = set()
        aliases.update(mother_forms)
        for maternal in ["maternal", "Maternal", "MATERNAL"]:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{maternal}{sep}{idf}")
        return sorted(aliases)

    @staticmethod
    def generate_nearest_gene_aliases() -> List[str]:
        prefixes = [
            "nearest",
            "Nearest",
            "NEAREST",
            "closest",
            "Closest",
            "CLOSEST",
            "next",
            "Next",
            "NEXT",
        ]
        gene_forms = ["gene", "Gene", "GENE"]
        aliases = set()
        for prefix in prefixes:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for gene in gene_forms:
                    aliases.add(f"{prefix}{sep1}{gene}")
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                            aliases.add(f"{prefix}{sep1}{gene}{sep2}{idf}")
        return sorted(aliases)

    @staticmethod
    def generate_n_aliases() -> List[str]:
        n_forms = ["N", "n"]
        sample_forms = ["Sample", "sample", "SAMPLE"]
        count_forms = ["Count", "count", "COUNT"]
        size_forms = ["Size", "size", "SIZE"]
        observation_forms = ["Observations", "observations", "OBSERVATIONS"]
        obs_forms = ["Obs", "obs", "OBS"]
        aliases = set(n_forms)
        for sample in sample_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for size in size_forms:
                    aliases.add(f"{sample}{sep}{size}")
        for sample in sample_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for count in count_forms:
                    aliases.add(f"{sample}{sep}{count}")
        for n in n_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                aliases.add(f"{n}{sep}samples")
                aliases.add(f"{n}{sep}samples_total")
                aliases.add(f"{n}{sep}total")
        for n in n_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for obs in observation_forms + obs_forms:
                    aliases.add(f"{n}{sep}{obs}")
        return sorted(aliases)

    @staticmethod
    def generate_nearest_gene_dist_aliases() -> List[str]:
        prefixes = [
            "nearest",
            "Nearest",
            "NEAREST",
            "closest",
            "Closest",
            "CLOSEST",
            "next",
            "Next",
            "NEXT",
            "Distance_to",
            "distance_to",
            "DistanceTo",
            "distanceto",
            "DISTANCE_TO",
        ]
        gene_forms = ["gene", "Gene", "GENE"]
        dist_forms = ["dist", "Dist", "DIST", "distance", "Distance", "DISTANCE"]
        aliases = set()
        for prefix in prefixes:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for gene in gene_forms:
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for dist in dist_forms:
                            aliases.add(f"{prefix}{sep1}{gene}{sep2}{dist}")
        return sorted(aliases)

    @staticmethod
    def generate_p_aliases() -> List[str]:
        p_forms = ["P", "p"]
        value_forms = ["value", "Value", "VALUE", "val", "Val", "VAL"]
        aliases = set(p_forms)
        for p in p_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for value in value_forms:
                    aliases.add(f"{p}{sep}{value}")
        for p in p_forms:
            for value in ["Value", "VALUE", "val", "Val", "VAL"]:
                aliases.add(f"{p}{value}")
        return sorted(aliases)

    @staticmethod
    def _generate_p_correction_aliases(
        correction_forms: Sequence[str], method_name: str
    ) -> List[str]:
        corrected_forms = ["corrected", "Corrected", "CORRECTED"]
        adjusted_forms = ["adjusted", "Adjusted", "ADJUSTED"]
        p_forms = ["P", "p"]
        adj_prefixes = ["adj", "Adj", "ADJ"]

        aliases = set(correction_forms)

        for p in p_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for corr in correction_forms:
                    aliases.add(f"{p}{sep}{corr}")

        for corr in correction_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for corrected in corrected_forms:
                    aliases.add(f"{corr}{sep}{corrected}")
                for adjusted in adjusted_forms:
                    aliases.add(f"{corr}{sep}{adjusted}")

        for pref in adj_prefixes:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for corr in correction_forms:
                    aliases.add(f"{pref}{sep1}{corr}")
                for p in p_forms:
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for corr in correction_forms:
                            aliases.add(f"{pref}{sep1}{p}{sep2}{corr}")
                            aliases.add(f"{p}{sep1}{pref}{sep2}{corr}")
                            aliases.add(f"{p}{sep1}{corr}{sep2}{pref}")

        for adj in adjusted_forms:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for corr in correction_forms:
                    aliases.add(f"{adj}{sep1}{corr}")
                for p in p_forms:
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for corr in correction_forms:
                            aliases.add(f"{adj}{sep1}{p}{sep2}{corr}")
                            aliases.add(f"{p}{sep1}{adj}{sep2}{corr}")
                            aliases.add(f"{p}{sep1}{corr}{sep2}{adj}")

        return sorted(aliases)

    @staticmethod
    def generate_p_bacon_aliases() -> List[str]:
        bacon_forms = ["BACON", "bacon", "Bacon"]
        empirical_forms = ["empirical", "Empirical", "EMPIRICAL"]
        corrected_forms = ["corrected", "Corrected", "CORRECTED"]
        p_forms = ["P", "p"]
        value_forms = ["value", "Value", "VALUE", "val", "Val", "VAL"]

        aliases = set()

        for bacon in bacon_forms:
            aliases.add(bacon)
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for corr in corrected_forms:
                    aliases.add(f"{bacon}{sep}{corr}")
                    for p in p_forms:
                        aliases.add(f"{bacon}{sep}{corr}{sep}{p}")
                        for value in value_forms:
                            aliases.add(f"{bacon}{sep}{corr}{sep}{p}{sep}{value}")
                            aliases.add(
                                f"{bacon}{sep}{corr}{sep}{p}{sep}{value.lower()}"
                            )
                            aliases.add(
                                f"{bacon}{sep}{corr}{sep}{p.lower()}{sep}{value}"
                            )
                            aliases.add(
                                f"{bacon}{sep}{corr}{sep}{p.lower()}{sep}{value.lower()}"
                            )

        for p in ["p", "P"]:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for value in ["value", "val", "Value", "Val", "VALUE", "VAL"]:
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for bacon in bacon_forms:
                            aliases.add(f"{p}{sep1}{value}{sep2}{bacon}")
                            aliases.add(f"{p}{sep1}{value}{sep2}{bacon}_corrected")
                            aliases.add(f"{p}{sep1}{value}{sep2}{bacon}{sep2}corrected")
                        aliases.add(f"{p}{value}{sep2}{bacon}")
                        aliases.add(f"{p}{value}{sep2}{bacon}_corrected")
                        aliases.add(f"{p}{value}{sep2}{bacon}{sep2}corrected")

        for p in ["p", "P"]:
            for sep in ["."]:
                for bacon in bacon_forms:
                    aliases.add(f"{p}{sep}{bacon}")
                    for corr in corrected_forms:
                        aliases.add(f"{p}{sep}{bacon}{sep}{corr}")
                        aliases.add(f"{p}{sep}{bacon}_{corr}")

        for empirical in empirical_forms:
            aliases.add(empirical)
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for p in p_forms:
                    aliases.add(f"{empirical}{sep1}{p}")
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for value in value_forms:
                            aliases.add(f"{empirical}{sep1}{p}{sep2}{value}")
                            aliases.add(f"{empirical}{sep1}{p.lower()}{sep2}{value}")
                            aliases.add(f"{empirical}{sep1}{p}{sep2}{value.lower()}")
                            aliases.add(
                                f"{empirical}{sep1}{p.lower()}{sep2}{value.lower()}"
                            )
                for value in value_forms:
                    aliases.add(f"{empirical}{sep1}{value}")
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for corr in corrected_forms:
                            aliases.add(f"{empirical}{sep1}{value}{sep2}{corr}")
                            aliases.add(f"{empirical}{sep1}{value}{sep2}{corr.lower()}")

        for empirical in empirical_forms:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for p in p_forms:
                    aliases.add(f"{empirical}{sep1}{p}")
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for corr in corrected_forms:
                            aliases.add(f"{empirical}{sep1}{p}{sep2}{corr}")
                            aliases.add(f"{empirical}{sep1}{p}{sep2}{corr.lower()}")
                for value in value_forms:
                    aliases.add(f"{empirical}{sep1}{value}")

        adj_prefixes = [
            "adj",
            "Adj",
            "ADJ",
            "adjusted",
            "Adjusted",
            "ADJUSTED",
            "adjust",
            "Adjust",
            "ADJUST",
        ]
        for adj in adj_prefixes:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for p in p_forms:
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for bacon in bacon_forms:
                            aliases.add(f"{adj}{sep1}{p}{sep2}{bacon}")
                            for val in value_forms:
                                aliases.add(f"{adj}{sep1}{p}{sep2}{val}{sep2}{bacon}")
                                aliases.add(f"{adj}{sep1}{p}{sep2}{bacon}{sep2}{val}")
                            for corr in corrected_forms:
                                aliases.add(f"{adj}{sep1}{p}{sep2}{bacon}{sep2}{corr}")
                                aliases.add(f"{adj}{sep1}{p}{sep2}{corr}{sep2}{bacon}")
                        for bacon in bacon_forms:
                            for corr in corrected_forms:
                                aliases.add(f"{adj}{sep1}{p}{sep2}{bacon}{sep2}{corr}")
                                aliases.add(f"{adj}{sep1}{p}{sep2}{bacon}_{corr}")

        return sorted(set(a for a in aliases if a))

    @staticmethod
    def generate_p_bonf_aliases() -> List[str]:
        return AliasUtils._generate_p_correction_aliases(
            ["BONF", "bonf", "Bonferroni", "bonferroni"], "bonferroni"
        )

    @staticmethod
    def generate_p_fdr_aliases() -> List[str]:
        return AliasUtils._generate_p_correction_aliases(["FDR", "fdr"], "fdr")

    @staticmethod
    def generate_p_holm_aliases() -> List[str]:
        return AliasUtils._generate_p_correction_aliases(
            ["HOLM", "holm", "Holm"], "holm"
        )

    @staticmethod
    def generate_phenotype_aliases() -> List[str]:
        pheno_forms = ["PHENOTYPE", "phenotype", "Phenotype", "PHENO", "pheno"]
        plural_forms = [f"{p}s" if not p.endswith("s") else p for p in pheno_forms]
        aliases = set(pheno_forms + plural_forms)
        return sorted(aliases)

    @staticmethod
    def generate_probelist_aliases() -> List[str]:
        prefixes = ["probe", "Probe", "PROBE", "cpg", "CpG", "Cpg", "CPG"]
        suffixes = [
            "list",
            "List",
            "LIST",
            *AliasUtils.STANDARD_ID_SUFFIXES,
            "identifier",
            "Identifier",
            "IDENTIFIER",
            "identifiers",
            "Identifiers",
            "IDENTIFIERS",
            "list",
            "List",
            "LIST",
            "s",
            "S",
        ]
        aliases = set()
        for prefix in prefixes:
            aliases.add(prefix)
            for suffix in suffixes:
                aliases.add(f"{prefix}{suffix}")
                aliases.add(f"{prefix}_{suffix}")
        return sorted(aliases)

    @staticmethod
    def generate_promoter_aliases() -> List[str]:
        promoter_forms = ["PROMOTER", "promoter", "Promoter"]
        aliases = set()
        for promoter in promoter_forms:
            aliases.add(promoter)
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{promoter}{sep}{idf}")
        return sorted(aliases)

    @staticmethod
    def generate_protein_aliases() -> List[str]:
        protein_forms = ["PROTEIN", "protein", "Protein"]

        aliases = set(protein_forms)

        for protein in protein_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{protein}{sep}{idf}")

        return sorted(aliases)

    @staticmethod
    def generate_ref_aliases() -> List[str]:
        ref_forms = ["REF", "ref", "Ref"]
        reference_forms = ["Reference", "reference", "REFERENCE"]
        allele_forms = ["Allele", "allele", "ALLELE"]

        aliases = set()

        aliases.update(ref_forms)

        for ref in ref_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for allele in allele_forms:
                    aliases.add(f"{ref}{sep}{allele}")

        for reference in reference_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for allele in allele_forms:
                    aliases.add(f"{reference}{sep}{allele}")

        aliases.update(reference_forms)

        return sorted(aliases)

    @staticmethod
    def generate_rsid_aliases() -> List[str]:
        rsid_forms = ["rs", "RS"]
        snp_forms = ["SNP", "snp", "Snp"]
        variant_forms = ["variant", "Variant", "VARIANT"]
        numbers = ["number", "Number", "NUMBER", "num", "Num", "NUM"]
        plurals = ["", "s", "S"]

        full_forms = [
            "SingleNucleotidePolymorphism",
            "Single_Nucleotide_Polymorphism",
            "Single-Nucleotide-Polymorphism",
            "Single Nucleotide Polymorphism",
            "single_nucleotide_polymorphism",
            "single-nucleotide-polymorphism",
            "single nucleotide polymorphism",
        ]

        aliases = set()
        aliases.update(rsid_forms)
        aliases.update(snp_forms)
        aliases.update(variant_forms)

        for base in rsid_forms + snp_forms + variant_forms:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for suf in AliasUtils.STANDARD_ID_SUFFIXES + numbers:
                    for plural in plurals:
                        if suf or plural:
                            aliases.add(f"{base}{sep1}{suf}{plural}")
            for plural in plurals:
                if plural:
                    aliases.add(f"{base}{plural}")

        for form in full_forms:
            aliases.add(form)
            if not form.endswith("s"):
                aliases.add(form + "s")
                aliases.add(form + "S")
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for suf in AliasUtils.STANDARD_ID_SUFFIXES + numbers:
                    if suf:
                        aliases.add(f"{form}{sep}{suf}")
                        if not form.endswith("s"):
                            aliases.add(f"{form}s{sep}{suf}")
                            aliases.add(f"{form}S{sep}{suf}")

        for base in rsid_forms + snp_forms + variant_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for num in numbers:
                    for plural in plurals:
                        aliases.add(f"{base}{sep}{num}{plural}")

        return sorted(aliases)

    @staticmethod
    def generate_samplelist_aliases() -> List[str]:
        sample_forms = ["Sample", "sample", "SAMPLE"]
        list_forms = ["List", "list", "LIST"]
        plural_forms = ["s", "S"]

        aliases = set()

        for sample in sample_forms:
            for listf in list_forms:
                aliases.add(f"{sample}{listf}")
                aliases.add(f"{sample}_{listf}")

        for sample in sample_forms:
            for plural in plural_forms:
                aliases.add(f"{sample}{plural}")

        for sample in sample_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{sample}{sep}{idf}")
                    for plural in plural_forms:
                        aliases.add(f"{sample}{sep}{idf}{plural}")

        for sample in sample_forms:
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for listf in list_forms:
                    for sep2 in AliasUtils.STANDARD_SEPARATORS:
                        for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                            aliases.add(f"{sample}{sep1}{listf}{sep2}{idf}")

        return sorted(aliases)

    @staticmethod
    def generate_se_aliases() -> List[str]:
        se_forms = ["SE", "se"]
        error_forms = [
            "Standard_Error",
            "standard_error",
            "StandardError",
            "standardError",
            "STANDARD_ERROR",
            "StdErr",
            "stderr",
            "STDERR",
            "STD_ERROR",
            "Std_Error",
            "std_error",
        ]
        plural_forms = ["s", "S", "es", "Es", "ES", "Errors", "errors", "ERRORS"]

        aliases = set(se_forms)

        for error in error_forms:
            aliases.add(error)
            for plural in plural_forms:
                aliases.add(f"{error}{plural}")
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for plural in plural_forms:
                    aliases.add(f"{error}{sep}{plural}")

        for se in se_forms:
            for plural in plural_forms:
                aliases.add(f"{se}{plural}")

        return sorted(aliases)

    @staticmethod
    def generate_sex_aliases() -> List[str]:
        sex_forms = ["sex", "Sex", "SEX"]
        gender_forms = ["gender", "Gender", "GENDER"]
        plural_forms = ["s", "es", "ES", "S"]
        aliases = set(sex_forms + gender_forms)

        for base in sex_forms + gender_forms:
            for plural in plural_forms:
                aliases.add(f"{base}{plural}")

        return sorted(aliases)

    @staticmethod
    def generate_start_aliases() -> List[str]:
        start_forms = ["START", "start", "Start"]
        builds = [
            "",
            "_hg38",
            "_HG38",
            "_hg19",
            "_HG19",
            "_grch38",
            "_GRCH38",
            "_grch37",
            "_GRCH37",
        ]

        aliases = set()

        for start in start_forms:
            for build in builds:
                aliases.add(f"{start}{build}")

        for start in start_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for build in [
                    "hg38",
                    "HG38",
                    "hg19",
                    "HG19",
                    "grch38",
                    "GRCH38",
                    "grch37",
                    "GRCH37",
                ]:
                    aliases.add(f"{start}{sep}{build}")

        return sorted(aliases)

    @staticmethod
    def generate_strand_aliases() -> List[str]:
        strand_forms = ["strand", "Strand", "STRAND"]
        orientation_forms = ["Orientation", "orientation", "ORIENTATION"]
        builds = [
            "hg38",
            "HG38",
            "hg19",
            "HG19",
            "grch38",
            "GRCH38",
            "grch37",
            "GRCH37",
        ]

        aliases = set(strand_forms + orientation_forms)

        for strand in strand_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for build in builds:
                    aliases.add(f"{strand}{sep}{build}")

        for orientation in orientation_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for build in builds:
                    aliases.add(f"{orientation}{sep}{build}")

        return sorted(aliases)

    @staticmethod
    def generate_tstat_aliases() -> List[str]:
        t_forms = ["T", "t"]
        stat_forms = ["statistic", "Statistic", "STATISTIC", "stat", "Stat", "STAT"]
        student_forms = [
            "student",
            "Student",
            "STUDENT",
            "student_t",
            "Student_t",
            "STUDENT_T",
            "student-t",
            "Student-T",
            "STUDENT-T",
            "student t",
            "Student t",
            "STUDENT T",
        ]
        welch_forms = [
            "welch",
            "Welch",
            "WELCH",
            "welch_t",
            "Welch_t",
            "WELCH_T",
            "welch-t",
            "Welch-T",
            "WELCH-T",
            "welch t",
            "Welch t",
            "WELCH T",
        ]

        aliases = set()

        for t in t_forms:
            aliases.add(t)
            for sep1 in AliasUtils.STANDARD_SEPARATORS:
                for stat in stat_forms:
                    aliases.add(f"{t}{sep1}{stat}")
                    aliases.add(f"{t}{sep1}{stat}istic")
                    aliases.add(f"{t}{sep1}{stat}istics")
            for stat in stat_forms:
                aliases.add(f"{t}{stat}")
                aliases.add(f"{t}{stat}istic")
                aliases.add(f"{t}{stat}istics")

        for t in t_forms:
            for stat in stat_forms:
                aliases.add(f"{t}{stat}")
                aliases.add(f"{t}{stat}istic")
                aliases.add(f"{t}{stat}istics")

        for student in student_forms:
            aliases.add(student)
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for t in t_forms:
                    aliases.add(f"{student}{sep}{t}")
                    for stat in stat_forms:
                        aliases.add(f"{student}{sep}{t}{sep}{stat}")
                        aliases.add(f"{student}{sep}{t}{sep}{stat}istic")
                        aliases.add(f"{student}{sep}{t}{sep}{stat}istics")

        for welch in welch_forms:
            aliases.add(welch)
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for t in t_forms:
                    aliases.add(f"{welch}{sep}{t}")
                    for stat in stat_forms:
                        aliases.add(f"{welch}{sep}{t}{sep}{stat}")
                        aliases.add(f"{welch}{sep}{t}{sep}{stat}istic")
                        aliases.add(f"{welch}{sep}{t}{sep}{stat}istics")

        return sorted(aliases)

    @staticmethod
    def generate_transcript_aliases() -> List[str]:
        transcript_forms = ["TRANSCRIPT", "transcript", "Transcript"]

        aliases = set(transcript_forms)

        for transcript in transcript_forms:
            for sep in AliasUtils.STANDARD_SEPARATORS:
                for idf in AliasUtils.STANDARD_ID_SUFFIXES:
                    aliases.add(f"{transcript}{sep}{idf}")

        return sorted(aliases)

    @staticmethod
    def generate_tss_aliases() -> List[str]:
        tss_forms = ["TSS", "tss"]
        full_forms = [
            "Transcription_Start_Site",
            "transcription_start_site",
            "TranscriptionStartSite",
            "transcriptionstartsite",
        ]
        plural_forms = ["s", "S", "Sites", "sites", "SITES"]
        aliases = set(tss_forms + full_forms)

        for base in tss_forms + full_forms:
            for plural in plural_forms:
                aliases.add(f"{base}{plural}")

        return sorted(aliases)

    @staticmethod
    def generate_tss_dist_aliases() -> List[str]:
        base_forms = [
            "TSS_DIST",
            "tss_dist",
            "TSS_Distance",
            "tss_distance",
            "TSSDistance",
            "TSS_Dist",
            "TSSdist",
            "tssdist",
            "Distance_to_TSS",
            "distance_to_tss",
            "DistanceToTSS",
            "distance_to_TSS",
        ]
        plural_forms = ["s", "S", "Distances", "distances", "DISTANCES"]
        aliases = set(base_forms)

        for base in base_forms:
            for plural in plural_forms:
                aliases.add(f"{base}{plural}")

        return sorted(aliases)

    @staticmethod
    def generate_z_aliases() -> List[str]:
        z_forms = [
            "Z",
            "z",
            "Z_SCORE",
            "z_score",
            "ZScore",
            "zscore",
            "Zstatistic",
            "zstatistic",
            "ZStat",
            "zstat",
        ]
        plural_forms = [
            "s",
            "S",
            "Scores",
            "scores",
            "SCORES",
            "Statistics",
            "statistics",
            "STATISTICS",
        ]
        aliases = set(z_forms)

        for base in z_forms:
            for plural in plural_forms:
                aliases.add(f"{base}{plural}")

        return sorted(aliases)

    @classmethod
    def _generate_field_aliases(cls, field: str) -> List[str]:
        """Generates aliases for a specific field, using cached values if available."""
        if field in cls.ALIASES:
            if cls.ALIASES[field] is None:
                method_name = cls._get_generator_method_name(field)
                generator_method = getattr(cls, method_name, None)
                if generator_method:
                    cls.ALIASES[field] = generator_method()
                else:
                    cls.ALIASES[field] = []
            return cls.ALIASES[field]
        return []

    @classmethod
    def _get_generator_method_name(cls, field: str) -> str:
        """Constructs the method name for generating aliases based on the field name."""
        clean_name = field.lower().replace(" ", "").replace("'", "").replace("-", "")
        return f"generate_{clean_name}_aliases"

    @classmethod
    def get_aliases(cls, fields: Union[str, Sequence[str]]) -> List[str]:
        """Retrieves all aliases for the given field(s), using caching for efficiency."""
        if fields is None:
            return []

        if isinstance(fields, (list, tuple)):
            cache_key = tuple(sorted(fields))
        else:
            cache_key = fields

        if cache_key not in cls._alias_cache:
            if isinstance(fields, (list, tuple)):
                aliases: List[str] = []
                for field in fields:
                    field_aliases = cls._generate_field_aliases(field)
                    aliases.extend(field_aliases)
                cls._alias_cache[cache_key] = aliases
            else:
                cls._alias_cache[cache_key] = cls._generate_field_aliases(fields)

        return cls._alias_cache[cache_key]

    @classmethod
    def get_field(cls, alias: Optional[str]) -> Optional[str]:
        """Finds the logical field name for a given alias."""
        if alias is None:
            return None

        for field in cls.ALIASES.keys():
            try:
                aliases = cls.get_aliases(field)
                if alias in aliases:
                    return field
            except Exception as e:
                log.debug(f"Error checking aliases for field {field}: {e}")
                continue

        if alias in cls.ALIASES:
            return alias

        return None

    @classmethod
    def find_keys(cls, group: Any, logical_field: str) -> Optional[str]:
        """Finds a key in the group that matches the logical field or its aliases."""
        if group is None or logical_field is None:
            return None

        try:
            aliases = cls.get_aliases(logical_field)
            for alias in aliases:
                if alias in group:
                    return alias

            for col in group:
                col_lower = col.lower()

                if col_lower.endswith(
                    f"_{logical_field.lower()}"
                ) or col_lower.endswith(f".{logical_field.lower()}"):
                    return col

                for alias in aliases:
                    alias_lower = alias.lower()
                    condition1 = col_lower.endswith(f"_{alias_lower}")
                    condition2 = col_lower.endswith(f".{alias_lower}")
                    condition3 = col_lower.endswith(f"-{alias_lower}")
                    condition4 = col_lower.endswith(f" {alias_lower}")
                    if condition1 or condition2 or condition3 or condition4:
                        return col

            group_lower = {item.lower(): item for item in group}
            for alias in aliases:
                if alias.lower() in group_lower:
                    found_item = group_lower[alias.lower()]
                    return found_item

        except Exception as e:
            log.debug(f"Error finding keys for {logical_field}: {e}")

        return None

    @classmethod
    def strip_numeric_suffix(cls, key: str) -> str:
        """Removes numeric or specific chromosome suffixes from a key."""
        base_key = re.sub(r"[0-9]+$|[XYM]+$|MT$", "", key, flags=re.IGNORECASE)
        return base_key

    @classmethod
    def get_complex_field(cls, field1: str, field2: str) -> str:
        """Determines the standardized complex field name for two given fields."""
        for sep in AliasUtils.STANDARD_SEPARATORS:
            complex_field = f"{field1}{sep}{field2}"
            if complex_field in cls.ALIASES:
                return complex_field

            complex_field_reversed = f"{field2}{sep}{field1}"
            if complex_field_reversed in cls.ALIASES:
                return complex_field_reversed

        complex_field = f"{field1}{field2}"
        if complex_field in cls.ALIASES:
            return complex_field

        complex_field_reversed = f"{field2}{field1}"
        if complex_field_reversed in cls.ALIASES:
            return complex_field_reversed

        standard_orders = {
            frozenset(["COEF", "SE"]): "COEF_SE",
            frozenset(["P", "FDR"]): "P_FDR",
            frozenset(["P", "BONF"]): "P_BONF",
            frozenset(["P", "HOLM"]): "P_HOLM",
            frozenset(["P", "BACON"]): "P_BACON",
            frozenset(["TSS", "DIST"]): "TSS_DIST",
            frozenset(["NEAREST_GENE", "DIST"]): "NEAREST_GENE_DIST",
        }

        field_set = frozenset([field1, field2])
        if field_set in standard_orders:
            return standard_orders[field_set]

        field_priority = {
            "COEF": 1,
            "BETA": 1,
            "OR": 1,
            "HR": 1,
            "P": 2,
            "T-STAT": 2,
            "Z": 2,
            "CHI2": 2,
            "SE": 3,
            "CI": 3,
            "VAR": 3,
            "RSID": 10,
            "CGID": 10,
            "GENE": 10,
            "CHR": 10,
            "BP": 10,
            "TSS": 11,
            "NEAREST_GENE": 11,
            "FDR": 20,
            "BONF": 20,
            "HOLM": 20,
            "BACON": 20,
            "DIST": 21,
            "DISTANCE": 21,
            "INFO": 30,
            "MAF": 30,
            "EAF": 30,
            "HWE": 30,
        }

        priority1 = field_priority.get(field1, 50)
        priority2 = field_priority.get(field2, 50)

        if priority1 < priority2:
            return f"{field1}_{field2}"
        elif priority2 < priority1:
            return f"{field2}_{field1}"
        else:
            if field1.lower() < field2.lower():
                return f"{field1}_{field2}"
            else:
                return f"{field2}_{field1}"

    @classmethod
    def find_complex_keys(cls, group: Any, field1: str, field2: str) -> Optional[str]:
        """Finds a complex key in the group based on combinations of two fields."""
        field1_aliases = cls.get_aliases(field1)
        field2_aliases = cls.get_aliases(field2)

        possible_combinations: Set[str] = set()

        for f1_alias in field1_aliases:
            for f2_alias in field2_aliases:
                for sep in AliasUtils.STANDARD_SEPARATORS:
                    possible_combinations.add(f"{f1_alias}{sep}{f2_alias}")
                    possible_combinations.add(f"{f2_alias}{sep}{f1_alias}")

        complex_field = cls.get_complex_field(field1, field2)
        complex_field_aliases = cls.get_aliases(complex_field)
        possible_combinations.update(complex_field_aliases)

        for alias in possible_combinations:
            if alias in group:
                return alias

        group_lower = {col.lower(): col for col in group}
        for alias in possible_combinations:
            if alias.lower() in group_lower:
                found_col = group_lower[alias.lower()]
                return found_col

        return None

    @classmethod
    def get_complex_aliases(cls, field1: str, field2: str) -> List[str]:
        """Generates all possible complex aliases for a combination of two fields."""
        field1_aliases = cls.get_aliases(field1)
        field2_aliases = cls.get_aliases(field2)

        complex_aliases: Set[str] = set()

        for f1_alias in field1_aliases:
            for f2_alias in field2_aliases:
                for sep in AliasUtils.STANDARD_SEPARATORS:
                    complex_aliases.add(f"{f1_alias}{sep}{f2_alias}")
                    complex_aliases.add(f"{f2_alias}{sep}{f1_alias}")

        return sorted(complex_aliases)

    @classmethod
    def get_all_fields_for_variable(
        cls, columns: Sequence[str], variable_name: str
    ) -> List[str]:
        """Retrieves all columns associated with a specific variable name."""
        if columns is None or len(columns) == 0 or not variable_name:
            return []

        result: List[str] = []
        variable_prefix = cls.standardize_variable_name(variable_name).lower()

        for col in columns:
            col_lower = col.lower()
            if col_lower.startswith(variable_prefix + "_") or col_lower.startswith(
                variable_prefix + "."
            ):
                result.append(col)

        return result

    @staticmethod
    def _generate_standard_aliases(
        base_forms: Sequence[str],
        suffixes: Optional[Sequence[str]] = None,
        separators: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Generates standard aliases by appending suffixes to base forms."""
        if separators is None:
            separators = AliasUtils.STANDARD_SEPARATORS
        if suffixes is None:
            suffixes = []

        aliases = set(base_forms)

        for base in base_forms:
            for suffix in suffixes:
                for sep in separators:
                    if suffix:
                        aliases.add(f"{base}{sep}{suffix}")

        return sorted(aliases)

    @staticmethod
    def _generate_compound_aliases(
        prefix_forms: Sequence[str],
        suffix_forms: Sequence[str],
        separators: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Generates compound aliases by combining prefixes and suffixes."""
        if separators is None:
            separators = AliasUtils.STANDARD_SEPARATORS

        aliases = set()
        for prefix in prefix_forms:
            for suffix in suffix_forms:
                for sep in separators:
                    aliases.add(f"{prefix}{sep}{suffix}")
        return sorted(aliases)

    @classmethod
    def auto_detect_variable(cls, df: pd.DataFrame) -> Optional[str]:
        """Automatically detects the primary variable in the DataFrame."""
        detected_variables = set()

        known_fields = ["P", "COEF", "SE", "P_FDR", "P_HOLM", "P_BONF", "P_BACON"]

        for field in known_fields:
            field_aliases = cls.get_aliases(field)
            for col in df.columns:
                col_parts = col.replace(".", "_").replace("-", "_").split("_")
                if len(col_parts) >= 2:
                    last_part = col_parts[-1]
                    if last_part in field_aliases or last_part.upper() in field_aliases:
                        var_name = "_".join(col_parts[:-1])
                        if var_name:
                            detected_variables.add(
                                cls.standardize_variable_name(var_name)
                            )

                    first_part = col_parts[0]
                    condition1 = first_part in field_aliases
                    condition2 = first_part.upper() in field_aliases
                    if condition1 or condition2:
                        var_name = "_".join(col_parts[1:])
                        if var_name:
                            detected_variables.add(
                                cls.standardize_variable_name(var_name)
                            )

        if detected_variables:
            priority_order = []

            has_cgid = cls.find_keys(dict.fromkeys(df.columns), "CGID")
            has_rsid = cls.find_keys(dict.fromkeys(df.columns), "RSID")

            if has_cgid:
                priority_order = [
                    "Methylation",
                    "methylation",
                    "METHYLATION",
                    "EWAS",
                    "ewas",
                ]
            elif has_rsid:
                priority_order = [
                    "Genotype",
                    "genotype",
                    "GENOTYPE",
                    "GWAS",
                    "gwas",
                    "Association",
                    "association",
                ]

            priority_order.extend(
                ["Analysis", "analysis", "Test", "test", "Model", "model"]
            )

            for priority_var in priority_order:
                if priority_var in detected_variables:
                    return priority_var

            variable_counts: Dict[str, int] = {}
            for var in detected_variables:
                count = 0
                for col in df.columns:
                    if col.lower().startswith(var.lower()):
                        count += 1
                variable_counts[var] = count

            if variable_counts:
                best_var = max(variable_counts.items(), key=lambda x: x[1])[0]
                return best_var

        return None

    @classmethod
    def find_variable_column(
        cls,
        df: pd.DataFrame,
        variable_name: str,
        target_field: str,
        existing_mappings: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Finds the column corresponding to a specific variable and target field."""
        if existing_mappings is None:
            existing_mappings = {}

        found_col = cls.find_complex_keys(df.columns, variable_name, target_field)
        if found_col and found_col not in existing_mappings:
            return found_col

        if target_field == "P":
            found_col = cls.find_p_column_comprehensive(
                df, existing_mappings, target_variable=variable_name
            )
            if found_col:
                return found_col
        elif target_field == "SE":
            found_col = cls.find_se_column_comprehensive(
                df, existing_mappings, target_variable=variable_name
            )
            if found_col:
                return found_col

        std_var = cls.standardize_variable_name(variable_name)
        var_lower = std_var.lower()
        field_aliases = cls.get_aliases(target_field)

        for col in df.columns:
            if col in existing_mappings:
                continue

            col_lower = col.lower()

            if col_lower.startswith(var_lower):
                for alias in field_aliases:
                    for sep in cls.STANDARD_SEPARATORS:
                        pattern = f"{var_lower}{sep}{alias.lower()}"
                        if col_lower == pattern:
                            return col

                for alias in field_aliases:
                    if col_lower == f"{var_lower}{alias.lower()}":
                        return col

        for col in df.columns:
            if col in existing_mappings:
                continue

            col_lower = col.lower()
            for alias in field_aliases:
                alias_lower = alias.lower()
                if col_lower.startswith(alias_lower):
                    for sep in cls.STANDARD_SEPARATORS:
                        pattern = f"{alias_lower}{sep}{var_lower}"
                        if col_lower == pattern:
                            return col

        standardized_col = cls.find_keys(dict.fromkeys(df.columns), target_field)
        if standardized_col and standardized_col not in existing_mappings:
            return standardized_col

        return None

    @classmethod
    def get_all_variable_columns(
        cls,
        df: pd.DataFrame,
        variable_name: str,
        existing_mappings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Retrieves all relevant columns for a given variable in the DataFrame."""
        if existing_mappings is None:
            existing_mappings = {}

        fields_to_find = [
            "P",
            "COEF",
            "SE",
            "P_FDR",
            "P_HOLM",
            "P_BONF",
            "P_BACON",
        ]
        variable_columns: Dict[str, str] = {}

        if not variable_name:
            return variable_columns

        std_var = cls.standardize_variable_name(variable_name)
        var_lower = std_var.lower() if std_var else variable_name.lower()

        for field in fields_to_find:
            try:
                found_col = cls.find_variable_column(
                    df, variable_name, field, existing_mappings
                )
            except Exception:
                found_col = None

            if found_col:
                if var_lower not in found_col.lower():
                    log.debug(
                        f"Ignoring {found_col} for variable '{variable_name}' since it does not contain the variable"
                    )
                    found_col = None

            if found_col:
                variable_columns[field] = found_col
                existing_mappings[found_col] = field

        for col in df.columns:
            if col in existing_mappings:
                continue
            col_lower = col.lower()
            if var_lower not in col_lower:
                continue

            for field in fields_to_find:
                if field in variable_columns:
                    continue
                aliases = cls.get_aliases(field) or [field]
                matched = False
                for alias in aliases:
                    alias_lower = alias.lower()
                    for sep in cls.STANDARD_SEPARATORS:
                        patterns = [
                            f"{var_lower}{sep}{alias_lower}",
                            f"{alias_lower}{sep}{var_lower}",
                            f"{var_lower}{alias_lower}",
                            f"{alias_lower}{var_lower}",
                        ]
                        if any(p in col_lower for p in patterns):
                            valid = True
                            try:
                                if field == "P":
                                    valid = cls._validate_p_column(df, col)
                                elif field == "SE":
                                    valid = cls._validate_se_column(df, col)
                                elif field == "COEF":
                                    valid = cls._validate_coef_column(df, col)
                            except Exception:
                                valid = False

                            if valid:
                                variable_columns[field] = col
                                existing_mappings[col] = field
                                matched = True
                                break
                    if matched:
                        break
                if matched:
                    break

        ci_columns = cls.find_all_standardized_ci_columns(df)
        for col, std_ci in ci_columns.items():
            if std_ci.startswith(std_var + "_") or (
                var_lower and var_lower in col.lower()
            ):
                variable_columns[std_ci] = col
                existing_mappings[col] = std_ci

        return variable_columns

    @classmethod
    def find_se_column_comprehensive(
        cls,
        df: pd.DataFrame,
        existing_mappings: Optional[Dict[str, str]] = None,
        target_variable: Optional[str] = None,
    ) -> Optional[str]:
        """Comprehensively detects the Standard Error (SE) column in the DataFrame."""
        if existing_mappings is None:
            existing_mappings = {}

        if target_variable:
            var_se_col = cls.find_complex_keys(df.columns, target_variable, "SE")
            if var_se_col and var_se_col not in existing_mappings:
                if cls._validate_se_column(df, var_se_col):
                    log.debug(f"Found variable-specific SE column: {var_se_col}")
                    return var_se_col

            target_lower = cls.standardize_variable_name(target_variable).lower()
            se_aliases = cls.get_aliases("SE")

            for col in df.columns:
                if col not in existing_mappings:
                    col_lower = col.lower()
                    if col_lower.startswith(target_lower):
                        for se_alias in se_aliases:
                            for sep in AliasUtils.STANDARD_SEPARATORS:
                                sep_str = f"{target_lower}{sep}{se_alias.lower()}"
                                condition = col_lower == sep_str
                                if condition:
                                    if cls._validate_se_column(df, col):
                                        log.debug(
                                            f"Found target variable SE column by pattern: {col}"
                                        )
                                        return col
                            if col_lower == f"{target_lower}{se_alias.lower()}":
                                if cls._validate_se_column(df, col):
                                    log.debug(
                                        f"Found target variable SE column by pattern: {col}"
                                    )
                                    return col

        se_col = cls.find_keys(dict.fromkeys(df.columns), "SE")
        if se_col and se_col not in existing_mappings:
            if cls._validate_se_column(df, se_col):
                return se_col

        complex_se_col = cls.find_complex_keys(df.columns, "COEF", "SE")
        if complex_se_col and complex_se_col not in existing_mappings:
            if cls._validate_se_column(df, complex_se_col):
                log.debug(
                    f"Found SE column using complex COEF+SE detection: {complex_se_col}"
                )
                return complex_se_col

        if not target_variable:
            detected_variables = set()
            for col in df.columns:
                if "_COEF" in col or "_coef" in col:
                    var_name = col.replace("_COEF", "").replace("_coef", "")
                    detected_variables.add(var_name)

            for var in detected_variables:
                var_se_col = cls.find_complex_keys(df.columns, var, "SE")
                if var_se_col and var_se_col not in existing_mappings:
                    if cls._validate_se_column(df, var_se_col):
                        log.debug(
                            f"Found auto-detected variable-specific SE column: {var_se_col}"
                        )
                        return var_se_col

        se_aliases = cls.generate_se_aliases()
        for se_alias in se_aliases[:15]:
            for col in df.columns:
                if col not in existing_mappings and se_alias.lower() in col.lower():
                    if cls._validate_se_column(df, col):
                        return col

        log.debug("Using aggressive pattern matching for SE detection")
        potential_cols: List[str] = []
        for col in df.columns:
            if col not in existing_mappings:
                col_lower = col.lower()
                se_patterns = [
                    "se",
                    "std_err",
                    "stderr",
                    "standard_error",
                    "standarderror",
                    "std_error",
                    "stderror",
                    "se_",
                    "_se",
                    "error",
                ]
                if any(pattern in col_lower for pattern in se_patterns):
                    if cls._validate_se_column(df, col):
                        potential_cols.append(col)

        if potential_cols:
            for col in potential_cols:
                col_lower = col.lower()
                if ("se" in col_lower and len(col) <= 15) or col_lower.endswith("_se"):
                    log.debug(
                        f"Found SE column through aggressive pattern matching: {col}"
                    )
                    return col
            log.debug(
                f"Found SE column through aggressive matching (first candidate): {potential_cols[0]}"
            )
            return potential_cols[0]

        log.debug("No SE column found using any detection method")
        return None

    @classmethod
    def _validate_se_column(cls, df: pd.DataFrame, col_name: str) -> bool:
        """Validates if the specified column in the DataFrame is a plausible SE column."""
        try:
            values = pd.to_numeric(df[col_name], errors="coerce").dropna()
            if len(values) == 0:
                return False

            negative_count = (values <= 0).sum()
            if negative_count > len(values) * 0.1:
                log.debug(
                    f"Column {col_name} has too many non-positive values: {negative_count}/{len(values)}"
                )
                return False

            min_threshold = 1e-10
            max_threshold = 1e6

            if values.max() > max_threshold or values.min() < min_threshold:
                log.debug(
                    f"Column {col_name} has unreasonable SE range: {values.min()} to {values.max()}"
                )
                return False

            if values.std() < 1e-15:
                log.debug(f"Column {col_name} has no variance (all same values)")
                return False

            if values.nunique() == 1:
                log.debug(f"Column {col_name} has only one unique value")
                return False

            log.debug(f"Column {col_name} passed SE validation: {len(values)} values")
            log.debug(f"SE range: {values.min():.6f} to {values.max():.6f}")
            return True
        except Exception as e:
            log.debug(f"Error validating SE column {col_name}: {e}")
            return False

    @classmethod
    def calculate_se_from_other_stats(
        cls, df: pd.DataFrame, existing_mappings: Optional[Dict[str, str]] = None
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        """Attempts to calculate SE from COEF and other statistics in the DataFrame."""
        if existing_mappings is None:
            existing_mappings = {}

        coef_col = None
        if "COEF" in df.columns:
            coef_col = "COEF"
        else:
            coef_aliases = cls.generate_coef_aliases()
            for col in df.columns:
                if col in coef_aliases:
                    coef_col = col
                    break

            if not coef_col:
                coef_col = cls.find_keys(dict.fromkeys(df.columns), "COEF")

        if not coef_col:
            log.debug("No COEF column found for SE calculation")
            return None, None

        tstat_col = cls.find_keys(dict.fromkeys(df.columns), "T-STAT")
        if tstat_col and tstat_col not in existing_mappings:
            try:
                coef_values = pd.to_numeric(df[coef_col], errors="coerce")
                tstat_values = pd.to_numeric(df[tstat_col], errors="coerce")

                calculated_se = abs(coef_values) / abs(tstat_values)
                calculated_se = calculated_se.replace([np.inf, -np.inf], np.nan)

                if calculated_se.dropna().std() > 0 and len(calculated_se.dropna()) > 0:
                    log.info(f"Successfully calculated SE from COEF/{tstat_col}")
                    return calculated_se, f"COEF/{tstat_col}"
            except Exception as e:
                log.debug(f"Failed to calculate SE from t-statistic: {e}")

        zstat_col = cls.find_keys(dict.fromkeys(df.columns), "Z")
        if zstat_col and zstat_col not in existing_mappings:
            try:
                coef_values = pd.to_numeric(df[coef_col], errors="coerce")
                zstat_values = pd.to_numeric(df[zstat_col], errors="coerce")

                calculated_se = abs(coef_values) / abs(zstat_values)
                calculated_se = calculated_se.replace([np.inf, -np.inf], np.nan)

                if calculated_se.dropna().std() > 0 and len(calculated_se.dropna()) > 0:
                    log.info(f"Successfully calculated SE from COEF/{zstat_col}")
                    return calculated_se, f"COEF/{zstat_col}"
            except Exception as e:
                log.debug(f"Failed to calculate SE from z-statistic: {e}")

        p_col = cls.find_keys(dict.fromkeys(df.columns), "P")
        if p_col and p_col not in existing_mappings:
            try:
                coef_values = pd.to_numeric(df[coef_col], errors="coerce")
                p_values = pd.to_numeric(df[p_col], errors="coerce")

                p_values = p_values.clip(lower=1e-300, upper=0.9999)
                z_scores = abs(norm.ppf(p_values / 2))

                calculated_se = abs(coef_values) / z_scores
                calculated_se = calculated_se.replace([np.inf, -np.inf], np.nan)

                valid_se = calculated_se.dropna()
                if len(valid_se) > 0 and valid_se.std() > 0:
                    if valid_se.max() < 100 and valid_se.min() > 1e-10:
                        log.info(
                            "Successfully calculated SE from COEF and p-values (less reliable method)"
                        )
                        return calculated_se, "COEF/P-value approximation"
            except Exception as e:
                log.debug(f"Failed to calculate SE from p-values: {e}")

        log.debug("Could not calculate SE from any available statistics")
        return None, None

    @classmethod
    def find_p_column_comprehensive(
        cls,
        df: pd.DataFrame,
        existing_mappings: Optional[Dict[str, str]] = None,
        target_variable: Optional[str] = None,
    ) -> Optional[str]:
        """Comprehensively detects the p-value column in the DataFrame."""
        if existing_mappings is None:
            existing_mappings = {}

        if target_variable:
            var_p_col = cls.find_complex_keys(df.columns, target_variable, "P")
            if var_p_col and var_p_col not in existing_mappings:
                if cls._validate_p_column(df, var_p_col):
                    log.debug(f"Found variable-specific p-value column: {var_p_col}")
                    return var_p_col

            target_lower = target_variable.lower()
            p_aliases = cls.get_aliases("P")

            for col in df.columns:
                if col not in existing_mappings:
                    col_lower = col.lower()
                    if col_lower.startswith(target_lower):
                        for p_alias in p_aliases:
                            for sep in AliasUtils.STANDARD_SEPARATORS:
                                if col_lower == f"{target_lower}{sep}{p_alias.lower()}":
                                    if cls._validate_p_column(df, col):
                                        log.debug(
                                            f"Found target variable p-value column by pattern: {col}"
                                        )
                                        return col
                            if col_lower == f"{target_lower}{p_alias.lower()}":
                                if cls._validate_p_column(df, col):
                                    log.debug(
                                        f"Found target variable p-value column by pattern: {col}"
                                    )
                                    return col

        p_col = cls.find_keys(dict.fromkeys(df.columns), "P")
        if p_col and p_col not in existing_mappings:
            if cls._validate_p_column(df, p_col):
                return p_col

        if not target_variable:
            detected_variables = set()
            for col in df.columns:
                if "_COEF" in col or "_coef" in col:
                    var_name = col.replace("_COEF", "").replace("_coef", "")
                    detected_variables.add(var_name)
                elif "_P" in col or "_p" in col:
                    var_name = col.replace("_P", "").replace("_p", "")
                    detected_variables.add(var_name)

            for var in detected_variables:
                var_p_col = cls.find_complex_keys(df.columns, var, "P")
                if var_p_col and var_p_col not in existing_mappings:
                    if cls._validate_p_column(df, var_p_col):
                        log.debug(
                            f"Found auto-detected variable-specific p-value column: {var_p_col}"
                        )
                        return var_p_col

        p_aliases = cls.generate_p_aliases()
        for p_alias in p_aliases[:20]:
            for col in df.columns:
                if col not in existing_mappings and p_alias.lower() in col.lower():
                    if cls._validate_p_column(df, col):
                        log.debug(
                            f"Found p-value column through alias pattern matching: {col}"
                        )
                        return col

        log.debug("Using aggressive pattern matching for p-value detection")
        potential_cols: List[str] = []
        for col in df.columns:
            if col not in existing_mappings:
                col_lower = col.lower()
                p_patterns = [
                    "pval",
                    "p_val",
                    "p.val",
                    "pvalue",
                    "p_value",
                    "p.value",
                    "p-value",
                    "pvals",
                    "p_vals",
                    "p.vals",
                    "pvalues",
                    "p_values",
                    "p.values",
                    "p-values",
                    "significance",
                    "sig",
                    "prob",
                    "probability",
                ]
                if any(pattern in col_lower for pattern in p_patterns):
                    if cls._validate_p_column(df, col):
                        potential_cols.append(col)
                elif col_lower in ["p", "P"] and cls._validate_p_column(df, col):
                    potential_cols.append(col)

        if potential_cols:
            for col in potential_cols:
                col_lower = col.lower()
                if ("pval" in col_lower or "p_value" in col_lower) and len(col) <= 20:
                    log.debug(
                        f"Found p-value column through aggressive pattern matching: {col}"
                    )
                    return col
            log.debug(
                f"Found p-value column through aggressive matching (first candidate): {potential_cols[0]}"
            )
            return potential_cols[0]

        log.debug("No p-value column found using any detection method")
        return None

    @classmethod
    def _validate_p_column(cls, df: pd.DataFrame, col_name: str) -> bool:
        """Validates whether the specified column in the DataFrame is a plausible p-value column."""
        try:
            values = pd.to_numeric(df[col_name], errors="coerce").dropna()
            if len(values) == 0:
                return False

            out_of_range = ((values < 0) | (values > 1)).sum()
            if out_of_range > len(values) * 0.05:
                log.debug(
                    f"Column {col_name} has too many out-of-range p-values: {out_of_range}/{len(values)}"
                )
                return False

            if values.nunique() < 2:
                log.debug(f"Column {col_name} has no variance (all same values)")
                return False

            if (values == 1.0).sum() > len(values) * 0.9:
                log.debug(f"Column {col_name} has too many values equal to 1.0")
                return False

            if (values == 0.0).sum() > len(values) * 0.1:
                log.debug(
                    f"Column {col_name} has many zero p-values (might be valid but suspicious)"
                )

            value_range = values.max() - values.min()
            if value_range < 1e-10:
                log.debug(f"Column {col_name} has extremely small range: {value_range}")
                return False

            q25, q50, q75 = values.quantile([0.25, 0.5, 0.75])

            if q25 > 0.9 and q50 > 0.95 and q75 > 0.99:
                log.debug(
                    f"Column {col_name} has suspicious distribution (most values near 1)"
                )
                return False

            log.debug(
                f"Column {col_name} passed p-value validation: {len(values)} values"
            )
            log.debug(f"p-value range: {values.min():.6f} to {values.max():.6f}")
            return True
        except Exception as e:
            log.debug(f"Error validating p-value column {col_name}: {e}")
            return False

    @classmethod
    def calculate_p_from_other_stats(
        cls, df: pd.DataFrame, existing_mappings: Optional[Dict[str, str]] = None
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        """Attempts to calculate p-values from other statistics in the DataFrame."""
        if existing_mappings is None:
            existing_mappings = {}

        coef_col = None
        if "COEF" in df.columns:
            coef_col = "COEF"
        else:
            coef_aliases = cls.generate_coef_aliases()
            for col in df.columns:
                if col in coef_aliases:
                    coef_col = col
                    break

            if not coef_col:
                coef_col = cls.find_keys(dict.fromkeys(df.columns), "COEF")

        tstat_col = cls.find_keys(dict.fromkeys(df.columns), "T-STAT")
        if tstat_col and tstat_col not in existing_mappings:
            try:
                tstat_values = pd.to_numeric(df[tstat_col], errors="coerce")

                n_col = cls.find_keys(dict.fromkeys(df.columns), "N")

                if n_col:
                    n_values = pd.to_numeric(df[n_col], errors="coerce")
                    df_values = n_values - 1
                else:
                    log.debug(
                        "No sample size found, using default df for t-test p-value calculation"
                    )
                    df_values = 100

                if isinstance(df_values, pd.Series):
                    calculated_p = 2 * (1 - stats.t.cdf(abs(tstat_values), df_values))
                    if isinstance(calculated_p, np.ndarray):
                        calculated_p = pd.Series(calculated_p, index=tstat_values.index)
                else:
                    calculated_p = 2 * (1 - stats.t.cdf(abs(tstat_values), df_values))
                    if isinstance(calculated_p, np.ndarray):
                        calculated_p = pd.Series(calculated_p, index=tstat_values.index)

                calculated_p = calculated_p.replace([np.inf, -np.inf], np.nan)
                calculated_p = calculated_p.clip(lower=1e-300, upper=1.0)

                valid_p = calculated_p.dropna()
                if len(valid_p) > 0 and valid_p.nunique() > 1:
                    log.info("Successfully calculated p-values from t-statistics")
                    df_desc = f"df={df_values if not isinstance(df_values, pd.Series) else 'variable'}"
                    return calculated_p, f"t-statistic ({df_desc})"
            except Exception as e:
                log.debug(f"Failed to calculate p-values from t-statistic: {e}")

        zstat_col = cls.find_keys(dict.fromkeys(df.columns), "Z")
        if zstat_col and zstat_col not in existing_mappings:
            try:
                zstat_values = pd.to_numeric(df[zstat_col], errors="coerce")

                calculated_p = 2 * (1 - stats.norm.cdf(abs(zstat_values)))

                if isinstance(calculated_p, np.ndarray):
                    calculated_p = pd.Series(calculated_p, index=zstat_values.index)

                calculated_p = calculated_p.replace([np.inf, -np.inf], np.nan)
                calculated_p = calculated_p.clip(lower=1e-300, upper=1.0)

                valid_p = calculated_p.dropna()
                if len(valid_p) > 0 and valid_p.nunique() > 1:
                    log.info("Successfully calculated p-values from z-statistics")
                    return calculated_p, "z-statistic (normal distribution)"
            except Exception as e:
                log.debug(f"Failed to calculate p-values from z-statistic: {e}")

        se_col = cls.find_keys(dict.fromkeys(df.columns), "SE")

        condition1 = se_col and se_col not in existing_mappings
        condition2 = coef_col and coef_col not in existing_mappings
        if condition1 and condition2:
            try:
                coef_values = pd.to_numeric(df[coef_col], errors="coerce")
                se_values = pd.to_numeric(df[se_col], errors="coerce")

                z_scores = coef_values / se_values
                z_scores = z_scores.replace([np.inf, -np.inf], np.nan)

                calculated_p = 2 * (1 - stats.norm.cdf(abs(z_scores)))

                if isinstance(calculated_p, np.ndarray):
                    calculated_p = pd.Series(calculated_p, index=z_scores.index)

                calculated_p = calculated_p.replace([np.inf, -np.inf], np.nan)
                calculated_p = calculated_p.clip(lower=1e-300, upper=1.0)

                valid_p = calculated_p.dropna()
                if len(valid_p) > 0 and valid_p.nunique() > 1:
                    log.info("Successfully calculated p-values from COEF/SE (z-test)")
                    return calculated_p, "COEF/SE z-test"
            except Exception as e:
                log.debug(f"Failed to calculate p-values from COEF/SE: {e}")

        log.debug("Could not calculate p-values from any available statistics")
        return None, None

    @classmethod
    def get_all_p_value_columns(
        cls, df: pd.DataFrame, existing_mappings: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Detects all p-value related columns in the DataFrame."""
        if existing_mappings is None:
            existing_mappings = {}

        p_value_fields = ["P", "P_FDR", "P_BONF", "P_HOLM", "P_BACON"]
        found_p_columns: Dict[str, str] = {}

        for field in p_value_fields:
            found_col = cls.find_keys(dict.fromkeys(df.columns), field)
            if found_col and found_col not in existing_mappings:
                if cls._validate_p_column(df, found_col):
                    found_p_columns[field] = found_col
                    log.debug(f"Found {field} column: {found_col}")

        return found_p_columns

    @classmethod
    def find_coef_column_comprehensive(
        cls,
        df: pd.DataFrame,
        existing_mappings: Optional[Dict[str, str]] = None,
        target_variable: Optional[str] = None,
    ) -> Optional[str]:
        """Comprehensively detects the coefficient column in the DataFrame."""
        if existing_mappings is None:
            existing_mappings = {}

        if target_variable:
            var_coef_col = cls.find_variable_column(
                df, target_variable, "COEF", existing_mappings
            )
            if var_coef_col:
                if cls._validate_coef_column(df, var_coef_col):
                    log.debug(
                        f"Found variable-specific coefficient column: {var_coef_col}"
                    )
                    return var_coef_col

        coef_col = cls.find_keys(dict.fromkeys(df.columns), "COEF")
        if coef_col and coef_col not in existing_mappings:
            if cls._validate_coef_column(df, coef_col):
                return coef_col

        if not target_variable:
            detected_variables = set()
            for col in df.columns:
                if "_COEF" in col or "_coef" in col:
                    var_name = col.replace("_COEF", "").replace("_coef", "")
                    detected_variables.add(var_name)
                elif "_P" in col or "_p" in col:
                    var_name = col.replace("_P", "").replace("_p", "")
                    detected_variables.add(var_name)

            for var in detected_variables:
                var_coef_col = cls.find_variable_column(
                    df, var, "COEF", existing_mappings
                )
                if var_coef_col:
                    if cls._validate_coef_column(df, var_coef_col):
                        log.debug(
                            f"Found auto-detected variable-specific coefficient column: {var_coef_col}"
                        )
                        return var_coef_col

        coef_aliases = cls.generate_coef_aliases()
        for coef_alias in coef_aliases[:20]:
            for col in df.columns:
                if col not in existing_mappings and coef_alias.lower() in col.lower():
                    if cls._validate_coef_column(df, col):
                        log.debug(
                            f"Found coefficient column through alias pattern matching: {col}"
                        )
                        return col

        log.debug("Using aggressive pattern matching for coefficient detection")
        potential_cols: List[str] = []
        for col in df.columns:
            if col not in existing_mappings:
                col_lower = col.lower()
                coef_patterns = [
                    "coef",
                    "coefficient",
                    "beta",
                    "effect",
                    "estimate",
                    "slope",
                    "regression",
                    "logfc",
                    "log_fc",
                    "fold_change",
                    "fc",
                ]
                if any(pattern in col_lower for pattern in coef_patterns):
                    if cls._validate_coef_column(df, col):
                        potential_cols.append(col)

        if potential_cols:
            for col in potential_cols:
                col_lower = col.lower()
                if ("coef" in col_lower or "beta" in col_lower) and len(col) <= 15:
                    log.debug(
                        f"Found coefficient column through aggressive pattern matching: {col}"
                    )
                    return col
            log.debug(
                f"Found coefficient column through aggressive matching (first candidate): {potential_cols[0]}"
            )
            return potential_cols[0]

        log.debug("No coefficient column found using any detection method")
        return None

    @classmethod
    def _validate_coef_column(cls, df: pd.DataFrame, col_name: str) -> bool:
        """Validates if a column can be considered a coefficient column."""
        try:
            values = pd.to_numeric(df[col_name], errors="coerce").dropna()
            if len(values) == 0:
                return False

            if values.std() < 1e-15:
                log.debug(f"Column {col_name} has no variance (all same values)")
                return False

            if values.nunique() == 1:
                log.debug(f"Column {col_name} has only one unique value")
                return False

            if abs(values).max() > 1e6:
                log.debug(
                    f"Column {col_name} has unreasonably large coefficient values"
                )
                return False

            non_zero_count = (values != 0).sum()
            if non_zero_count < len(values) * 0.1:
                log.debug(
                    f"Column {col_name} has too many zero values for coefficients"
                )
                return False

            log.debug(
                f"Column {col_name} passed coefficient validation: {len(values)} values"
            )
            log.debug(f"Coefficient range: {values.min():.6f} to {values.max():.6f}")
            return True
        except Exception as e:
            log.debug(f"Error validating coefficient column {col_name}: {e}")
            return False

    @classmethod
    def parse_ci_column(cls, col_name: str) -> Optional[Dict[str, Any]]:
        """Parses a confidence interval bound column name into its components."""
        col_norm = col_name.lower().replace(" ", "_")
        pattern = re.compile(
            r"""
            ^(?P<var>[a-zA-Z0-9\.\-]+)?
            [_\.\-\s]*
            (?P<ci_num>99|95|90)?%?
            [_\.\-\s]*
            (ci|confidence[_\.\-\s]?interval)
            [_\.\-\s]*
            (?P<bound>lower|low|lb|min|l|upper|high|ub|max|u)?$
            """,
            re.VERBOSE,
        )
        m = pattern.match(col_norm)
        if not m:
            pattern2 = re.compile(
                r"""
                ^(?P<ci_num>99|95|90)%?
                [_\.\-\s]*
                (?P<var>[a-zA-Z0-9\.\-]+)?
                [_\.\-\s]*
                (ci|confidence[_\.\-\s]?interval)
                [_\.\-\s]*
                (?P<bound>lower|low|lb|min|l|upper|high|ub|max|u)?$
                """,
                re.VERBOSE,
            )
            m = pattern2.match(col_norm)
        if not m:
            return None

        groups = m.groupdict()
        var = cls.standardize_variable_name((groups.get("var") or "").strip("._- "))
        ci_num = groups.get("ci_num")
        bound = groups.get("bound")
        if not bound:
            return None
        lower_aliases = {"lower", "low", "lb", "min", "l"}
        upper_aliases = {"upper", "high", "ub", "max", "u"}
        if bound in lower_aliases:
            bound_std = "lower"
        elif bound in upper_aliases:
            bound_std = "upper"
        else:
            return None
        return {"variable": var, "ci_num": ci_num, "bound": bound_std}

    @classmethod
    def standardize_ci_column_name(cls, col_name: str) -> Optional[str]:
        """Standardizes a confidence interval bound column name to a canonical format."""
        parsed = cls.parse_ci_column(col_name)
        if not parsed:
            return None
        var = cls.standardize_variable_name(parsed["variable"])
        ci_num = parsed["ci_num"]
        bound = parsed["bound"]
        parts = []
        if var:
            parts.append(var)
        if ci_num:
            parts.append(ci_num)
        parts.append("CI_LOWER" if bound == "lower" else "CI_UPPER")
        return "_".join(parts)

    @classmethod
    def find_all_standardized_ci_columns(
        cls, df: pd.DataFrame, validate: bool = True
    ) -> Dict[str, str]:
        """Finds all confidence interval bound columns in the DataFrame and returns a mapping of their names."""
        result: Dict[str, str] = {}
        for col in df.columns:
            std = cls.standardize_ci_column_name(col)
            if std:
                if not validate or cls._validate_ci_column(df, col):
                    result[col] = std
        return result

    @classmethod
    def _validate_ci_column(cls, df: pd.DataFrame, col_name: str) -> bool:
        """Validates if a column can be considered a confidence interval bound column."""
        try:
            values = df[col_name].dropna()
            if len(values) == 0:
                return False
            numeric = values.apply(lambda x: isinstance(x, (int, float)))
            if numeric.mean() < 0.8:
                return False
            return True
        except Exception:
            return False

    @classmethod
    def standardize_variable_name(cls, var: Optional[str]) -> str:
        """Standardizes a variable name to its canonical form."""
        if not var:
            return ""
        std = cls.get_field(var.upper()) or cls.get_field(var.lower())
        return std if std else var.upper()
