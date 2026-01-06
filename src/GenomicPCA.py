#!/usr/bin/env python
# Import required modules
import h5py
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Optional, Tuple
from utils.AliasUtils import AliasUtils
from utils.CLIFramework import CLIFramework, OptionConfig
from utils.H5Utils import CachedH5Utils
from utils.LoggingUtils import log


def GenomicPCA(
    input: str,
    output: str,
    n_components: int = 10,
    data_type: str = "Methylation",
    residuals: bool = False,
    batch_size: int = 5000,
    scale: bool = True,
    ld_pruned: bool = False,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    try:
        log.info(f"Running PCA on {data_type} data with {n_components} components")

        if residuals:
            residuals_df = pd.read_csv(input, sep=",", header=0)
            log.info(f"Read residuals data from {input}")

            sample_id_col = AliasUtils.find_keys(residuals_df.columns, "IID")
            if not sample_id_col:
                sample_id_col = AliasUtils.find_keys(residuals_df.columns, "SampleList")

            if sample_id_col:
                log.info(f"Found sample ID column: {sample_id_col}")
                sample_ids = residuals_df[sample_id_col].tolist()
                residuals_df.set_index(sample_id_col, inplace=True)
            else:
                log.warn(
                    "No sample ID column found. Assuming first column contains sample IDs."
                )
                sample_ids = residuals_df.iloc[:, 0].tolist()
                residuals_df.set_index(residuals_df.columns[0], inplace=True)

            data_values = residuals_df.values

            if np.isnan(data_values).any():
                data_values = SimpleImputer(strategy="mean").fit_transform(data_values)

            max_components = min(
                n_components, min(data_values.shape[0] - 1, data_values.shape[1])
            )
            pca = PCA(n_components=max_components)
            pca_result = pca.fit_transform(data_values)
            explained_var_ratio = pca.explained_variance_ratio_

            pca_columns = [f"resiPC{i + 1}" for i in range(pca_result.shape[1])]

        else:
            with h5py.File(input, "r") as infile:
                h5_utils = CachedH5Utils(infile)
                all_keys = h5_utils.get_chromosomes()
                chr_keys = []
                for key in all_keys:
                    condition1 = isinstance(infile[key], h5py.Group)
                    condition2 = key.lower().startswith(
                        "chr"
                    ) or key.upper().startswith("CHR")
                    if condition1 and condition2:
                        chr_keys.append(key)

                log.info(f"Chromosome groups found: {chr_keys}")

                sample_ids = []

                try:
                    file_info = h5_utils.get_data_info()
                    condition1 = "sample_path" in file_info
                    condition2 = file_info["sample_path"] in infile
                    if condition1 and condition2:
                        sample_ids = [
                            s.decode("utf-8") if isinstance(s, bytes) else s
                            for s in infile[file_info["sample_path"]]
                        ]
                        log.info(
                            f"Found {len(sample_ids)} samples from CachedH5Utils info: {file_info['sample_path']}"
                        )
                except Exception as e:
                    log.debug(f"Error getting sample IDs from file_info: {e}")

                if not sample_ids:
                    metadata_aliases = AliasUtils.generate_metadata_aliases()
                    metadata_path = None

                    for meta_alias in metadata_aliases:
                        if meta_alias in infile:
                            metadata_path = meta_alias
                            log.info(f"Found metadata group: {metadata_path}")

                            sample_found = False

                            sample_fields = [
                                "sample_list",
                                "sampleList",
                                "Samples",
                                "samples",
                                "IID",
                                "iid",
                                "SampleID",
                                "sampleid",
                                "ID",
                                "id",
                                "SampleIDs",
                                "sampleids",
                                "IDs",
                                "ids",
                            ]

                            for field in sample_fields:
                                if field in infile[metadata_path]:
                                    sample_ids = [
                                        s.decode("utf-8") if isinstance(s, bytes) else s
                                        for s in infile[metadata_path][field]
                                    ]
                                    log.info(
                                        f"Found {len(sample_ids)} samples in {metadata_path}/{field}"
                                    )
                                    sample_found = True
                                    break

                            if not sample_found:
                                log.debug(
                                    f"Metadata fields: {list(infile[metadata_path].keys())}"
                                )

                            break

                if not sample_ids:
                    samplelist_aliases = AliasUtils.generate_samplelist_aliases()
                    for sample_alias in samplelist_aliases:
                        if sample_alias in infile:
                            sample_ids = [
                                s.decode("utf-8") if isinstance(s, bytes) else s
                                for s in infile[sample_alias]
                            ]
                            log.info(
                                f"Found {len(sample_ids)} samples in root/{sample_alias}"
                            )
                            break

                if not sample_ids and chr_keys:
                    chr_name = chr_keys[0]

                    if data_type.lower() == "methylation":
                        data_fields = [
                            "betas",
                            "beta",
                            "Beta",
                            "BETA",
                            "Methylation",
                            "methylation",
                            "M",
                            "m",
                        ]
                    else:
                        data_fields = [
                            "genotypes",
                            "genotype",
                            "Genotype",
                            "Genotypes",
                            "GT",
                            "gt",
                            "dosage",
                            "Dosage",
                        ]

                    for field in data_fields:
                        if field in infile[chr_name] and isinstance(
                            infile[chr_name][field], h5py.Dataset
                        ):
                            data_shape = infile[chr_name][field].shape
                            log.debug(
                                f"Found data field {field} with shape {data_shape}"
                            )

                            if data_type.lower() == "methylation":
                                n_samples = data_shape[0]
                            else:
                                if len(data_shape) > 1:
                                    n_samples = data_shape[1]
                                else:
                                    n_samples = data_shape[0]

                            sample_ids = [f"Sample_{i + 1}" for i in range(n_samples)]
                            log.warn(
                                f"No sample IDs found in file. Created {n_samples} placeholder IDs."
                            )
                            break

                if not sample_ids:
                    if data_type.lower() == "genotype" and "Genotypes" in infile:
                        try:
                            data_shape = infile["Genotypes"].shape
                            n_samples = (
                                data_shape[1] if len(data_shape) > 1 else data_shape[0]
                            )
                            sample_ids = [f"Sample_{i + 1}" for i in range(n_samples)]
                            log.warn(
                                f"Using direct Genotypes dataset, created {n_samples} placeholder IDs."
                            )
                        except Exception as e:
                            log.debug(f"Error with direct Genotypes dataset: {e}")

                if not sample_ids:
                    raise ValueError(
                        "Could not determine sample IDs from the input file"
                    )

                n_samples = len(sample_ids)
                max_components = (
                    min(n_components, n_samples - 1) if n_samples > 1 else 1
                )

                if data_type.lower() == "methylation":
                    all_data = []

                    for grp in tqdm(chr_keys, desc="Processing Methylation data"):
                        beta_field = None
                        for field in [
                            "betas",
                            "beta",
                            "Beta",
                            "BETA",
                            "methylation",
                            "Methylation",
                        ]:
                            if field in infile[grp]:
                                beta_field = field
                                break

                        if not beta_field:
                            log.warn(
                                f"No methylation data found in chromosome {grp}. Skipping."
                            )
                            continue

                        data_shape = infile[grp][beta_field].shape
                        n_probes = data_shape[1]

                        sample_size = min(50, n_probes)
                        if n_probes <= 50:
                            indices = range(n_probes)
                        else:
                            indices = np.linspace(
                                0, n_probes - 1, sample_size, dtype=int
                            )

                        probe_data = np.array(infile[grp][beta_field][:, indices])
                        all_data.append(probe_data)

                    if not all_data:
                        raise ValueError(
                            "No valid methylation data found in any chromosome"
                        )

                    combined_data = np.hstack(all_data)

                    if np.isnan(combined_data).any():
                        imputer = SimpleImputer(strategy="mean")
                        combined_data = imputer.fit_transform(combined_data)

                    if scale:
                        scaler = StandardScaler()
                        combined_data = scaler.fit_transform(combined_data)

                    pca = PCA(n_components=max_components)
                    pca_result = pca.fit_transform(combined_data)
                    explained_var_ratio = pca.explained_variance_ratio_

                    pca_columns = [f"PC{i + 1}" for i in range(pca_result.shape[1])]

                elif data_type.lower() == "genotype":
                    log.info(
                        f"Processing {len(chr_keys)} chromosomes for genotype data"
                    )

                    all_variants = []

                    for chr_name in tqdm(chr_keys, desc="Processing Genotype data"):
                        genotype_field = None
                        for field in [
                            "genotypes",
                            "genotype",
                            "Genotype",
                            "Genotypes",
                            "GT",
                            "gt",
                        ]:
                            if field in infile[chr_name]:
                                genotype_field = field
                                break

                        if not genotype_field:
                            log.warn(
                                f"No genotype data found in chromosome {chr_name}. Skipping."
                            )
                            continue

                        data_shape = infile[chr_name][genotype_field].shape
                        log.debug(f"Genotype data shape for {chr_name}: {data_shape}")

                        n_first_dim = data_shape[0]
                        n_second_dim = data_shape[1] if len(data_shape) > 1 else 1

                        variants_in_rows = n_second_dim == n_samples

                        sample_size = min(50, n_first_dim)
                        if n_first_dim <= 50:
                            indices = range(n_first_dim)
                        else:
                            indices = np.linspace(
                                0, n_first_dim - 1, sample_size, dtype=int
                            )

                        if variants_in_rows:
                            variant_data = np.array(
                                infile[chr_name][genotype_field][indices, :]
                            )
                            log.debug(
                                f"Extracted {variant_data.shape} with variants in rows"
                            )
                        else:
                            variant_data = np.array(
                                infile[chr_name][genotype_field][:, indices]
                            )
                            variant_data = variant_data.T
                            log.debug(
                                f"Extracted and transposed to {variant_data.shape}"
                            )

                        variant_data = np.where(
                            variant_data == -1, np.nan, variant_data
                        )

                        all_variants.append(variant_data)

                    if not all_variants:
                        raise ValueError(
                            "No valid genotype data found in any chromosome"
                        )

                    combined_variants = np.vstack(all_variants)
                    log.info(f"Combined genotype data shape: {combined_variants.shape}")

                    features_matrix = combined_variants.T
                    log.info(f"Transposed data for PCA, shape: {features_matrix.shape}")

                    if np.isnan(features_matrix).any():
                        imputer = SimpleImputer(strategy="mean")
                        features_matrix = imputer.fit_transform(features_matrix)

                    if scale:
                        scaler = StandardScaler()
                        features_matrix = scaler.fit_transform(features_matrix)

                    pca = PCA(n_components=max_components)
                    pca_result = pca.fit_transform(features_matrix)
                    explained_var_ratio = pca.explained_variance_ratio_

                    pca_columns = [f"PC{i + 1}" for i in range(pca_result.shape[1])]

                else:
                    raise ValueError(
                        f"Unsupported data type: {data_type}. Use 'Methylation' or 'Genotype'."
                    )

        pca_df = pd.DataFrame(pca_result, columns=pca_columns).round(5)
        pca_df.index = sample_ids[: pca_result.shape[0]]
        pca_df.index.name = "IID"

        pca_df.to_csv(output)

        component_names = pca_columns
        explained_var = pd.DataFrame(
            {"Component": component_names, "ExplainedVariance": explained_var_ratio}
        )

        variance_output = os.path.splitext(output)[0] + "_ExplainedVariance.csv"
        explained_var.to_csv(variance_output, index=False)

        log.success(f"PCA completed with {len(pca_columns)} components")
        log.success(f"PC values saved to: {output}")
        log.success(f"Explained variance saved to: {variance_output}")
        log.info(
            f"Total variance explained: {explained_var['ExplainedVariance'].sum() * 100:.2f}%"
        )
        return pca_df, explained_var

    except Exception as e:
        log.error(f"Error in GenomicPCA: {e}")
        return None


options = [
    OptionConfig(flags=["-i", "--input"], type=str),
    OptionConfig(flags=["-o", "--output"], type=str),
    OptionConfig(flags=["-n", "--n_components"], type=int, default=10),
    OptionConfig(flags=["-d", "--data_type"], type=str, default="Methylation"),
    OptionConfig(flags=["-r", "--residuals"], action="store_true"),
    OptionConfig(flags=["-b", "--batch_size"], type=int, default=5000),
    OptionConfig(flags=["-s", "--scale"], action="store_true", default=True),
    OptionConfig(flags=["-p", "--ld_pruned"], action="store_true"),
]

if __name__ == "__main__":
    framework = CLIFramework(option_list=options, script_name="GenomicPCA")
    opt = framework.run()

    GenomicPCA(
        input=opt.input,
        output=opt.output,
        n_components=opt.n_components,
        data_type=opt.data_type,
        residuals=opt.residuals,
        batch_size=opt.batch_size,
        scale=opt.scale,
        ld_pruned=opt.ld_pruned,
    )
