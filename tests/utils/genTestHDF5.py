import os
import numpy as np
import h5py
import logging


def mixed_betas(n, shape1_1, shape2_1, shape1_2, shape2_2, mix_ratio):
    n1 = int(round(n * mix_ratio))
    n2 = n - n1
    return np.concatenate(
        [np.random.beta(shape1_1, shape2_1, n1), np.random.beta(shape1_2, shape2_2, n2)]
    )


def gen_test_hdf5(
    output, n_chr=22, n_probe=None, n_sample=5, na_rate=0.05, na_threshold=0.05
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gen_test_hdf5")

    if n_probe is None:
        n_probe = [5000] * n_chr
    elif isinstance(n_probe, str):
        # Parse string like "2x5000 3x1000"
        n_probe = n_probe.split()
        parsed = []
        for x in n_probe:
            if "x" in x:
                count, val = x.split("x")
                parsed.extend([int(val)] * int(count))
            else:
                parsed.append(int(x))
        n_probe = parsed

    if os.path.exists(output):
        logger.info(f"Removing existing file: {output}")
        os.remove(output)

    logger.info(f"Creating HDF5 file: {output}")
    with h5py.File(output, "w") as f:
        total_cpgs = sum(n_probe)
        num_cpgs_with_missing = int(round(total_cpgs * na_rate))

        # Calculate CpG indices
        all_cpg_indices = []
        for chr_num in range(n_chr):
            start_idx = sum(n_probe[:chr_num])
            end_idx = sum(n_probe[: chr_num + 1])
            all_cpg_indices.extend(range(start_idx, end_idx))
        all_cpg_indices = np.array(all_cpg_indices)

        # Sample CpGs with missing values
        cpgs_with_missing = np.random.choice(
            all_cpg_indices, num_cpgs_with_missing, replace=False
        )

        for chr_num in range(n_chr):
            chr_group = f"/chr{chr_num + 1}"
            logger.info(f"Creating group for chromosome: {chr_group}")
            grp = f.create_group(chr_group)

            # Generate synthetic methylation β-values with a bimodal distribution
            logger.info(
                f"Generating synthetic methylation β-values for chromosome: {chr_num + 1}"
            )
            betas = np.empty((n_probe[chr_num], n_sample))
            betas[:] = np.nan
            for i in range(n_sample):
                betas[:, i] = mixed_betas(
                    n_probe[chr_num],
                    shape1_1=2,
                    shape2_1=20,
                    shape1_2=20,
                    shape2_2=2,
                    mix_ratio=0.5,
                )

            # Introduce missing values with varying thresholds
            chr_start_idx = sum(n_probe[:chr_num])
            chr_end_idx = sum(n_probe[: chr_num + 1])
            chr_cpgs_with_missing = [
                idx - chr_start_idx
                for idx in cpgs_with_missing
                if chr_start_idx <= idx < chr_end_idx
            ]

            for cpg_idx in chr_cpgs_with_missing:
                random_threshold = np.random.uniform(0, na_threshold)
                num_missing = int(np.ceil(n_sample * random_threshold))
                missing_indices = np.random.choice(n_sample, num_missing, replace=False)
                betas[cpg_idx, missing_indices] = np.nan

            # Cap β-values between 0 and 1
            betas = np.clip(betas, 0, 1)

            # Write datasets
            logger.info(f"Writing datasets for chromosome: {chr_num + 1}")
            grp.create_dataset(
                "betas",
                data=betas,
                chunks=(min(1000, n_probe[chr_num]), n_sample),
                compression="gzip",
                compression_opts=9,
            )

            probe_list = np.array(
                [f"cg{idx + 1:08d}" for idx in range(n_probe[chr_num])], dtype="S"
            )
            grp.create_dataset(
                "probeList",
                data=probe_list,
                chunks=(min(1000, len(probe_list)),),
                compression="gzip",
                compression_opts=9,
            )

            probewise_mean = np.nanmean(betas, axis=1)
            grp.create_dataset(
                "probewiseMean",
                data=probewise_mean,
                chunks=(min(1000, len(probewise_mean)),),
                compression="gzip",
                compression_opts=9,
            )

            probewise_stdev = np.nanstd(betas, axis=1)
            grp.create_dataset(
                "probewiseStdev",
                data=probewise_stdev,
                chunks=(min(1000, len(probewise_stdev)),),
                compression="gzip",
                compression_opts=9,
            )

        # Write metadata group
        logger.info("Writing metadata group.")
        meta_grp = f.create_group("metadata")
        sample_list = np.random.randint(1000000, 10000000, n_sample)
        meta_grp.create_dataset(
            "sampleList",
            data=sample_list,
            chunks=(n_sample,),
            compression="gzip",
            compression_opts=9,
        )

        logger.info(f"Synthetic HDF5 file created successfully: {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic HDF5 file for methylation data."
    )
    parser.add_argument("--output")
    parser.add_argument("--n_chr", type=int, default=22)
    parser.add_argument("--n_probe", type=str, default=None)
    parser.add_argument("--n_sample", type=int, default=5)
    parser.add_argument("--na_rate", type=float, default=0.05)
    parser.add_argument("--na_threshold", type=float, default=0.05)
    args = parser.parse_args()

    gen_test_hdf5(
        output=args.output,
        n_chr=args.n_chr,
        n_probe=args.n_probe,
        n_sample=args.n_sample,
        na_rate=args.na_rate,
        na_threshold=args.na_threshold,
    )
