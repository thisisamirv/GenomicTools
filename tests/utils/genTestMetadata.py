import numpy as np
import pandas as pd
import h5py
import logging
from datetime import datetime
import random


def define_constants():
    common_values = [1, 2, 3, np.nan]
    smoke_levels = [1, 2, 3, 4, 5, np.nan]
    extensive_units = [
        "10e3/uL",
        "/cmm",
        "/cu mm",
        "/cumm",
        "/uL",
        "#/cmm",
        "#/hpf",
        "#/mm3",
        "#/uL",
        "x1000/uL",
        "10.e3/uL",
        "10(3)/mcL",
        "10**3",
        "10*3/uL",
        "1000/uL",
        "1000/ul",
        "10E3/mcL",
        "10e3/ul",
        "10E9/L",
        "10x3/cmm",
        "10x3cumm",
        "BILL/L",
        "cells/uL",
        "cumm",
        "G/DL",
        "K/cmm",
        "k/cmm",
        "K/ccm",
        "K/CMM",
        "K/cu mm",
        "K/cumm",
        "K/CUMM",
        "K/CUMM.",
        "K/mcL",
        "K/ml",
        "K/mm-3",
        "K/mm3",
        "K/MM3",
        "k/uL",
        "K/ul",
        "K/UL",
        "t/cmm",
        "T/uL",
        "thou/cumm",
        "THOU/CUMM",
        "thou/ul",
        "thou/uL",
        "thous./mm3",
        "uL",
        "X(10)3",
        "X10-3",
        "X10-3/uL",
        "X10-3/ul",
        "X10(9)/L",
        "x10*3/uL",
        "X1000/cmm",
        "x1000/uL",
        "x10E3/ul",
        "x10E3/uL",
    ]

    def generate_icd10_codes():
        codes = []
        for prefix in ["A0", "A1", "B0", "B1"]:
            for i in range(10):
                codes.append(f"{prefix}{i}")
        return codes

    possible_values = {
        "possible_gia": ["AFR", "EUR", "EAS", "LA"],
        "possible_scannerid": [
            "N0875",
            "N0961",
            "N1016",
            "N1017",
            "N1199",
            "N171",
            "N299",
            "N326",
        ],
        "labchemtestname": ["Albumin", "ALBUMIN", "CRP", "Wbc", "WBC", ""],
        "value": ["", "CURRENT", "FORMER", "NEVER"],
        "smoker_survey": common_values,
        "lss_smoke": [0, 1, 2, np.nan],
        "smkchew": common_values,
        "smkcigar": common_values,
        "smkcur": [0, 1, np.nan],
        "smkcuramt": smoke_levels,
        "smkexphome": smoke_levels,
        "smkexpoth": smoke_levels,
        "smkpastamt": smoke_levels,
        "smkdly": [0, 1, np.nan],
        "smknow": common_values,
        "smkyrs": list(range(90)) + [np.nan],
    }
    params = {
        "genomicpc_params": dict(
            mean=-1.67648e-05, sd=0.0047904, min=-0.022373, max=0.019019
        ),
        "technicalmetric_params": dict(
            mean=-0.029918, sd=2.77918, min=-7.28499, max=20.50821
        ),
    }
    date_ranges = {
        "enrolled_d": dict(
            start=datetime(2011, 2, 10),
            end=datetime(2019, 9, 24),
            mean=datetime(2015, 5, 5),
        ),
        "baseline_scan_dt": dict(
            start=datetime(2011, 1, 13),
            end=datetime(2023, 9, 10),
            mean=datetime(2014, 8, 20),
        ),
        "lifestyle_scan_dt": dict(
            start=datetime(2011, 4, 6),
            end=datetime(2023, 8, 27),
            mean=datetime(2015, 4, 3),
        ),
        "blooddraw_dt": dict(
            start=datetime(2011, 2, 10),
            end=datetime(2019, 9, 24),
            mean=datetime(2015, 5, 6),
        ),
        "dxdate": dict(
            start=datetime(1994, 3, 31),
            end=datetime(2018, 10, 12),
            mean=datetime(2012, 9, 22),
        ),
        "lab_date": dict(
            start=datetime(1999, 9, 28),
            end=datetime(2023, 9, 28),
            mean=datetime(2009, 7, 3),
        ),
    }
    icd_codes = {
        "icd9": [str(i) for i in range(1, 21)],
        "icd10": generate_icd10_codes(),
    }
    houseman_params = {
        "houseman_bcell": dict(mean=0.040, sd=0.038, min=0, max=0.781),
        "houseman_cd4t": dict(mean=0.115, sd=0.057, min=0, max=0.451),
        "houseman_cd8t": dict(mean=0.037, sd=0.042, min=0, max=0.534),
        "houseman_mono": dict(mean=0.084, sd=0.027, min=0, max=0.514),
        "houseman_neu": dict(mean=0.630, sd=0.103, min=0, max=0.955),
        "houseman_nk": dict(mean=0.093, sd=0.048, min=0, max=0.593),
    }

    def units_values(test_name):
        if test_name == "Wbc":
            return random.choice(extensive_units)
        elif test_name == "ALBUMIN":
            return random.choice(
                [
                    "/uL",
                    "g/dl",
                    "g/dL",
                    "G/dL",
                    "G/DL",
                    "g/L",
                    "gm/dl",
                    "gm/dL",
                    "Gm/dl",
                    "GM/DL",
                    "GMS/DL",
                    "K/cmm",
                    "mg/dl",
                    "thou/cumm",
                    "ug/min.",
                ]
            )
        elif test_name == "Albumin":
            return random.choice(["g/dL", "gm/dL", "mg/L"])
        elif test_name == "CRP":
            return random.choice(["g/dL", "K/cmm", "mg/dL", "mg/L"])
        else:
            return np.nan

    return dict(
        possible_values=possible_values,
        params=params,
        date_ranges=date_ranges,
        icd_codes=icd_codes,
        houseman_params=houseman_params,
        units_values=units_values,
    )


def generate_column(df, column_name, values, na_percentage=0):
    col = np.random.choice(values, len(df), replace=True)
    if na_percentage > 0:
        na_indices = np.random.choice(
            len(df), int(na_percentage * len(df)), replace=False
        )
        col[na_indices] = np.nan
    df[column_name] = col
    return df


def generate_age_column(df, mean, sd, minv, maxv):
    age = np.clip(np.round(np.random.normal(mean, sd, len(df))), minv, maxv)
    na_indices = np.random.choice(len(df), int(0.05 * len(df)), replace=False)
    age[na_indices] = np.nan
    df["age"] = age
    return df


def generate_houseman_df(total_rows, houseman_params):
    np.random.seed(123)
    houseman_df = pd.DataFrame()
    for k, params in houseman_params.items():
        values = np.clip(
            np.random.normal(params["mean"], params["sd"], total_rows),
            params["min"],
            params["max"],
        )
        na_indices = np.random.choice(
            total_rows, int(np.ceil(total_rows * 0.10)), replace=False
        )
        values[na_indices] = np.random.uniform(
            -8.8672e-19, -6.5487e-17, len(na_indices)
        )
        houseman_df[k] = values
    # Normalize rows to sum to ~1 (with random scaling)
    total = houseman_df.sum(axis=1)
    for col in houseman_df.columns:
        houseman_df[col] = (
            houseman_df[col] / total * np.random.uniform(0.987, 1.0, total_rows)
        )
    return houseman_df


def generate_values(n, mean, sd, minv, maxv):
    return np.clip(np.random.normal(mean, sd, n), minv, maxv)


def generate_dates(n, start_date, end_date, mean_date, na_percentage=0):
    date_seq = pd.date_range(start_date, end_date, freq="D")
    mean_index = np.where(date_seq == pd.Timestamp(mean_date))[0][0]
    spread = min(mean_index, len(date_seq) - mean_index - 1)
    indices = np.sort(
        np.random.choice(
            range(mean_index - spread, mean_index + spread + 1), n, replace=True
        )
    )
    dates = date_seq[indices]
    if na_percentage > 0:
        na_indices = np.random.choice(n, int(na_percentage * n), replace=False)
        dates = dates.astype("O")
        for idx in na_indices:
            dates[idx] = pd.NaT
    return dates


def generate_date_column(df, column_name, date_range, na_percentage=0):
    df[column_name] = generate_dates(
        len(df),
        date_range["start"],
        date_range["end"],
        date_range["mean"],
        na_percentage,
    )
    return df


def generate_icd_columns(df, icd_codes):
    all_icd_codes = icd_codes["icd9"] + icd_codes["icd10"]
    sampled_codes = np.random.choice(all_icd_codes, len(df), replace=True)
    df["icdcode"] = sampled_codes
    df["codetype"] = np.where(
        np.isin(sampled_codes, icd_codes["icd9"]), "ICD9", "ICD10"
    )
    return df


def generate_yob(n, mean, sd, minv, maxv):
    return np.clip(np.round(np.random.normal(mean, sd, n)), minv, maxv)


def generate_additional_columns(df, possible_values):
    columns = [
        "smkchew",
        "smkcigar",
        "smkcur",
        "smkcuramt",
        "smkexphome",
        "smkexpoth",
        "smkpastamt",
        "smkdly",
        "smknow",
    ]
    for col in columns:
        df[col] = np.random.choice(possible_values[col], len(df), replace=True)
    return df


def generate_normal_values(test_name, n):
    normal_ranges = {
        "Wbc": (4.0, 11.0),
        "WBC": (4.0, 11.0),
        "ALBUMIN": (3.5, 5.0),
        "Albumin": (3.5, 5.0),
        "CRP": (0.0, 1.0),
    }
    if test_name in normal_ranges:
        low, high = normal_ranges[test_name]
        return np.random.uniform(low, high, n)
    else:
        return np.random.uniform(0, 100, n)


def gen_test_metadata(h5_file, output):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gen_test_metadata")
    constants = define_constants()
    possible_values = constants["possible_values"]
    params = constants["params"]
    date_ranges = constants["date_ranges"]
    icd_codes = constants["icd_codes"]
    houseman_params = constants["houseman_params"]
    units_values = constants["units_values"]

    logger.info("Loading metadata from HDF5.")
    with h5py.File(h5_file, "r") as f:
        mvp004_id = f["/metadata/sampleList"][:]
    df = pd.DataFrame({"mvp004_id": mvp004_id})

    logger.info("Generating GIA.")
    df = generate_column(df, "GIA", possible_values["possible_gia"])
    logger.info("Generating age.")
    df = generate_age_column(df, 17, 5, 18, 99)
    logger.info("Generating sex.")
    df = generate_column(df, "sex", [1, 2])
    logger.info("Generating methylation age.")
    df["methylation_age_horvath"] = [
        (
            np.random.choice(range(18, 100))
            if pd.isna(x) or x == -9
            else x + np.random.uniform(-8.3544, 8.3544)
        )
        for x in df["age"]
    ]
    logger.info("Generating scannerid.")
    df = generate_column(df, "scannerid", possible_values["possible_scannerid"])
    logger.info("Generating days_stored.")
    days_stored = np.concatenate([np.arange(743, 4097), [np.nan] * int(0.10 * len(df))])
    df = generate_column(df, "days_stored", days_stored)
    logger.info("Generating houseman data.")
    houseman_df = generate_houseman_df(len(df), houseman_params)
    df = pd.concat([df, houseman_df], axis=1)

    logger.info("Generating principal components.")
    for i in range(1, 21):
        df[f"genomicpc{i}"] = generate_values(
            len(df),
            params["genomicpc_params"]["mean"],
            params["genomicpc_params"]["sd"],
            params["genomicpc_params"]["min"],
            params["genomicpc_params"]["max"],
        )
        vals = generate_values(
            len(df),
            params["technicalmetric_params"]["mean"],
            params["technicalmetric_params"]["sd"],
            params["technicalmetric_params"]["min"],
            params["technicalmetric_params"]["max"],
        )
        na_indices = np.random.choice(len(df), int(0.08 * len(df)), replace=False)
        vals[na_indices] = np.nan
        df[f"technicalmetric_pc{i}"] = vals

    logger.info("Generating dates.")
    for col, drange in date_ranges.items():
        na_pct = (
            0.05
            if col in ["baseline_scan_dt", "lifestyle_scan_dt", "dxdate", "lab_date"]
            else 0
        )
        df = generate_date_column(df, col, drange, na_percentage=na_pct)

    logger.info("Generating ICD columns.")
    df = generate_icd_columns(df, icd_codes)
    logger.info("Generating yob.")
    df["yob"] = generate_yob(len(df), 1950, 15, 1913, 1998)
    logger.info("Generating lab data.")
    df = generate_column(df, "labchemtestname", possible_values["labchemtestname"])
    logger.info("Generating lab data columns.")
    df["units"] = [units_values(x) for x in df["labchemtestname"]]
    df["phenotypeid"] = 2
    logger.info("Generating lab_value columns.")
    df = generate_column(df, "value", possible_values["value"])
    df["lab_value"] = [
        generate_normal_values(test_name, 1)[0] for test_name in df["labchemtestname"]
    ]
    logger.info("Generating smoking data.")
    df = generate_column(df, "smoker_survey", possible_values["smoker_survey"])
    df = generate_column(df, "lss_smoke", possible_values["lss_smoke"])
    df = generate_age_column(df, 27, 16, 0, 89)
    df = generate_additional_columns(df, possible_values)

    df["smkyrs"] = np.minimum(
        df["age"] - 10, generate_age_column(df.copy(), 0, 10, 0, 89)["age"]
    )
    df["units"] = [x if x is not None else np.nan for x in df["units"]]

    logger.info(f"Writing output to {output}")
    df.to_csv(output, index=False)
    logger.info(f"Synthetic metadata generation complete: {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic metadata CSV from HDF5 sample list."
    )
    parser.add_argument("--h5_file")
    parser.add_argument("--output")
    args = parser.parse_args()
    gen_test_metadata(args.h5_file, args.output)
