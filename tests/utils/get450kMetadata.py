# Description: This script downloads the metadata for the GSE77696 dataset from GEO and saves it to a CSV file.

# Import libraries
import os
import GEOparse
import pandas as pd
import tempfile

# Download files
files_to_download = [
    # Metadata
    (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77696/soft/GSE77696_family.soft.gz",
        "tmp/rawdata/GSE77696_metadata.soft.gz",
    )
]


# Download the files in a temp dir and unzip them
def download_files(files):
    gse = None
    for url, local_path in files:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.close()
            os.system(f"wget -O {temp_file.name} {url}")
            os.system(f"gunzip -c {temp_file.name} > {local_path}")
            gse = GEOparse.get_GEO(filepath=local_path, silent=True)
            print("Downloaded and extracted")
            os.remove(temp_file.name)
    return gse


# Download and extract the GEO dataset, assign to gse
gse = download_files(files_to_download)

# Extract the metadata from the GEO data
metadata = gse.metadata.copy()

# Shorten the sample_id and supplementary_file lists for display
metadata["sample_id"] = (
    metadata["sample_id"][:10] + ["..."] + metadata["sample_id"][-10:]
)
metadata["supplementary_file"] = metadata["supplementary_file"][:3]


# Function to format the metadata dictionary
def format_metadata(metadata, gse):
    formatted = ""
    for key, value in metadata.items():
        if isinstance(value, list):
            formatted += f"* {key}: {', '.join(value)}\n"
        else:
            formatted += f"* {key}: {value}\n"

    # Get headers from the metadata
    characteristics_list = []
    for gsm_name, gsm in gse.gsms.items():
        for characteristic in gsm.metadata.get("characteristics_ch1", []):
            name = characteristic.split(":")[0]
            characteristics_list.append(name)
        break

    formatted += "\n* Metadata headers:\n"
    formatted += ", ".join(characteristics_list)

    return formatted


# Print the formatted metadata
print(format_metadata(metadata, gse))

# Define the fields to extract from the sample metadata
fields_to_extract = [
    "hiv",
    "agebl",
    "racecomg",
    "dmgsex",
    "wbc_new",
    "bcell",
    "cd4t",
    "cd8t",
    "gran",
    "mono",
    "nk",
]

# Add the control probe and residual PC fields to the list
control_probe_fields = [f"control_probe_pc{i}" for i in range(1, 31)]
fields_to_extract.extend(control_probe_fields)
resipc_fields = [f"resipc_{i}" for i in range(1, 6)]
fields_to_extract.extend(resipc_fields)

# Extract the sample metadata
sample_metadata = []
for gsm_name, gsm in gse.gsms.items():
    metadata = gsm.metadata
    sample_info = {}
    for field in fields_to_extract:
        sample_info[field] = None
        for characteristic in metadata.get("characteristics_ch1", []):
            if characteristic.startswith(f"{field}:"):
                sample_info[field] = characteristic.split(": ")[1]
                break
    # Extract sample_id from Sample_description
    description = metadata.get("description", [""])[0]
    sample_info["sample_id"] = description.split(": Sample ")[-1]
    sample_metadata.append(sample_info)

# Move sample_id to the first position
sample_metadata = [{**{"sample_id": x["sample_id"]}, **x} for x in sample_metadata]

# Convert the sample metadata to a DataFrame
phenotype = pd.DataFrame(sample_metadata)

# Subset only the first 10 samples
phenotype = phenotype.head(10).copy()

# Change sample names from 1, 2, etc. to sample_1, sample_2, etc.
phenotype["sample_id"] = phenotype["sample_id"].apply(lambda x: f"sample_{x}")

# Save the phenotype data to a CSV file
phenotype.to_csv("450k_metadata.csv", sep=",", index=False)

# Print a message indicating the completion of the function
print("Metadata saved to 450k_metadata.csv")
