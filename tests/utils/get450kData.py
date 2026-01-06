# Description: This script downloads and processes the data for the DNA methylation analysis.

# Import libraries
import os
import pandas as pd
import tempfile

# Download files
files_to_download = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77696/suppl/GSE77696%5FMatrixProcessed.txt.gz"


# Download the file in a temporary directory, unzip it, and read it into a pandas DataFrame
def download_and_extract(url):
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, os.path.basename(url))
        os.system(f"wget -O {file_path} {url}")
        os.system(f"gunzip {file_path}")
        return pd.read_csv(file_path[:-3], sep="\t", skiprows=1, comment="#")


methylation_data = download_and_extract(files_to_download)

# Drop columns that contain 'Detection Pval' in their names
detection_pval_columns = methylation_data.filter(like="Detection Pval").columns
methylation_data.drop(columns=detection_pval_columns, inplace=True)

# Rename the 'ID_REF' column to 'probe_id'
methylation_data.rename(columns={"ID_REF": "probe_id"}, inplace=True)

# Subset only the first 10 samples
methylation_data = methylation_data.iloc[:, :10]

# Subset only the first 10000 probes
methylation_data = methylation_data.iloc[:10000, :]

# Reset the index of the dataframe
methylation_data.reset_index(drop=True, inplace=True)

# Save the processed data
methylation_data.to_csv("test/data/450k_data.csv")

# Print a message indicating the completion of the function
print("Data downloaded and saved into test/data/450k_data.csv")
