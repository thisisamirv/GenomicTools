#!/usr/bin/env Rscript
# Import the necessary libraries
suppressPackageStartupMessages({
    library(data.table)
    library(progress)
    library(rhdf5)
})

# Main function
log2FC <- function(input, group1, group2, probe_file, output = NULL, file = TRUE) {
    tryCatch({
        # Open the HDF5 file for reading
        log_info("Opening HDF5 file:", input)
        h5_file <- H5Fopen(input, flags = "H5F_ACC_RDONLY")
        
        # Get list of chromosomes, excluding metadata
        log_debug("Fetching chromosome list from input file.")
        chromosome_list <- setdiff(h5ls(h5_file, recursive = FALSE)$name, "metadata")
        log_debug("Chromosome list:", chromosome_list)
        
        # Read probe names from input file
        log_debug("Reading probes from probe file:", probe_file)
        probes <- fread(probe_file)[, V1]
        log_debug("Number of probes read:", length(probes))
        
        # Create data.table to store probe information
        log_debug("Creating probe information data.table")
        probes_df <- data.table(probe = probes, chromosome = NA_character_, index = NA_integer_)
        
        # Read sample groups
        log_info("Reading sample groups")
        group1 <- fread(group1)[, V1]
        group2 <- fread(group2)[, V1]
        log_debug("Group 1 samples:", length(group1))
        log_debug("Group 2 samples:", length(group2))
        
        # Map probes to their chromosomes and indices
        log_info("Mapping probes to chromosomes")
        probes_df <- map_probes_to_chromosomes(h5_file, chromosome_list, probes_df)
        
        # Get sample indices for both groups
        log_debug("Getting sample indices")
        sample_list <- h5read(h5_file, "metadata/sampleList")
        group1_indices <- which(sample_list %in% group1)
        group2_indices <- which(sample_list %in% group2)
        
        # Validate sample indices
        if (length(group1_indices) == 0 || length(group2_indices) == 0) {
            stop("No matching samples found in one or both groups")
        }
        if (length(group1_indices) != length(group1) || length(group2_indices) != length(group2)) {
            log_warn("Some samples were not found in the dataset")
        }
        
        # Initialize results data.table and progress bar
        log_info("Calculating log2 fold change")
        log2FC <- data.table(probe = probes_df$probe, log2fc = numeric(nrow(probes_df)))
        cat("Calculating log2 fold changes...\n")
        pb <- progress_bar$new(total = length(chromosome_list))
        
        # Calculate log2 fold change for each probe
        for (chromosome in chromosome_list) {
            log_debug("Processing chromosome:", chromosome)
            chromosome_probes <- probes_df[chromosome == chromosome]
            log_debug("Number of probes in chromosome:", nrow(chromosome_probes))
            group1_data <- h5read(
            h5_file,
            paste0("/", chromosome, "/betas"),
            index = list(chromosome_probes$index, group1_indices)
            )
            group2_data <- h5read(
            h5_file,
            paste0("/", chromosome, "/betas"),
            index = list(chromosome_probes$index, group2_indices)
            )
            log_debug("Group 1 data dimensions:", dim(group1_data))
            log_debug("Group 2 data dimensions:", dim(group2_data))
            group1_means <- rowMeans(group1_data, na.rm = TRUE)
            group2_means <- rowMeans(group2_data, na.rm = TRUE)
            log2fc_values <- ifelse(
            group1_means > 0 & group2_means > 0,
            round(log2(group2_means / group1_means), 4),
            NA
            )
            log_debug("Some of Log2FC values:", head(log2fc_values))
            set(log2FC, i = which(probes_df$chromosome == chromosome), j = "log2fc", value = log2fc_values)
            pb$tick()
        }
        
        if (!file) {
            return(log2FC)
        } else {
            # Save results to output file
            log_info("Writing results to:", output)
            fwrite(log2FC, output)
            success_message("Log2FC calculation completed successfully and written to:", output)
        }
    
    }, error = function(e) {
        log_error(e$message)
    }, warning = function(w) {
        log_warn(w$message)
    }, finally = {
        if (exists("h5_file")) {
            H5Fclose(h5_file)
        }
    })
}

# Helper function to map probes to chromosomes
map_probes_to_chromosomes <- function(h5_file, chromosome_list, probes_df) {
    log_info("Mapping probes to chromosomes")
    # Iterate through each chromosome in the dataset
    for (chromosome in chromosome_list) {
        log_debug("Processing chromosome:", chromosome)
        # Read the probe list for the current chromosome
        probe_list <- h5read(h5_file, paste0("/", chromosome, "/probeList"))
        # Find indices of probes that exist in our input probe list
        probe_indices <- which(probe_list %in% probes_df$probe)
        
        # If any matching probes were found
        if (length(probe_indices) > 0) {
            log_debug("Found", length(probe_indices), "probes in chromosome", chromosome)
            # Update the chromosome and index information for matching probes
            probes_df$chromosome[probes_df$probe %in% probe_list[probe_indices]] <- chromosome
            probes_df$index[probes_df$probe %in% probe_list[probe_indices]] <- probe_indices
        }
    }
    
    return(probes_df)
}

# Source helper functions
if (!interactive()) {
    args <- commandArgs(trailingOnly = FALSE)
    script_path <- sub("--file=", "", args[grep("--file=", args)])
    script_dir <- dirname(normalizePath(script_path))
} else {
    script_dir <- dirname(normalizePath(sys.frame(1)$ofile))
}
utils_dir <- file.path(script_dir, "utils")
utils_files <- c("loggingUtils.R")
for (util_file in utils_files) {
    util_path <- file.path(utils_dir, util_file)
    source(util_path)
}

# Parse command line arguments
options <- list(
    list(flags = c("-i", "--input"), type = "character"),
    list(flags = c("-g", "--group1"), type = "character"),
    list(flags = c("-j", "--group2"), type = "character"),
    list(flags = c("-p", "--probe_file"), type = "character"),
    list(flags = c("-o", "--output"), type = "character", default = NULL),
    list(flags = c("-f", "--file"), type = "logical", default = TRUE)
)

if (!interactive()) {
    source(file.path(script_dir, "utils/initializeScript.R"))
    opt <- initializeScript(option_list = options, script_name = "log2FC")
    log2FC(
        input = opt$input,
        group1 = opt$group1,
        group2 = opt$group2,
        probe_file = opt$probe_file,
        output = opt$output,
        file = opt$file
    )
}
