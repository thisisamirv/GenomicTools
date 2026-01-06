#!/usr/bin/env Rscript
# Import the necessary libraries
suppressPackageStartupMessages({
    library(cowplot)
    library(data.table)
    library(ggplot2)
    library(pROC)
    library(Rcpp)
    library(rhdf5)
})

# Main function
calculatePGMS <- function(
    coefficients,
    counts,
    features,
    metadata,
    group,
    output,
    figure = NULL,
    group_labels = "{'case': 'NULL', 'control': 'NULL'}",
    colors = "red, blue",
    set_col = "set"
) {
    tryCatch({
        # Split strings if comma-separated
        colors <- stringToList(colors)
        group_labels <- dictToList(group_labels)
        
        # If all group_lables list values are NULL, set group_labels to NULL
        if (all(unlist(group_labels) == "NULL")) {
            group_labels <- NULL
        }
        
        
        # Read EWAS results and count data
        log_info("Reading EWAS results from", coefficients)
        coefficients <- fread(coefficients, sep = ",", header = TRUE)
        
        # Open the input HDF5 file
        log_info("Opening the input HDF5 file:", counts)
        h5_file <- H5Fopen(counts, flags = "H5F_ACC_RDONLY")
        
        # Get chromosome list
        log_info("Retrieving chromosome list from HDF5 file.")
        chr_list <- getChromosomeList(h5_file)
        
        # Read the metadata
        log_info("Reading the metadata from", metadata)
        metadata_df <- fread(metadata)
        log_debug("Metadata dimensions:", nrow(metadata_df), "x", ncol(metadata_df))
        
        # Filter metadata for the "group" column to only include "group_labels", if it is not NULL
        if (is.null(group_labels)) {
            log_info("No group labels provided, skipping filtering.")
            group_labels <- unique(metadata_df[[group]])
        } else {
            log_info("Filtering metadata for the group column.")
            log_debug("Group column head before filtering:", head(metadata_df[[group]]))
            group_labels_unlist <- unlist(group_labels)
            metadata_df <- metadata_df[get(group) %in% group_labels_unlist, ]
            log_debug("Filtered metadata dimensions:", nrow(metadata_df), "x", ncol(metadata_df))
            log_debug("Group column head after filtering:", head(metadata_df[[group]]))
        }
        
        # Read sample names for indices
        log_info("Reading sample names")
        train_samples_list <- metadata_df[get(set_col) == "train", sample_id]
        test_samples_list <- metadata_df[get(set_col) == "test", sample_id]
        
        # Get sample indices for efficient reading
        log_info("Obtaining sample indices for training samples.")
        train_indices <- getSampleIndices(h5_file, train_samples_list)
        log_info("Obtaining sample indices for test samples.")
        test_indices <- getSampleIndices(h5_file, test_samples_list)
        
        # Read the features file
        log_info("Reading features from", features)
        features <- fread(features, header = FALSE)[[1]]
        
        # Read data
        log_info("Reading data from HDF5 file.")
        data_list <- lapply(chr_list, function(chr) {
            probe_indices <- getProbeIndices(h5_file, chr, features)
            if (!is.null(probe_indices)) {
                chr_data_train <- readChromosomeData(h5_file, chr, probe_indices, train_indices)
                chr_data_test <- readChromosomeData(h5_file, chr, probe_indices, test_indices)
                return(list(train = chr_data_train, test = chr_data_test))
            }
            return(NULL)
        })
        
        # Combine the data
        log_info("Combining training and test data.")
        train_data <- do.call(rbind, lapply(data_list, function(x) if (!is.null(x)) x$train else NULL))
        test_data <- do.call(rbind, lapply(data_list, function(x) if (!is.null(x)) x$test else NULL))
        
        # Sort data and coefficients based on features
        log_info("Sorting count data and EWAS results based on features.")
        log_debug("Train data shape:", dim(train_data))
        train_data <- train_data[probe_id %in% features, , drop = FALSE]
        test_data <- test_data[probe_id %in% features, , drop = FALSE]
        coefficients <- coefficients[probe_id %in% features]
        log_debug("Train data shape after filtering:", dim(train_data))
        
        # Normalize beta values per sample
        log_info("Normalizing beta values per sample.")
        train_count_sums <- colSums(train_data[, !c("probe_id"), with=FALSE], na.rm = TRUE)
        test_count_sums <- colSums(test_data[, !c("probe_id"), with=FALSE], na.rm = TRUE)
        train_data[, names(train_count_sums) := Map("/", .SD, train_count_sums), .SDcols = names(train_count_sums)]
        test_data[, names(test_count_sums) := Map("/", .SD, test_count_sums), .SDcols = names(test_count_sums)]
        
        # Normalize coefficients
        log_info("Normalizing coefficients.")
        coefficients[, coef := coef / sum(coef, na.rm = TRUE)]
        
        # Match coefficients to features
        log_debug("Matching coefficients to features.")
        coefficients <- coefficients[match(train_data[["probe_id"]], probe_id), coef]
        
        # Drop probe_id column
        train_data <- train_data[, !c("probe_id"), with=FALSE]
        test_data <- test_data[, !c("probe_id"), with=FALSE]
        
        # Calculate PGMS for training samples
        log_info("Calculating PGMS for training samples.")
        train_score_per_probe <- sweep(train_data, 1, coefficients, "*")
        train_score_per_individual <- as.vector(colSums(train_score_per_probe, na.rm = TRUE))
        
        # Calculate PGMS for test samples
        log_info("Calculating PGMS for test samples.")
        test_score_per_probe <- sweep(test_data, 1, coefficients, "*")
        test_score_per_individual <- as.vector(colSums(test_score_per_probe, na.rm = TRUE))
        
        # Scale scores to 0-1
        log_info("Scaling scores to 0-1.")
        train_score_per_individual <- scaleZeroOne(train_score_per_individual)
        test_score_per_individual <- scaleZeroOne(test_score_per_individual)
        
        # Subset relevant columns from metadata
        original_metadata_df <- copy(metadata_df)
        metadata_df <- metadata_df[, c("sample_id", group), with=FALSE]
        
        # Divide metadata into groups
        log_info("Dividing metadata into training and test groups.")
        train_metadata <- metadata_df[sample_id %in% train_samples_list]
        test_metadata <- metadata_df[sample_id %in% test_samples_list]
        log_debug("Number of training samples:", nrow(train_metadata))
        log_debug("Number of test samples:", nrow(test_metadata))
        
        # Add PGMS to metadata
        log_info("Adding PGMS to training metadata.")
        train_score_per_individual <- as.vector(unlist(train_score_per_individual))
        test_score_per_individual <- as.vector(unlist(test_score_per_individual))
        names(train_score_per_individual) <- train_samples_list
        names(test_score_per_individual) <- test_samples_list
        train_metadata[, PGMS := train_score_per_individual[as.character(sample_id)]]
        log_info("Adding PGMS to test metadata.")
        test_metadata[, PGMS := test_score_per_individual[as.character(sample_id)]]
        log_debug("Train metadata dimensions:", dim(train_metadata))
        log_debug("Test metadata dimensions:", dim(test_metadata))
        
        if(!is.null(figure)) {
            # Check if there are more than 2 levels in group column
            if (length(unique(train_metadata[[group]])) > 2 || length(unique(test_metadata[[group]])) > 2) {
                log_error("Plotting is only supported for binary classification (2 levels) in the group column.")
                stop("If only getting PGMS values is desired, please run the script without the --figure option.")
            } else {
                # Perform ROC analysis on training data
                log_info("Performing ROC analysis on training data.")
                roc_results <- rocAnalysis(
                    train_metadata,
                    group,
                    "PGMS",
                    smooth = FALSE,
                    plot = FALSE,
                    threshold = TRUE
                )
                threshold <- roc_results$threshold
                cat("The optimal threshold for PGMS is:", round(threshold, 2), "\n")
                
                # Perform ROC analysis with plotting
                log_info("Performing ROC analysis with plotting.")
                roc_results_plot <- rocAnalysis(
                    train_metadata,
                    group,
                    "PGMS",
                    smooth = TRUE,
                    plot = TRUE,
                    threshold = FALSE
                )
                
                # Prepare plot data
                plot_data <- train_metadata[, .(sample_id, get(group), PGMS)]
                setnames(plot_data, 2, group)
                
                # Generate density plot, if group_labels are not NULL
                if (is.null(group_labels)) {
                    log_debug("Group labels are NULL, skipping density plot.")
                } else {
                    log_info("Generating density plot.")
                    density_plot <- densityDTA(
                        data = plot_data,
                        group_column = group,
                        score_column = "PGMS",
                        group_labels = group_labels,
                        colors = colors,
                        threshold = threshold,
                        x_limits = c(-0.3, 1.3)
                    )
                    
                    # Combine density plot and ROC plot
                    log_info("Combining density plot and ROC plot.")
                    final_plot <- plot_grid(density_plot, roc_results_plot$roc_plot, ncol = 2)
                    
                    # Save the plots
                    log_info("Saving the combined plot to:", figure)
                    ggsave(filename = figure, plot = final_plot, width = 12, height = 6)
                }
            }
        }
        
        # Combine metadata
        log_info("Combining metadata.")
        final_metadata <- rbindlist(list(train_metadata, test_metadata))
        
        # Drop the group column from metadata and merge with the original metadata
        log_info("Dropping the group column from metadata.")
        final_metadata <- final_metadata[, !c(group), with=FALSE]
        
        log_debug("Merging final metadata with original metadata.")
        final_metadata <- merge(
            original_metadata_df,
            final_metadata,
            by = "sample_id",
            all.x = TRUE
        )
        log_debug("Final metadata dimensions:", dim(final_metadata))
        
        # Write the final metadata to output
        log_info("Writing the final metadata to:", output)
        fwrite(final_metadata, output)
        
        success_message("PGMS calculation and plotting completed successfully.")
        
    }, error = function(e) {
        log_error(e$message)
    }, warning = function(w) {
        log_warn(w$message)
    }, finally = {
        # Close the input HDF5 file if it's still open
        if (exists("h5_file")) {
            log_debug("Closing the input HDF5 file.")
            H5Fclose(h5_file)
        }
    })
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
utils_files <- c(
    "loggingUtils.R", "densityDTA.R", "dictToList.R", "getChromosomeList.R", "getProbeIndices.R",
    "getSampleIndices.R", "readChromosomeData.R", "rocAnalysis.R", "scaleZeroOne.R", "stringToList.R"
)
for (util_file in utils_files) {
    util_path <- file.path(utils_dir, util_file)
    source(util_path)
}

# Parse command line arguments
options <- list(
    list(flags = c("-e", "--coefficients"), type = "character"),
    list(flags = c("-c", "--counts"), type = "character"),
    list(flags = c("-u", "--features"), type = "character"),
    list(flags = c("-m", "--metadata"), type = "character"),
    list(flags = c("-g", "--group"), type = "character"),
    list(flags = c("-o", "--output"), type = "character"),
    list(flags = c("-f", "--figure"), type = "character", default = NULL),
    list(flags = c("-p", "--group_labels"), type = "character", default = "{'case': 'NULL', 'control': 'NULL'}"),
    list(flags = c("-r", "--colors"), type = "character", default = "red, blue"),
    list(flags = c("-s", "--set_col"), type = "character", default = "set")
)

if (!interactive()) {
    source(file.path(script_dir, "utils/initializeScript.R"))
    opt <- initializeScript(option_list = options, script_name = "calculatePGMS")
    calculatePGMS(
        coefficients= opt$coefficients,
        counts = opt$counts,
        features = opt$features,
        metadata = opt$metadata,
        group = opt$group,
        figure = opt$figure,
        output = opt$output,
        group_labels = opt$group_labels,
        colors = opt$colors,
        set_col = opt$set_col
    )
}
