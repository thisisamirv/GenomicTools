#!/usr/bin/env Rscript
# Helper function to process missing values
# Import the necessary libraries
suppressPackageStartupMessages({
    library(data.table)
    library(rhdf5)
    library(parallel)
})

# Main function
getMissingData <- function(h5_file, breaks, labels, chunk, aggregate = c("Probes", "Individual")) {
    log_info("Calculating missing values for", aggregate, "aggregation")
    
    # Validate inputs
    aggregate <- match.arg(aggregate)
    if (!is.numeric(breaks) || length(breaks) < 2) stop("Invalid breaks")
    if (length(labels) != length(breaks)-1) stop("Labels/Breaks length mismatch")
    
    # Setup parallel processing
    cores <- detectCores() - 1
    log_info("Using", cores, "cores for parallel processing")
    
    # Get chromosome list using helper
    chromosome_list <- getChromosomeList(h5_file)
    if (is.null(chromosome_list)) stop("No valid chromosomes found")
    
    # Preload sample names for Individual mode
    if (aggregate == "Individual") {
        all_sample_names <- h5read(h5_file, "/metadata/sampleList")
        sample_indices <- getSampleIndices(h5_file, all_sample_names)
        if (is.null(sample_indices) || length(sample_indices) == 0) {
            stop("No valid samples found")
        }
    }
    
    # Process chromosomes in parallel
    results_list <- mclapply(chromosome_list, function(chromosome) {
        log_debug("Processing chromosome:", chromosome)
        tryCatch({
            if (aggregate == "Probes") {
                # PROBE-LEVEL PROCESSING
                probe_indices <- getProbeIndices(h5_file, chromosome)
                if (is.null(probe_indices)) return(NULL)
                
                # Read all probe IDs once per chromosome
                probe_ids <- h5read(h5_file, paste0("/", chromosome, "/probeList"))
                missing_prop <- numeric(length(probe_ids))
                
                # Process in chunks
                for (i in seq(1, length(probe_indices), by = chunk)) {
                    end <- min(i + chunk - 1, length(probe_indices))
                    chunk_idx <- probe_indices[i:end]
                    
                    data <- readChromosomeData(h5_file, chromosome, cpg_indices = chunk_idx)
                    if (is.null(data)) next
                    
                    # Calculate missing proportions for this chunk
                    mat <- as.matrix(data[, -"probe"])
                    missing_prop[chunk_idx] <- rowMeans(is.na(mat))
                }
                
                data.table(
                    CpG = paste(chromosome, probe_ids, sep = "_"),
                    missing_proportion = missing_prop
                )
                
            } else {
                # INDIVIDUAL-LEVEL PROCESSING
                # Get total probes for this chromosome
                probe_indices <- getProbeIndices(h5_file, chromosome)
                if (is.null(probe_indices)) return(NULL)
                n_probes <- length(probe_indices)
                
                # Initialize storage
                missing_counts <- integer(length(sample_indices))
                
                # Process samples in chunks
                for (i in seq_along(sample_indices)) {
                    chunk_start <- i
                    chunk_end <- min(i + chunk - 1, length(sample_indices))
                    current_samples <- sample_indices[chunk_start:chunk_end]
                    
                    data <- readChromosomeData(h5_file, chromosome, sample_indices = current_samples)
                    if (is.null(data)) next
                    
                    # Calculate missing counts for this chunk
                    mat <- as.matrix(data[, -"probe"])
                    missing_counts[chunk_start:chunk_end] <- colSums(is.na(mat))
                }
                
                data.table(
                    sample = all_sample_names[sample_indices],
                    missing_count = missing_counts,
                    total_count = n_probes
                )
            }
        }, error = function(e) {
            log_error("Chromosome", chromosome, ":", e$message)
            return(NULL)
        })
    }, mc.cores = cores, mc.preschedule = FALSE)
    
    # Combine results
    all_results <- rbindlist(results_list, use.names = TRUE)
    
    # Final aggregation
    if (aggregate == "Individual") {
        all_results <- all_results[, .(
            missing_count = sum(missing_count, na.rm = TRUE),
            total_count = sum(total_count, na.rm = TRUE)
        ), by = .(sample)]
        all_results[, missing_proportion := missing_count / total_count]
    }
    
    # Bin results using factor levels
    all_results[, Missing_Range := cut(
        missing_proportion,
        breaks = breaks,
        labels = labels,
        include.lowest = TRUE,
        ordered_result = TRUE
    )]
    
    # Ensure all bins are represented
    final_counts <- data.table(Missing_Range = factor(labels, levels = labels, ordered = TRUE))
    final_counts <- merge(
        final_counts,
        all_results[, .N, by = .(Missing_Range)],
        all.x = TRUE
    )
    final_counts[is.na(N), N := 0][order(Missing_Range)]
    
    return(final_counts)
}

# Source helper functions
if (!interactive()) {
    args <- commandArgs(trailingOnly = FALSE)
    script_dir <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
    source(file.path(script_dir, "utils/loggingUtils.R"))
    source(file.path(script_dir, "utils/getChromosomeList.R"))
    source(file.path(script_dir, "utils/getProbeIndices.R"))
    source(file.path(script_dir, "utils/getSampleIndices.R"))
    source(file.path(script_dir, "utils/readChromosomeData.R"))
} else {
    script_dir <- dirname(normalizePath(parent.frame(2)$ofile))
    source(file.path(script_dir, "loggingUtils.R"))
    source(file.path(script_dir, "getChromosomeList.R"))
    source(file.path(script_dir, "getProbeIndices.R"))
    source(file.path(script_dir, "getSampleIndices.R"))
    source(file.path(script_dir, "readChromosomeData.R"))
}
