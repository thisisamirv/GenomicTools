#!/usr/bin/env Rscript
# Helper function to process beta values
# Import the necessary libraries
suppressPackageStartupMessages({
    library(data.table)
    library(rhdf5)
    library(parallel)
})

# Main function
getBetaDist <- function(h5_file, breaks, labels, chunk, mode = c("individual", "mean")) {
    log_info("Aggregating by", capitalize(mode))
    
    # Set the number of cores to use for parallel processing
    cores <- detectCores() - 1
    log_info("Using", cores, "cores for parallel processing.")
    
    cat(sprintf("Processing beta %s values...\n", mode))
    
    # Get the list of chromosomes
    chromosome_list <- getChromosomeList(h5_file)
    if (is.null(chromosome_list)) {
        stop("No valid chromosomes found in HDF5 file")
    }
    
    # Initialize variables based on mode
    if (mode == "individual") {
        # Read all sample names and determine number of samples
        sample_names <- h5read(h5_file, "/metadata/sampleList")
        n_cols <- length(sample_names)
    } else {
        # Initialize data.table for mean mode
        cpg_counts <- data.table(Value_Range = labels, Count = integer(length(labels)))
    }
    
    # Process chromosomes in parallel
    results_list <- mclapply(chromosome_list, function(chromosome) {
        log_debug("Processing chromosome:", chromosome)
        tryCatch({
            if (mode == "individual") {
                results <- list()
                # Process samples in chunks
                for (i in seq(1, n_cols, by = chunk)) {
                    end <- min(i + chunk - 1, n_cols)
                    current_samples <- sample_names[i:end]
                    
                    # Get sample indices for current chunk
                    sample_indices <- getSampleIndices(h5_file, current_samples)
                    if (is.null(sample_indices)) {
                        log_warn("Skipping sample chunk", i, "-", end)
                        next
                    }
                    
                    # Read data for current chunk
                    chr_data <- readChromosomeData(
                        h5_file, chromosome, 
                        cpg_indices = NULL, 
                        sample_indices = sample_indices
                    )
                    
                    if (is.null(chr_data)) {
                        log_warn("Empty data chunk for samples", i, "-", end)
                        next
                    }
                    
                    # Melt data and calculate counts
                    melted <- melt(
                        chr_data, 
                        id.vars = "probe", 
                        variable.name = "Sample", 
                        value.name = "Beta"
                    )
                    
                    melted[, Value_Range := cut(
                        Beta,
                        breaks = breaks,
                        labels = labels,
                        include.lowest = TRUE
                    )]
                    chunk_counts <- melted[, .N, by = .(Sample, Value_Range)]
                    results <- c(results, list(chunk_counts))
                }
                return(if (length(results) > 0) rbindlist(results) else NULL)
                
            } else {
                # Get total number of probes for chromosome
                probe_list <- h5read(h5_file, paste0("/", chromosome, "/probeList"))
                n_rows <- length(probe_list)
                
                # Process probes in chunks
                for (i in seq(1, n_rows, by = chunk)) {
                    end <- min(i + chunk - 1, n_rows)
                    
                    # Read data for current probe chunk
                    chr_data <- readChromosomeData(
                        h5_file, chromosome,
                        cpg_indices = i:end,
                        sample_indices = NULL
                    )
                    
                    if (is.null(chr_data)) {
                        log_warn("Empty data chunk for probes", i, "-", end)
                        next
                    }
                    
                    # Calculate counts for current chunk
                    betas_data <- as.vector(unlist(chr_data[, -"probe"]))
                    value_classes <- cut(
                        betas_data,
                        breaks = breaks,
                        labels = labels,
                        include.lowest = TRUE
                    )
                    counts <- table(value_classes)
                    
                    # Update counts safely
                    cpg_counts[, Count := Count + counts[as.character(Value_Range)]]
                }
                return(cpg_counts)
            }
        }, error = function(e) {
            log_error("Error in chromosome", chromosome, ":", e$message)
            return(NULL)
        })
    }, mc.cores = cores, mc.preschedule = FALSE)
    
    # Combine results
    if (mode == "individual") {
        combined <- rbindlist(results_list)
        # Handle cases with no data
        if (is.null(combined) || nrow(combined) == 0) {
            return(data.table(Value_Range = labels)[, cbind(.SD, setNames(as.list(rep(0, n_cols)), sample_names))])
        }
        
        # Ensure all factor levels are present
        final_dt <- dcast(
            combined,
            Value_Range ~ Sample,
            value.var = "N",
            fill = 0,
            drop = FALSE
        )
        
        # Add missing levels if any
        missing_levels <- setdiff(labels, final_dt$Value_Range)
        if (length(missing_levels) > 0) {
            final_dt <- rbindlist(list(
                final_dt,
                data.table(Value_Range = missing_levels)
            ), fill = TRUE)
        }
        final_dt[is.na(final_dt)] <- 0
        return(final_dt[order(factor(Value_Range, levels = labels))])
        
    } else { # Mean mode
        final_counts <- rbindlist(results_list)[, .(Count = sum(Count, na.rm = TRUE)), by = .(Value_Range)]
        # Ensure all labels are present
        final_counts <- final_counts[.(labels), on = "Value_Range"][
            is.na(Count), Count := 0
        ]
        return(final_counts)
    }
}

# Source helper functions
if (!interactive()) {
    args <- commandArgs(trailingOnly = FALSE)
    script_dir <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
    source(file.path(script_dir, "utils/loggingUtils.R"))
    source(file.path(script_dir, "utils/getChromosomeList.R"))
    source(file.path(script_dir, "utils/getSampleIndices.R"))
    source(file.path(script_dir, "utils/readChromosomeData.R"))
} else {
    script_dir <- dirname(normalizePath(parent.frame(2)$ofile))
    source(file.path(script_dir, "loggingUtils.R"))
    source(file.path(script_dir, "getChromosomeList.R"))
    source(file.path(script_dir, "getSampleIndices.R"))
    source(file.path(script_dir, "readChromosomeData.R"))
}
