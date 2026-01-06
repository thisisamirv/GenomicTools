#!/usr/bin/env Rscript
# Import required modules
suppressPackageStartupMessages({
    library(data.table)
    library(limma)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
    stop("Usage: PowerLimma.R input_csv n_cnt n_tx output_csv")
}

input_csv <- args[1]
n_cnt <- as.integer(args[2])
n_tx  <- as.integer(args[3])
output_csv <- args[4]

# Read chunk of data
mat <- tryCatch({
    as.matrix(fread(input_csv, header = FALSE))
}, error = function(e) {
    stop(sprintf("Failed to read input CSV: %s", e$message))
})

# Check matrix dimensions and transpose if needed
n_samples <- n_cnt + n_tx
if (ncol(mat) != n_samples) {
    if (nrow(mat) == n_samples && ncol(mat) != n_samples) {
        mat <- t(mat)
    } else {
        stop(sprintf("Matrix dimensions mismatch: expected %d samples but got %d", n_samples, ncol(mat)))
    }
}

# Create design matrix
group <- c(rep(0L, n_cnt), rep(1L, n_tx))
design <- cbind(Intercept = 1, Group = as.numeric(group))

# Handle trivial cases
if (ncol(mat) < 2 || nrow(mat) == 0) {
    pvals <- rep(NA_real_, nrow(mat))
    fwrite(data.table(pval = pvals), output_csv)
    quit(status = 0)
}

# Fit model and compute p-values
result_pvals <- tryCatch({
    fit <- lmFit(mat, design)
    eb <- eBayes(fit)
    if (!is.null(eb$p.value) && ncol(eb$p.value) >= 2) {
        pvals <- eb$p.value[, 2]
    } else {
        pvals <- rep(NA_real_, nrow(mat))
    }
    pvals
}, error = function(e) {
    warning(sprintf("limma/eBayes failed: %s", e$message))
    rep(NA_real_, nrow(mat))
})

# Write output p-values
fwrite(data.table(pval = result_pvals), output_csv, col.names = TRUE)
