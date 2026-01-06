#!/usr/bin/env Rscript
# Import required modules
load_packages <- function() {
    suppressPackageStartupMessages({
        library(argparse)
        library(compiler)
        library(MASS)
        library(Matrix)
    })
    enableJIT(4)
}


# Function to setup logging by sourcing LoggingUtils.R
setup_logging <- function() {
    .logging_placeholders <- c(
        "COLORS", "colorize", "initialize_log_file", "get_log_file_paths",
        "store_log_files", "get_stored_log_files", "create_console_appender",
        "resolve_level_name", "format_log_args", "normalize_log_level",
        "format_log_line", "append_log_to_files", "setup_logger",
        "log_info", "log_debug", "log_warn", "log_error", "log_success"
    )
    for (nm in .logging_placeholders) {
        if (!exists(nm, envir = .GlobalEnv, inherits = FALSE)) {
            assign(nm, NULL, envir = .GlobalEnv)
        }
    }

    script_path <- commandArgs(trailingOnly = FALSE)
    script_path <- script_path[grep("--file=", script_path)]
    script_path <- sub("--file=", "", script_path)
    script_dir <- dirname(script_path)
    source(file.path(script_dir, "LoggingUtils.R"))
    assign("COLORS", get0("COLORS", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("colorize", get0("colorize", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("initialize_log_file", get0("initialize_log_file", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("get_log_file_paths", get0("get_log_file_paths", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("store_log_files", get0("store_log_files", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("get_stored_log_files", get0("get_stored_log_files", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("create_console_appender", get0("create_console_appender", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("resolve_level_name", get0("resolve_level_name", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("format_log_args", get0("format_log_args", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("normalize_log_level", get0("normalize_log_level", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("format_log_line", get0("format_log_line", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("append_log_to_files", get0("append_log_to_files", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("setup_logger", get0("setup_logger", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("log_info", get0("log_info", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("log_debug", get0("log_debug", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("log_warn", get0("log_warn", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("log_error", get0("log_error", envir = .GlobalEnv), envir = .GlobalEnv)
    assign("log_success", get0("log_success", envir = .GlobalEnv), envir = .GlobalEnv)
}

# Function to compute non-truncated (homogeneous) test statistic
Non_Truncated_TestScore <- function(X, SampleSize, CorrMatrix) {
    # Create weight vector from sample sizes
    Wi <- matrix(SampleSize, nrow = 1)

    # Normalize weights by their Euclidean norm
    sumW <- sqrt(sum(Wi^2))
    W <- Wi / sumW

    # Compute generalized inverse of correlation matrix
    Sigma <- ginv(CorrMatrix)

    # Apply test statistic calculation to each row of X (each study set)
    XX <- apply(X, 1, function(x) {
        # Convert effect size vector to row matrix
        x1 <- matrix(x, ncol = length(x), nrow = 1)

        # Compute numerator: weighted sum of effect sizes
        numerator <- W %*% Sigma %*% t(x1)

        # Compute test statistic: numerator squared divided by variance
        # This follows the form of a Wald test statistic
        test_stat <- (numerator * numerator) / (W %*% Sigma %*% t(W))

        test_stat[1, 1]
    })
    XX
}

# Compile the homogeneous test function for better performance
SHom <- compiler::cmpfun(Non_Truncated_TestScore)

# Function to compute truncated (heterogeneous) test statistic
Truncated_TestScore <- function(
    X,
    SampleSize,
    CorrMatrix,
    correct = 1,
    startCutoff = 0,
    endCutoff = 1,
    CutoffStep = 0.05,
    isAllpossible = TRUE
) {

    # Get number of studies
    N <- dim(X)[2]

    # Create and normalize weight vector
    Wi <- matrix(SampleSize, nrow = 1)
    sumW <- sqrt(sum(Wi^2))
    W <- Wi / sumW

    # Apply truncated test statistic to each row (study set)
    XX <- apply(X, 1, function(x) {
        max_test_score <- -1

        # Define threshold values to test
        if (isAllpossible) {
            # Use all unique absolute effect sizes as potential thresholds
            cutoff <- sort(unique(abs(x)))
        } else {
            # Use regular grid of threshold values
            cutoff <- seq(startCutoff, endCutoff, CutoffStep)
        }

        # Test each threshold value
        for (threshold in cutoff) {
            x1 <- x

            # Find studies with effect sizes below threshold (to be excluded)
            index <- which(abs(x1) < threshold)

            # If all studies would be excluded, break
            if (length(index) == N) break

            # Initialize matrices for this threshold
            A <- CorrMatrix
            W1 <- W

            # Remove studies below threshold
            if (length(index) != 0) {
                x1 <- x1[-index]
                A <- A[-index, -index]
                W1 <- W[-index]
            }

            # Apply sign correction if requested
            if (correct == 1) {
                negative_index <- which(x1 < 0)
                if (length(negative_index) != 0) {
                    # Flip weights for negative effect sizes to test for consistency
                    W1[negative_index] <- -W1[negative_index]
                }
            }

            # Compute test statistic for this subset
            A <- ginv(A)
            x1 <- matrix(x1, nrow = 1)
            W1 <- matrix(W1, nrow = 1)
            stat_val <- W1 %*% A %*% t(x1)
            stat_val <- (stat_val * stat_val) / (W1 %*% A %*% t(W1))

            # Keep track of maximum test statistic across all thresholds
            if (max_test_score < stat_val[1, 1]) {
                max_test_score <- stat_val[1, 1]
            }
        }

        max_test_score
    })

    return(XX)
}

# Compile the heterogeneous test function for better performance
SHet <- compiler::cmpfun(Truncated_TestScore)

# Function to estimate gamma distribution parameters for null distribution
EstimateGamma <- function(
    N = 1E6,
    SampleSize,
    CorrMatrix,
    correct = 1,
    startCutoff = 0,
    endCutoff = 1,
    CutoffStep = 0.05,
    isAllpossible = TRUE
) {

    # Generate random samples from null distribution (multivariate normal)
    Permutation <- mvrnorm(
        n = N,
        mu = rep(0, length(SampleSize)),
        Sigma = CorrMatrix,
        tol = 1e-8,
        empirical = FALSE
    )

    # Compute test statistics for all simulated samples
    Stat <- Truncated_TestScore(
        X = Permutation,
        SampleSize = SampleSize,
        CorrMatrix = CorrMatrix,
        correct = correct,
        startCutoff = startCutoff,
        endCutoff = endCutoff,
        CutoffStep = CutoffStep,
        isAllpossible = isAllpossible
    )

    # Initial parameter estimates using method of moments
    a <- min(Stat) * 3 / 4
    ex3 <- mean(Stat^3)
    V <- var(Stat)

    # Iterative refinement of parameter estimates (Newton-Raphson style)
    for (i in 1:100) {
        E <- mean(Stat) - a
        k <- E^2 / V
        theta <- V / E

        # Pre-compute repeated expressions for efficiency
        k_plus_1 <- k + 1
        k_plus_2 <- k + 2
        theta_squared <- theta^2
        theta_cubed <- theta^3
        theta_fourth <- theta^4

        # Solve cubic equation for location parameter 'a'
        # This comes from matching the third moment of shifted gamma distribution
        first_term <- 9 * k^2 * k_plus_1^2 * theta_fourth

        inner_expression <- k * k_plus_1 * k_plus_2 * theta_cubed - ex3
        second_term <- 12 * k * theta * inner_expression

        discriminant <- first_term - second_term

        numerator_first_part <- -3 * k * k_plus_1 * theta_squared
        numerator_second_part <- sqrt(discriminant)
        denominator <- 6 * k * theta

        # Update location parameter
        a <- (numerator_first_part + numerator_second_part) / denominator
    }

    # Return estimated parameters: shape, scale, location
    para <- c(k, theta, a)
    return(para)
}

# Function to generate empirical null distribution
EmpDist <- function(
    N = 1E6,
    SampleSize,
    CorrMatrix,
    correct = 1,
    startCutoff = 0,
    endCutoff = 1,
    CutoffStep = 0.05,
    isAllpossible = TRUE
) {

    # Generate random samples from null distribution
    Permutation <- mvrnorm(
        n = N,
        mu = rep(0, length(SampleSize)),
        Sigma = CorrMatrix,
        tol = 1e-8,
        empirical = FALSE
    )

    # Compute and return test statistics for empirical distribution
    Stat <- Truncated_TestScore(
        X = Permutation,
        SampleSize = SampleSize,
        CorrMatrix = CorrMatrix,
        correct = correct,
        startCutoff = startCutoff,
        endCutoff = endCutoff,
        CutoffStep = CutoffStep,
        isAllpossible = isAllpossible
    )

    return(Stat)
}

# Run Multi-Trait Association Analysis using CPASSOC
RunCPASSOC <- function(
    input_files,
    correlation_matrix,
    sample_sizes,
    output_file,
    marker_col = "CGID",
    t_col = "T-STAT",
    cols_to_add = NULL,
    traits = c("trait1", "trait2"),
    alpha = 0.05
) {
    # Input validation
    if (length(input_files) < 2) {
        stop("At least 2 files are required for multi-trait analysis")
    }

    if (length(sample_sizes) != length(input_files)) {
        stop("Number of sample sizes must match number of files")
    }

    # Check if all input files exist
    missing_files <- input_files[!file.exists(input_files)]
    if (length(missing_files) > 0) {
        stop(paste("The following files do not exist:", paste(missing_files, collapse = ", ")))
    }

    # Validate marker/t column specification: allow length 1 (broadcast) or exactly num files
    n_files <- length(input_files)
    if (!(length(marker_col) %in% c(1, n_files))) {
        stop("marker_col must be length 1 or match number of input files")
    }
    if (!(length(t_col) %in% c(1, n_files))) {
        stop("t_col must be length 1 or match number of input files")
    }

    log_info("Starting multi-trait association analysis...")
    log_info("Number of traits:", length(input_files))
    log_info("Sample sizes:", paste(sample_sizes, collapse = ", "))

    # Load correlation matrix
    if (is.character(correlation_matrix)) {
        log_info("Loading correlation matrix from file:", correlation_matrix)
        if (!file.exists(correlation_matrix)) {
            stop(paste("Correlation matrix file does not exist:", correlation_matrix))
        }
        corr_matrix <- as.matrix(read.csv(correlation_matrix, row.names = 1))
    } else if (is.matrix(correlation_matrix)) {
        corr_matrix <- correlation_matrix
    } else {
        stop("correlation_matrix must be either a matrix or path to CSV file")
    }

    # Validate correlation matrix dimensions
    if (nrow(corr_matrix) != ncol(corr_matrix)) {
        stop("Correlation matrix must be square")
    }

    if (nrow(corr_matrix) != length(input_files)) {
        stop("Correlation matrix dimensions must match number of files")
    }

    # Check if correlation matrix is symmetric and has 1s on diagonal
    if (!isSymmetric(corr_matrix)) {
        stop("Correlation matrix must be symmetric")
    }

    if (!all(abs(diag(corr_matrix) - 1) < 1e-10)) {
        stop("Correlation matrix diagonal elements must be 1")
    }

    log_info("Correlation matrix validation passed")

    # Read summary statistics files
    log_info("Reading summary statistics files...")

    input_data_list <- list()

    # marker_col and t_col may be vectors (one per input file) or length-1 (broadcast)
    for (i in seq_along(input_files)) {
        log_info("  Reading file", i, ":", basename(input_files[i]))

        # Read the file
        data <- read.csv(input_files[i], stringsAsFactors = FALSE, check.names = FALSE)

        # Select per-file marker and t column names (broadcast if length==1)
        marker_i <- if (length(marker_col) == 1) marker_col[1] else marker_col[i]
        t_i <- if (length(t_col) == 1) t_col[1] else t_col[i]
        trait_i <- if (!is.null(traits) && length(traits) >= i) traits[i] else paste0("Trait", i)

        # Check required columns for this file
        required_cols <- c(marker_i, t_i)
        missing_cols <- required_cols[!required_cols %in% colnames(data)]
        if (length(missing_cols) > 0) {
            stop(paste("Missing columns in", input_files[i], ":", paste(missing_cols, collapse = ", ")))
        }

        # Store data and column info
        input_data_list[[i]] <- list(
            data = data,
            marker_col = marker_i,
            t_col = t_i,
            trait = trait_i
        )

        log_info("    Rows:", nrow(data))
        log_info("    Marker sites with missing t values:", sum(is.na(data[[t_i]])))
        log_info("    Marker sites with missing p values:", sum(is.na(data[[p_i]])))
    }

    # Find common marker sites across all files
    log_info("Finding common marker sites across all files...")

    marker_sets <- lapply(input_data_list, function(x) x$data[[x$marker_col]])
    common_markers <- Reduce(intersect, marker_sets)

    if (length(common_markers) == 0) {
        stop("No common marker sites found across all files")
    }

    log_info("  Total common marker sites:", length(common_markers))
    for (i in seq_along(input_files)) {
        log_info("  Marker sites in file", i, ":", length(marker_sets[[i]]))
    }

    # Prepare t matrix for common markers
    log_info("Preparing t matrix...")

    t_matrix <- matrix(NA, nrow = length(common_markers), ncol = length(input_files))
    p_matrix <- matrix(NA, nrow = length(common_markers), ncol = length(input_files))
    rownames(t_matrix) <- common_markers
    rownames(p_matrix) <- common_markers
    colnames(t_matrix) <- paste0("Trait_", seq_along(input_files))
    colnames(p_matrix) <- paste0("Trait_", seq_along(input_files))

    for (i in seq_along(input_data_list)) {
        data <- input_data_list[[i]]$data
        marker_name <- input_data_list[[i]]$marker_col
        t_name <- input_data_list[[i]]$t_col
        marker_indices <- match(common_markers, data[[marker_name]])
        t_rhs <- rep(NA_real_, length(marker_indices))
        p_rhs <- rep(NA_real_, length(marker_indices))
        valid <- which(!is.na(marker_indices) & marker_indices >= 1 & marker_indices <= nrow(data))
        if (length(valid) > 0) {
            t_rhs[valid] <- as.numeric(data[[t_name]][marker_indices[valid]])
        }
        t_matrix[, i] <- t_rhs
        p_matrix[, i] <- p_rhs
    }

    # Remove rows with any missing t values
    complete_rows <- complete.cases(t_matrix)
    if (sum(complete_rows) == 0) stop("No complete markers with t across traits")

    t_matrix <- t_matrix[complete_rows, , drop = FALSE]
    p_matrix <- p_matrix[complete_rows, , drop = FALSE]
    final_markers <- common_markers[complete_rows]

    log_info("Running CPASSOC (SHom) analysis...")

    # Run CPASSOC homogeneous test
    cpassoc_stat <- SHom(
        X = t_matrix,
        SampleSize = sample_sizes,
        CorrMatrix = corr_matrix
    )

    # Calculate p-values from chi-squared distribution
    p_values <- pchisq(cpassoc_stat, df = 1, lower.tail = FALSE)

    log_info("Preparing results...")

    # Determine output column name for markers
    marker_col_names <- sapply(input_data_list, function(x) x$marker_col)
    marker_output_name <- if (length(unique(marker_col_names)) == 1) marker_col_names[1] else "Marker"

    # Create results data frame with dynamic marker column name
    results_df <- data.frame(
        tempMarker = final_markers,
        SHom_ChiSq = cpassoc_stat,
        SHom_P = p_values,
        stringsAsFactors = FALSE
    )
    colnames(results_df)[colnames(results_df) == "tempMarker"] <- marker_output_name

    # Add requested extra columns from input files (cols_to_add)
    if (!is.null(cols_to_add) && length(cols_to_add) > 0 && !all(cols_to_add == "")) {
        requested_cols <- cols_to_add
        # Count occurrences of each requested column across input files
        occ <- sapply(requested_cols, function(col) {
            sum(sapply(input_data_list, function(x) col %in% colnames(x$data)))
        }, USE.NAMES = TRUE)

        for (col in requested_cols) {
            if (is.na(col) || col == "") next
            nocc <- occ[[col]]
            if (is.null(nocc) || nocc == 0) {
                log_warn("Requested column not found in any input file:", col)
                next
            }

            if (nocc == 1) {
                # Add single column with original name
                for (i in seq_along(input_data_list)) {
                    if (!(col %in% colnames(input_data_list[[i]]$data))) next
                    data_i <- input_data_list[[i]]$data
                    marker_name <- input_data_list[[i]]$marker_col
                    idx <- match(final_markers, data_i[[marker_name]])
                    vals <- rep(NA, length(idx))
                    valid <- which(!is.na(idx) & idx >= 1 & idx <= nrow(data_i))
                    if (length(valid) > 0) vals[valid] <- data_i[[col]][idx[valid]]
                    label <- col
                    safe_label <- gsub("[^A-Za-z0-9_]", "_", label)
                    if (safe_label %in% colnames(results_df)) safe_label <- paste0(safe_label, "_", i)
                    results_df[[safe_label]] <- vals
                    break
                }
            } else {
                # Appears in multiple files: add one column per file with suffix _[trait]
                for (i in seq_along(input_data_list)) {
                    if (!(col %in% colnames(input_data_list[[i]]$data))) next
                    data_i <- input_data_list[[i]]$data
                    marker_name <- input_data_list[[i]]$marker_col
                    idx <- match(final_markers, data_i[[marker_name]])
                    vals <- rep(NA, length(idx))
                    valid <- which(!is.na(idx) & idx >= 1 & idx <= nrow(data_i))
                    if (length(valid) > 0) vals[valid] <- data_i[[col]][idx[valid]]
                    trait_label <- input_data_list[[i]]$trait
                    label <- paste0(col, "_", trait_label)
                    safe_label <- gsub("[^A-Za-z0-9_]", "_", label)
                    if (safe_label %in% colnames(results_df)) safe_label <- paste0(safe_label, "_", i)
                    results_df[[safe_label]] <- vals
                }
            }
        }
    }

    # Run CPASSOC heterogeneous test
    log_info("Running CPASSOC (SHet) analysis...")
    Test_shet <- SHet(
        X = t_matrix,
        SampleSize = sample_sizes,
        CorrMatrix = corr_matrix
    )

    # Estimate gamma distribution parameters for null distribution
    para <- tryCatch(
        EstimateGamma(1e4, sample_sizes, corr_matrix),
        error = function(e) {
            log_warn("Gamma estimation failed — using empirical null.")
            Stat <- EmpDist(1e6, sample_sizes, corr_matrix)
            list(empirical = Stat)
        }
    )
    if (!is.list(para) || !("empirical" %in% names(para))) {
        # Calculate p-values from gamma distribution
        log_info("Using gamma distribution for SHet p-values.")
        p_shet <- pgamma(Test_shet - para[3], shape = para[1], scale = para[2], lower.tail = FALSE)
    } else {
        # Fallback to empirical p-values
        log_info("Using empirical distribution for SHet p-values.")
        p_shet <- sapply(Test_shet, function(x) mean(para$empirical >= x))
    }

    # Add SHet results to data frame
    results_df$SHet_P <- p_shet

    # Add individual trait p-values
    for (i in seq_along(input_files)) {
        trait_label <- input_data_list[[i]]$trait
        col_label <- paste0(trait_label, "_P")
        safe_label <- gsub("[^A-Za-z0-9_]", "_", col_label)
        if (safe_label %in% colnames(results_df)) {
            safe_label <- paste0(safe_label, "_", i)
        }
        results_df[[safe_label]] <- p_matrix[, i]
    }

    # Apply multiple testing correction
    mtests <- nrow(results_df)
    results_df$P_BONF_SHom <- pmin(1, results_df$SHom_P * mtests)
    results_df$P_FDR_SHom <- p.adjust(results_df$SHom_P, method = "BH")
    results_df$P_BONF_SHet <- pmin(1, results_df$SHet_P * mtests)
    results_df$P_FDR_SHet <- p.adjust(results_df$SHet_P, method = "BH")

    # Sort by SHom_P (ascending)
    results_df <- results_df[order(results_df$SHom_P), ]

    first_cols <- c(marker_output_name, "SHom_ChiSq", "SHom_P", "SHet_P")
    first_cols <- c(first_cols, "P_BONF_SHom", "P_FDR_SHom", "P_BONF_SHet", "P_FDR_SHet")

    other_cols <- setdiff(colnames(results_df), first_cols)
    results_df <- results_df[, c(first_cols, other_cols)]

    # Save results
    log_info("Saving results to:", output_file)

    # Create output directory if it doesn't exist
    output_dir <- dirname(output_file)
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }

    write.csv(results_df, output_file, row.names = FALSE)

    log_info("Analysis completed successfully!")
    log_info("Results saved to:", output_file)
    log_info("Total marker sites analyzed:", nrow(results_df))

    # Report counts for multiple-testing significance
    n_bonf_sig_shom <- sum(results_df$P_BONF_SHom < alpha, na.rm = TRUE)
    n_fdr_sig_shom <- sum(results_df$P_FDR_SHom < alpha, na.rm = TRUE)
    n_bonf_sig_shet <- sum(results_df$P_BONF_SHet < alpha, na.rm = TRUE)
    n_fdr_sig_shet <- sum(results_df$P_FDR_SHet < alpha, na.rm = TRUE)
    log_info("Significant marker sites for SHom (Bonferroni):", n_bonf_sig_shom)
    log_info("Significant marker sites for SHom (FDR):", n_fdr_sig_shom)
    log_info("Significant marker sites for SHet (Bonferroni):", n_bonf_sig_shet)
    log_info("Significant marker sites for SHet (FDR):", n_fdr_sig_shet)

    return(results_df)
}

#' Parse command-line arguments
#' input_files: Comma-separated paths to CSV files
#' correlation_matrix: Path to correlation matrix CSV file or R matrix
#' sample_sizes: Comma-separated sample sizes for each trait
#' output_file: Path to output CSV file
#' marker_col: Name of marker ID column
#' t_col: Name of t value column
#' cols_to_add: Other columns from the input files to add to output
#' traits: Comma-separated trait names
#' alpha: Significance threshold
#' log_level: Log level (INFO, DEBUG, WARN, ERROR)
parse_args <- function() {
    parser <- argparse::ArgumentParser(description = "Run CPASSOC multi-trait Association Analysis")
    parser$add_argument("--input", required = TRUE)
    parser$add_argument("--correlation_matrix", required = TRUE)
    parser$add_argument("--sample_sizes", required = TRUE)
    parser$add_argument("--output_file", required = TRUE)
    parser$add_argument("--marker_col", default = "CGID")
    parser$add_argument("--t_col", default = "T-STAT")
    parser$add_argument("--cols_to_add", default = NULL)
    parser$add_argument("--traits", default = NULL)
    parser$add_argument("--alpha", type = "double", default = 0.05)
    parser$add_argument("--log_level", default = "INFO")
    args <- parser$parse_args()

    # split comma-separated fields
    args$input <- strsplit(args$input, ",")[[1]]
    args$sample_sizes <- as.numeric(strsplit(args$sample_sizes, ",")[[1]])
    args$marker_col <- strsplit(args$marker_col, ",")[[1]]
    args$t_col <- strsplit(args$t_col, ",")[[1]]
    args$cols_to_add <- strsplit(args$cols_to_add, ",")[[1]]

    # Handle traits: if NULL or empty string, generate default Trait1..TraitN
    n_files <- length(args$input)
    if (is.null(args$traits) || identical(args$traits, "")) {
        args$traits <- paste0("Trait", seq_len(n_files))
    } else {
        args$traits <- strsplit(args$traits, ",")[[1]]
        if (length(args$traits) == 1 && identical(args$traits, "")) {
            args$traits <- paste0("Trait", seq_len(n_files))
        }
    }
    args$correlation_matrix <- args$correlation_matrix

    args
}

main <- function() {
    print("Starting CPASSOC R script...")
    load_packages()
    setup_logging()
    args <- parse_args()
    setup_logger(args$log_level)

    print("Running CPASSOC analysis...")
    res <- RunCPASSOC(
        input_files = args$input,
        correlation_matrix = args$correlation_matrix,
        sample_sizes = args$sample_sizes,
        output_file = args$output_file,
        marker_col = args$marker_col,
        t_col = args$t_col,
        cols_to_add = args$cols_to_add,
        traits = args$traits,
        alpha = args$alpha
    )

    invisible(res)
}

main()
