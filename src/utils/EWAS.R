#!/usr/bin/env Rscript
# Import required modules
load_packages <- function() {
    suppressPackageStartupMessages({
        library(argparse)
        library(bacon)
        library(data.table)
        library(doParallel)
        library(dplyr)
        library(foreach)
        library(glue)
        library(limma)
        library(parallel)
        library(progressr)
        library(rhdf5)
        library(tibble)
    })
}

initialize_environment <- function() {
    set.seed(123)
    options(matprod = "blas")
    rhdf5::h5disableFileLocking()
    progressr::handlers("txtprogressbar")
    progressr::handlers(global = TRUE)
}

optimize_memory <- function(memory_per_core = 2.0) {
    memory_limit_gb <- memory_per_core * 0.95
    if (.Platform$OS.type == "windows") {
        if (memory_limit_gb * 1024 > 0) {
            tryCatch({
                utils::memory.limit(size = ceiling(memory_limit_gb * 1024))
                log_info("R memory limit set to %.2f GB (%.0f MB)", memory_limit_gb, memory_limit_gb * 1024)
            }, error = function(e) {
                log_warn("Could not set memory limit: %s", e$message)
            })
        }
    } else {
        log_info("Memory limit setting skipped (not supported on %s)", .Platform$OS.type)
    }
    gc_settings <- list(
        gc.level = 2,
        nsize = memory_limit_gb * 0.05 * 1e6,
        vsize = memory_limit_gb * 0.7 * 1e6
    )
    tryCatch({
        do.call(options, gc_settings)
        log_info("GC optimization configured")
    }, error = function(e) {
        log_warn("Could not configure GC settings: %s", e$message)
    })
    gc(full = TRUE)
}

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

read_group_probes <- function(h5_file, group_name, marker_list_name) {
    tryCatch({
        probes <- rhdf5::h5read(h5_file, glue("{group_name}/{marker_list_name}"))
        probes
    }, error = function(e) {
        stop(glue("Error reading probes for group {group_name}: {e$message}"))
    }, finally = {
        rhdf5::h5closeAll()
    })
}

load_beta_chunk <- function(
    h5_file,
    chunk_range,
    group_probes,
    sample_ix,
    group_name,
    mvalue = TRUE,
    full_sample_list = NULL,
    marker_list_name = "probeList",
    betas_name = "betas",
    data_orientation = "markers_as_rows",
    max_attempts = 5
) {
    probe_start <- chunk_range[1]
    probe_end <- chunk_range[2]
    chunk_probes <- group_probes[probe_start:probe_end]
    n_probes <- length(chunk_probes)

    log_debug(
        "Loading beta chunk from %s/%s (range %d–%d, %d probes, orientation=%s)",
        group_name, betas_name, probe_start, probe_end, n_probes, data_orientation
    )

    betas <- NULL
    for (attempt in seq_len(max_attempts)) {
        rhdf5::h5closeAll()
        betas <- tryCatch({
            if (data_orientation == "markers_as_rows") {
                rhdf5::h5read(
                    h5_file,
                    glue("{group_name}/{betas_name}"),
                    index = list(probe_start:probe_end, NULL)
                )
            } else if (data_orientation == "samples_as_rows") {
                rhdf5::h5read(
                    h5_file,
                    glue("{group_name}/{betas_name}"),
                    index = list(NULL, probe_start:probe_end)
                )
            } else {
                stop("Unknown data_orientation: ", data_orientation)
            }
        }, error = function(e) {
            log_warn(
                "Attempt %d failed for %s (%d–%d): %s",
                attempt, group_name, probe_start, probe_end, e$message
            )
            Sys.sleep(2)
            NULL
        })

        if (!is.null(betas)) break
    }
    if (is.null(betas)) {
        log_error("Failed to load beta chunk %d–%d from %s", probe_start, probe_end, group_name)
        return(NULL)
    }

    if (data_orientation == "samples_as_rows") {
        betas <- t(betas)
    }

    rownames(betas) <- chunk_probes

    if (!is.null(full_sample_list)) {
        colnames(betas) <- full_sample_list
    }

    if (length(sample_ix) > 0) {
        betas <- betas[, sample_ix, drop = FALSE]
    }

    if (mvalue) {
        log_debug("Transforming beta values to M-values")
        betas <- mvalue_transform(betas)
    }

    return(betas)
}

valid_and_simplify_vars <- function(covariates_table, validate_strict = TRUE) {
    if (is.null(covariates_table) || ncol(covariates_table) == 0) {
        return(NULL)
    }

    log_info("Validating and simplifying %d variables", ncol(covariates_table))

    validated_vars <- list()

    for (col_name in names(covariates_table)) {
        col_data <- covariates_table[[col_name]]
        original_class <- class(col_data)[1]

        if (is.numeric(col_data)) {
            validated_vars[[col_name]] <- col_data

        } else if (is.factor(col_data)) {
            if (validate_strict) {
                n_levels <- length(levels(col_data))

                if (n_levels == 2) {
                    if (!is.ordered(col_data)) {
                        validated_vars[[col_name]] <- as.numeric(col_data) - 1
                        log_info("Variable '%s': binary factor converted to numeric (0/1)", col_name)
                    } else {
                        validated_vars[[col_name]] <- col_data
                        log_debug("Variable '%s': ordered binary factor - validated", col_name)
                    }
                } else if (is.ordered(col_data)) {
                    validated_vars[[col_name]] <- col_data
                    log_debug("Variable '%s': ordered factor (%d levels) - validated", col_name, n_levels)
                } else {
                    log_warn(
                        "Variable '%s': unordered factor with %d levels may cause interpretation issues",
                        col_name,
                        n_levels
                    )
                    validated_vars[[col_name]] <- col_data
                }
            } else {
                validated_vars[[col_name]] <- col_data
                log_debug("Variable '%s': factor - validated (strict validation disabled)", col_name)
            }

        } else if (is.character(col_data)) {
            col_factor <- as.factor(col_data)
            n_levels <- length(levels(col_factor))

            if (validate_strict) {
                if (n_levels == 2) {
                    validated_vars[[col_name]] <- as.numeric(col_factor) - 1
                    log_info("Variable '%s': character converted to binary numeric (0/1)", col_name)
                } else if (n_levels <= 10) {
                    validated_vars[[col_name]] <- col_factor
                    log_info("Variable '%s': character converted to factor (%d levels)", col_name, n_levels)
                } else {
                    log_warn(
                        "Variable '%s': character with %d unique values - may need preprocessing",
                        col_name,
                        n_levels
                    )
                    validated_vars[[col_name]] <- col_factor
                }
            } else {
                validated_vars[[col_name]] <- col_factor
                log_debug("Variable '%s': character converted to factor", col_name)
            }

        } else if (is.logical(col_data)) {
            validated_vars[[col_name]] <- as.numeric(col_data)
            log_info("Variable '%s': logical converted to numeric (0/1)", col_name)

        } else {
            if (validate_strict) {
                log_warn(
                    "Variable '%s': unsupported type '%s' - attempting conversion to factor",
                    col_name,
                    original_class
                )
            }
            validated_vars[[col_name]] <- as.factor(col_data)
        }
    }

    result_df <- as.data.frame(validated_vars)
    log_info("Variable validation complete: %d variables processed", ncol(result_df))

    return(result_df)
}

load_metadata_value <- function(source, field, h5_file, metadata_group) {
    if (source != "NONE" && file.exists(source)) {
        meta <- tryCatch(fread(source), error = function(e) NULL)
        if (!is.null(meta) && field %in% names(meta)) {
            return(meta[[field]])
        } else {
            log_warn("'%s' not found in external metadata, trying HDF5", field)
        }
    }
    tryCatch({
        path <- glue("/{metadata_group}/{field}")
        rhdf5::h5read(h5_file, path)
    }, error = function(e) {
        stop(glue("Failed to read {field} from {metadata_group} sources: {e$message}"))
    })
}

make_chunks <- function(items, items_per_chunk) {
    if (items_per_chunk <= 0) stop("items_per_chunk must be positive")
    if (length(items) <= items_per_chunk) {
        return(list(c(1, length(items))))
    }
    n_chunks <- ceiling(length(items) / items_per_chunk)
    lapply(seq_len(n_chunks), function(i) {
        start <- (i - 1) * items_per_chunk + 1
        end <- min(i * items_per_chunk, length(items))
        c(start, end)
    })
}

mvalue_transform <- function(beta_matrix) {
    mvals <- log2((beta_matrix + 1e-6) / (1 - beta_matrix + 1e-6))
    mvals[!is.finite(mvals)] <- NA
    mvals
}

combine_limma_results <- function(a, b) {
    if (is.null(a) || is.null(a$results)) return(b)
    if (is.null(b) || is.null(b$results)) return(a)
    a$results <- rbind(a$results, b$results)
    a
}

calculate_genomic_inflation <- function(p_values, n_samples = NULL, n_cases = NULL, n_controls = NULL) {
    if (length(p_values) == 0 || all(is.na(p_values))) {
        log_warn("No valid p-values for genomic inflation calculation")
        return(list(lambda_raw = NA, lambda_scaled = NA))
    }

    valid_p <- p_values[!is.na(p_values)]
    if (length(valid_p) == 0) {
        log_warn("No valid p-values after removing NAs")
        return(list(lambda_raw = NA, lambda_scaled = NA))
    }

    chi_obs <- qchisq(valid_p, df = 1, lower.tail = FALSE)
    lambda_raw <- median(chi_obs, na.rm = TRUE) / qchisq(0.5, df = 1)

    lambda_scaled <- NA_real_
    condition1 <- !is.null(n_cases) && !is.null(n_controls)
    condition2 <- is.finite(n_cases) && is.finite(n_controls)
    condition3 <- n_cases > 0 && n_controls > 0
    if (condition1 && condition2 && condition3) {
        n_effective <- (4 * n_cases * n_controls) / (n_cases + n_controls)
        if (is.finite(n_effective) && n_effective > 0) {
            lambda_scaled <- 1 + (lambda_raw - 1) * (1000 / n_effective)
        } else {
            lambda_scaled <- lambda_raw
        }

    } else if (!is.null(n_samples) && is.finite(n_samples) && n_samples > 0) {
        # Non-binary or no case/control counts: scale by total sample size
        lambda_scaled <- 1 + (lambda_raw - 1) * (1000 / n_samples)
    } else {
        lambda_scaled <- lambda_raw
    }

    list(lambda_raw = lambda_raw, lambda_scaled = lambda_scaled)
}

calculate_empirical_null <- function(t_values) {
    if (length(t_values) == 0 || all(is.na(t_values))) {
        log_warn("No valid t-statistics for empirical null calculation")
        return(list(bias = NA, inflation = NA))
    }
    valid_t <- t_values[!is.na(t_values)]
    if (length(valid_t) < 100) {
        log_warn("Too few valid t-values for empirical null estimation")
        return(list(bias = NA, inflation = NA))
    }

    bc <- bacon::bacon(teststatistics = valid_t)
    bias_val <- bacon::bias(bc)
    infl_val <- bacon::inflation(bc)

    list(bias = bias_val, inflation = infl_val)
}

parse_args <- function() {
    parser <- argparse::ArgumentParser(description = "EWAS with limma (lmFit + eBayes)")
    parser$add_argument("--methylation_betas_file", required = TRUE)
    parser$add_argument("--output", required = TRUE)
    parser$add_argument("--metadata", default = "NONE")
    parser$add_argument("--sample_id", default = NULL)
    parser$add_argument("--processes", type = "integer", default = 2)
    parser$add_argument("--chunk_size", type = "integer", default = 500)
    parser$add_argument("--chrom_groups", required = TRUE)
    parser$add_argument("--marker_list_name", default = "probeList")
    parser$add_argument("--sample_list_name", default = "sampleList")
    parser$add_argument("--metadata_group", default = "metadata")
    parser$add_argument("--betas_name", default = "betas")
    parser$add_argument("--data_orientation", default = "markers_as_rows")
    parser$add_argument("--log_level", default = "INFO")
    parser$add_argument("--covariate_names", default = NULL)
    parser$add_argument("--stat_var", default = NULL)
    parser$add_argument("--memory_per_core", type = "double", default = 2.0)
    parser$add_argument("--temp_dir", default = NULL)

    args <- parser$parse_args()
    if (args$processes < 1) stop("Number of processes must be positive")
    if (args$chunk_size < 1) stop("Number of probes per chunk must be positive")
    args$chrom_groups <- strsplit(args$chrom_groups, ",")[[1]]
    if (length(args$chrom_groups) == 0) stop("No chromosome groups provided")
    if (!is.null(args$temp_dir) && nchar(args$temp_dir) > 0 && args$temp_dir != "NONE") {
    }

    args
}

generate_qc_summary <- function(results_table, output_prefix, stat_var = NULL) {
    log_info("Generating quality control summary")

    if (!is.null(stat_var) && nchar(stat_var) > 0) {
        stat_vars <- trimws(unlist(strsplit(stat_var, ",")))
        num_stat_vars <- length(stat_vars)
        n_probes <- nrow(results_table) / num_stat_vars
    } else {
        n_probes <- nrow(results_table)
    }

    n_terms <- length(unique(results_table$Term))

    log_info("====== EWAS Quality Control Summary ======")
    log_info("Generated: %s", Sys.time())
    log_info("")
    log_info("Overall Statistics:")
    log_info("  Total probes analyzed: %d", n_probes)
    log_info("  Number of terms: %d", n_terms)
    log_info("")

    sig_summary <- results_table[, {
        pvals <- .SD[["P_Value"]]
        basic_sig <- sum(pvals < 0.05, na.rm = TRUE)

        fdr_sig <- if ("adj_p_fdr" %in% names(.SD)) sum(.SD[["adj_p_fdr"]] < 0.05, na.rm = TRUE) else NA
        bonf_sig <- if ("adj_p_bonferroni" %in% names(.SD)) sum(.SD[["adj_p_bonferroni"]] < 0.05, na.rm = TRUE) else NA
        holm_sig <- if ("adj_p_holm" %in% names(.SD)) sum(.SD[["adj_p_holm"]] < 0.05, na.rm = TRUE) else NA

        list(
            n_basic_sig = basic_sig,
            n_fdr_sig = fdr_sig,
            n_bonferroni_sig = bonf_sig,
            n_holm_sig = holm_sig,
            n_total = .N
        )
    }, by = "Term"]

    log_info("Significance Counts (p < 0.05):")
    for (i in seq_len(nrow(sig_summary))) {
        row <- sig_summary[i, ]
        term <- row$Term
        n_total <- row$n_total
        n_basic <- row$n_basic_sig
        n_fdr <- if (!is.na(row$n_fdr_sig)) row$n_fdr_sig else "N/A"
        n_bonf <- if (!is.na(row$n_bonferroni_sig)) row$n_bonferroni_sig else "N/A"
        n_holm <- if (!is.na(row$n_holm_sig)) row$n_holm_sig else "N/A"
        log_info(
            "  %s: Basic = %s/%s, FDR = %s, Bonferroni = %s, Holm = %s",
            term,
            n_basic,
            n_total,
            n_fdr,
            n_bonf,
            n_holm
        )
    }
    log_info("=========================================")

    invisible(NULL)
}

start_log_tailer <- function(log_file, poll_interval = 0.25) {
    if (is.null(log_file) || !nzchar(log_file)) return(NULL)
    log_file <- normalizePath(log_file, winslash = "/", mustWork = FALSE)
    dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)
    if (!file.exists(log_file)) file.create(log_file)
    stop_file <- paste0(log_file, ".tailstop")
    if (file.exists(stop_file)) file.remove(stop_file)
    job <- parallel::mcparallel({
        con <- NULL
        repeat {
            if (file.exists(log_file)) {
                try({
                    con <- file(log_file, open = "r", encoding = "UTF-8")
                    seek(con, where = 0, origin = "end")
                }, silent = TRUE)
            }
            if (!is.null(con)) break
            Sys.sleep(poll_interval)
        }
        on.exit({
            try(close(con), silent = TRUE)
        }, add = TRUE)
        repeat {
            if (!file.exists(stop_file)) break
            lines <- tryCatch(readLines(con, n = 200), error = function(e) character(0))
            if (length(lines) > 0) {
                for (ln in lines) {
                    cat(ln, "\n", sep = "")
                }
            } else {
                Sys.sleep(poll_interval)
            }
        }
        invisible(NULL)
    })
    list(job = job, stop_file = stop_file)
}

stop_log_tailer <- function(handle) {
    if (is.null(handle)) return(invisible(TRUE))
    try({
        if (file.exists(handle$stop_file)) file.remove(handle$stop_file)
        if (!is.null(handle$job)) parallel::mccollect(handle$job, wait = TRUE)
    }, silent = TRUE)
    invisible(TRUE)
}

run_ewas <- function(args) {
    full_sample_path <- paste0(
        args$metadata_group,
        "/",
        args$sample_list_name
    )
    full_sample_list <- as.character(
        rhdf5::h5read(
            args$methylation_betas_file,
            full_sample_path
        )
    )
    rhdf5::h5closeAll()

    log_info("Loading covariates")
    meta_tbl <- if (args$metadata != "NONE" && file.exists(args$metadata)) {
        tryCatch(fread(args$metadata), error = function(e) NULL)
    } else {
        NULL
    }

    covariate_names <- character(0)
    if (!is.null(args$covariate_names)) {
        covariate_names <- unlist(strsplit(args$covariate_names, ","))
    } else if (!is.null(meta_tbl)) {
        covariate_names <- names(meta_tbl)[-1]
    }

    covariates_table <- NULL
    if (length(covariate_names) > 0) {
        covariate_list <- list()
        for (cov in covariate_names) {
            if (!is.null(meta_tbl) && cov %in% names(meta_tbl)) {
                covariate_list[[cov]] <- meta_tbl[[cov]]
            } else {
                log_warn("Covariate '%s' not found in metadata file, trying HDF5", cov)
                covariate_list[[cov]] <- tryCatch({
                    rhdf5::h5read(args$methylation_betas_file, glue("/{args$metadata_group}/{cov}"))
                }, error = function(e) {
                    log_warn("Failed to load covariate '%s' from HDF5: %s", cov, e$message)
                    rep(NA, length(full_sample_list))
                })
            }
        }
        covariates_table <- as.data.frame(covariate_list)
        log_info("Loaded %d covariates: %s", ncol(covariates_table), paste(names(covariates_table), collapse = ", "))
    } else {
        log_info("No covariates provided or found in metadata sources")
    }
    if (!is.null(covariates_table)) {
        covariates_table <- valid_and_simplify_vars(
            covariates_table,
            validate_strict = TRUE
        )
    }

    log_info("Aligning sample IDs between HDF5 and metadata")
    sample_col <- NULL
    if (!is.null(args$sample_id) && nchar(args$sample_id) > 0 && args$sample_id != "NONE") {
        if (!is.null(meta_tbl) && args$sample_id %in% names(meta_tbl)) {
            sample_col <- args$sample_id
            log_info("Using explicit sample ID column: '%s'", sample_col)
        } else {
            log_warn(
                "Specified sample ID column '%s' not found in metadata; falling back to first column",
                args$sample_id
            )
            sample_col <- if (!is.null(meta_tbl)) names(meta_tbl)[1] else "SampleID"
        }
    } else {
        sample_col <- if (!is.null(meta_tbl)) names(meta_tbl)[1] else "SampleID"
        log_warn("Sample ID column not specified. Attempting automatic detection...")
        log_debug("Sample IDs from HDF5: %s", paste(head(full_sample_list, 5), collapse = "', '"))
        if (!is.null(meta_tbl)) {
            log_info("Auto-detected sample ID column: '%s'", sample_col)
        }
    }

    if (!is.null(meta_tbl) && sample_col %in% names(meta_tbl)) {
        meta_samples <- as.character(meta_tbl[[sample_col]])
        common_samples <- intersect(full_sample_list, meta_samples)
    } else {
        common_samples <- full_sample_list
    }

    if (length(common_samples) < 1) {
        stop(
            glue(
                "No overlapping samples between HDF5 ({length(full_sample_list)}) ",
                "and metadata ({ifelse(is.null(meta_tbl), 0, nrow(meta_tbl))})"
            )
        )
    }

    sample_ix <- match(common_samples, full_sample_list)
    names(sample_ix) <- common_samples

    if (!is.null(meta_tbl) && sample_col %in% names(meta_tbl)) {
        meta_tbl <- meta_tbl[meta_tbl[[sample_col]] %in% common_samples, , drop = FALSE]
        meta_tbl <- meta_tbl[match(common_samples, meta_tbl[[sample_col]]), , drop = FALSE]

        if (!is.null(covariates_table)) {
            covariates_table <- covariates_table[match(common_samples, meta_samples), , drop = FALSE]
        }
    }

    log_info("Aligned %d samples successfully", length(common_samples))

    chrom_groups <- args$chrom_groups
    n_probes_total <- sum(
        sapply(
            chrom_groups,
            function(chr) {
                probes <- read_group_probes(
                    args$methylation_betas_file,
                    chr,
                    args$marker_list_name
                )
                length(probes)
            }
        )
    )
    log_info("Total probes: %d", n_probes_total)
    log_info("Setting up parallel processing with %d workers", args$processes)
    cluster <- parallel::makeCluster(args$processes, outfile = "")
    log_file <- Sys.getenv("GT_LOG_FILE", "")
    parallel::clusterExport(cluster, "log_file", envir = environment())
    parallel::clusterEvalQ(cluster, {
        if (nzchar(log_file)) Sys.setenv(GT_LOG_FILE = log_file)
        script_path <- commandArgs(trailingOnly = FALSE)
        script_path <- script_path[grep("--file=", script_path)]
        if (length(script_path) > 0) {
            script_dir <- dirname(sub("--file=", "", script_path[1]))
            logging_path <- file.path(script_dir, "LoggingUtils.R")
            if (file.exists(logging_path)) {
                try(source(logging_path), silent = TRUE)
            }
        }
        try(setup_logger("INFO"), silent = TRUE)
        rhdf5::h5disableFileLocking()
    })
    use_snow_progress <- FALSE
    if (requireNamespace("doSNOW", quietly = TRUE)) {
        doSNOW::registerDoSNOW(cluster)
        use_snow_progress <- TRUE
        log_info("Registered cluster with doSNOW backend (progress callbacks enabled)")
    } else {
        doParallel::registerDoParallel(cluster)
        log_warn("doSNOW not available; progress callbacks via .options.snow will be ignored. Install 'doSNOW' to enable progress bars for foreach.")
    }
    logging_functions <- c(
        "COLORS", "colorize", "initialize_log_file", "get_log_file_paths",
        "store_log_files", "get_stored_log_files", "create_console_appender",
        "resolve_level_name", "format_log_args", "normalize_log_level",
        "format_log_line", "append_log_to_files", "setup_logger",
        "log_info", "log_warn", "log_debug", "log_error", "log_success"
    )
    cluster_env <- list(
        sample_ix = sample_ix,
        full_sample_list = full_sample_list,
        covariates_table = covariates_table,
        validate_variables = args$validate_variables
    )
    required_packages <- c("limma", "rhdf5", "glue", "data.table", "futile.logger")
    parallel::clusterExport(cluster, c(
        logging_functions,
        "process_chrom_group",
        "read_group_probes",
        "load_beta_chunk",
        "make_chunks",
        "mvalue_transform",
        "load_metadata_value",
        "valid_and_simplify_vars"
    ))
    parallel::clusterExport(cluster, names(cluster_env), envir = list2env(cluster_env))
    parallel::clusterEvalQ(cluster, {
        suppressPackageStartupMessages({
            library(futile.logger)
            library(limma)
            library(rhdf5)
            library(glue)
            library(data.table)
        })
        script_path <- commandArgs(trailingOnly = FALSE)
        script_path <- script_path[grep("--file=", script_path)]
        if (length(script_path) > 0) {
            script_path <- sub("--file=", "", script_path)
            script_dir <- dirname(script_path)
            tryCatch({
                source(file.path(script_dir, "LoggingUtils.R"))
            }, error = function(e) {
                log_info <- function(...) cat(sprintf(...), "\n")
                log_warn <- function(...) cat("WARN:", sprintf(...), "\n")
                log_error <- function(...) cat("ERROR:", sprintf(...), "\n")
                log_debug <- function(...) cat("DEBUG:", sprintf(...), "\n")
            })
        }
        setup_logger("INFO")
        rhdf5::h5disableFileLocking()
    })
    log_info("Running EWAS in parallel mode across %d chromosome groups...", length(chrom_groups))
    pb <- utils::txtProgressBar(min = 0, max = length(chrom_groups), style = 3)
    results_list <- NULL
    tryCatch({
        foreach_args <- list(
            group_name = chrom_groups,
            .combine = combine_limma_results,
            .packages = required_packages
        )
        if (use_snow_progress) {
            foreach_args$.options.snow <- list(progress = function(n) {
                try(utils::setTxtProgressBar(pb, n), silent = TRUE)
            })
        }

        results_list <- do.call(foreach::foreach, foreach_args) %dopar% {
            current_group <- get("group_name", envir = environment())
            log_info("Processing chromosome group: %s", current_group)
            group_results <- process_chrom_group(
                group_name = current_group,
                h5_file = args$methylation_betas_file,
                marker_list_name = args$marker_list_name,
                betas_name = args$betas_name,
                chunk_size = args$chunk_size,
                sample_ix = sample_ix,
                full_sample_list = full_sample_list,
                mvalue = TRUE,
                covariates_table = covariates_table,
                data_orientation = args$data_orientation,
                stat_var = args$stat_var
            )
            gc()
            group_results
        }
    }, finally = {
        try(close(pb), silent = TRUE)
    })

    parallel::stopCluster(cluster)
    if (!is.null(results_list) && !is.null(results_list$results) && nrow(results_list$results) > 0) {
        full_res <- results_list$results

        dt <- data.table::as.data.table(full_res)
        n_samples <- length(unique(names(sample_ix)))

        term_stats <- dt[, {
            pvec <- get("P_Value")
            tvec <- get("t")

            n_cases_vec <- unique(.SD[["n_cases"]])
            n_controls_vec <- unique(.SD[["n_controls"]])
            n_cases <- if (length(n_cases_vec) > 0) n_cases_vec[!is.na(n_cases_vec)][1] else NA
            n_controls <- if (length(n_controls_vec) > 0) n_controls_vec[!is.na(n_controls_vec)][1] else NA

            lambda_vals <- calculate_genomic_inflation(
                pvec,
                n_samples = n_samples,
                n_cases = n_cases,
                n_controls = n_controls
            )
            emp_null <- calculate_empirical_null(tvec)
            if (!is.na(lambda_vals$lambda_raw) && n_samples >= 1000) {
                log_info("Calculating scaled lambda for term '%s' with n_samples=%d", .BY[[1]], n_samples)
                scaled_n <- ifelse(
                    !is.na(n_cases) && !is.na(n_controls),
                    round((4 * n_cases * n_controls) / (n_cases + n_controls)),
                    n_samples
                )
                if (!is.na(n_cases) && !is.na(n_controls)) {
                    cat(
                        paste0(
                            "Term: ", .BY[[1]],
                            " | λ = ", formatC(lambda_vals$lambda_raw, digits = 4),
                            " | Cases = ", n_cases,
                            " | Controls = ", n_controls,
                            " | Scaled λ (N=", scaled_n,
                            ") = ", formatC(lambda_vals$lambda_scaled, digits = 4),
                            " | Empirical Bias = ", formatC(emp_null$bias, digits = 4),
                            " | Empirical Inflation = ", formatC(emp_null$inflation, digits = 4)
                        ),
                        "\n"
                    )
                } else {
                    cat(
                        paste0(
                            "Term: ", .BY[[1]],
                            " | λ = ", formatC(lambda_vals$lambda_raw, digits = 4),
                            " | Samples = ", n_samples,
                            " | Scaled λ (N=", scaled_n,
                            ") = ", formatC(lambda_vals$lambda_scaled, digits = 4),
                            " | Empirical Bias = ", formatC(emp_null$bias, digits = 4),
                            " | Empirical Inflation = ", formatC(emp_null$inflation, digits = 4)
                        ),
                        "\n"
                    )
                }
            } else if (!is.na(lambda_vals$lambda_raw)) {
                cat(paste0(
                    "Term: ", .BY[[1]],
                    " | λ = ", formatC(lambda_vals$lambda_raw, digits = 4),
                    " | Empirical Bias = ", formatC(emp_null$bias, digits = 4),
                    " | Empirical Inflation = ", formatC(emp_null$inflation, digits = 4),
                    "\n"
                ))
            } else {
                cat(paste0(
                    "Term: ", .BY[[1]],
                    " | λ = NA",
                    " | Empirical Bias = ", formatC(emp_null$bias, digits = 4),
                    " | Empirical Inflation = ", formatC(emp_null$inflation, digits = 4),
                    "\n"
                ))
            }
            list(lambda_basic = lambda_vals$lambda_scaled)
        }, by = "Term"]

        dt <- merge(dt, term_stats, by = "Term", all.x = TRUE, sort = FALSE)
        dt[, ("adj_p_fdr") := p.adjust(get("P_Value"), method = "fdr"), by = "Term"]
        dt[, ("adj_p_holm") := p.adjust(get("P_Value"), method = "holm"), by = "Term"]
        dt[, ("adj_p_bonferroni") := p.adjust(get("P_Value"), method = "bonferroni"), by = "Term"]
        dt[, ("lambda_corrected") := NA_real_]

        full_res <- as.data.frame(dt)
        results_list$results <- full_res
    }
    if (is.null(results_list) || is.null(results_list$results) || nrow(results_list$results) == 0) {
        stop("No valid results were generated by the EWAS analysis")
    }
    results_list
}

process_chrom_group <- function(
    group_name,
    h5_file,
    marker_list_name,
    betas_name,
    chunk_size,
    sample_ix,
    full_sample_list,
    mvalue,
    covariates_table,
    data_orientation,
    stat_var = NULL
) {
    rhdf5::h5closeAll()

    probes_in_group <- read_group_probes(
        h5_file,
        group_name,
        marker_list_name
    )
    if (!length(probes_in_group)) {
        log_warn("No probes found for group %s, skipping", group_name)
        return(NULL)
    }

    chunks_for_group <- make_chunks(probes_in_group, chunk_size)
    results_for_group <- NULL

    for (chunk_range in chunks_for_group) {
        betas <- load_beta_chunk(
            h5_file = h5_file,
            chunk_range = chunk_range,
            group_probes = probes_in_group,
            sample_ix = sample_ix,
            group_name = group_name,
            mvalue = mvalue,
            full_sample_list = full_sample_list,
            marker_list_name = marker_list_name,
            betas_name = betas_name,
            data_orientation = data_orientation
        )

        if (is.null(betas) || !nrow(betas)) next

        # ---- Create design matrix ----
        if (is.null(covariates_table) || ncol(covariates_table) == 0) {
            design <- matrix(1, nrow = ncol(betas), ncol = 1)
            colnames(design) <- "(Intercept)"
        } else {
            design <- tryCatch({
                cov_df <- as.data.frame(covariates_table)
                for (col in names(cov_df)) {
                    if (!is.numeric(cov_df[[col]]) && !is.factor(cov_df[[col]])) {
                        cov_df[[col]] <- as.factor(cov_df[[col]])
                    }
                }
                model.matrix(~ ., data = cov_df)
            }, error = function(e) {
                log_error("Error creating design matrix: %s", e$message)
                stop(paste("Design matrix error:", e$message))
            })
        }
        # ---- Fit model ----
        fit <- limma::lmFit(betas, design)
        fit <- limma::eBayes(fit)

        # ---- Identify which terms to process ----
        terms_to_process <- setdiff(colnames(design), "(Intercept)")
        if (!is.null(stat_var) && nchar(stat_var) > 0) {
            stat_vars <- trimws(unlist(strsplit(stat_var, ",")))
            matched_terms <- unlist(lapply(stat_vars, function(v) grep(paste0("^", v), terms_to_process, value = TRUE)))
            if (length(matched_terms) > 0) terms_to_process <- matched_terms
        }

        # ---- Identify binary variables ----
        binary_terms <- character(0)
        for (term in terms_to_process) {
            vals <- design[, term]
            if (is.numeric(vals) && all(vals %in% c(0, 1))) {
                binary_terms <- c(binary_terms, term)
            }
        }

        # ---- Compute shrunken fold changes ----
        shrunken_fc <- NULL
        if (length(binary_terms) > 0) {
            shrunken_fc <- tryCatch({
                limma::predFCm(fit)
            }, error = function(e) {
                log_warn("predFCm failed for %s: %s", group_name, e$message)
                NULL
            })
        }
        # ---- Collect results ----
        results_list <- list()
        for (term in terms_to_process) {
            coef <- fit$coefficients[, term]
            tval <- fit$t[, term]
            pval <- fit$p.value[, term]
            se <- sqrt(fit$s2.post) * fit$stdev.unscaled[, term]

            # compute per-term case/control counts if binary
            vals <- NULL
            n_cases <- NA_integer_
            n_controls <- NA_integer_
            n_effective <- NA_real_
            if (term %in% colnames(design)) {
                vals <- design[, term]
                if (is.numeric(vals) && all(vals %in% c(0, 1, NA))) {
                    n_cases <- sum(vals == 1, na.rm = TRUE)
                    n_controls <- sum(vals == 0, na.rm = TRUE)
                    if ((n_cases + n_controls) > 0) {
                        n_effective <- (4 * n_cases * n_controls) / (n_cases + n_controls)
                    }
                }
            }

            emp_fc <- if (!is.null(shrunken_fc) && term %in% binary_terms) {
                if (is.matrix(shrunken_fc)) {
                    if (term %in% colnames(shrunken_fc)) {
                        shrunken_fc[, term]
                    } else if (ncol(shrunken_fc) == 1) {
                        as.numeric(shrunken_fc[, 1])
                    } else {
                        rep(NA_real_, nrow(fit$coefficients))
                    }
                } else if (is.atomic(shrunken_fc) && length(shrunken_fc) == nrow(fit$coefficients)) {
                    as.numeric(shrunken_fc)
                } else {
                    rep(NA_real_, nrow(fit$coefficients))
                }
            } else {
                NA
            }
            term_df <- data.frame(
                Term = rep(term, length(coef)),
                Probe_ID = rownames(fit$coefficients),
                Coefficient = coef,
                eLOG2FC = emp_fc,
                Std_Error = se,
                t = tval,
                P_Value = pval,
                n_cases = rep(n_cases, length(coef)),
                n_controls = rep(n_controls, length(coef)),
                n_effective = rep(n_effective, length(coef)),
                stringsAsFactors = FALSE
            )

            results_list[[term]] <- term_df
        }

        results_df <- do.call(rbind, results_list)
        rownames(results_df) <- NULL

        if (is.null(results_for_group)) {
            results_for_group <- list(results = results_df)
        } else {
            results_for_group$results <- rbind(results_for_group$results, results_df)
        }

        gc()
    }

    results_for_group
}


save_results <- function(ewas_results, args) {
    results_table <- as.data.table(ewas_results$results)
    if (!"Probe_ID" %in% names(results_table)) {
        pids <- rownames(ewas_results$results)
        if (is.null(pids)) stop("Probe IDs unavailable: no rownames found")
        results_table[, ("Probe_ID") := pids]
    }

    cols_to_remove <- c(
        "lambda_basic", "lambda_corrected", "lambda_uncorrected",
        "empirical_sd", "empirical_mean", "CI_Upper", "CI_Lower"
    )
    cols_to_remove <- c(cols_to_remove, "n_cases", "n_controls", "n_effective")

    for (col in cols_to_remove) {
        if (col %in% names(results_table)) {
            results_table[, (col) := NULL]
        }
    }

    generate_qc_summary(results_table, NULL, args$stat_var)

    if ("Term" %in% names(results_table)) {
        results_table[, ("Term") := as.character(.SD$Term)]

        value_vars <- setdiff(names(results_table), c("Probe_ID", "Term"))

        wide_dt <- tryCatch({
            data.table::dcast(results_table, Probe_ID ~ Term, value.var = value_vars)
        }, error = function(e) {
            log_warn("Failed to pivot results to wide format: %s. Falling back to long format.", e$message)
            NULL
        })

        if (!is.null(wide_dt)) {
            elog_cols <- grep("^eLOG2FC", names(wide_dt), value = TRUE)
            if (length(elog_cols) > 0) {
                for (cname in elog_cols) {
                    if (all(is.na(wide_dt[[cname]]))) wide_dt[[cname]] <- NULL
                }
            }
            setcolorder(wide_dt, c("Probe_ID", setdiff(names(wide_dt), "Probe_ID")))
            data.table::fwrite(wide_dt, file = args$output)
            log_info("Saved EWAS results (wide) to %s", args$output)
            return(invisible(NULL))
        }
    }

    priority_cols <- c("Probe_ID", "Term", "Coefficient", "Std_Error", "t", "P_Value", "eLOG2FC")
    adj_p_cols <- grep("^adj_p_", names(results_table), value = TRUE)
    existing_priority <- intersect(priority_cols, names(results_table))
    other_cols <- setdiff(names(results_table), c(existing_priority, adj_p_cols))
    setcolorder(results_table, c(existing_priority, adj_p_cols, other_cols))

    data.table::fwrite(results_table, file = args$output)
    log_info("Saved EWAS results to %s", args$output)

    invisible(NULL)
}

main <- function() {
    load_packages()
    initialize_environment()
    setup_logging()
    args <- parse_args()
    setup_logger(args$log_level)
    tail_handle <- NULL
    try({
        gt_log <- Sys.getenv("GT_LOG_FILE", "")
        if (nzchar(gt_log)) {
            tail_handle <- start_log_tailer(gt_log)
        }
    }, silent = TRUE)
    optimize_memory(args$memory_per_core)
    log_info("Starting EWAS analysis")
    ewas_results <- run_ewas(args)
    save_results(ewas_results, args)
    try({
        stop_log_tailer(tail_handle)
    }, silent = TRUE)
}

main()
