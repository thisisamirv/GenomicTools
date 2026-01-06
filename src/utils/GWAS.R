#!/usr/bin/env Rscript
# Import required modules
load_packages <- function() {
    suppressPackageStartupMessages({
        library(argparse)
        library(Biobase)
        library(data.table)
        library(doParallel)
        library(dplyr)
        library(foreach)
        library(GENESIS)
        library(glue)
        library(GWASTools)
        library(SNPRelate)
        library(parallel)
        library(progressr)
        library(rhdf5)
        library(SeqArray)
        library(SeqVarTools)
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

list_chromosomes <- function(h5_file, chrom_prefix) {
    groups <- rhdf5::h5ls(h5_file, recursive = FALSE)
    chrom_groups <- groups$name[grep(paste0("^", chrom_prefix), groups$name)]
    chrom_groups
}

read_variant_ids <- function(h5_file, chrom, marker_list_name) {
    tryCatch({
        rhdf5::h5read(h5_file, glue("{chrom}/{marker_list_name}"))
    }, error = function(e) {
        stop(glue("Failed to read variant IDs for {chrom}: {e$message}"))
    }, finally = rhdf5::h5closeAll())
}

load_chunk_data <- function(
    h5_file,
    chunk_range,
    chrom,
    sample_ids,
    geno_name,
    marker_list_name,
    a1_path = NULL,
    a2_path = NULL,
    bp_path = NULL,
    data_orientation = "markers_as_rows",
    max_attempts = 5,
    sample_indices = NULL
) {
    get_h5_dimensions <- function(file, dataset_path) {
        fid <- rhdf5::H5Fopen(file)
        did <- rhdf5::H5Dopen(fid, dataset_path)
        sid <- rhdf5::H5Dget_space(did)
        dims <- rhdf5::H5Sget_simple_extent_dims(sid)$size
        rhdf5::H5Sclose(sid)
        rhdf5::H5Dclose(did)
        rhdf5::H5Fclose(fid)
        dims
    }
    all_variant_ids <- rhdf5::h5read(h5_file, glue("{chrom}/{marker_list_name}"))
    total_variants <- length(all_variant_ids)
    variant_start <- max(1, chunk_range[1])
    variant_end <- min(chunk_range[2], total_variants)
    n_variants <- variant_end - variant_start + 1
    if (variant_start > total_variants) {
        log_warn("Chunk range starts beyond available variants — skipping (%d > %d)", variant_start, total_variants)
        return(NULL)
    }
    dataset_path <- glue("{chrom}/{geno_name}")
    real_dims <- get_h5_dimensions(h5_file, dataset_path)
    log_info("Detected HDF5 dimensions for %s/%s: %s", chrom, geno_name, paste(real_dims, collapse = " × "))
    n_rows <- real_dims[1]
    n_cols <- real_dims[2]
    log_debug("Reading chunk with data_orientation=%s (rows=%d, cols=%d)", data_orientation, n_rows, n_cols)
    geno <- NULL
    for (attempt in seq_len(max_attempts)) {
        rhdf5::h5closeAll()
        geno <- tryCatch({
            if (data_orientation == "markers_as_rows") {
                col_indices <- if (!is.null(sample_indices)) sample_indices else 1:n_cols
                rhdf5::h5read(h5_file, dataset_path, index = list(variant_start:variant_end, col_indices))
            } else {
                row_indices <- if (!is.null(sample_indices)) sample_indices else 1:n_rows
                rhdf5::h5read(h5_file, dataset_path, index = list(row_indices, variant_start:variant_end))
            }
        }, error = function(e) {
            log_warn("Attempt %d failed for %s (%d–%d): %s", attempt, chrom, variant_start, variant_end, e$message)
            Sys.sleep(1)
            NULL
        })
        if (!is.null(geno)) break
    }
    if (is.null(geno)) {
        log_error("Failed to load genotype chunk %d–%d from %s", variant_start, variant_end, chrom)
        return(NULL)
    }
    if (data_orientation == "samples_as_rows") {
        geno <- t(geno)
        log_debug("Transposed genotype matrix to ensure variants as rows")
    }
    if (length(all_variant_ids[variant_start:variant_end]) != nrow(geno)) {
        log_warn(
            "Variant ID count (%d) does not match genotype rows (%d) — fixing mismatch",
            length(all_variant_ids[variant_start:variant_end]),
            nrow(geno)
        )
    }
    if (length(sample_ids) != ncol(geno)) {
        log_warn(
            "Sample ID count (%d) does not match genotype columns (%d) — truncating to minimum length",
            length(sample_ids),
            ncol(geno)
        )
        min_len <- min(length(sample_ids), ncol(geno))
        sample_ids <- sample_ids[seq_len(min_len)]
        geno <- geno[, seq_len(min_len), drop = FALSE]
    }
    rownames(geno) <- all_variant_ids[variant_start:variant_end]
    colnames(geno) <- sample_ids
    variant_info <- data.frame(
        SNP = all_variant_ids[variant_start:variant_end],
        CHR = sub("^CHR", "", chrom),
        stringsAsFactors = FALSE
    )
    if (!is.null(bp_path) && !is.na(bp_path) && nchar(bp_path) > 0 && bp_path != "None") {
        tryCatch({
            bp_dataset <- glue("{chrom}/{bp_path}")
            bp_index <- list(variant_start:variant_end)
            bp_values <- rhdf5::h5read(h5_file, bp_dataset, index = bp_index)
            variant_info$POS <- bp_values
            log_debug("Loaded %d base pair positions from %s", n_variants, bp_path)
        }, error = function(e) {
            log_warn("Failed to read BP positions from '%s': %s; using sequential positions", bp_path, e$message)
            variant_info$POS <- seq_len(n_variants)
        })
    } else {
        log_debug("No BP path provided; using sequential positions")
        variant_info$POS <- seq_len(n_variants)
    }
    if (!is.null(a1_path) && !is.na(a1_path) && nchar(a1_path) > 0 && a1_path != "None") {
        tryCatch({
            a1_dataset <- glue("{chrom}/{a1_path}")
            a1_index <- list(variant_start:variant_end)
            variant_info$A1 <- rhdf5::h5read(
                h5_file,
                a1_dataset,
                index = a1_index
            )
            log_debug("Loaded %d A1 alleles from %s", n_variants, a1_path)
        }, error = function(e) {
            log_warn("Failed to read A1 alleles from '%s': %s", a1_path, e$message)
        })
    }
    if (!is.null(a2_path) && !is.na(a2_path) && nchar(a2_path) > 0 && a2_path != "None") {
        tryCatch({
            a2_dataset <- glue("{chrom}/{a2_path}")
            a2_index <- list(variant_start:variant_end)
            variant_info$A2 <- rhdf5::h5read(h5_file, a2_dataset, index = a2_index)
            log_debug("Loaded %d A2 alleles from %s", n_variants, a2_path)
        }, error = function(e) {
            log_warn("Failed to read A2 alleles from '%s': %s", a2_path, e$message)
        })
    }
    log_info("Loaded genotype matrix: %d variants × %d samples", nrow(geno), ncol(geno))
    list(genotype = geno, variant_info = variant_info)
}

load_metadata_value <- function(source, field, h5_file, metadata_group) {
    if (source != "NONE" && file.exists(source)) {
        meta <- tryCatch(fread(source), error = function(e) NULL)
        if (!is.null(meta) && field %in% names(meta)) {
            log_info("Found '%s' in external metadata file", field)
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

combine_assoc_results <- function(a, b) {
    if (is.null(a) || nrow(a) == 0) return(b)
    if (is.null(b) || nrow(b) == 0) return(a)
    rbind(a, b)
}

close_all_gds_connections <- function() {
    tryCatch({
        if (requireNamespace("SNPRelate", quietly = TRUE)) {
            try(SNPRelate::snpgdsClose(verbose = FALSE), silent = TRUE)
        }
        if (requireNamespace("gdsfmt", quietly = TRUE)) {
            try({
                files <- gdsfmt::showfile.gds()
                if (length(files) > 0) {
                    for (f in files) gdsfmt::closefn.gds(f)
                }
            }, silent = TRUE)
        }
    }, error = function(e) {
        message("Note: GDS connection cleanup encountered issues: ", e$message)
    })
}

run_assoc_test <- function(
    geno_data,
    dependent_var_vector,
    covariates_data = NULL,
    variant_info,
    test_type = "linear",
    random_effects = NULL,
    interaction_term = NULL,
    stat_var = NULL
) {
    n_variants <- nrow(geno_data)
    n_samples <- ncol(geno_data)
    stat_vars <- NULL
    if (!is.null(stat_var) && nchar(stat_var) > 0) {
        stat_vars <- trimws(unlist(strsplit(stat_var, ",")))
        log_info("Will extract statistics for specific variables: %s", paste(stat_vars, collapse=", "))
    }
    rhdf5::h5closeAll()
    close_all_gds_connections()
    gds_file <- tempfile(pattern = "geno_chunk_", fileext = ".gds")
    if (file.exists(gds_file)) unlink(gds_file)
    if ("A1" %in% names(variant_info) && "A2" %in% names(variant_info)) {
        snp_allele <- paste(variant_info$A1, variant_info$A2, sep = "/")
    } else {
        snp_allele <- rep("A/G", n_variants)
        log_debug("Using dummy A/G alleles for SeqArray conversion")
    }
    SNPRelate::snpgdsCreateGeno(
        gds_file,
        genmat = as.matrix(geno_data),
        sample.id = colnames(geno_data),
        snp.id = variant_info$SNP,
        snp.chromosome = as.integer(variant_info$CHR),
        snp.position = as.integer(variant_info$POS),
        snp.allele = snp_allele,
        snpfirstdim = TRUE,
        compress.annotation = "ZIP",
        compress.geno = "ZIP"
    )
    seqarray_file <- tempfile(pattern = "seqarray_chunk_", fileext = ".gds")
    if (file.exists(seqarray_file)) unlink(seqarray_file)
    tryCatch({
        log_debug("Converting SNP GDS to SeqArray format")
        SeqArray::seqSNP2GDS(gds_file, seqarray_file, verbose = FALSE)
    }, error = function(e) {
        log_error("Failed to convert to SeqArray format: %s", e$message)
        unlink(gds_file)
        NULL
    })
    gds_obj <- tryCatch({
        SeqArray::seqOpen(seqarray_file)
    }, error = function(e) {
        log_error("Failed to open SeqArray file: %s", e$message)
        unlink(c(gds_file, seqarray_file))
        NULL
    })
    if (is.null(gds_obj)) {
        log_error("Could not create SeqArray object")
        try(unlink(c(gds_file, seqarray_file)), silent = TRUE)
        NULL
    }
    gds_sample_ids <- SeqArray::seqGetData(gds_obj, "sample.id")
    pheno_data <- data.frame(sample.id = gds_sample_ids, stringsAsFactors = FALSE)
    outcome_values <- dependent_var_vector[match(gds_sample_ids, names(dependent_var_vector))]
    if (any(is.na(outcome_values))) {
        log_error("Sample mismatch between genotypes and phenotypes")
        try(SeqArray::seqClose(gds_obj), silent = TRUE)
        try(unlink(c(gds_file, seqarray_file)), silent = TRUE)
        NULL
    }
    pheno_data$outcome <- outcome_values
    if (!is.null(covariates_data) && ncol(covariates_data) > 0) {
        for (cov in names(covariates_data)) {
            pheno_data[[cov]] <- covariates_data[match(gds_sample_ids, rownames(covariates_data)), cov]
        }
    }
    sample_annotation <- Biobase::AnnotatedDataFrame(pheno_data)
    seqvar_data <- tryCatch({
        SeqVarTools::SeqVarData(gds_obj, sampleData = sample_annotation)
    }, error = function(e) {
        log_error("Failed to create SeqVarData object: %s", e$message)
        NULL
    })
    if (is.null(seqvar_data)) {
        try(SeqArray::seqClose(gds_obj), silent = TRUE)
        try(unlink(c(gds_file, seqarray_file)), silent = TRUE)
        NULL
    }
    iterator <- tryCatch({
        log_debug("Creating SeqVarBlockIterator")
        SeqVarTools::SeqVarBlockIterator(seqvar_data, verbose = FALSE)
    }, error = function(e) {
        log_error("Failed to create iterator: %s", e$message)
        NULL
    })
    if (is.null(iterator)) {
        try(SeqArray::seqClose(gds_obj), silent = TRUE)
        try(unlink(c(gds_file, seqarray_file)), silent = TRUE)
        NULL
    }
    covar_names <- setdiff(names(pheno_data), c("sample.id", "outcome"))
    if (!is.null(interaction_term) && interaction_term %in% covar_names) {
        log_info("Interaction term '%s' will be tested with genotype", interaction_term)
        covars_to_use <- covar_names
    } else {
        covars_to_use <- covar_names
    }
    null_model <- tryCatch({
        if (!is.null(random_effects) && random_effects %in% names(pheno_data)) {
            group_factor <- as.factor(pheno_data[[random_effects]])
            n_groups <- length(unique(group_factor))
            if (n_groups == 1) {
                log_warn(
                    "Random effect '%s' has only one level — treating as fixed effect",
                    random_effects
                )
                GENESIS::fitNullModel(
                    pheno_data,
                    outcome = "outcome",
                    covars = if (length(covars_to_use) > 0) covars_to_use else NULL,
                    family = ifelse(test_type == "logistic", "binomial", "gaussian")
                )
            } else if (n_groups == n_samples) {
                log_warn(
                    "Random effect '%s' has unique values for each sample — skipping random effects",
                    random_effects
                )
                GENESIS::fitNullModel(
                    pheno_data,
                    outcome = "outcome",
                    covars = if (length(covars_to_use) > 0) covars_to_use else NULL,
                    family = ifelse(test_type == "logistic", "binomial", "gaussian")
                )
            } else {
                log_info("Fitting mixed model with random effect '%s' (%d groups)", random_effects, n_groups)
                GENESIS::fitNullModel(
                    pheno_data,
                    outcome = "outcome",
                    covars = if (length(covars_to_use) > 0) covars_to_use else NULL,
                    family = ifelse(test_type == "logistic", "binomial", "gaussian"),
                    cov.mat = NULL,
                    group.var = random_effects
                )
            }
        } else {
            log_info("Fitting standard %s model (no random effects)", test_type)
            GENESIS::fitNullModel(
                pheno_data,
                outcome = "outcome",
                covars = if (length(covars_to_use) > 0) covars_to_use else NULL,
                family = ifelse(test_type == "logistic", "binomial", "gaussian")
            )
        }
    }, error = function(e) {
        log_error("fitNullModel failed: %s", e$message)
        NULL
    })
    if (is.null(null_model)) {
        try(SeqArray::seqClose(gds_obj), silent = TRUE)
        try(unlink(c(gds_file, seqarray_file)), silent = TRUE)
        return(NULL)
    }
    log_info("Null model successfully fitted (%s)", test_type)
    assoc_results <- tryCatch({
        log_debug("Running assocTestSingle with SeqVarBlockIterator")
        if (!is.null(interaction_term) && interaction_term %in% names(pheno_data)) {
            log_info("Testing genotype × %s interaction", interaction_term)
            GENESIS::assocTestSingle(iterator, null.model = null_model, test = "Score", ivars = interaction_term)
        } else {
            GENESIS::assocTestSingle(iterator, null.model = null_model, test = "Score")
        }
    }, error = function(e) {
        log_error("assocTestSingle failed: %s", e$message)
        NULL
    })
    try(SeqVarTools::resetIterator(iterator), silent = TRUE)
    try(SeqArray::seqClose(gds_obj), silent = TRUE)
    try(close_all_gds_connections(), silent = TRUE)
    try(unlink(c(gds_file, seqarray_file)), silent = TRUE)
    if (is.null(assoc_results)) return(NULL)
    results_df <- tryCatch({
        df <- as.data.frame(assoc_results)
        if ("Score.Var" %in% names(df)) {
            df$Score.Var <- as.numeric(df$Score.Var)
            if (any(!is.na(df$Score.Var) & df$Score.Var >= 0)) {
                df$Score.SE <- sqrt(df$Score.Var)
            }
        }
        if (!is.null(stat_vars) && length(stat_vars) > 0) {
            null_model_coefs <- null_model$betaCov
            if (!is.null(null_model_coefs)) {
                log_debug("Found null model coefficients for additional terms")
                term_mapping <- list()
                model_terms <- rownames(null_model_coefs)
                for (requested_var in stat_vars) {
                    if (requested_var %in% model_terms) {
                        term_mapping[[requested_var]] <- requested_var
                    } else {
                        factor_matches <- grep(paste0("^", requested_var), model_terms, value = TRUE)
                        if (length(factor_matches) > 0) {
                            for (fm in factor_matches) {
                                term_mapping[[fm]] <- requested_var
                            }
                            log_info(
                                "Matched term(s) for '%s': %s",
                                requested_var,
                                paste(factor_matches, collapse = ", ")
                            )
                        }
                    }
                }
                for (term in names(term_mapping)) {
                    if (term %in% rownames(null_model_coefs)) {
                        display_term <- term_mapping[[term]]
                        idx <- which(rownames(null_model_coefs) == term)
                        coef_val <- null_model$beta[idx]
                        se_val <- sqrt(diag(null_model_coefs)[idx])
                        t_val <- coef_val / se_val
                        p_val <- 2 * pnorm(-abs(t_val))
                        df[[paste0("coef_", display_term)]] <- rep(coef_val, nrow(df))
                        df[[paste0("se_", display_term)]] <- rep(se_val, nrow(df))
                        df[[paste0("t_", display_term)]] <- rep(t_val, nrow(df))
                        df[[paste0("p_", display_term)]] <- rep(p_val, nrow(df))
                    }
                }
            }
        }
        attr(df, "has_random_effects") <- !is.null(random_effects)
        attr(df, "has_interaction") <- !is.null(interaction_term)
        attr(df, "stat_vars") <- stat_vars
        df
    }, error = function(e) {
        log_error("Failed to process association results: %s", e$message)
        as.data.frame(assoc_results)
    })
    return(results_df)
}

parse_args <- function() {
    parser <- argparse::ArgumentParser(description = "GWAS with GENESIS")
    parser$add_argument("--genotype_file", required = TRUE)
    parser$add_argument("--output", required = TRUE)
    parser$add_argument("--metadata", default = "NONE")
    parser$add_argument("--dependent_var", required = TRUE)
    parser$add_argument("--sample_id", default = NULL)
    parser$add_argument("--processes", type = "integer", default = 2)
    parser$add_argument("--chunk_size", type = "integer", default = 1000)
    parser$add_argument("--chrom_groups", required = TRUE)
    parser$add_argument("--marker_list_name", default = "RSID")
    parser$add_argument("--sample_list_name", default = "IID")
    parser$add_argument("--metadata_group", default = "metadata")
    parser$add_argument("--geno_name", default = "Genotype")
    parser$add_argument("--data_orientation", default = "markers_as_rows")
    parser$add_argument("--test_type", default = "linear")
    parser$add_argument("--log_level", default = "INFO")
    parser$add_argument("--interaction_term", default = NULL)
    parser$add_argument("--random_effects", default = NULL)
    parser$add_argument("--a1_path", default = NULL)
    parser$add_argument("--a2_path", default = NULL)
    parser$add_argument("--bp_path", default = NULL)
    parser$add_argument("--covariate_names", default = NULL)
    parser$add_argument("--stat_var", default = NULL)
    parser$add_argument("--memory_per_core", type = "double", default = 2.0)
    parser$add_argument("--temp_dir", default = NULL, help = "Custom temp directory")
    args <- parser$parse_args()
    for (param in c("a1_path", "a2_path", "bp_path")) {
        if (!is.null(args[[param]]) && (args[[param]] == "None" || args[[param]] == "NONE")) {
            args[[param]] <- NULL
        }
    }
    if (args$processes < 1) stop("Processes must be positive")
    if (args$chunk_size < 1) stop("Chunk size must be positive")
    if (!args$test_type %in% c("linear", "logistic")) stop("Invalid test_type")
    if (!is.null(args$covariate_names)) {
        args$covariate_names <- unlist(strsplit(args$covariate_names, ","))
    } else {
        args$covariate_names <- character(0)
    }
    args$chrom_groups <- strsplit(args$chrom_groups, ",")[[1]]
    if (length(args$chrom_groups) == 0) stop("No chromosome groups provided")
    if (!is.null(args$temp_dir) && nchar(args$temp_dir) > 0 && args$temp_dir != "NONE") {
        if (dir.exists(args$temp_dir)) {
            Sys.setenv(TMPDIR = args$temp_dir)
            log_info("Using custom temp directory: %s", args$temp_dir)
        } else {
            log_warn("Provided temp directory doesn't exist: %s", args$temp_dir)
        }
    }
    args
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
                for (ln in lines) cat(ln, "\n", sep = "")
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

process_chrom <- function(
    chrom,
    h5_file,
    valid_samples,
    dependent_var_vector,
    covariates_data,
    marker_list_name,
    geno_name,
    chunk_size,
    a1_path,
    a2_path,
    bp_path,
    data_orientation,
    test_type,
    random_effects,
    interaction_term,
    metadata_group,
    sample_list_name,
    stat_var = NULL
) {
    variants <- read_variant_ids(h5_file, chrom, marker_list_name)
    if (!length(variants)) {
        log_warn("No variants in %s", chrom)
        return(NULL)
    }

    full_sample_list <- as.character(rhdf5::h5read(h5_file, paste0(metadata_group, "/", sample_list_name)))
    sample_indices <- match(valid_samples, full_sample_list)
    if (any(is.na(sample_indices))) {
        missing_samples <- valid_samples[is.na(sample_indices)]
        stop(glue("Samples missing in genotype data for %s: %s", chrom, paste(missing_samples, collapse = ", ")))
    }

    if (length(dependent_var_vector) != length(valid_samples)) {
        dep_length <- length(dependent_var_vector)
        stop(glue("Dependent variable length ({dep_length}) does not match sample count ({length(valid_samples)})"))
    }

    chunks <- make_chunks(variants, chunk_size)
    results_for_chrom <- NULL

    for (chunk_range in chunks) {
        chunk_data <- load_chunk_data(
            h5_file,
            chunk_range,
            chrom,
            valid_samples,
            geno_name,
            marker_list_name,
            a1_path,
            a2_path,
            bp_path,
            data_orientation = data_orientation,
            sample_indices = sample_indices
        )

        if (is.null(chunk_data) || nrow(chunk_data$genotype) == 0) next

        geno <- chunk_data$genotype
        variant_var <- apply(geno, 1, function(x) var(x, na.rm = TRUE))
        keep_idx <- which(!is.na(variant_var) & variant_var > 0)

        if (length(keep_idx) < nrow(geno)) {
            dropped <- nrow(geno) - length(keep_idx)
            log_warn("Dropping %d monomorphic or NA-only variants in this chunk", dropped)
            geno <- geno[keep_idx, , drop = FALSE]
            chunk_data$variant_info <- chunk_data$variant_info[keep_idx, , drop = FALSE]
        }

        sample_na <- colSums(is.na(geno)) == nrow(geno)
        if (any(sample_na)) {
            log_warn("Dropping %d samples with all missing genotypes", sum(sample_na))
            geno <- geno[, !sample_na, drop = FALSE]
        }

        if (nrow(geno) == 0 || ncol(geno) == 0) {
            log_warn("Empty genotype matrix after filtering — skipping chunk")
            next
        }

        chunk_data$genotype <- geno

        res <- tryCatch({
            run_assoc_test(
                geno_data = chunk_data$genotype,
                dependent_var_vector = dependent_var_vector,
                covariates_data = covariates_data,
                variant_info = chunk_data$variant_info,
                test_type = test_type,
                random_effects = random_effects,
                interaction_term = interaction_term,
                stat_var = stat_var
            )
        }, error = function(e) {
            log_error("Chunk %d-%d failed on %s: %s", chunk_range[1], chunk_range[2], chrom, e$message)
            NULL
        })

        if (is.null(res)) {
            gc()
            next
        }

        results_for_chrom <- combine_assoc_results(results_for_chrom, res)
        for (attr_name in c("has_random_effects", "has_interaction", "stat_vars")) {
            attr(results_for_chrom, attr_name) <- attr(res, attr_name)
        }

        gc()
    }

    results_for_chrom
}

process_chromosomes_parallel <- function(chrom_groups, args, valid_samples, dependent_var_vector, covariates_data) {
    log_info("Setting up parallel processing with %d workers", args$processes)
    cl <- parallel::makeCluster(args$processes, outfile = "")
    doParallel::registerDoParallel(cl)
    required_functions <- c(
        "process_chrom",
        "read_variant_ids",
        "make_chunks",
        "load_chunk_data",
        "combine_assoc_results",
        "run_assoc_test",
        "log_info",
        "log_warn",
        "log_error",
        "log_debug",
        "colorize",
        "COLORS",
        "close_all_gds_connections"
    )
    log_file <- Sys.getenv("GT_LOG_FILE", "")
    parallel::clusterExport(cl, "log_file", envir = environment())
    parallel::clusterExport(
        cl,
        c("args", "valid_samples", "dependent_var_vector", "covariates_data", required_functions), envir = environment()
    )
    parallel::clusterEvalQ(cl, {
        suppressPackageStartupMessages({
            library(glue)
            library(rhdf5)
            library(GENESIS)
            library(GWASTools)
            library(SNPRelate)
            library(data.table)
            library(SeqArray)
            library(SeqVarTools)
            library(Biobase)
            library(futile.logger)
        })
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
    log_info("Starting parallel GWAS across %d chromosomes", length(chrom_groups))
    `%dopar%` <- foreach::`%dopar%`
    results_raw <- foreach::foreach(
        chrom = chrom_groups,
        .packages = c(
            "glue",
            "rhdf5",
            "GENESIS",
            "GWASTools",
            "SNPRelate",
            "data.table",
            "SeqArray",
            "SeqVarTools",
            "Biobase",
            "foreach"
        ),
        .errorhandling = "pass",
        .verbose = FALSE
    ) %dopar% {
        current_chrom <- get("chrom", envir = environment())
        tryCatch({
            res <- process_chrom(
                chrom = current_chrom,
                h5_file = args$genotype_file,
                valid_samples = valid_samples,
                dependent_var_vector = dependent_var_vector,
                covariates_data = covariates_data,
                marker_list_name = args$marker_list_name,
                geno_name = args$geno_name,
                chunk_size = args$chunk_size,
                a1_path = args$a1_path,
                a2_path = args$a2_path,
                bp_path = args$bp_path,
                data_orientation = args$data_orientation,
                test_type = args$test_type,
                random_effects = args$random_effects,
                interaction_term = args$interaction_term,
                metadata_group = args$metadata_group,
                sample_list_name = args$sample_list_name,
                stat_var = args$stat_var
            )
            gc(verbose = FALSE)
            list(chrom = current_chrom, results = res, error = NULL)
        }, error = function(e) {
            list(chrom = current_chrom, results = NULL, error = e$message)
        })
    }
    parallel::stopCluster(cl)
    log_info("Parallel processing complete")
    results_raw
}

run_gwas <- function(args) {
    full_sample_list <- as.character(
        rhdf5::h5read(args$genotype_file, paste0(args$metadata_group, "/", args$sample_list_name))
    )
    rhdf5::h5closeAll()
    log_info("Loaded %d sample IDs from HDF5 metadata", length(full_sample_list))

    meta_tbl <- if (args$metadata != "NONE" && file.exists(args$metadata)) {
        tryCatch(fread(args$metadata), error = function(e) {
            log_warn("Failed to read metadata file '%s': %s", args$metadata, e$message)
            NULL
        })
    } else {
        NULL
    }

    sample_col <- NULL
    if (!is.null(args$sample_id) && nchar(args$sample_id) > 0 && args$sample_id != "NONE") {
        if (!is.null(meta_tbl) && args$sample_id %in% names(meta_tbl)) {
            sample_col <- args$sample_id
            log_info("Using explicit sample ID column: '%s'", sample_col)
        } else {
            log_warn(
                "Specified sample ID column '%s' not found in metadata; using first column when available",
                args$sample_id
            )
            if (!is.null(meta_tbl)) {
                sample_col <- names(meta_tbl)[1]
            }
        }
    } else if (!is.null(meta_tbl)) {
        sample_col <- names(meta_tbl)[1]
        log_info("No sample ID column specified, using '%s'", sample_col)
    }

    if (!is.null(meta_tbl) && !is.null(sample_col) && sample_col %in% names(meta_tbl)) {
        meta_tbl[[sample_col]] <- as.character(meta_tbl[[sample_col]])
        meta_samples <- meta_tbl[[sample_col]]
    } else {
        meta_samples <- character(0)
    }

    if (length(meta_samples) > 0) {
        common_samples <- intersect(full_sample_list, meta_samples)
    } else {
        common_samples <- full_sample_list
    }

    if (length(common_samples) < 1) {
        stop("No overlapping samples between HDF5 and metadata")
    }

    log_info("Aligned %d samples between genotypes and metadata", length(common_samples))

    sample_ix <- match(common_samples, full_sample_list)
    names(sample_ix) <- common_samples

    if (!is.null(meta_tbl) && length(meta_samples) > 0) {
        meta_tbl <- meta_tbl[meta_tbl[[sample_col]] %in% common_samples, , drop = FALSE]
        meta_tbl <- meta_tbl[match(common_samples, meta_tbl[[sample_col]]), , drop = FALSE]
    }

    dependent_var_vector <- NULL
    if (!is.null(meta_tbl) && args$dependent_var %in% names(meta_tbl)) {
        log_info("Dependent variable '%s' found in metadata file", args$dependent_var)
        dependent_var_vector <- meta_tbl[[args$dependent_var]]
    } else {
        if (args$metadata != "NONE" && file.exists(args$metadata)) {
            log_warn(
                "Dependent variable '%s' not found in metadata file, reading from HDF5",
                args$dependent_var
            )
        }
        dependent_raw <- tryCatch({
            rhdf5::h5read(args$genotype_file, glue("/{args$metadata_group}/{args$dependent_var}"))
        }, error = function(e) {
            stop(glue("Failed to load dependent variable '{args$dependent_var}' from metadata: {e$message}"))
        })
        dependent_var_vector <- dependent_raw[sample_ix]
    }

    log_info("Loaded dependent variable '%s' for %d samples", args$dependent_var, length(dependent_var_vector))

    covariate_names <- args$covariate_names
    if (length(covariate_names) == 0 && !is.null(meta_tbl)) {
        covariate_names <- setdiff(names(meta_tbl), c(sample_col, args$dependent_var))
    }

    covariates_data <- NULL
    if (length(covariate_names) > 0) {
        covariate_list <- list()
        for (cov in covariate_names) {
            if (!is.null(meta_tbl) && cov %in% names(meta_tbl)) {
                covariate_list[[cov]] <- meta_tbl[[cov]]
            } else {
                if (args$metadata != "NONE" && file.exists(args$metadata)) {
                    log_warn("Covariate '%s' not found in metadata file, trying HDF5", cov)
                }
                cov_vector <- tryCatch({
                    rhdf5::h5read(args$genotype_file, glue("/{args$metadata_group}/{cov}"))
                }, error = function(e) {
                    log_warn("Failed to load covariate '%s' from HDF5: %s", cov, e$message)
                    rep(NA, length(full_sample_list))
                })
                covariate_list[[cov]] <- cov_vector[sample_ix]
            }
        }
        covariates_data <- as.data.frame(covariate_list, stringsAsFactors = FALSE)
        log_info("Loaded %d covariates: %s", ncol(covariates_data), paste(names(covariates_data), collapse = ", "))
    } else {
        log_info("No covariates provided or found in metadata sources")
    }

    if (!is.null(covariates_data)) {
        for (col in names(covariates_data)) {
            if (is.character(covariates_data[[col]])) {
                covariates_data[[col]] <- factor(covariates_data[[col]])
            }
        }
    }

    numeric_dep <- suppressWarnings(as.numeric(as.character(dependent_var_vector)))
    if (all(is.na(numeric_dep))) {
        stop("Dependent variable could not be converted to numeric values")
    }
    dependent_var_vector <- numeric_dep

    invalid_mask <- is.na(dependent_var_vector) | dependent_var_vector == -9
    if (any(invalid_mask)) {
        log_info("Removing %d invalid observations (NA or -9) from dependent variable", sum(invalid_mask))
    }
    valid_mask <- !invalid_mask
    valid_samples <- common_samples[valid_mask]
    dependent_var_vector <- dependent_var_vector[valid_mask]
    if (!is.null(covariates_data)) {
        covariates_data <- covariates_data[valid_mask, , drop = FALSE]
    }

    unique_vals <- sort(unique(na.omit(dependent_var_vector)))
    if (length(unique_vals) == 2) {
        if (setequal(unique_vals, c(1, 2))) {
            log_info("Converting dependent variable from {1,2} to {0,1}")
            dependent_var_vector[dependent_var_vector == 1] <- 0
            dependent_var_vector[dependent_var_vector == 2] <- 1
        } else if (setequal(unique_vals, c(2, 3))) {
            log_info("Converting dependent variable from {2,3} to {0,1}")
            dependent_var_vector[dependent_var_vector == 2] <- 0
            dependent_var_vector[dependent_var_vector == 3] <- 1
        }
    }

    if (length(unique(dependent_var_vector)) == 2) {
        log_info("Detected binary dependent variable — switching to logistic regression")
        args$test_type <- "logistic"
    } else {
        log_info("Detected continuous dependent variable — using linear regression")
        args$test_type <- "linear"
    }

    if (!is.null(covariates_data)) {
        keep_cov <- complete.cases(covariates_data)
        if (any(!keep_cov)) {
            log_info("Dropping %d samples with missing covariates", sum(!keep_cov))
            covariates_data <- covariates_data[keep_cov, , drop = FALSE]
            dependent_var_vector <- dependent_var_vector[keep_cov]
            valid_samples <- valid_samples[keep_cov]
        }
    }

    condition1 <- !is.null(args$interaction_term)
    condition2 <- !is.null(covariates_data)
    condition3 <- args$interaction_term %in% names(covariates_data)
    if (condition1 && condition2 && condition3) {
        keep_int <- !is.na(covariates_data[[args$interaction_term]])
        if (any(!keep_int)) {
            log_info("Dropping %d samples with missing interaction term '%s'", sum(!keep_int), args$interaction_term)
            covariates_data <- covariates_data[keep_int, , drop = FALSE]
            dependent_var_vector <- dependent_var_vector[keep_int]
            valid_samples <- valid_samples[keep_int]
        }
    }

    if (!is.null(args$random_effects) && !is.null(covariates_data) && args$random_effects %in% names(covariates_data)) {
        keep_re <- !is.na(covariates_data[[args$random_effects]])
        if (any(!keep_re)) {
            log_info("Dropping %d samples with missing random effect '%s'", sum(!keep_re), args$random_effects)
            covariates_data <- covariates_data[keep_re, , drop = FALSE]
            dependent_var_vector <- dependent_var_vector[keep_re]
            valid_samples <- valid_samples[keep_re]
        }
    }

    if (length(dependent_var_vector) != length(valid_samples)) {
        dep_var_length <- length(dependent_var_vector)
        stop(glue("Post-filtering mismatch: outcome n={dep_var_length} vs samples n={length(valid_samples)}"))
    }

    if (!is.null(covariates_data) && nrow(covariates_data) != length(valid_samples)) {
        stop(glue("Post-filtering mismatch: covariates n={nrow(covariates_data)} vs samples n={length(valid_samples)}"))
    }

    if (args$test_type == "logistic" && length(unique(dependent_var_vector)) < 2) {
        stop("After filtering, the dependent variable has a single class; logistic regression cannot be run.")
    }

    if (!is.null(covariates_data)) {
        rownames(covariates_data) <- valid_samples
    }
    names(dependent_var_vector) <- valid_samples

    log_info("Performing data integrity checks before analysis")
    if (any(is.na(dependent_var_vector))) {
        stop("Dependent variable contains missing values after filtering")
    }
    if (length(unique(dependent_var_vector)) == 1) {
        stop("Dependent variable has no variance — all values identical")
    }

    if (!is.null(covariates_data) && any(is.na(covariates_data))) {
        log_warn("Covariates contain missing values after filtering; affected samples were removed earlier")
    }

    log_info(
        "Final analysis set: %d samples, %d covariates",
        length(valid_samples),
        ifelse(is.null(covariates_data), 0, ncol(covariates_data))
    )

    test_chrom <- args$chrom_groups[1]
    if (!is.null(test_chrom) && nzchar(test_chrom)) {
        geno_probe <- tryCatch({
            rhdf5::h5read(
                args$genotype_file,
                glue("{test_chrom}/{args$geno_name}"),
                index = list(seq_len(10), seq_len(min(10, length(valid_samples))))
            )
        }, error = function(e) {
            log_warn("Could not load small test block from %s: %s", test_chrom, e$message)
            NULL
        })
        if (!is.null(geno_probe)) {
            if (any(is.na(geno_probe))) {
                frac_na <- mean(is.na(geno_probe))
                log_info("Genotype probe contains %.2f%% missing values", frac_na * 100)
                if (frac_na > 0.2) {
                    log_warn("High missingness in genotype matrix (>20%%)")
                }
            }
            variant_var <- apply(geno_probe, 1, function(x) var(x, na.rm = TRUE))
            monomorphic <- sum(variant_var == 0 | is.na(variant_var))
            if (monomorphic > 0) {
                log_warn("%d/%d test variants are monomorphic (no variation)", monomorphic, length(variant_var))
            } else {
                log_info("All tested genotype variants show expected variability")
            }
        } else {
            log_warn("Skipped genotype probe test (no test data loaded)")
        }
    }

    chrom_groups <- args$chrom_groups
    log_debug("Using %d chromosome groups: %s", length(chrom_groups), paste(chrom_groups, collapse = ", "))

    if (args$processes > 1 && length(chrom_groups) > 1) {
        log_info("Starting GWAS with %d parallel processes", args$processes)
        results_raw <- process_chromosomes_parallel(
            chrom_groups,
            args,
            valid_samples,
            dependent_var_vector,
            covariates_data
        )
    } else {
        log_info("Starting GWAS (processes=%d, chromosomes=%d)", args$processes, length(chrom_groups))
        results_raw <- list()
        for (chrom in chrom_groups) {
            log_info("Processing chromosome %s", chrom)
            tryCatch({
                log_debug("Starting chromosome %s analysis with %d samples", chrom, length(valid_samples))
                res <- process_chrom(
                    chrom = chrom,
                    h5_file = args$genotype_file,
                    valid_samples = valid_samples,
                    dependent_var_vector = dependent_var_vector,
                    covariates_data = covariates_data,
                    marker_list_name = args$marker_list_name,
                    geno_name = args$geno_name,
                    chunk_size = args$chunk_size,
                    a1_path = args$a1_path,
                    a2_path = args$a2_path,
                    bp_path = args$bp_path,
                    data_orientation = args$data_orientation,
                    test_type = args$test_type,
                    random_effects = args$random_effects,
                    interaction_term = args$interaction_term,
                    metadata_group = args$metadata_group,
                    sample_list_name = args$sample_list_name,
                    stat_var = args$stat_var
                )
                if (is.null(res)) {
                    log_warn("Chromosome %s returned NULL results", chrom)
                } else {
                    log_info("Chromosome %s processed successfully with %d variants", chrom, nrow(res))
                }
                results_raw[[chrom]] <- list(chrom = chrom, results = res, error = NULL)
            }, error = function(e) {
                log_error("Chromosome %s failed: %s", chrom, e$message)
                results_raw[[chrom]] <- list(chrom = chrom, results = NULL, error = e$message)
            })
            gc(verbose = FALSE)
            log_info("Memory cleanup performed after chromosome %s", chrom)
        }
    }

    errors_found <- vapply(results_raw, function(x) !is.null(x$error), logical(1))
    if (any(errors_found)) {
        log_error("Errors occurred in %d/%d chromosome analyses:", sum(errors_found), length(results_raw))
        for (i in which(errors_found)) {
            log_error("Chromosome %s: %s", results_raw[[i]]$chrom, results_raw[[i]]$error)
        }
    }

    results_elems <- lapply(results_raw, function(x) x$results)
    results_elems <- Filter(Negate(is.null), results_elems)

    if (length(results_elems) == 0) {
        log_warn("No valid results from any chromosome")
        log_warn("This may occur when:")
        log_warn("  - All variants are monomorphic (no variation)")
        log_warn("  - All variants failed quality control filters")
        log_warn("  - Sample size is too small for the analysis")
        empty_results <- data.frame(
            variant.id = character(0),
            chr = integer(0),
            pos = integer(0),
            n.obs = integer(0),
            freq = numeric(0),
            MAC = numeric(0),
            Est = numeric(0),
            Est.SE = numeric(0),
            Score.Stat = numeric(0),
            Score = numeric(0),
            Score.SE = numeric(0),
            Score.pval = numeric(0),
            PVE = numeric(0),
            stringsAsFactors = FALSE
        )
        log_info("Saving empty results file with proper structure")
        results_list <- empty_results
    } else if (length(results_elems) == 1) {
        results_list <- results_elems[[1]]
    } else {
        results_list <- Reduce(combine_assoc_results, results_elems)
    }

    if (!is.null(results_list) && length(results_elems) > 0) {
        for (attr_name in c("has_random_effects", "has_interaction", "stat_vars")) {
            attr_val <- attr(results_elems[[1]], attr_name)
            if (!is.null(attr_val)) {
                attr(results_list, attr_name) <- attr_val
            }
        }
    }

    if (is.null(results_list)) {
        log_error("Results list is NULL - this should not happen")
        stop("Failed to create results structure")
    }
    if (nrow(results_list) == 0) {
        log_warn("GWAS complete with 0 variants passing filters")
    } else {
        log_info("GWAS complete, total variants analyzed: %d", nrow(results_list))
    }

    results_list
}

save_results <- function(gwas_results, args) {
    results_table <- data.table::copy(as.data.table(gwas_results))
    if (nrow(results_table) == 0) {
        log_warn("No results to save - creating empty output file with headers")
        if (args$test_type == "logistic") {
            results_table <- data.table(
                RSID = character(0),
                CHR = integer(0),
                BP = integer(0),
                EFFECT_ALLELE = character(0),
                N = integer(0),
                EAF = numeric(0),
                MAF = numeric(0),
                OR = numeric(0),
                OR_SE = numeric(0),
                U_STAT = numeric(0),
                U_SE = numeric(0),
                P = numeric(0),
                R2 = numeric(0)
            )
        } else {
            results_table <- data.table(
                RSID = character(0),
                CHR = integer(0),
                BP = integer(0),
                EFFECT_ALLELE = character(0),
                N = integer(0),
                EAF = numeric(0),
                MAF = numeric(0),
                COEF = numeric(0),
                COEF_SE = numeric(0),
                U_STAT = numeric(0),
                U_SE = numeric(0),
                P = numeric(0),
                R2 = numeric(0)
            )
        }
        fwrite(results_table, file = args$output, sep = ",")
        log_info("Saved empty GWAS results file to %s", args$output)
        return(invisible(NULL))
    }
    if ("allele.index" %in% names(results_table)) {
        log_info("Performing allele.index sanity check")
        sample_indices <- head(which(!is.na(results_table$allele.index)), 10)
        if (length(sample_indices) > 0) {
            log_debug("Allele index verification (first %d variants)", length(sample_indices))
            has_zeros <- any(results_table$allele.index == 0, na.rm = TRUE)
            has_ones <- any(results_table$allele.index == 1, na.rm = TRUE)
            has_twos <- any(results_table$allele.index == 2, na.rm = TRUE)
            if (has_zeros && !has_twos) {
                log_info("Detected 0-based allele indexing (0=A1, 1=A2)")
                results_table[, ("EFFECT_ALLELE") := ifelse(.SD$allele.index == 0, "A1", "A2")]
            } else if (has_ones && has_twos && !has_zeros) {
                log_info("Detected 1-based allele indexing (1=A1, 2=A2)")
                results_table[, ("EFFECT_ALLELE") := ifelse(.SD$allele.index == 1, "A1", "A2")]
            } else if (has_ones && !has_twos && !has_zeros) {
                log_debug("Detected uniform allele indexing (all 1s = A1)")
                results_table[, ("EFFECT_ALLELE") := "A1"]
            } else {
                log_warn("Unclear allele.index pattern - using generic ALLELE label")
                results_table[, ("EFFECT_ALLELE") := paste0("ALLELE", .SD$allele.index)]
            }
        } else {
            log_warn("No valid allele.index values found")
            results_table[, ("EFFECT_ALLELE") := NA_character_]
        }
        results_table[, ("allele.index") := NULL]
    }
    if ("MAC" %in% names(results_table) && "n.obs" %in% names(results_table)) {
        results_table[, ("MAF") := .SD$MAC / (2 * .SD$n.obs)]
        results_table[, ("MAC") := NULL]
    } else if ("MAC" %in% names(results_table)) {
        log_warn("MAC present but n.obs missing; cannot calculate MAF accurately")
        setnames(results_table, "MAC", "MAF")
    }
    stat_vars <- attr(gwas_results, "stat_vars")
    has_var_stats <- !is.null(stat_vars) && any(grepl("^(coef|se|t|p)_", names(results_table)))
    if (has_var_stats) {
        log_info("Processing variable-specific statistics for: %s", paste(stat_vars, collapse=", "))
        var_cols <- grep("^(coef|se|t|p)_", names(results_table), value = TRUE)
        for (col in var_cols) {
            matches <- regexpr("^(coef|se|t|p)_(.+)$", col, perl = TRUE)
            if (matches > 0) {
                type <- regmatches(col, regexec("^(coef|se|t|p)_(.+)$", col))[[1]][2]
                var_name <- regmatches(col, regexec("^(coef|se|t|p)_(.+)$", col))[[1]][3]
                if (type == "coef") {
                    new_name <- if (args$test_type == "logistic") paste0("OR_", var_name) else paste0("COEF_", var_name)
                    if (args$test_type == "logistic") {
                        results_table[[col]] <- exp(results_table[[col]])
                    }
                } else if (type == "se") {
                    if (args$test_type == "logistic") {
                        new_name <- paste0("OR_SE_", var_name)
                    } else {
                        new_name <- paste0("COEF_SE_", var_name)
                    }
                    if (args$test_type == "logistic") {
                        coef_col <- paste0("coef_", var_name)
                        if (coef_col %in% names(results_table)) {
                            results_table[[col]] <- results_table[[col]] * exp(results_table[[coef_col]])
                        }
                    }
                } else if (type == "t") {
                    new_name <- paste0("T_", var_name)
                } else if (type == "p") {
                    new_name <- paste0("P_", var_name)
                } else {
                    new_name <- toupper(col)
                }
                setnames(results_table, col, new_name)
            }
        }
    }
    if (args$test_type == "logistic") {
        if ("Est" %in% names(results_table)) {
            results_table[, c("OR", "OR_SE") := {
                or_val <- exp(.SD$Est)
                list(or_val, .SD$Est.SE * or_val)
            }]
            log_info("Converted LOG_OR to OR (exponentiated effect estimate)")
        }
        rename_map <- list(
            variant.id = "RSID",
            chr = "CHR",
            pos = "BP",
            n.obs = "N",
            freq = "EAF",
            PVE = "R2",
            Score.pval = "P",
            Score = "U_STAT",
            Score.SE = "U_SE"
        )
    } else {
        rename_map <- list(
            variant.id = "RSID",
            chr = "CHR",
            pos = "BP",
            n.obs = "N",
            freq = "EAF",
            PVE = "R2",
            Est = "COEF",
            Est.SE = "COEF_SE",
            Score.pval = "P",
            Score = "U_STAT",
            Score.SE = "U_SE"
        )
    }
    for (old in names(rename_map)) {
        if (old %in% names(results_table)) {
            setnames(results_table, old, rename_map[[old]])
        }
    }
    cols_to_remove <- c("Est", "Est.SE", "Score.Stat")
    if (args$test_type == "logistic") {
        for (col in cols_to_remove) {
            if (col %in% names(results_table)) {
                results_table[, (col) := NULL]
            }
        }
    } else {
        if ("Score.Stat" %in% names(results_table)) {
            results_table[, ("Score.Stat") := NULL]
        }
    }
    base_cols <- c("RSID", "CHR", "BP", "EFFECT_ALLELE", "N", "EAF", "MAF")
    if (args$test_type == "logistic") {
        effect_cols <- c("OR", "OR_SE")
    } else {
        effect_cols <- c("COEF", "COEF_SE")
    }
    stat_cols <- c("U_STAT", "U_SE", "P", "R2")
    if (has_var_stats) {
        var_stat_cols <- grep("_(COEF|OR|SE|T|P)_", names(results_table), value = TRUE)
        stat_cols <- c(stat_cols, var_stat_cols)
    }
    cols_to_order <- intersect(c(base_cols, effect_cols, stat_cols), names(results_table))
    if (length(cols_to_order) > 0) {
        remaining_cols <- setdiff(names(results_table), cols_to_order)
        cols_to_order <- c(cols_to_order, remaining_cols)
        setcolorder(results_table, cols_to_order)
    }
    fwrite(results_table, file = args$output, sep = ",")
    log_debug(
        "Saved GWAS results to %s (%d variants, %d columns)",
        args$output,
        nrow(results_table),
        ncol(results_table)
    )
    if ("P" %in% names(results_table)) {
        sig_variants <- sum(results_table$P < 5e-8, na.rm = TRUE)
        log_info("Found %d genome-wide significant variants (P < 5e-8)", sig_variants)
    }
    if ("R2" %in% names(results_table)) {
        max_r2 <- max(results_table$R2, na.rm = TRUE)
        mean_r2 <- mean(results_table$R2, na.rm = TRUE)
        log_info("R² statistics: max = %.4f, mean = %.6f", max_r2, mean_r2)
    }
    if (has_var_stats) {
        for (var in stat_vars) {
            p_col <- paste0("P_", var)
            if (p_col %in% names(results_table)) {
                var_sig <- sum(results_table[[p_col]] < 5e-8, na.rm = TRUE)
                if (!is.na(var_sig)) {
                    log_info("Variable '%s': found %d significant variants (P < 5e-8)", var, var_sig)
                }
            }
        }
    }
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
        if (nzchar(gt_log)) tail_handle <- start_log_tailer(gt_log)
    }, silent = TRUE)
    optimize_memory(args$memory_per_core)
    log_info("Starting GWAS analysis")
    gwas_results <- run_gwas(args)
    save_results(gwas_results, args)
    try({
        stop_log_tailer(tail_handle)
    }, silent = TRUE)
}
main()
