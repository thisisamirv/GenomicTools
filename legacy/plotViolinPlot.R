#!/usr/bin/env Rscript
# Import the necessary libraries
suppressPackageStartupMessages({
    library(data.table)
    library(ggplot2)
    library(progress)
    library(rhdf5)
})

# Ensure null PDF device
pdf(file = NULL)

# Main function
plotViolinPlot <- function(
    input,
    metadata,
    probes,
    output = NULL,
    groups = "{'group': 'group_var', 'case': 'case_group', 'control': 'control_group'}",
    colors = "#FF0000, #0000FF",
    annotation = NULL,
    file = TRUE
) {
    tryCatch({
        # Split strings into lists
        colors <- stringToList(colors)
        groups <- dictToList(groups)
        group <- groups$group
        case <- groups$case
        control <- groups$control
        
        # Open the HDF5 file
        log_info("Opening HDF5 file:", input)
        h5_file <- H5Fopen(input, flags = "H5F_ACC_RDONLY")
        
        # Get chromosome list
        log_debug("Getting chromosome list")
        chromosomes <- getChromosomeList(h5_file)
        
        # Load the probes
        probes <- fread(probes, header = FALSE)[[1]]
        
        # Load the metadata
        metadata <- fread(metadata)
        
        # Keep relevant columns
        metadata <- metadata[, .(sample_id, get(group))]
        setnames(metadata, c("sample_id", group))
        
        # Get sample indices
        samples <- unique(metadata[["sample_id"]])
        sample_indices <- getSampleIndices(h5_file, samples)
        sample_indices <- na.omit(sample_indices)
        
        # Load the annotation
        if (!is.null(annotation)) {
            annotation <- fread(annotation)
        }
        
        # Loop through chromosomes, get probe indices, and read data
        log_info("Reading data")
        data <- lapply(chromosomes, function(chr) {
            log_debug("Reading data for chromosome:", chr)
            cpg_indices <- getProbeIndices(h5_file, chr, probes)
            log_debug("cpg_indices for", chr, ":", paste(cpg_indices, collapse = ", "))
            if (is.null(cpg_indices) || length(cpg_indices) == 0) {
                log_debug("No matching probes in chromosome", chr, "- skipping")
                return(NULL)
            }
            chr_data <- readChromosomeData(h5_file, chr, cpg_indices, sample_indices)
            if (is.null(chr_data)) {
                return(NULL)
            }
            chr_data$chromosome <- chr
            return(chr_data)
        })
        data <- rbindlist(data, use.names = TRUE, fill = TRUE)
        
        # Prepare the data for plotting
        log_info("Preparing data for plotting")
        measure_vars <- setdiff(names(data), c("probe", "chromosome"))
        data <- melt(
            data,
            id.vars = c("probe", "chromosome"),
            measure.vars = measure_vars,
            variable.name = "sample_id",
            value.name = "beta"
        )
        data[, chromosome := NULL]
        data <- merge(data, metadata, by = "sample_id")
        
        # Get gene names
        if (!is.null(annotation)) { 
            log_info("Getting gene names")
            data <- merge(data, annotation[, .(probe, gene_name)], by.x = "probe", by.y = "probe", all.x = TRUE)
        }
        
        # Conduct t-tests
        log_info("Conducting t-tests")
        t_tests <- data.table(t = numeric(), df = numeric(), p = numeric(), probe = character(), gene = character())
        for (i in probes) {
            genes = unique(data[probe == i, gene_name])
            for (j in genes) {
                sub_data <- data[probe == i & gene_name == j]
                t_test <- t.test(
                    sub_data[sub_data[[group]] == case, "beta"],
                    sub_data[sub_data[[group]] == control, "beta"]
                )
                t_tests <- rbind(
                    t_tests,
                    data.table(t = t_test$statistic, df = t_test$parameter, p = t_test$p.value, probe = i, gene = j)
                )
            }
        }
            
        
        # Fix probe names
        probes <- gsub("_", ":", probes)
        data[["probe"]] <- gsub("_", ":", data[["probe"]])
        t_tests[["probe"]] <- gsub("_", ":", t_tests[["probe"]])
        log_debug("Data header:", head(data))
        
        # Plot the data
        log_info("Plotting data")
        if (file && !dir.exists(output)) {
            dir.create(output, recursive = TRUE)
        }
        plots <- list()
        pb <- txtProgressBar(min = 0, max = length(probes), style = 3)
        counter <- 0
        for (i in probes) {
            genes = unique(data[probe == i, gene_name])
            log_debug("Genes for probe", i, ":", paste(genes, collapse = ", "))
            for (j in genes) {
                gene <- j
                
                # Subset data
                sub_data <- data[probe == i & gene_name == j]
                t_sub <- t_tests[probe == i & gene == j]
                
                ### Set subtitle ###
                # Function to convert exponent to superscript
                superscript_map <- function(x) {
                    mapping <- c(
                        "+" = "\u207A", "-" = "\u207B", "0" = "\u2070", "1" = "\u00B9",
                        "2" = "\u00B2", "3" = "\u00B3", "4" = "\u2074", "5" = "\u2075",
                        "6" = "\u2076", "7" = "\u2077", "8" = "\u2078", "9" = "\u2079"
                    )
                    chars <- unlist(strsplit(x, split = ""))
                    paste(sapply(chars, function(y) mapping[[y]]), collapse = "")
                }
                # Format p-value for scientific notation
                t_fmt <- format(t_sub$p, scientific = TRUE, digits = 2)
                # Extract parts of the formatted p-value:
                base_part <- sub("e.*", "", t_fmt)
                exp_part <- sub(".*e", "", t_fmt)
                # Trim exponent part to sign + 2 digits (total 3 characters)
                exp_part <- substr(exp_part, 1, 3)
                # Convert exponent into superscript using the helper function
                sup_exp <- superscript_map(exp_part)
                sub <- paste0("t = ", round(t_sub$t, 2), ", p = ", base_part, " x 10", sup_exp)
                
                # Set lower limit
                lower_limit <- min(sub_data$beta, na.rm = TRUE) * 0.97
                
                # Convert group to factor
                sub_data[[group]] <- factor(sub_data[[group]], levels = c(case, control))
                
                # Create the plot
                p <- ggplot(sub_data, aes(x = .data[[group]], y = beta, fill = .data[[group]])) +
                    geom_violin(alpha = 0.8, color = "black") +
                    labs(
                        title = paste0(i, "-", gene),
                        subtitle = sub,
                        x = "",
                        y = "β Value",
                        fill = NULL
                    ) +
                    scale_y_continuous(limits = c(lower_limit, NA)) +
                    scale_fill_manual(values = colors) +
                    cleanThemeSub() +
                    theme(legend.text = element_text(size = 16)) +
                    stat_summary(
                        fun = median, geom = "point", shape = 21, size = 5,
                        fill = "white", color = "black"
                    ) +
                    stat_summary(
                        fun.data = function(y) {
                            data.frame(
                                y = median(y),
                                ymin = quantile(y, 0.25),
                                ymax = quantile(y, 0.75)
                            )
                        },
                        geom = "errorbar", width = 0.3, linewidth = 2, color = "black"
                    )
                plots[[i]] <- p
                
                # Save the plot
                if (file) {
                    log_debug("Saving plot for probe:", i)
                    file_name <- gsub(":", "_", i)
                    file_name <- paste0(j, "_", file_name)
                    ggsave(
                        filename = paste0(output, "/", file_name, ".png"),
                        plot = p,
                        width = 8,
                        height = 6
                    )
                }
            }
            
            counter <- counter + 1
            setTxtProgressBar(pb, counter)
        }
        close(pb)
        
        if (!file) {
            return(plots)
        } else {
            success_message("Violin plots saved to:", output)
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
    "loggingUtils.R", "cleanThemeSub.R", "dictToList.R", "getChromosomeList.R", "getProbeIndices.R",
    "getSampleIndices.R", "readChromosomeData.R", "stringToList.R"
)
for (util_file in utils_files) {
    util_path <- file.path(utils_dir, util_file)
    source(util_path)
}

# Parse command line arguments
options <- list(
    list(flags = c("-i", "--input"), type = "character"),
    list(flags = c("-m", "--metadata"), type = "character"),
    list(flags = c("-p", "--probes"), type = "character"),
    list(flags = c("-o", "--output"), type = "character", default = NULL),
    list(
        flags = c("-g", "--groups"),
        type = "character",
        default = "{'group': 'group_var', 'case': 'case_group', 'control': 'control_group'}"
    ),
    list(flags = c("-c", "--colors"), type = "character", default = "#FF0000, #0000FF"),
    list(flags = c("-a", "--annotation"), type = "character", default = NULL),
    list(flags = c("-f", "--file"), type = "logical", default = TRUE)
)

if (!interactive()) {
    source(file.path(script_dir, "utils/initializeScript.R"))
    opt <- initializeScript(option_list = options, script_name = "plotViolinPlot")
    plotViolinPlot(
        input = opt$input,
        metadata = opt$metadata,
        probes = opt$probes,
        output = opt$output,
        groups = opt$groups,
        colors = opt$colors,
        annotation = opt$annotation,
        file = opt$file
    )
}
