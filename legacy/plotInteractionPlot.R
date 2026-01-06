#!/usr/bin/env Rscript
# Import the necessary libraries
suppressPackageStartupMessages({
    library(ggplot2)
    library(reshape2)
})

# Ensure null PDF device
pdf(file = NULL)

# Main function
plotInteractionPlot <- function(
    input,
    groups,
    cat_var,
    vars_to_plot,
    colors,
    output = NULL,
    highlight = NULL,
    rect_width = 0.5,
    labels = NULL,
    file = TRUE
) {
    tryCatch({
        # Split the input strings
        log_debug("Splitting input strings")
        data <- stringToList(input)
        groups <- stringToList(groups)
        colors <- stringToList(colors)
        vars_to_plot <- stringToList(vars_to_plot)
        
        # Load data
        log_debug("Loading data files")
        data_list <- lapply(seq_along(data), function(i) {
            data_path <- data[i]
            
            # Read data
            log_debug("Reading file:", data_path)
            data_df <- read.csv(data_path, stringsAsFactors = FALSE)
            
            # Extract the category variable and variables to plot
            log_debug("Extracting category variable and variables to plot")
            data_df <- data_df[, c(cat_var, vars_to_plot), drop = FALSE]
            
            # Add groups name prefix to all column names except category variable
            cols <- names(data_df)
            cols[cols != cat_var] <- paste0(groups[i], "_", cols[cols != cat_var])
            names(data_df) <- cols
            return(data_df)
        })
        
        # Merge data by category variable
        log_debug("Merging data by category variable")
        data <- Reduce(function(x, y) merge(x, y, by = cat_var, all = TRUE), data_list)
        
        # Find the set to highlight
        log_debug("Finding the set to highlight")
        if (is.null(highlight)) {
            log_debug("No set to highlight provided")
            highlight <- NULL
        } else {
            highlight <- as.character(highlight)
        }
        
        # Reshape the data for plotting
        log_debug("Reshaping data for plotting")
        data_long <- reshape2::melt(data, id.vars = cat_var)
        data_long$group <- sub("_.*", "", data_long$variable)
        data_long$metric <- sub(".*_", "", data_long$variable)
        vars_to_plot <- sub(".*_", "", vars_to_plot)
        data_long$metric <- factor(data_long$metric, levels = vars_to_plot)
        data_long$variable <- NULL
        log_debug("Plot data reshaped")
        
        # Set plot parameters
        if (length(vars_to_plot) == 1) {
            linetypes <- c("solid")
        } else if (length(vars_to_plot) == 2) {
            linetypes <- c("solid", "dashed")
        } else if (length(vars_to_plot) == 3) {
            linetypes <- c("solid", "dashed", "dotted")
        } else {
            stop("Number of variables to plot must be between 1 and 3")
        }
        
        if (is.null(labels)) {
            x_label <- "Set"
            y_label <- "Value"
            color_label <- "Group"
            title_label <- ""
            line_labels <- unique(data_long$metric)
        } else {
            labels <- dictToList(labels)
            x_label <- labels$x
            y_label <- labels$y
            color_label <- labels$color
            title_label <- labels$title
            line_labels <- c(unlist(labels$line))
            names(line_labels) <- unique(data_long$metric)
        }
        
        # Calculate the mean of all values at highlight set
        log_debug("Calculating the mean of all values at highlight set")
        if (!is.null(highlight)) {
            highlight_data <- data_long[data_long[[cat_var]] == highlight, ]
            highlight_data <- aggregate(value ~ group + metric, data = highlight_data, FUN = mean)
            highlight_data[[cat_var]] <- highlight
            
            # Calculate the mean for each metric regardless of group
            highlight_data_all <- aggregate(value ~ metric, data = highlight_data, FUN = mean)
            highlight_data_all[[cat_var]] <- highlight
            cat("Mean values at highlight set:\n")
            for (i in highlight_data_all$metric) {
                cat(paste0("   ", i, ": ", round(highlight_data_all$value[highlight_data_all$metric == i], 2), "\n"))
            }
        }
        
        # Generate the plot
        log_info("Generating plot")
        plot <- ggplot(
            data_long,
            aes(x = !!sym(cat_var), y = value, color = group, linetype = metric)
        ) +
            geom_line(linewidth = 1.3) +
            labs(x = x_label, y = y_label, color = color_label, title = title_label, linetype = "Metric") +
            scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
            scale_x_continuous(breaks = unique(data_long[, cat_var])) +
            scale_linetype_manual(values = linetypes, labels = line_labels) +
            geom_rect(
                data = data_long,
                aes(
                    xmin = as.numeric(highlight) - rect_width,
                    xmax = as.numeric(highlight) + rect_width,
                    ymin = -Inf,
                    ymax = Inf
                ),
                fill = NA,
                color = "black",
                linetype = "dashed",
                linewidth = 0.7
            ) +
            scale_color_manual(values = colors) +
            theme(
                plot.title = element_text(hjust = 0.5, size = 24),
                axis.title.x = element_text(size = 22, margin = margin(t = 10)),
                axis.title.y = element_text(size = 22, margin = margin(r = 10)),
                axis.line.x = element_line(linewidth = 0.5, linetype = "solid", colour = "black"),
                axis.line.y = element_line(linewidth = 0.5, linetype = "solid", colour = "black"),
                panel.background = element_rect(fill = "white", colour = "white"),
                panel.border = element_blank(),
                panel.grid.major = element_line(linetype = "dotted", colour = "gray"),
                panel.grid.minor = element_line(linetype = "dotted", colour = "gray"),
                legend.position = "right",
                legend.text = element_text(size = 18, family = "Arial Black", face = "bold"),
                legend.title = element_text(size = 18, family = "Arial Black", face = "bold"),
                legend.key.size = unit(1.25, "cm"),
                legend.key = element_rect(fill = "white", color = NA),
                legend.background = element_rect(fill = "white", color = NA),
                axis.text.y = element_text(size = 20),
                axis.text.x = element_text(
                    angle = 90, vjust = 0.4, hjust = 1,
                    face = "plain",
                    size = 18,
                    color = "black"
                )
            )
        
        if (!file) {
            return(plot)
        } else {
            # Save the plot
            log_info("Saving plot to:", output)
            ggsave(file = output, plot = plot, width = 13, height = 9)
            # Print success message
            success_message("Interaction plot generated successfully and saved to", output)
        }
        
    }, error = function(e) {
        log_error(e$message)
    }, warning = function(w) {
        log_warn(w$message)
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
utils_files <- c("loggingUtils.R", "dictToList.R", "stringToList.R")
for (util_file in utils_files) {
    util_path <- file.path(utils_dir, util_file)
    source(util_path)
}

# Parse command line arguments
options <- list(
    list(flags = c("-i", "--input"), type = "character"),
    list(flags = c("-g", "--groups"), type = "character"),
    list(flags = c("-a", "--cat_var"), type = "character"),
    list(flags = c("-b", "--vars_to_plot"), type = "character"),
    list(flags = c("-c", "--colors"), type = "character"),
    list(flags = c("-o", "--output"), type = "character", default = NULL),
    list(flags = c("-t", "--highlight"), type = "character", default = NULL),
    list(flags = c("-w", "--rect_width"), type = "numeric", default = 0.5),
    list(flags = c("-e", "--labels"), type = "character", default = NULL),
    list(flags = c("-f", "--file"), type = "logical", default = TRUE)
)

if (!interactive()) {
    source(file.path(script_dir, "utils/initializeScript.R"))
    opt <- initializeScript(option_list = options, script_name = "plotInteractionPlot")
    plotInteractionPlot(
        input = opt$input,
        groups = opt$groups,
        cat_var = opt$cat_var,
        vars_to_plot = opt$vars_to_plot,
        colors = opt$colors,
        output = opt$output,
        highlight = opt$highlight,
        rect_width = opt$rect_width,
        labels = opt$labels,
        file = opt$file
    )
}
