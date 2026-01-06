#!/usr/bin/env Rscript
# Import the necessary libraries
suppressPackageStartupMessages({
    library(data.table)
    library(ggplot2)
    library(logger)
    library(rlang)
    library(utils)
})

# Main function
plotDist <- function(
    input,
    title,
    breaks,
    labels,
    output = NULL,
    chunk = 1000,
    width = 15,
    height = 10,
    metric = c("Beta", "Missing"),
    x_interval = 10,
    aggregate = c("Mean", "Individual"),
    file = TRUE
) {
    metric <- match.arg(metric)
    aggregate <- match.arg(aggregate)
    
    tryCatch({
        # Generate distribution data using appropriate function
        log_info("Generating distribution data for", metric)
        distribution <- if (metric == "Beta") {
            getBetaDist(
                h5_file = h5_file,
                breaks = breaks,
                labels = labels,
                chunk = chunk,
                mode = tolower(aggregate)
            )
        } else {
            getMissingData(
                h5_file = h5_file,
                breaks = breaks,
                labels = labels,
                chunk = chunk,
                aggregate = if (aggregate == "Individual") "Individual" else "Probes",
                output = tempfile()
            )
        }

        # Process data based on metric
        if (metric == "Beta") {
            # Beta value processing
            if (aggregate == "Mean") {
                distribution[, Count := as.numeric(Count)]
                distribution <- na.omit(distribution, cols = "Count")
            } else {
                numeric_cols <- setdiff(names(distribution), "Value_Range")
                distribution[, (numeric_cols) := lapply(.SD, as.numeric), .SDcols = numeric_cols]
                distribution <- na.omit(distribution, cols = numeric_cols)
            }
            
            # Clean value range labels
            distribution[, Value_Range := fcase(
                grepl("1$", Value_Range), "1.0",
                default = gsub(" -.*", "", Value_Range)
            )]
            
            # Create Beta plot
            plot <- if (aggregate == "Mean") {
                ggplot(distribution, aes(x = Value_Range, y = Count)) +
                    geom_col(fill = "#0072B2", width = 0.8) +
                    labs(title = title, x = "Beta Value Range", y = "Count (Millions)") +
                    scale_y_continuous(labels = function(x) format(x/1e6, digits = 2))
            } else {
                melted <- melt(
                    distribution,
                    id.vars = "Value_Range",
                    variable.name = "Sample",
                    value.name = "Count"
                )
                
                ggplot(melted, aes(x = Value_Range, y = Count, group = Sample)) +
                    geom_smooth(aes(color = Sample), method = "loess", se = FALSE, span = 0.1) +
                    labs(title = title, x = "Beta Value Range", y = "Count") +
                    scale_color_viridis_d(option = "plasma") +
                    theme(legend.position = "none")
            }
            
        } else {
            # Missing value processing
            distribution[, Missing_Range := factor(
                Missing_Range,
                levels = labels,
                ordered = TRUE
            )]
            
            # Create percentage labels for key bins
            distribution[, Percentage_Label := fcase(
                as.numeric(Missing_Range) <= 0.2, 
                sprintf("%.0f%%", as.numeric(Missing_Range)*100),
                default = NA_character_
            )]
            
            # Create Missing plot
            plot <- ggplot(distribution[!is.na(Percentage_Label)], aes(x = Missing_Range, y = N)) +
                geom_col(fill = "#D55E00", width = 0.8) +
                labs(title = title, x = "Missing Proportion", y = "Count (Thousands)") +
                scale_y_continuous(labels = function(x) format(x/1e3, digits = 2)) +
                scale_x_discrete(labels = distribution[!is.na(Percentage_Label)]$Percentage_Label)
        }

        # Add common theme elements
        plot <- plot + 
            cleanTheme() +
            theme(
                axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
                panel.grid.major.x = element_blank(),
                plot.title = element_text(hjust = 0.5, face = "bold"),
                axis.title = element_text(face = "bold")
            ) +
            scale_x_discrete(breaks = function(x) {
                x[seq(1, length(x), by = x_interval)]
            })

        if (!file){
            return(plot)
        } else {
            # Save plot
            log_info("Saving plot to:", output)
            ggsave(
                plot = plot,
                filename = output,
                width = width,
                height = height,
                units = "in",
                dpi = 300
            )
            success_message("Successfully created visualization:", output)
        }
        
    }, error = function(e) {
        log_error("Visualization failed:", e$message)
    }, warning = function(w) {
        log_warn("Visualization warning:", w$message)
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
utils_files <- c("loggingUtils.R", "cleanTheme.R", "getBetaDist.R", "getMissingDist.R")
for (util_file in utils_files) {
    util_path <- file.path(utils_dir, util_file)
    source(util_path)
}

# Parse command line arguments
options <- list(
    list(flags = c("-i", "--input"), type = "character"),
    list(flags = c("-t", "--title"), type = "character"),
    list(flags = c("-b", "--breaks"), type = "numeric"),
    list(flags = c("-a", "--labels"), type = "character"),
    list(flags = c("-o", "--output"), type = "character", default = NULL),
    list(flags = c("-c", "--chunk"), type = "numeric", default = 1000),
    list(flags = c("-w", "--width"), type = "numeric", default = 15),
    list(flags = c("-e", "--height"), type = "numeric", default = 10),
    list(flags = c("-m", "--metric"), type = "character", default = "Beta"),
    list(flags = c("-x", "--x_interval"), type = "numeric", default = 10),
    list(flags = c("-g", "--aggregate"), type = "character", default = "Mean"),
    list(flags = c("-f", "--file"), type = "logical", default = TRUE)
)

if (!interactive()) {
    source(file.path(script_dir, "utils/initializeScript.R"))
    opt <- initializeScript(option_list = options, script_name = "plotDist")
    plotDist(
        input = opt$input,
        title = opt$title,
        breaks = opt$breaks,
        labels = opt$labels,
        output = opt$output,
        chunk = opt$chunk,
        width = opt$width,
        height = opt$height,
        metric = opt$metric,
        x_interval = opt$x_interval,
        aggregate = opt$aggregate,
        file = opt$file
    )
}
