#!/usr/bin/env Rscript
# Import the necessary libraries
suppressPackageStartupMessages({
    library(data.table)
    library(cowplot)
    library(ggplot2)
    library(reshape2)
})

# Ensure null PDF device
pdf(file = NULL)

# Main function
plotFacetedDeltaBarPlots <- function(
    input,
    baseline_value,
    cat_var,
    cont_var,
    colors,
    output = NULL,
    grouping_vars = "{'group1': 'group1', 'group2': 'group2'}",
    plot_params = "{'cat_var_name': 'x', 'cont_var_name': 'y', 'group2_name': 'group2'}",
    file = TRUE
) {
    tryCatch({
        # Split the input strings
        log_debug("Splitting the input strings")
        grouping_vars <- dictToList(grouping_vars)
        plot_params <- dictToList(plot_params)
        colors <- stringToList(colors)
        
        # Read the input data
        log_info("Reading the input data")
        data <- fread(input)
        data <- data[,
            .(sample_id, get(cat_var), get(cont_var), get(grouping_vars[['group1']]), get(grouping_vars[['group2']]))
        ]
        setnames(data, c("sample_id", cat_var, cont_var, grouping_vars[['group1']], grouping_vars[['group2']]))
        
        # Get group levels
        log_debug("Getting group levels")
        group1_levels <- as.character(data[, unique(get(grouping_vars[['group1']]))])
        group2_levels <- as.character(data[, unique(get(grouping_vars[['group2']]))])
        
        # Drop NA values from group levels
        log_debug("Dropping NA values from group levels")
        group1_levels <- group1_levels[!is.na(group1_levels)]
        group2_levels <- group2_levels[!is.na(group2_levels)]
        
        # Ensure the grouping levels are 2 and 2
        log_debug("Ensuring the grouping levels are 2 and 2")
        if (length(group1_levels) != 2 || length(group2_levels) != 2) {
            stop("The grouping variables must have exactly 2 levels each")
        }
        
        # Create the equation and title plots
        log_debug("Creating the equation and title plots")
        equation <- as.call( 
            bquote(paste(Delta, .(plot_params[["cont_var_name"]]), " = ",
              .(plot_params[["cont_var_name"]])[.(plot_params[["cat_var_name"]])], " - ", 
              .(plot_params[["cont_var_name"]])["Baseline"])))
        title <- ggplot() + annotate("text", x = 1, y = 1, size = 8, parse = TRUE, label = deparse(equation)) + emptyTheme()
        title1 <- ggplot() +
            annotate(
                "text",
                x = 1,
                y = 1,
                size = 12,
                label = paste(plot_params[["group2_name"]], group2_levels[1])
            ) + emptyTheme()
        title2 <- ggplot() +
            annotate(
                "text",
                x = 1,
                y = 1,
                size = 12,
                label = paste(plot_params[["group2_name"]], group2_levels[2])
            ) + emptyTheme()
        
        # Double the colors
        log_debug("Doubling the colors")
        colors <- c(colors[1], colors[1], colors[2], colors[2])
        
        # Loop over the group levels
        plot_list <- list()
        loop <- 0
        for (value2 in group2_levels) {
            for (value1 in group1_levels) {
                loop <- loop + 1
                # Subset the data
                log_debug("Subsetting the data")
                sub_data <- data[get(grouping_vars[['group1']]) == value1]
                
                # Subset baseline scores
                log_info("Subsetting baseline scores")
                baseline <- sub_data[
                    get(cat_var) == baseline_value,
                    .(sample_id, get(cat_var), get(cont_var), get(grouping_vars[['group2']]))
                ]
                setnames(baseline, c("sample_id", cat_var, cont_var, grouping_vars[['group2']]))
                baseline <- unique(baseline, by = "sample_id")
                other <- sub_data[
                    get(cat_var) != baseline_value,
                    .(sample_id, get(cat_var), get(cont_var))
                ]
                setnames(other, c("sample_id", cat_var, cont_var))
                
                # Calculate changes from baseline
                log_info("Calculating changes from baseline")
                changes <- data.table(
                    sample_id = character(0),
                    cat_var = integer(0),
                    Change = numeric(0),
                    group2 = character(0)
                )
                setnames(changes, "group2", grouping_vars[['group2']])
                
                log_debug("Iterating over the unique sample IDs")
                changes <- merge(other, baseline, by = "sample_id", suffixes = c("", "_baseline"))
                changes[, Change := get(paste0(cont_var)) - get(paste0(cont_var, "_baseline"))]
                changes <- changes[, .(
                    sample_id,
                    cat_var = get(paste0(cat_var)),
                    Change,
                    group2 = get(grouping_vars[['group2']])
                )]
                setnames(changes, c("sample_id", cat_var, "Change", grouping_vars[['group2']]))
                
                # Prepare the plot data
                log_debug("Preparing the plot data")
                plot_data <- data.table(
                    variable = character(0),
                    mean = numeric(0),
                    sd = numeric(0),
                    cat_var = character(0)
                )
                for (i in unique(changes[[cat_var]])) {
                    group_data <- changes[
                        get(grouping_vars[['group2']]) == value2 & get(cat_var) == i, Change
                    ]
                    mean_val <- mean(group_data, na.rm = TRUE)
                    sd_val <- sd(group_data, na.rm = TRUE)
                    plot_data <- rbind(
                        plot_data,
                        data.table(variable = "Diff", mean = mean_val, sd = sd_val, cat_var = i)
                    )
                }
                setnames(plot_data, "variable", cont_var)
                
                # Plot the data
                log_info("Plotting the data")
                y_range <- range(changes$Change, na.rm = TRUE)
                # Convert cat_var to numeric, sort values, then back to character for proper ordering
                ordered_levels <- as.character(sort(as.numeric(unique(plot_data$cat_var))))
                log_debug(paste("Ordered levels: ", paste(ordered_levels, collapse=", ")))
                
                # Apply the ordered factor to all relevant data frames
                plot_data$cat_var <- factor(plot_data$cat_var, levels = ordered_levels)
                changes[[cat_var]] <- factor(changes[[cat_var]], levels = ordered_levels)
                
                # Print debug info to verify data structure
                log_debug("Plot data head: ")
                log_debug(head(plot_data))
                
                plot <- ggplot(plot_data, aes(x = cat_var, y = mean)) +
                    geom_bar(stat = "identity", fill = colors[loop], width = 0.5) +
                    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2, color = "black") +
                    labs(
                        title = value1,
                        subtitle = paste0(
                            "Mean Δ", plot_params[["cont_var_name"]], ": ", round(mean_val, 2), 
                            " (SD: ", round(sd_val, 2), ")"
                        ),
                        x = plot_params[["cat_var_name"]],
                        y = paste0("Δ", plot_params[["cont_var_name"]], " ± SD")
                    ) +
                    # Be explicit about the axis ordering - use both breaks and limits
                    scale_x_discrete(breaks = ordered_levels, limits = ordered_levels) +
                    scale_y_continuous(breaks = seq(-50, 10, by = 10), limits = y_range) +
                    cleanThemeSub() +
                    # Force axis to respect the factor order
                    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
                    
                # Append the plot to the list
                plot_list <- c(plot_list, list(plot))
            }
        }
        
        # Concatenate the plots
        log_debug("Concatenating the plots")
        plot1 <- plot_grid(plot_list[[1]], plot_list[[2]], ncol = 2, nrow = 1)
        plot2 <- plot_grid(plot_list[[3]], plot_list[[4]], ncol = 2, nrow = 1)
        plot1 <- plot_grid(title1, plot1, ncol = 1, nrow = 2, rel_heights = c(0.15, 1))
        plot2 <- plot_grid(title2, plot2, ncol = 1, nrow = 2, rel_heights = c(0.15, 1))
        plot <- plot_grid(plot1, plot2, ncol = 1, nrow = 2)
        plot <- plot_grid(plot, title, ncol = 1, nrow = 2, rel_heights = c(1, 0.07))
        
        if (!file) {
            return(plot)
        } else {
            # Save the plot
            log_debug("Saving the plot")
            ggsave(output, plot, width = 15, height = 15)
            # Print success message
            success_message("Faceted delta bar plots generated successfully and saved to", output)
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
utils_files <- c("loggingUtils.R", "cleanThemeSub.R", "dictToList.R", "emptyTheme.R", "stringToList.R")
for (util_file in utils_files) {
    util_path <- file.path(utils_dir, util_file)
    source(util_path)
}

# Parse command line arguments
options <- list(
    list(flags = c("-i", "--input"), type = "character"),
    list(flags = c("-s", "--baseline_value"), type = "character"),
    list(flags = c("-a", "--cat_var"), type = "character"),
    list(flags = c("-b", "--cont_var"), type = "character"),
    list(flags = c("-c", "--colors"), type = "character"),
    list(flags = c("-o", "--output"), type = "character", default = NULL),
    list(
        flags = c("-g", "--grouping_vars"),
        type = "character",
        default = "{'group1': 'group1', 'group2': 'group2'}"
    ),
    list(
        flags = c("-p", "--plot_params"),
        type = "character",
        default = "{'cat_var_name': 'x', 'cont_var_name': 'y', 'group2_name': 'group2'}"
    ),
    list(flags = c("-f", "--file"), type = "logical", default = TRUE)
)

if (!interactive()) {
    source(file.path(script_dir, "utils/initializeScript.R"))
    opt <- initializeScript(option_list = options, script_name = "plotFacetedDeltaBarPlots")
    plotFacetedDeltaBarPlots(
        input = opt$input,
        baseline_value = opt$baseline_value,
        cat_var = opt$cat_var,
        cont_var = opt$cont_var,
        colors = opt$colors,
        output = opt$output,
        grouping_vars = opt$grouping_vars,
        plot_params = opt$plot_params,
        file = opt$file
    )
}
