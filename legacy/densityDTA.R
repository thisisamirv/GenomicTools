#!/usr/bin/env Rscript
# Helper function to generate density plot with threshold and annotation for DTA analysis
# Import the necessary libraries
suppressPackageStartupMessages({
    library(cowplot)
    library(data.table)
    library(ggplot2)
})

# Main function
densityDTA <- function(
    data,
    group_column,
    score_column,
    group_labels = c(case = "R", control = "NR"),
    colors,
    threshold,
    x_limits = c(-0.3, 1.3)
) {
    tryCatch({
        # Change group column values from group labels to case/control
        data <- data[as.character(get(group_column)) %in% c(group_labels[["case"]], group_labels[["control"]])]
        data[as.character(get(group_column)) == group_labels[["case"]], (group_column) := "case"]
        data[as.character(get(group_column)) == group_labels[["control"]], (group_column) := "control"]
        log_debug("Data group column head after conversion:", head(data[[group_column]]))
        
        # Ensure grouping variable is a factor
        data[[group_column]] <- factor(data[[group_column]])
        
        # Check for at least two groups
        groups <- levels(data[[group_column]])
        if (length(groups) < 2) {
            stop("Grouping variable must have at least two levels for statistical tests.")
        }
        
        # Extract scores for each group
        log_debug("Extracting scores for each group.")
        case_scores <- data[[score_column]][data[[group_column]] == "case"]
        control_scores <- data[[score_column]][data[[group_column]] == "control"]
        
        # Perform T-test
        t_test_result <- t.test(case_scores, control_scores)
        t_statistic <- round(t_test_result$statistic, 1)
        t_p_value <- sprintf("%.2e", t_test_result$p.value)
        
        # Predict responses based on threshold
        data$predicted_response <- ifelse(data[[score_column]] >= threshold, "control", "case")
        data$predicted_response <- factor(data$predicted_response)
        
        # Calculate accuracy
        log_debug("Calculating accuracy.")
        correct_predictions <- sum(data$predicted_response == data[[group_column]], na.rm = TRUE)
        total_samples <- sum(!is.na(data$predicted_response))
        accuracy <- correct_predictions / total_samples
        log_info("Accuracy:", round(accuracy,2 ))
        
        # Calculate sensitivity and specificity
        log_debug("Calculating sensitivity and specificity.")
        true_positives <- sum(data$predicted_response == "control" & data[[group_column]] == "control", na.rm = TRUE)
        false_negatives <- sum(data$predicted_response == "case" & data[[group_column]] == "control", na.rm = TRUE)
        true_negatives <- sum(data$predicted_response == "case" & data[[group_column]] == "case", na.rm = TRUE)
        false_positives <- sum(data$predicted_response == "control" & data[[group_column]] == "case", na.rm = TRUE)
        
        sensitivity <- true_positives / (true_positives + false_negatives)
        specificity <- true_negatives / (true_negatives + false_positives)
        print(paste("Sensitivity:", round(sensitivity, 2)))
        print(paste("Specificity:", round(specificity)))
        print(paste("Accuracy:", round(accuracy)))
        
        # Confusion matrix and Chi-square test
        log_debug("Creating confusion matrix.")
        confusion_matrix <- table(
            Predicted = data$predicted_response, 
            Actual = data[[group_column]]
        )
        
        if (all(dim(confusion_matrix) >= c(2, 2))) {
            chi_sq_test <- chisq.test(confusion_matrix)
            chi_sq_stat <- round(chi_sq_test$statistic, 1)
            chi_sq_p <- sprintf("%.2e", chi_sq_test$p.value)
        } else {
            chi_sq_stat <- NA
            chi_sq_p <- NA
            log_warn("Chi-square test not applicable for the given confusion matrix.")
        }
        
        # Create annotation texts
        log_debug("Creating annotation texts.")
        footnote_text <- paste0(
            "X\u00b2 = ", 
            ifelse(is.na(chi_sq_stat), "NA", chi_sq_stat),
            ", p = ", 
            ifelse(is.na(chi_sq_p), "NA", chi_sq_p), 
            ", Accuracy = ", 
            round(accuracy * 100, 1), "%"
        )
        subtitle_text <- paste0("t = ", t_statistic, ", p = ", t_p_value)
        
        # Change group column values from group case/control to group labels
        log_debug("Changing group column values from case/control to group labels.")
        data[, (group_column) := ifelse(get(group_column) == "case", group_labels[1], group_labels[2])]
        
        # Convert group column back to factor
        data[[group_column]] <- as.factor(as.character(data[[group_column]]))
        
        # Create density plot
        log_debug("Creating density plot.")
        density_plot <- ggplot(data, aes(x = .data[[score_column]], fill = .data[[group_column]], group = .data[[group_column]])) +
            geom_density(alpha = 0.4, adjust = 1.5) +
            geom_vline(xintercept = threshold, linetype = "dashed", color = "black", linewidth = 0.5) +
            labs(title = "", subtitle = subtitle_text, x = "PGMS") +
            xlim(x_limits) +
            scale_fill_manual(values = colors) +
            cleanTheme() +
            theme(
                plot.title = element_text(hjust = 0.5, size = 14),
                plot.subtitle = element_text(hjust = 0.5, size = 18),
                axis.text.x = element_text(size = 12),
                axis.text.y = element_blank(),
                axis.title.x = element_text(size = 12),
                axis.title.y = element_blank(),
                axis.ticks.y = element_blank(),
                axis.line.y = element_blank(),
                legend.position = "left",
                legend.text = element_text(size = 14),
                legend.title = element_blank(),
                panel.grid.major.x = element_blank(),
                panel.grid.major.y = element_blank(),
                panel.grid.minor.x = element_blank(),
                panel.grid.minor.y = element_blank()
            )
        
        # Create footnote plot
        footnote_plot <- ggplot() +                                                  
            annotate("text", x = 0.5, y = 0.5, label = footnote_text, hjust = 0.5, size = 5.5, color = "black") +
            emptyTheme()                                                                  
        
        # Combine density plot and footnote
        final_plot <- plot_grid(density_plot, footnote_plot, ncol = 1, rel_heights = c(10, 1.5))
        
        return(final_plot)
        
    }, error = function(e) {
        log_error(e$message)
        return(NULL)
    }, warning = function(w) {
        log_warn(w$message)
    })
}

# Source helper functions
if (!interactive()) {
    args <- commandArgs(trailingOnly = FALSE)
    script_dir <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
    source(file.path(script_dir, "utils/loggingUtils.R"))
    source(file.path(script_dir, "utils/cleanTheme.R"))
    source(file.path(script_dir, "utils/emptyTheme.R"))
} else {
    script_dir <- dirname(normalizePath(parent.frame(2)$ofile))
    source(file.path(script_dir, "loggingUtils.R"))
    source(file.path(script_dir, "cleanTheme.R"))
    source(file.path(script_dir, "emptyTheme.R"))
}
