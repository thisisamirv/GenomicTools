#!/usr/bin/env Rscript
# Helper function to perform ROC analysis
# Import the necessary libraries
suppressPackageStartupMessages({
    library(data.table)
    library(pROC)
    library(ggplot2)
    library(reshape2)
})

# Main function
rocAnalysis <- function(data, group_column, score_column, smooth = TRUE, plot = TRUE, threshold = TRUE) {
    tryCatch({
        # Store original group names
        group_names <- unique(data[[group_column]])
        
        # Convert group to numeric
        data[, group_numeric := as.numeric(as.factor(get(group_column)))]
        
        # Generate ROC curve
        roc_curve <- roc(data[["group_numeric"]], data[[score_column]])
        auroc <- round(auc(roc_curve), 2)
        ci_auc <- ci.auc(roc_curve, conf.level = 0.95)
        
        # Check for perfect ROC, step function, or ties/numerical issues
        is_perfect_roc <- (auroc == 1 || auroc == 0)
        is_step_function <- (
            length(unique(roc_curve$sensitivities)) <= 2 ||
            length(unique(roc_curve$specificities)) <= 2
        )
        has_ties_or_numerical_issues <- (length(unique(data[[score_column]])) < nrow(data))
        all_thresholds_unique <- (length(unique(roc_curve$thresholds)) == nrow(data))
        
        if (is_perfect_roc || is_step_function || has_ties_or_numerical_issues) {
            set.seed(42) # for reproducibility
            data[[score_column]] <- data[[score_column]] + rnorm(nrow(data), mean = 0, sd = 1e-6)
            roc_curve <- roc(data[["group_numeric"]], data[[score_column]])
            auroc <- round(auc(roc_curve), 2)
            ci_auc <- ci.auc(roc_curve, conf.level = 0.95)
        }
        
        # Smooth ROC curve if required
        if (smooth) {
            smooth_methods <- c("binormal", "density", "fitdistr")
            smoothed <- FALSE
            for (method in smooth_methods) {
                smoothed_curve <- tryCatch(
                    smooth(roc_curve, method = method),
                    error = function(e) {
                        return(NULL)
                    }
                )
                if (!is.null(smoothed_curve)) {
                    smoothed <- TRUE
                    break
                }
            }
            if (!smoothed) {
                print("ROC curve could not be smoothed with any method and will be shown as a step function.")
            }
        } else {
            smoothed_curve <- roc_curve
        }
        
        # Extract sensitivities and specificities
        sensitivities <- smoothed_curve$sensitivities
        specificities <- smoothed_curve$specificities
        print(paste("AUROC:", auroc))
        print(paste("95% CI for AUROC:", paste(round(ci_auc, 2), collapse = " - ")))
        
        # Calculate number of positive and negative samples
        n_pos <- data[group_numeric == 1, .N]
        n_neg <- data[group_numeric == 2, .N]
        
        # Z-score for 95% confidence interval
        z <- 1.96
        
        # Calculate standard errors
        se_sens <- sqrt((sensitivities * (1 - sensitivities)) / n_pos)
        se_spec <- sqrt((specificities * (1 - specificities)) / n_neg)
        
        # Calculate confidence intervals
        ci_sens_lower <- sensitivities - z * se_sens
        ci_sens_upper <- sensitivities + z * se_sens
        ci_spec_lower <- specificities - z * se_spec
        ci_spec_upper <- specificities + z * se_spec
        
        # Adjust Specificity for plotting
        one_minus_specificity <- 1 - specificities
        ci_spec_lower_1_minus <- 1 - ci_spec_upper
        ci_spec_upper_1_minus <- 1 - ci_spec_lower
        
        # Create ROC data table
        roc_data <- data.table(
            Threshold = 1:length(sensitivities),
            Sensitivity = sensitivities,
            Specificity = specificities,
            OneMinusSpecificity = one_minus_specificity,
            SensitivityLower = pmax(ci_sens_lower, 0),
            SensitivityUpper = pmin(ci_sens_upper, 1),
            SpecificityLower = pmax(ci_spec_lower_1_minus, 0),
            SpecificityUpper = pmin(ci_spec_upper_1_minus, 1)
        )
        
        # Find the optimal threshold if required
        if (threshold) {
    optimal_coords <- coords(smoothed_curve, "best", ret = "threshold", transpose = FALSE)
    threshold <- as.numeric(optimal_coords)
} else {
    threshold <- NULL
}
        
        if (plot) {
            # Create ROC plot
            roc_plot <- ggroc(
                data = smoothed_curve,
                legacy.axes = TRUE,
                color = "blue",
                linewidth  = 1.2
            ) +
            labs(
                title = paste(
                    "AUROC = ", auroc,
                    " (95% CI: ", round(ci_auc[1], 2), "-", round(ci_auc[3], 2), ")"
                ),
                x = "1 - Specificity",
                y = "Sensitivity"
            ) +
            cleanTheme() +
            theme(plot.title = element_text(hjust = 0.5, size = 18))
            
            # Add 95% confidence intervals to the ROC plot
            roc_plot <- roc_plot + 
                geom_ribbon(
                    data = roc_data,
                    aes(x = OneMinusSpecificity, ymin = SensitivityLower, ymax = SensitivityUpper),
                    fill = "grey",
                    alpha = 0.2
                )
        } else {
            roc_plot <- NULL
        }
        
        # Remove the temporary group_numeric column
        data[, group_numeric := NULL]
        
        return(list(
    roc_curve = smoothed_curve,
    auroc = auroc,
    ci_auc = ci_auc,
    roc_data = roc_data,
    roc_plot = roc_plot,
    threshold = threshold
))
        
    }, error = function(e) {
        return(NULL)
    }, warning = function(w) {
    })
}
