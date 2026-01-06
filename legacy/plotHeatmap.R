#!/usr/bin/env Rscript
# Import the necessary libraries
suppressPackageStartupMessages({
    library(data.table)
    library(corrplot)
    library(magick)
    library(RColorBrewer)
    library(rhdf5)
})

# Ensure null PDF device
pdf(file = NULL)

# Main function
plotHeatmap <- function(
    input,
    metadata,
    group,
    output = NULL,
    filters = NULL,
    cut_value = 100,
    single_sample_cluster = TRUE,
    metadata_output = NULL,
    z_score = TRUE,
    methods = "{distance: 'euclidean', clustering: 'complete'}",
    plot_params = "{'mirror': 'TRUE', 'cex': '2', 'clusterColors': ['#ff6969', '#8088ff'], 'xlab': 'x', 'ylab': 'y'}",
    file = TRUE
) {
    tryCatch({
        # Convert filters and methods to lists
        filters <- dictToList(filters)
        methods <- dictToList(methods)
        plot_params <- dictToList(plot_params)
        
        # Read the input data
        if (grepl("\\.(h5|H5|hdf5|HDF5)$", input)) {
            # Open the input HDF5 file
            log_info("Opening the input HDF5 file.")
            h5_file <- H5Fopen(input, flags = "H5F_ACC_RDONLY")
            
            # Get the list of chromosomes
            chr_list <- getChromosomeList(h5_file)
            
            # Read the HDF5 data
            log_info("Reading the HDF5 data.")
            data_list <- lapply(chr_list, function(chr) {
            log_debug("Reading data for chromosome:", chr)
            chr_data <- readChromosomeData(h5_file = h5_file, chr = chr)
            return(chr_data)
            })
            
            # Combine all chromosome data and convert to matrix
            data_combined <- rbindlist(data_list)
            
            # Set rownames and remove first column 
            probes <- data_combined[["probe_id"]]
            data_combined[, probe_id := NULL]
            data_matrix <- as.matrix(data_combined)
        } else {
            # Read the input data
            log_info("Reading the input data.")
            data <- fread(input)
            rownames(data) <- data[, 1]
            data <- data[, -1]
            data_matrix <- as.matrix(data)
            data_matrix <- t(data_matrix)
            log_debug("Data dimensions:", dim(data_matrix))
        }
        
        # Read the metadata
        log_info("Reading the metadata.")
        metadata <- fread(metadata)
        log_debug("Metadata dimensions:", dim(metadata))
        
        # Filter metadata
        if (!is.null(filters)) {
            log_info("Filtering metadata.")
            filtered_metadata <- filterDataFrame(metadata, filters)
            log_debug("Filtered metadata dimensions:", dim(filtered_metadata))
        } else {
            filtered_metadata <- copy(metadata)
        }
        
        # Create output directory if it doesn't exist
        log_info("Creating output directory and combining images.")
        output_dir <- dirname(output)
        if (!dir.exists(output_dir)) {
            dir.create(output_dir, recursive = TRUE)
            log_debug("Created output directory:", output_dir)
        }
        
        if (z_score) {
            # Convert data to z-scores
            log_info("Converting data to z-scores.")
            data_scaled <- scaleZ(data_matrix)
            log_debug("Z-score conversion complete. Data range:", range(data_scaled, na.rm = TRUE))
            
            # Scale the data to 0 to 1
            log_info("Scaling data to 0 to 1.")
            data_scaled <- scaleZeroOne(data_scaled)
            log_debug("Data range after scaling:", range(data_scaled, na.rm = TRUE))
            
            # Convert to matrix
            data_matrix <- as.matrix(data_scaled)
            
            # Set legend title
            legend_title <- "z-score"
        } else {
            # Set legend title
            legend_title <- "Value"
        }
        
        # Clustering and distance calculation
        log_debug("Distance method:", methods[["distance"]])
        log_debug("Clustering method:", methods[["clustering"]])
        distance_method <- gsub("'", "", methods[["distance"]])
        clustering_method <- gsub("'", "", methods[["clustering"]])
        
        if (!single_sample_cluster) {
            log_info("Removing single-sample clusters.")
            # Get dendrogram first
            col_names <- colnames(data_matrix)
            dist_matrix <- dist(t(data_matrix), method = distance_method)
            log_debug("Distance matrix shape: ", nrow(as.matrix(dist_matrix)), " x ", ncol(as.matrix(dist_matrix)))
            hc <- hclust(dist_matrix, method = clustering_method)
            clusters <- cutree(hc, h = cut_value)
            
            # Remove single-sample clusters and print removed samples
            log_debug("Data matrix dimensions before removing single-sample clusters:", dim(data_matrix))
            cluster_sizes <- table(clusters)
            valid_clusters <- which(cluster_sizes > 1)
            valid_samples <- which(clusters %in% valid_clusters)
            removed_samples <- which(!clusters %in% valid_clusters)
            if (length(removed_samples) > 0) {
                removed_sample_names <- col_names[removed_samples]
                cat(paste("Removed", length(removed_samples), "samples:", paste(removed_sample_names, collapse=", "), "\n"))
            }
            data_matrix <- data_matrix[, valid_samples]
            colnames(data_matrix) <- col_names[valid_samples]
            
            # Remove single-sample clusters from metadata
            filtered_metadata <- filtered_metadata[get("sample_id") %in% colnames(data_matrix)]
            log_debug("Data matrix dimensions after removing single-sample clusters:", dim(data_matrix))
            log_debug("Metadata dimensions after removing single-sample clusters:", dim(filtered_metadata))
        } else {
            filtered_metadata <- copy(metadata)
            if (!is.data.table(filtered_metadata)) setDT(filtered_metadata)
        }
        
        # Plot the heatmap
        log_info("Plotting the heatmap.")
        tmp1_path <- tempfile(pattern = "heatmap", fileext = ".png")
        png(tmp1_path, width = 10, height = 10, units = "in", res = 200) 
        plot <- annotatedHeatmap(
            input = data_matrix,
            ColSideCut = cut_value,
            distance_method = distance_method,
            clustering_method = clustering_method,
            ColSideWidth = 0.5,
            ColSideAnn = data.frame(as.factor(filtered_metadata[[group]])),
            xlab = plot_params[["xlab"]],
            ylab = plot_params[["ylab"]],
            margins = c(1.75, 1.75),
            clusterColors = plot_params[["clusterColors"]],
            mirror = plot_params[["mirror"]],
            lab_size = plot_params[["cex"]]
        )
        dev.off()
        
        # Chi-square test
        log_info("Performing chi-square test on cluster assignments.")
        cluster_assignments <- plot$cutColoumIndList
        group_labels <- filtered_metadata[[group]]
        group_levels <- unique(filtered_metadata[[group]])
        log_debug("Group levels:", group_levels)
        cluster_distribution <- matrix(0, nrow=1, ncol=length(group_levels))
        colnames(cluster_distribution) <- group_levels
        log_debug("Cluster distribution matrix:", cluster_distribution)
        cluster_distribution[1,] <- table(factor(group_labels[cluster_assignments[[1]]], levels=group_levels))
        
        # Create cluster assignments vector for metadata
        log_info("Adding cluster assignments to metadata.")
        filtered_metadata[, cluster := NA_integer_]
        for (cluster_idx in 1:length(cluster_assignments)) {
            cluster_num <- length(cluster_assignments) - cluster_idx + 1
            valid_indices <- cluster_assignments[[cluster_idx]][cluster_assignments[[cluster_idx]] <= nrow(filtered_metadata)]
            filtered_metadata[valid_indices, cluster := cluster_num]
        }
        
        log_debug("Cluster distribution matrix after first cluster:", cluster_distribution)
        if (length(cluster_assignments) > 1) {
            for (cluster_idx in 2:length(cluster_assignments)) {
                valid_indices <- cluster_assignments[[cluster_idx]][cluster_assignments[[cluster_idx]] <= length(group_labels)]
                new_row <- table(factor(group_labels[valid_indices], levels=group_levels))
                cluster_distribution <- rbind(cluster_distribution, new_row)
            }
        }
        chi_square_test <- chisq.test(cluster_distribution, simulate.p.value = TRUE, B = 10000)
        log_debug("Chi-square test - statistic:", chi_square_test$statistic, "p-value:", chi_square_test$p.value)
        
        # Write updated metadata with cluster assignments
        if (!is.null(metadata_output)) {
            # Find the sample column in the metadata
            log_debug("Finding the sample column in the metadata.")
            setDT(metadata)
            col_name <- names(metadata)[which.max(sapply(metadata, function(x) 
            length(intersect(x, colnames(data_matrix))) / length(colnames(data_matrix)) > 0.8))]
            
            if (is.null(col_name)) {
                stop("Could not find matching sample column in metadata")
            }
            
            # Merge cluster assignments with the original metadata
            log_debug("Merging cluster assignments with the original metadata.")
            metadata[filtered_metadata, cluster := i.cluster, on = col_name]
            
            if (file) {
                # Write the updated metadata to a file
                log_info("Writing updated metadata to a file.")
                fwrite(metadata, metadata_output)
            }
        }
        
        # Convert cluster assignments to a matrix and calculate proportions
        log_info("Calculating cluster proportions and creating barplot.")
        DT <- as.data.table(cluster_distribution)
        setnames(DT, colnames(cluster_distribution))
        
        # Add cluster labels as a column
        DT[, Cluster := paste("Cluster", .N:1)]
        setkey(DT, Cluster)
        
        # Convert to matrix format required for barplot
        cluster_proportions <- as.matrix(DT[, .SD, .SDcols = !c("Cluster")])
        rownames(cluster_proportions) <- DT$Cluster
        
        # Calculate percentages within each column (group)
        cluster_proportions <- apply(cluster_proportions, 2, function(x) x/sum(x) * 100)
        
        # Plot the barplot
        tmp2_path <- tempfile(pattern = "barplot", fileext = ".png")
        png(filename = tmp2_path, width = 10, height = 6, units = "in", res = 200)
        layout(matrix(c(1,2), nrow = 2), heights = c(5,1))
        par(mar = c(2, 9, 2, 9))
        cluster_colors <- adjustcolor(plot_params[["clusterColors"]], alpha.f = 0.4)
        if (nrow(cluster_proportions) > 2) {
            cluster_colors <- c(cluster_colors, 
            colorRampPalette(brewer.pal(n = 8, name = "Paired"))(nrow(cluster_proportions) - 2))
        }
        barplot(cluster_proportions,
            beside = FALSE,
            col = if (plot_params[["mirror"]]) cluster_colors else rev(cluster_colors),
            xlab = "",
            ylab = "Percentage",
            cex.lab = 2,
            cex.names = 1.3,
            ylim = c(0, 100),
            yaxp = c(0, 100, 5),
            space = 1.5,
            main = bquote(X^2 == .(round(chi_square_test$statistic, 2)) ~
            ", p =" ~
            .(sprintf("%.2e", chi_square_test$p.value))),
            cex.main = 1.5)

        par(mar = c(0, 0, 0, 0))
        plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), axes = FALSE, xlab = "", ylab = "")
        legend(
            "top", 
            legend = rownames(cluster_proportions),
            cex = 2,
            fill = if (plot_params[["mirror"]]) cluster_colors else rev(cluster_colors),
            title = "", 
            horiz = TRUE,
            inset = c(0.05, -0.5),
            box.lwd = 0
        )
        dev.off()
        
        # Combine all images vertically into a single output file
        image1 <- image_read(tmp1_path)
        image2 <- image_read(tmp2_path)
        stacked_image <- image_append(c(image1, image2), stack = TRUE)
        if (file) {
            image_write(stacked_image, path = output)
            log_debug("Wrote final image to:", output)
            # Print success message
            success_message("Heatmap plot generated successfully and saved to:", output)
        } else {
            return(list(metadata = metadata, plot = stacked_image))
        }
    
    }, error = function(e) {
        log_error(e$message)
    }, warning = function(w) {
        log_warn(w$message)
    }, finally = {
        if (exists("h5_file")) {
            H5Fclose(h5_file)
        }
        if (exists("tmp1_path")) {
            invisible(file.remove(tmp1_path))
        }
        if (exists("tmp2_path")) {
            invisible(file.remove(tmp2_path))
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
    "loggingUtils.R", "annotatedHeatmap.R", "dictToList.R", "filterDataFrame.R",
    "getChromosomeList.R", "readChromosomeData.R", "scaleZ.R", "scaleZeroOne.R"
)
for (util_file in utils_files) {
    util_path <- file.path(utils_dir, util_file)
    source(util_path)
}

# Parse command line arguments
options <- list(
    list(flags = c("-i", "--input"), type = "character"),
    list(flags = c("-m", "--metadata"), type = "character"),
    list(flags = c("-o", "--output"), type = "character", default = NULL),
    list(flags = c("-g", "--group"), type = "character", default = "group"),
    list(flags = c("-r", "--filters"), type = "character", default = NULL),
    list(flags = c("-t", "--cut_value"), type = "numeric", default = 500),
    list(flags = c("-s", "--single_sample_cluster"), type = "logical", default = TRUE),
    list(flags = c("-e", "--metadata_output"), type = "character", default = NULL),
    list(flags = c("-z", "--z_score"), type = "logical", default = TRUE),
    list(
        flags = c("-d", "--methods"),
        type = "character",
        default = "{distance: 'euclidean', clustering: 'complete'}"
    ),
    list(
        flags = c("-p", "--plot_params"),
        type = "character",
        default = "{'mirror': 'TRUE', 'cex': '2', 'clusterColors': ['#ff6969', '#8088ff'], 'xlab': 'x', 'ylab': 'y'}"
    ),
    list(flags = c("-f", "--file"), type = "logical", default = TRUE)
)

if (!interactive()) {
    source(file.path(script_dir, "utils/initializeScript.R"))
    opt <- initializeScript(option_list = options, script_name = "plotHeatmap")
    plotHeatmap(
        input = opt$input,
        metadata = opt$metadata,
        group = opt$group,
        output = opt$output,
        filter = opt$filters,
        cut_value = opt$cut_value,
        single_sample_cluster = opt$single_sample_cluster,
        metadata_output = opt$metadata_output,
        z_score = opt$z_score,
        methods = opt$methods,
        plot_params = opt$plot_params,
        file = opt$file
    )
}
