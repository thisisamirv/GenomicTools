#!/usr/bin/env Rscript
# Helper function to plot an enhanced annotated heatmap
#' @description An enhanced version of the heatmap3 function.
#'
#' This function is based on the heatmap3 function from the heatmap3 package.
#' The original heatmap3 package was written by Shilin Zhao, Linlin Yin, Yan Guo, Quanhu Sheng, and Yu Shyr.
#'
#' @details
#' This is a modified version of the `heatmap3` function from the heatmap3 package.
#' The original function was published under the GPL (>= 2) license. This function has been modified to better integrate with the rest of the
#' package codes and suit specific purposes related to the GenomicTools package. 
#'
#' @references
#' Zhao, Shilin, et al. "Heatmap3: an improved heatmap package with more powerful and convenient features." BMC bioinformatics 15 (2014): 1-2.
#' URL: https://github.com/slzhao/heatmap3
#' 
#' @license GPL (>= 2)

# Import the necessary libraries
suppressPackageStartupMessages({
    library(corrplot)
    library(graphics)
    library(grDevices)
    library(RColorBrewer)
    library(stats)
})

# Main function
annotatedHeatmap <- function(
    input, 
    distance_method,
    clustering_method,
    ColSideAnn,
    ColSideWidth = 0.4,
    ColSideCut,
    xlab = NULL, 
    ylab = NULL,
    margins = c(5, 5),
    clusterColors = NULL,
    mirror = FALSE,
    lab_size = 1
) {
    # Print citation message
    cat("=============================================\n")
    cat("This function is based on the heatmap3 function from the heatmap3 package.\n")
    cat("The original function was published under the GPL (>= 2) license.\n")
    cat("This function has been modified to better integrate with the rest of the package codes and suit specific purposes related to the GenomicTools package.\n")
    cat("If you use this function, please consider citing the original authors of the heatmap3 package.\n")
    cat("Reference: Zhao, Shilin, et al. 'Heatmap3: an improved heatmap package with more powerful and convenient features.' BMC bioinformatics 15 (2014): 1-2.\n")
    cat("=============================================\n")
    
    # Initialize dimensions and means
    log_debug("Initializing dimensions and means")
    numRows <- dim(input)[1]
    numCols <- dim(input)[2]
    log_debug("Computing row means")
    rowMeans <- rowMeans(input, na.rm = TRUE)
    log_debug("Computing column means")
    colMeans <- colMeans(input, na.rm = TRUE)
    
    # Compute row and column clustering
    log_debug("Computing row clustering")
    row_clustering <- compute_clustering(input, distance_method, clustering_method, rowMeans)
    log_debug("Computing column clustering")
    col_clustering <- compute_clustering(t(input), distance_method, clustering_method, colMeans)
    
    # Extract results
    log_debug("Extracting clustering results")
    rowDendrogram <- row_clustering$dendrogram
    rowIndices <- row_clustering$indices
    rowHierarchicalClust <- row_clustering$hclust
    
    colDendrogram <- col_clustering$dendrogram
    colIndices <- col_clustering$indices
    colHierarchicalClust <- col_clustering$hclust
    
    # Calculate heatmap breaks
    log_debug("Calculating heatmap breaks")
    breaks <- calculate_heatmap_breaks(input)
    
    # Generate color palette
    log_debug("Generating color palette")
    col <- generate_color_palette(breaks)
    
    # Apply normalization
    log_debug("Applying data normalization")
    input <- normalize_heatmap_data(input, rowIndices, colIndices)

    # Layout setup
    log_debug("Setting up plot layout")
    layoutMatrix <- rbind(c(NA, 3, NA), c(2, 1, NA), c(NA, NA, NA))
    layoutHeights <- c(0.5, 4)
    layoutMatrix <- rbind(layoutMatrix[1, ] + 1, c(NA, 1, NA), layoutMatrix[2, ] + 1)
    layoutHeights <- c(layoutHeights[1], ColSideWidth, layoutHeights[2])
    layoutMatrix <- layoutMatrix + 1
    layoutMatrix[is.na(layoutMatrix)] <- 0
    layoutMatrix[1, 1] <- 1

    # Graphics setup
    log_debug("Setting up graphics parameters")
    dev.hold()
    on.exit(dev.flush())
    originalPar <- par(no.readonly = TRUE)
    on.exit(par(originalPar), add = TRUE)
    
    # Create layout and plot components
    log_debug("Creating plot layout")
    graphics::layout(layoutMatrix, widths = c(0.5, 4, 0.7), heights = layoutHeights, respect = TRUE)
    
    # Plot annotation panel
    log_debug("Plotting annotation panel")
    par(mar = c(0, 0, 0, 0), xpd = NA)
    plot.new()
    
    # Process column side cuts
    log_debug("Processing column side cuts")
    colClusters <- cut(colDendrogram, ColSideCut)$lower
    colClusterIndices <- lapply(colClusters, order.dendrogram)
    if (is.null(clusterColors)) {
        colClusterColors <- rainbow(length(colClusters), alpha = 0.4)
    } else {
        # Make cluster colors transparent
        log_debug("Making cluster colors transparent")
        colClusterColors <- adjustcolor(clusterColors, alpha.f = 0.4)
    }
    
    # Plot column annotations
    log_debug("Plotting column annotations")
    columnCount <- (numCols - 1)
    par(mar = c(0.5, 0, 0, margins[2]))
    colAnnotations <- ColSideAnn[colIndices, , drop = F]
    annotationHeight <- showAnn(colAnnotations, mirror = mirror)
    annotationYLimits <- par("usr")[3:4]
    
    # Draw column side rectangles
    log_debug("Drawing column side rectangles")
    draw_column_side_rectangles(colClusters, columnCount, annotationYLimits, colClusterColors, mirror = mirror)
    
    # Reverse the plot on the y-axis if needed
    if (mirror) {
        log_debug("Reversing plot on y-axis")
        # Reverse column indices and data
        colIndices <- rev(colIndices)
        input <- input[, ncol(input):1]
        
        # Adjust column dendrogram and annotations
        colDendrogram <- rev(colDendrogram)
        if (!is.null(ColSideAnn)) {
            ColSideAnn <- ColSideAnn[ncol(input):1, , drop=FALSE]
        }
        
        # Reverse column clusters
        colClusters <- rev(colClusters)
        
        # Reverse column labels if they exist
        if (!is.null(xlab)) {
            xlab <- rev(xlab)
        }
    }
    
    # Plot heatmap
    log_debug("Drawing main heatmap")
    par(mar = c(margins[1], 0, 0, margins[2]))
    image(
        x = 1:numCols,
        y = 1:numRows,
        z = t(input),
        xlim = 0.5 + c(0, numCols),
        ylim = 0.5 + c(0, numRows),
        axes = FALSE,
        xlab = "",
        ylab = "",
        col = col,
        breaks = breaks
    )
    
    # Add axes and labels
    log_debug("Adding axes and labels")
    axis(1, 1:numCols, labels = FALSE, las = 2, line = -0.5, tick = 0, cex.axis = 0.2 + 1 / log10(numCols))
    mtext(xlab, side = 1, line = margins[1] - (0.25 * lab_size), cex = lab_size)
    axis(4, 1:numRows, labels = FALSE, las = 2, line = -0.5, tick = 0, cex.axis = 0.2 + 1 / log10(numRows))
    mtext(ylab, side = 4, line = margins[2] - (0.25 * lab_size), cex = lab_size)

    # Plot dendrograms
    log_debug("Plotting dendrograms")
    par(mar = c(margins[1], 0, 0, 0))
    plot(rowDendrogram, horiz = TRUE, axes = FALSE, yaxs = "i", leaflab = "none")
    par(mar = c(margins[1], 0, 1, margins[2]))
    plot(colDendrogram, horiz = FALSE, axes = FALSE, xaxs = "i", leaflab = "none")
    
    # Draw cluster rectangles
    log_debug("Drawing cluster rectangles")
    draw_cluster_rectangles(colClusters, colClusterColors, ColSideCut)
    
    # Clear layout and draw legend
    layout(1)
    draw_legend()
    
    # Return results
    log_debug("Returning results")
    invisible(list(
        rowInd = rowIndices,
        colInd = colIndices,
        Colv = colDendrogram,
        cutColoumIndList = colClusterIndices,
        hcr = rowHierarchicalClust,
        hcc = colHierarchicalClust
    ))
}

# Helprt function for data normalization
normalize_heatmap_data <- function(data, row_indices, col_indices) {
    # Reorder data according to indices
    log_debug("Reordering data according to indices")
    data <- data[row_indices, col_indices]
    
    # Center the data by subtracting row means
    log_debug("Centering the data by subtracting row means")
    data <- sweep(data, 1, rowMeans(data, na.rm = TRUE), check.margin = FALSE)
    
    # Scale the data by dividing by row standard deviations
    log_debug("Scaling the data by dividing by row standard deviations")
    data <- sweep(data, 1, apply(data, 1, sd, na.rm = TRUE), "/", check.margin = FALSE)
    
    return(data)
}

# Helper function for computing clustering
compute_clustering <- function(data, distance_method, clustering_method, means) {
    log_debug("Computing distance matrix using method:", distance_method)
    dist_matrix <- dist(data, method = distance_method)
    
    log_debug("Performing hierarchical clustering using method:", clustering_method)
    hier_clust <- hclust(dist_matrix, method = clustering_method)
    
    log_debug("Converting hierarchical clustering to dendrogram")
    dendro <- as.dendrogram(hier_clust)
    
    log_debug("Reordering dendrogram based on means")
    dendro <- reorder(dendro, means)
    
    log_debug("Getting indices from dendrogram")
    indices <- order.dendrogram(dendro)
    
    list(dendrogram = dendro, indices = indices, hclust = hier_clust)
}

# Helper function for calculating heatmap breaks
calculate_heatmap_breaks <- function(input, br_range1=5, br_range2=0.5, br_range3=0.01) {
    log_debug("Calculating data range")
    d_range <- range(as.numeric(input), na.rm = TRUE)
    
    log_debug("Creating break sequence for range:", d_range[1], "to", d_range[2])
    breaks <- unique(c(
        seq(d_range[1], -br_range1, length = 5),
        seq(-br_range1, -br_range2, length = 10),
        seq(-br_range2, -br_range3, length = 20),
        0,
        seq(br_range3, br_range2,  length = 20),
        seq(br_range2, br_range1,  length = 10),
        seq(br_range1, d_range[2], length = 5)
    ))
    
    log_debug("Sorting and rounding break values")
    sort(signif(breaks, 2))
}

# Helper function for generating color palette
generate_color_palette <- function(breaks, c_length = as.integer((length(breaks) - 1) / 2)) {
    log_debug("Generating blue color palette")
    col1 <- colorRampPalette(rev(brewer.pal(9, "Blues")))(c_length)
    
    log_debug("Generating red color palette")
    col2 <- colorRampPalette(brewer.pal(9, "Reds"))(c_length)
    
    log_debug("Combining and returning unique colors")
    unique(c(col1, col2))
}

# Helper function to draw column side rectangles
draw_column_side_rectangles <- function(colClusters, columnCount, annotationYLimits, colClusterColors, mirror = FALSE) {
    log_debug("Calculating cluster lengths")
    cluster_lengths <- sapply(colClusters, function(x) length(unlist(x)))
    
    log_debug("Computing cumulative lengths of clusters")
    cumulative_lengths <- cumsum(cluster_lengths)
    
    # Calculate y-coordinates for narrow rectangles
    narrow_height <- (annotationYLimits[2] - annotationYLimits[1]) * 0.25
    space_between <- (annotationYLimits[2] - annotationYLimits[1]) * 0.02
    narrow_y_bottom <- annotationYLimits[2] + space_between
    narrow_y_top <- narrow_y_bottom + narrow_height
    
    # Calculate text positions
    text_y <- (narrow_y_bottom + narrow_y_top) / 2
    cluster_numbers <- 1:length(colClusters)
    
    if (mirror) {
        # Draw original rectangles
        rect(
            1 + 1/columnCount/2 - 1/columnCount * cumulative_lengths,
            annotationYLimits[1],
            c(1 + 1/columnCount/2, 1 + 1/columnCount/2 - 1/columnCount * cumulative_lengths[-length(colClusters)]),
            annotationYLimits[2],
            col = rev(colClusterColors)
        )
        # Draw narrow rectangles
        rect(
            1 + 1/columnCount/2 - 1/columnCount * cumulative_lengths,
            narrow_y_bottom,
            c(1 + 1/columnCount/2, 1 + 1/columnCount/2 - 1/columnCount * cumulative_lengths[-length(colClusters)]),
            narrow_y_top,
            col = rev(colClusterColors)
        )
        # Add text labels
        text_x <- 1 + 1/columnCount/2 - 1/columnCount * (cumulative_lengths - cluster_lengths/2)
        text(text_x, text_y, paste("Cluster", rev(cluster_numbers)), cex = 2)
    } else {
        # Draw original rectangles
        rect(
            c(0 - 1/columnCount/2, (0 - 1/columnCount/2) + 1/columnCount * cumulative_lengths[-length(colClusters)]),
            annotationYLimits[1],
            (0 - 1/columnCount/2) + 1/columnCount * cumulative_lengths,
            annotationYLimits[2],
            col = colClusterColors
        )
        # Draw narrow rectangles
        rect(
            c(0 - 1/columnCount/2, (0 - 1/columnCount/2) + 1/columnCount * cumulative_lengths[-length(colClusters)]),
            narrow_y_bottom,
            (0 - 1/columnCount/2) + 1/columnCount * cumulative_lengths,
            narrow_y_top,
            col = colClusterColors
        )
        # Add text labels
        text_x <- (0 - 1/columnCount/2) + 1/columnCount * (cumulative_lengths - cluster_lengths/2)
        text(text_x, text_y, paste("Cluster", cluster_numbers), cex = 2)
    }
}

# Helper function to draw cluster rectangles
draw_cluster_rectangles <- function(colClusters, colClusterColors, ColSideCut) {
    log_debug("Calculating cluster lengths")
    cluster_lengths <- sapply(colClusters, function(x) length(unlist(x)))
    
    log_debug("Computing cumulative lengths of clusters")
    cumulative_lengths <- cumsum(cluster_lengths)
    
    log_debug("Drawing cluster rectangles")
    y_offset <- 0.5
    rect(
        c(0.5, 0.5 + cumulative_lengths[-length(colClusters)]),
        y_offset,
        cumulative_lengths + 0.5,
        ColSideCut + y_offset,
        col = colClusterColors
    )
}

# Helper function to show annotations
showAnn <- function(annData, mirror = FALSE) {
    log_debug("Calculating total number of lines needed for annotations")
    # Calculate total number of lines needed for annotations
    factor_columns <- which(sapply(annData, class) == "factor")
    all_levels <- sapply(annData[, factor_columns, drop=F], levels)
    total_levels <- length(unlist(all_levels))
    total_lines <- (ncol(annData) - length(factor_columns)) * 2 + total_levels
    
    log_debug("Setting up plot area with", total_lines, "lines")
    # Set up the plot area
    plot(
        x = c(0 - 1/nrow(annData)/2, 1 + 1/nrow(annData)/2),
        y = c(1, total_lines + 1),
        type = "n",
        xaxt = "n",
        yaxt = "n",
        xlab = "",
        ylab = "",
        bty = "n",
        axes = FALSE,
        xaxs = "i"
    )
    
    # Process each column in annotation data
    offset <- 1
    for (col_index in 1:ncol(annData)) {
        # Handle factor columns
        log_debug("Converting factors to matrix representation")
        factor_matrix <- factor2Matrix(annData[, col_index], colnames(annData)[col_index])
        
        if (mirror) {
            log_debug("Mirroring factor matrix")
            factor_matrix <- factor_matrix[nrow(factor_matrix):1,]
        }
        
        # Plot factor levels
        log_debug("Plotting factor levels")
        image(
            x = seq(0, 1, length.out = nrow(annData)),
            y = (1:ncol(factor_matrix)) + offset - 0.5,
            z = factor_matrix,
            col = c("white", "black"),
            add = TRUE
        )
        
        # Add grid lines
        log_debug("Drawing grid lines")
        segments(
            x0 = seq(0 - 1 / nrow(annData) / 2, 1 + 1 / nrow(annData) / 2, length.out = nrow(annData) + 1),
            y0 = offset,
            x1 = seq(0 - 1 / nrow(annData) / 2, 1 + 1 / nrow(annData) / 2, length.out = nrow(annData) + 1),
            y1 = ncol(factor_matrix) + offset,
            col = "white"
        )
        abline(h = c(0, ncol(factor_matrix)) + offset, col = "white")
        
        # Add labels (levels only)
        log_debug("Adding level labels")
        mtext(
            text = sub("^[^=]+=", "", colnames(factor_matrix)), 
            side = 2,
            line = 0.5,
            at = (1:ncol(factor_matrix)) + offset - 0.5, 
            las = 1,
            cex = 1.4
        )
        offset <- offset + ncol(factor_matrix)
    }
    
    log_debug("Returning plotting limits")
    return(c(0.6, offset))
}

# Helper function to convert factor data to matrix
factor2Matrix <- function(factorData, colName) {
    log_debug("Creating factor matrix for column:", colName)
    
    # Create an empty matrix with dimensions: number of data points x number of factor levels
    log_debug("Initializing empty matrix with dimensions:", length(factorData), "x", length(levels(factorData)))
    factorMatrix <- matrix(0, nrow = length(factorData), ncol = length(levels(factorData)))
    
    # Create index pairs for the non-zero elements
    log_debug("Creating index pairs for non-zero elements")
    temp <- cbind(1:nrow(factorMatrix), as.numeric(factorData))
    
    # Set corresponding elements to 1
    log_debug("Setting matrix values based on factor levels")
    factorMatrix[temp] <- 1
    
    # Name the columns with format "colName=level"
    log_debug("Setting column names with format 'colName=level'")
    colnames(factorMatrix) <- paste(colName, levels(factorData), sep = "=")
    
    log_debug("Returning factor matrix")
    return(factorMatrix)
}

# Helper function to draw the legend
draw_legend <- function(zlim = c(0, 1)) {
    par(new = TRUE, fig = c(0.9, 1.0, 0.2, 0.8))
    
    # Create and add legend
    log_debug("Adding color legend to the plot")
    custom_palette <- colorRampPalette(c("blue", "white", "red"))(100)
    plot.new()
    colorlegend(
        colbar = custom_palette,
        labels = seq(min(zlim[1], na.rm = TRUE), max(zlim[2], na.rm = TRUE), 0.25),
        vertical = TRUE,
        xlim = c(0, 1),
        ylim = c(0.3, 0.8)
    )
    
    # Add legend title
    text(x = 0.37, y = 0.85, labels = "Z score", cex = 1, font = 2)
}

# Source helper functions
if (!interactive()) {
    args <- commandArgs(trailingOnly = FALSE)
    script_dir <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
    source(file.path(script_dir, "utils/loggingUtils.R"))
} else {
    script_dir <- dirname(normalizePath(parent.frame(2)$ofile))
    source(file.path(script_dir, "loggingUtils.R"))
}
