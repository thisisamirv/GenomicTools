#!/usr/bin/env Rscript
# Helper function to create a gene-concept network from gene set enrichment results
# Import the necessary libraries
suppressPackageStartupMessages({
    library(data.table)
    library(grDevices)
    library(igraph)
    library(png)
})

# Main function
enrichGeneConceptNet <- function(enrich_results, pathways) {
    tryCatch({
        # Select rows matching pathways
        log_debug("Selecting rows matching pathways")
        selectedPaths <- enrich_results[enrich_results$Description %in% pathways, ]
        
        # Initialize an empty data frame to store edges
        log_debug("Initializing an empty data frame to store edges")
        edgeList <- data.frame(from = character(), to = character(), stringsAsFactors = FALSE)
        
        # Loop over each selected pathway to extract edges
        log_debug("Looping over each selected pathway to extract edges")
        for (i in 1:nrow(selectedPaths)) {
            # Extract the pathway name
            pathway <- selectedPaths$Description[i]
            
            # Split the gene IDs into a list
            genes <- unlist(strsplit(selectedPaths$geneID[i], "/"))
            
            # Create a temporary edge list for the current pathway
            tmpEdges <- data.frame(
                from = rep(pathway, length(genes)),
                to = genes,
                stringsAsFactors = FALSE
            )
            
            # Append the temporary edge list to the main edge list
            edgeList <- rbind(edgeList, tmpEdges)
        }
        
        # Create a graph from the edge list
        log_debug("Creating a graph from the edge list")
        graph <- graph_from_data_frame(edgeList, directed = FALSE)
        
        # Set graph attributes
        log_debug("Setting graph attributes")
        V(graph)$type <- V(graph)$name %in% unique(edgeList$from)
        uniquePaths <- unique(edgeList$from)
        pathwayColors <- setNames(rainbow(length(uniquePaths)), uniquePaths)
        V(graph)$color <- ifelse(V(graph)$name %in% uniquePaths, pathwayColors[V(graph)$name], "orange")
        E(graph)$color <- pathwayColors[edgeList$from]
        V(graph)$size <- ifelse(V(graph)$name %in% uniquePaths, 10, 4)
        E(graph)$curved <- 0.2
        E(graph)$width <- 2
        V(graph)$label <- ifelse(V(graph)$name %in% uniquePaths, "", V(graph)$name)
        
        # Create an off-screen device to capture the plot
        tmp <- tempfile(fileext = ".png")
        png(filename = tmp, width = 1000, height = 1000)
        
        # Increase margin to allow labels outside plot area (right margin increased)
        old_mar <- par("mar")
        par(mar = c(5, 5, 5.5, 8))
        
        # Set the layout parameters
        log_debug("Setting the layout parameters")
        layout_circular <- igraph::layout_in_circle(graph)
        angles <- atan2(layout_circular[, 2], layout_circular[, 1])
        angles_deg <- (angles * 180 / pi) %% 360

        custom_label_deg <- sapply(angles_deg, function(theta) {
            if (theta >= 0 && theta < 90) {
                # Nodes on the right side: rotation decreases from 0 to -45 degrees
                - (theta / 2)
            } else if (theta >= 90 && theta < 180) {
                # Nodes on the top: rotation decreases from 45 to 0 degrees
                45 - (theta - 90) / 2
            } else if (theta >= 180 && theta < 270) {
                # Nodes on the left: rotation decreases from 0 to -45 degrees
                - (theta - 180) / 2
            } else {
                # Nodes on the bottom: rotation decreases from 45 to 0 degrees
                45 - (theta - 270) / 2
            }
        })

        # Plot the graph without vertex labels
        log_debug("Plotting the graph without vertex labels")
        igraph::plot.igraph(
            graph,
            layout = layout_circular,
            rescale = FALSE,
            vertex.size = V(graph)$size,
            vertex.label = NA,
            edge.color = E(graph)$color,
            edge.curved = E(graph)$curved,
            edge.width = E(graph)$width,
            main = ""
        )

        # Add a legend to the plot
        legend(
            "topright",
            legend = uniquePaths,
            col = pathwayColors,
            pch = 20,
            cex = 1.3,
            text.font = 2,
            title = "Pathways",
            inset = c(-0.10, -0.10),
            xpd = TRUE,
            bty = "n"
        )

        # Reset the plotting parameters
        par(mar = old_mar)
        
        # Add vertex labels manually with alignment adjusted based on node position
        vertex_labels <- ifelse(V(graph)$name %in% uniquePaths, "", V(graph)$name)
        nonEmpty <- vertex_labels != ""
        xy <- layout_circular[nonEmpty, , drop = FALSE]
        lbl <- vertex_labels[nonEmpty]
        rot <- custom_label_deg[nonEmpty] * -1
        angles_nonEmpty <- angles[nonEmpty]

        offset_val <- 0.03

        if (length(lbl) > 0) {
            for (i in seq_along(lbl)) {
                new_x <- xy[i, 1] + offset_val * cos(angles_nonEmpty[i])
                new_y <- xy[i, 2] + offset_val * sin(angles_nonEmpty[i])
                
                # Adjust the horizontal alignment based on the node position
                horiz_adj <- ifelse(cos(angles_nonEmpty[i]) >= 0, 0, 1)
                text(
                    x = new_x,
                    y = new_y,
                    labels = lbl[i],
                    cex = 1.4,
                    col = "black",
                    srt = rot[i],
                    adj = c(horiz_adj, 0.5),
                    xpd = NA
                )
            }
        }
        # Capture the plot as a raster image
        dev.off()
        plot <- readPNG(tmp)
        file.remove(tmp)
        
        return(plot)
    
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
} else {
    script_dir <- dirname(normalizePath(parent.frame(2)$ofile))
    source(file.path(script_dir, "loggingUtils.R"))
}
