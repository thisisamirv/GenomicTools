#!/usr/bin/env Rscript
# Helper function to run GSEA analysis
# Import the necessary libraries
suppressPackageStartupMessages({
    library(data.table)
    library(clusterProfiler)
    library(org.Hs.eg.db)
})

# Main function
enrichGSEA <- function(data, log2fc, database = "KEGG", ontology = NULL) {
    tryCatch({
        # Merge with data by probe
        log_debug("Merging log2fc data with gene data")
        data <- merge.data.table(data, log2fc, by = "probe", all.x = TRUE)
        
        # In cases of duplicate entz_id, keep the one with the lowest distance_to_tss
        log_debug("Removing duplicate ENTREZ IDs")
        setorder(data, distance_to_tss)
        data <- unique(data, by = "entz_id")
        
        # Prepare ranked gene list
        log_debug("Preparing ranked gene list")
        ranked_genes <- data$log2fc
        names(ranked_genes) <- data$entz_id
        ranked_genes <- sort(ranked_genes, decreasing = TRUE)
        
        # Add small noise to break ties
        log_debug("Adding noise to break ties")
        set.seed(123)
        ranked_genes <- ranked_genes + rnorm(length(ranked_genes), sd = 0.0001)
        ranked_genes <- sort(ranked_genes, decreasing = TRUE)
        
        if (database == "KEGG") {
            # Run GSEA-KEGG analysis
            log_info("Running GSEA-KEGG analysis")
            gsea_result <- gseKEGG(
                geneList = ranked_genes,
                organism = "hsa",
                minGSSize = 10,
                maxGSSize = 500,
                pvalueCutoff = 1
            )
        } else if (database == "GO") {
            # Run GSEA-GO analysis
            log_debug("Running GSEA-GO analysis")
            gsea_result <- gseGO(
                geneList = ranked_genes,
                ont = ontology,
                OrgDb = org.Hs.eg.db,
                keyType = "ENTREZID",
                minGSSize = 10,
                maxGSSize = 500,
                pvalueCutoff = 1
            )
        }
        
        # Extract NES scores
        log_debug("Extracting NES scores")
        nes_scores <- data.table(
            ID = gsea_result@result$ID,
            Description = gsea_result@result$Description,
            NES = gsea_result@result$NES,
            pvalue = gsea_result@result$pvalue,
            p.adjust = gsea_result@result$p.adjust
        )
        
        # Sort by absolute NES value first
        log_debug("Sorting results by NES and p-value")
        setorder(nes_scores, -abs(NES))
        
        # Truncate Description if longer than 7 words
        log_debug("Truncating Description if longer than 7 words")
        nes_scores[, Description := sapply(strsplit(Description, " "), function(x) {
            if (length(x) > 7) paste(x[1:7], collapse = " ") else paste(x, collapse = " ")
        })]
        
        # Then sort by p-value
        log_debug("Sorting results by NES and p-value")
        setorder(nes_scores, pvalue)
        
        return(list(gsea_result = gsea_result, nes_scores = nes_scores))
    
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
