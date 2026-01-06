#!/usr/bin/env Rscript
# Import required modules
suppressPackageStartupMessages({
    library(futile.logger)
})

COLORS <- list(
    RESET = "\033[0m",
    BOLD = "\033[1m",
    RED = "\033[1;31m",
    GREEN = "\033[1;32m",
    YELLOW = "\033[1;33m",
    BLUE = "\033[1;34m",
    MAGENTA = "\033[1;35m",
    CYAN = "\033[1;36m",
    WHITE = "\033[1;37m",
    BRIGHT_CYAN = "\033[1;96m",
    BRIGHT_MAGENTA = "\033[1;95m",
    ORANGE = "\033[1;38;5;208m"
)

colorize <- function(text, color) {
    paste0(color, text, COLORS$RESET)
}

initialize_log_file <- function(path) {
    if (is.null(path) || !nzchar(path)) {
        return(character())
    }
    path <- path.expand(path)
    dir_path <- dirname(path)
    if (!dir.exists(dir_path)) {
        dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
    }
    if (!file.exists(path)) {
        file.create(path, showWarnings = FALSE)
    }
    normalizePath(path, winslash = "/", mustWork = FALSE)
}

get_log_file_paths <- function() {
    paths <- unique(c(
        Sys.getenv("GT_LOG_FILE", ""),
        Sys.getenv("GT_LAST_LOG", "")
    ))
    paths <- paths[nzchar(paths)]
    if (!length(paths)) {
        return(character())
    }
    initialized <- vapply(paths, initialize_log_file, character(1), USE.NAMES = FALSE)
    initialized[nzchar(initialized)]
}

store_log_files <- function(log_files) {
    assign("GT_LOG_FILES", log_files, envir = .GlobalEnv)
}

get_stored_log_files <- function() {
    files <- try(get("GT_LOG_FILES", envir = .GlobalEnv), silent = TRUE)
    if (inherits(files, "try-error")) {
        character()
    } else {
        files
    }
}

create_console_appender <- function() {
    function(line) {
        tryCatch({
            cat(line, file = stderr())
        }, error = function(e) {
            cat(line, "\n")
        })
        invisible(TRUE)
    }
}

resolve_level_name <- function(level) {
    level_map <- c(
        `1` = "FATAL",
        `2` = "ERROR",
        `3` = "ERROR",
        `4` = "WARN",
        `5` = "WARN",
        `6` = "INFO",
        `7` = "DEBUG",
        `8` = "TRACE"
    )
    level_key <- as.character(level)
    resolved <- level_map[[level_key]]
    if (is.null(resolved)) {
        toupper(level_key)
    } else {
        resolved
    }
}

format_log_args <- function(message, ...) {
    args <- list(...)
    if (length(args) == 0) {
        return(message)
    }
    tryCatch(
        do.call(sprintf, c(list(message), args)),
        warning = function(w) do.call(paste, c(list(message), args)),
        error = function(e) do.call(paste, c(list(message), args))
    )
}

normalize_log_level <- function(level) {
    if (is.null(level) || !nzchar(level)) {
        return("INFO")
    }
    upper_level <- toupper(trimws(level))
    mapping <- c(
        "CRITICAL" = "FATAL",
        "FATAL" = "FATAL",
        "ERROR" = "ERROR",
        "ERR" = "ERROR",
        "WARNING" = "WARN",
        "WARN" = "WARN",
        "INFO" = "INFO",
        "DEBUG" = "DEBUG",
        "TRACE" = "TRACE",
        "ALL" = "ALL"
    )
    mapped <- mapping[[upper_level]]
    if (is.null(mapped)) "INFO" else mapped
}

format_log_line <- function(level_name, message, colored = TRUE) {
    level_lower <- tolower(level_name)
    level_color <- switch(
        level_lower,
        debug = COLORS$CYAN,
        info = COLORS$WHITE,
        warn = COLORS$YELLOW,
        error = COLORS$RED,
        fatal = COLORS$RED,
        COLORS$WHITE
    )
    message_color <- switch(
        level_lower,
        debug = COLORS$CYAN,
        info = COLORS$WHITE,
        warn = COLORS$YELLOW,
        error = COLORS$RED,
        fatal = COLORS$RED,
        COLORS$WHITE
    )
    level_text <- sprintf("%-8s", level_name)
    if (colored) {
        level_text <- colorize(level_text, level_color)
        message <- colorize(message, message_color)
    }
    paste0(level_text, " ", message, "\n")
}

append_log_to_files <- function(level_name, message) {
    log_files <- get_stored_log_files()
    if (!length(log_files)) {
        log_files <- get_log_file_paths()
        store_log_files(log_files)
    }
    if (!length(log_files)) {
        return(invisible(TRUE))
    }
    line <- format_log_line(level_name, message, colored = TRUE)
    for (file in log_files) {
        cat(line, file = file, append = TRUE)
    }
    invisible(TRUE)
}

setup_logger <- function(log_level = "INFO") {
    normalized_level <- normalize_log_level(log_level)
    futile.logger::flog.threshold(normalized_level)

    log_files <- get_log_file_paths()
    store_log_files(log_files)
    appender <- create_console_appender()
    futile.logger::flog.appender(appender)

    futile.logger::flog.layout(function(level, msg, ...) {
        level_name <- resolve_level_name(level)
        msg_text <- paste(unlist(msg), collapse = "")
        format_log_line(level_name, msg_text, colored = TRUE)
    })
}

log_info <- function(...) {
    msg <- format_log_args(...)
    append_log_to_files("INFO", msg)
    futile.logger::flog.info(msg)
}

log_debug <- function(...) {
    msg <- format_log_args(...)
    append_log_to_files("DEBUG", msg)
    futile.logger::flog.debug(msg)
}

log_warn <- function(...) {
    msg <- format_log_args(...)
    append_log_to_files("WARN", msg)
    futile.logger::flog.warn(msg)
}

log_error <- function(...) {
    msg <- format_log_args(...)
    append_log_to_files("ERROR", msg)
    futile.logger::flog.error(msg)
}

log_success <- function(...) {
    msg <- format_log_args(...)
    prefix <- colorize("✓ SUCCESS:", COLORS$GREEN)
    combined <- sprintf("%s %s", prefix, msg)
    append_log_to_files("INFO", combined)
    futile.logger::flog.info(combined)
}
