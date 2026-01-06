#!/bin/bash
set -euo pipefail

# Determine the script directory
SCRIPT_PATH="$(readlink -f "$0")"
SRC_DIR="$(dirname "$SCRIPT_PATH")/../src"
SRC_DIR="$(cd "$SRC_DIR" && pwd)"

# Get the name of the script being called
SCRIPT_NAME=$(basename "$0")

# Remove "GT-" prefix
SCRIPT_NAME=${SCRIPT_NAME#GT-}

# Function to display an error message and exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Set the appropriate interpreter for the script
if [[ -f "$SRC_DIR/$SCRIPT_NAME.py" ]]; then
    INTERPRETER="python3"
    SCRIPT_PATH_FULL="$SRC_DIR/$SCRIPT_NAME.py"
    SCRIPT_EXT=".py"
else
    echo "Unknown script name: $SCRIPT_NAME" >&2
    exit 1
fi

# Function to display help
display_help() {
    $INTERPRETER "$SCRIPT_PATH_FULL" --help
}

# Check for help flag
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    display_help
    exit 0
fi

# Execute the appropriate script
$INTERPRETER "$SCRIPT_PATH_FULL" "$@" || error_exit "$INTERPRETER script '$SCRIPT_NAME$SCRIPT_EXT' failed."
