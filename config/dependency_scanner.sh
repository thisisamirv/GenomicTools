#!/bin/bash

###########################################
# Package Dependencies Scanner
# - Scans Python files for imports
# - Filters out standard libraries
# - Creates a nicely formatted report
###########################################

# --- Configuration ---
OUTPUT_FILE="config/requirements.txt"

# --- Create working directories ---
mkdir -p config/temp

# --- Helper functions ---
function cleanup() {
    echo "Cleaning up temporary files..."
    rm -rf config/temp
}

function scan_python_imports() {
    echo "Scanning Python imports..."
    # Find import statements in Python files
    find src/ -type f -name "*.py" -print0 | xargs -0 grep -l "^\s*\(from\|import\)\s" | while read -r file; do 
        grep -E "^\s*(import|from)\s+" "$file" >> config/temp/raw_python_imports.txt
    done

    # Clean and filter imports
    sed 's/^[ \t]*//' config/temp/raw_python_imports.txt | grep -v "^from utils" | sort -u > config/temp/clean_python_imports.txt

    # Extract base module names
    (
        awk '/^from/ {print $2}' config/temp/clean_python_imports.txt | awk -F"." '{print $1}';
        awk '/^import/ {print $2}' config/temp/clean_python_imports.txt | awk -F"[.,]" '{print $1}';
    ) | sort -u > config/temp/python_modules.txt

    # Filter standard Python packages - FIX THE INDENTATION HERE
    python3 -c "
import sys, os, pkgutil
std_libs = set(sys.builtin_module_names)
std_libs.update(m.name for m in pkgutil.iter_modules([os.path.dirname(os.__file__)]) if not m.name.startswith('_'))
std_libs.update(['math'])
with open('config/temp/python_modules.txt') as f:
    modules = [l.strip() for l in f if l.strip()]
with open('config/temp/std_filtered_modules.txt', 'w') as f:
    f.write('\n'.join(m for m in modules if m not in std_libs))
"
}

function generate_report() {
    echo "Generating report..."
    
    # Create header with metadata
    cat > "$OUTPUT_FILE" << EOF
[python]
EOF

    # Add Python modules
    cat config/temp/std_filtered_modules.txt >> "$OUTPUT_FILE"

    # Replace common package names with their pip equivalents
    sed -i 's/^sklearn$/scikit-learn/g' "$OUTPUT_FILE"
    sed -i 's/^PIL$/Pillow/g' "$OUTPUT_FILE"
    sed -i 's/^cv2$/opencv/g' "$OUTPUT_FILE"
    
    # Sort the final output
    sort -u "$OUTPUT_FILE" -o "$OUTPUT_FILE"
    
    # Remove [python] tag if it exists
    sed -i '/^\[python\]$/d' "$OUTPUT_FILE"
    
    # Add [python] tag to the top
    sed -i '1s/^/\[python\]\n/' "$OUTPUT_FILE"
    
    # Add [shell] tag
    echo "" >> "$OUTPUT_FILE"
    echo "[shell]" >> "$OUTPUT_FILE"
    
    # Add IMPUTE2
    echo "IMPUTE2" >> "$OUTPUT_FILE"
}

# --- Main script execution ---
echo "Starting dependency scan..."

# Remove old files
rm -f "$OUTPUT_FILE"
rm -f config/temp/*

# Scan for dependencies
scan_python_imports

# Generate the final report
generate_report

# Clean up
cleanup

echo "Dependency scan complete!"
