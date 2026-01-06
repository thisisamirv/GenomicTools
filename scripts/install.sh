#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--dev]"
    exit 1
}

DEV_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

install_dev_dependencies_into_GT() {
    local DEV_REQ_FILE="$PWD/config/dev_dependencies.txt"

    if [ "$DEV_MODE" != true ]; then
        return 0
    fi

    if [ ! -f "$DEV_REQ_FILE" ]; then
        echo "No development dependency file found at $DEV_REQ_FILE. Skipping dev installation."
        return 0
    fi

    echo "Installing development dependencies from $DEV_REQ_FILE into 'GT' using pip..."
    if ! mamba run -n GT pip install -r "$DEV_REQ_FILE"; then
        echo "Error: Failed to install development dependencies from $DEV_REQ_FILE."
        exit 1
    fi
    echo "Development dependencies installed."
}

install_requirements_into_GT() {
    local REQ_FILE="$PWD/requirements.txt"
    if [ -f "$REQ_FILE" ]; then
        echo "Installing packages from $REQ_FILE into 'GT' using mamba (channels: conda-forge, bioconda, r, defaults)..."
        if ! mamba install -n GT -y --file "$REQ_FILE" -c conda-forge -c bioconda -c r -c defaults; then
            echo "Warning: mamba failed to install some packages from $REQ_FILE."
            echo "Attempting pip fallback for Python packages inside the 'GT' environment..."
            if ! mamba run -n GT pip install -r "$REQ_FILE"; then
                echo "Error: pip fallback also failed to install packages from $REQ_FILE."
                exit 1
            fi
        fi
        echo "Package installation into 'GT' completed."
    else
        echo "No requirements.txt found at $PWD/requirements.txt. Skipping package installation."
    fi

    install_dev_dependencies_into_GT
}

# Check for mamba
echo "Checking for mamba..."
MAMBA_AVAILABLE=true
if ! command_exists "mamba"; then
    MAMBA_AVAILABLE=false
    echo "Error: 'mamba' is not installed or not available in your shell."
    echo "If your system uses environment modules, you may need to load Miniconda first, e.g.:"
    echo "  module load miniconda"
    echo "If you don't have mamba, install it from the following:"
    echo "  https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html"
    echo ""
    echo "WARNING: It is strongly advised to install mamba for proper dependency management."
    echo "Without mamba, the GT environment and package dependencies will not be set up."
    
    while true; do
        read -r -p "Do you want to continue without mamba? (yes/no): " answer
        answer="${answer,,}"  # normalize to lowercase
        case "$answer" in
            y|yes )
                echo "Continuing installation without mamba. Environment setup will be skipped."
                break
                ;;
            n|no )
                echo "Installation aborted. Please install mamba and try again."
                exit 1
                ;;
            * )
                echo "Please answer yes or no."
                ;;
        esac
    done
fi

# Only proceed with environment setup if mamba is available
if [ "$MAMBA_AVAILABLE" = true ]; then
    # Check if GT environment exists
    ENV_EXISTS=false
    if command_exists mamba; then
        if mamba env list >/dev/null 2>&1; then
            if mamba env list | grep -E '(^|/)(envs/)?GT([[:space:]]|$)' >/dev/null 2>&1; then
                ENV_EXISTS=true
            fi
        else
            echo "Error: Unable to query mamba environments. Ensure mamba is functional and accessible."
            exit 1
        fi
    else
        echo "Error: mamba is not available in PATH."
        echo "If your system uses environment modules, you may need to load Miniconda first, e.g.:"
        echo "  module load miniconda"
        echo "If you don't have mamba, install it from the following:"
        echo "  https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html"
        exit 1
    fi

    # Create or recreate GT environment
    if [ "$ENV_EXISTS" = false ]; then
        echo "No existing Mamba 'GT' environment detected."
        echo "Creating 'GT' with python and pip (python=3.13.*)..."
        mamba create -n GT python=3.13.* pip -y || { echo "Error: Failed to create 'GT' environment."; exit 1; }
        echo "Mamba environment 'GT' created successfully."

        # Install packages from requirements.txt into newly created GT
        install_requirements_into_GT
    else
        echo "Mamba 'GT' environment detected."
        while true; do
            read -r -p "Do you want to recreate the 'GT' environment? (yes[recommended for a fresh install]/no): " answer
            answer="${answer,,}"  # normalize to lowercase
            case "$answer" in
                y|yes )
                    echo "Removing existing 'GT' environment..."
                    mamba env remove -n GT -y || { echo "Error: Failed to remove existing 'GT' environment."; exit 1; }
                    echo "Creating 'GT' with python and pip (python=3.13.*)..."
                    mamba create -n GT python=3.13.* pip -y || { echo "Error: Failed to create 'GT' environment."; exit 1; }
                    echo "Mamba environment 'GT' recreated successfully."

                    # Install packages from requirements.txt into recreated GT
                    install_requirements_into_GT
                    break
                    ;;
                n|no )
                    echo "Skipping GT environment recreation as requested."
                    break
                    ;;
                * )
                    echo "Please answer yes or no."
                    ;;
            esac
        done
    fi
else
    echo "Skipping mamba environment setup and package installation as requested."
fi

# Determine installation directories
SETUP_DIR="$(pwd)"
SCRIPT_DIR="$SETUP_DIR/scripts"
SRC_DIR="$SETUP_DIR/src"
DOCS="$SETUP_DIR/docs"
DOCS_API_DIR="$SETUP_DIR/docs/api"
PKG_DIR="$HOME/.local/bin/GenomicTools"
PKG_BIN="$PKG_DIR/bin"
PYTHONRC_FILE="$HOME/.pythonrc.py"

# Set permissions for the dispatcher
chmod +x "$SCRIPT_DIR/dispatch.sh"
chmod 750 "$SCRIPT_DIR/dispatch.sh"

# Set permissions for the uninstaller
chmod +x "$SCRIPT_DIR/uninstall.sh"

# Get user confirmation
echo "The scripts will be installed to $PKG_DIR."
read -r -p "Do you want to continue with the installation? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation aborted."
    exit 1
fi

# Copy scripts
echo "Copying scripts..."
mkdir -p "$PKG_DIR/scripts"
mkdir -p "$PKG_DIR/src"
if [ -d "$SCRIPT_DIR" ] && [ -d "$SRC_DIR" ]; then
    cp -r "$SCRIPT_DIR" "$PKG_DIR"
    cp -r "$SRC_DIR" "$PKG_DIR"
else
    echo "Error: Required directories not found."
    exit 1
fi

# Create bin directory
echo "Creating bin directory..."
mkdir -p "$PKG_BIN"

# Remove old GenomicTools symlinks
echo "Cleaning existing symbolic links..."
find "$PKG_BIN" -type l -name 'GT-*' -exec rm -f {} +

# Determine the shell
SHELL_NAME=$(basename "$SHELL")
PROFILE_FILE=""
PROFILE_EXPORT_LINE=""
case "$SHELL_NAME" in
    "bash")
        PROFILE_FILE="$HOME/.bashrc"
        PROFILE_EXPORT_LINE="export GT_LAST_LOG=\"\$HOME/.local/bin/GenomicTools/LAST_LOG.log\""
        ;;
    "zsh")
        PROFILE_FILE="$HOME/.zshrc"
        PROFILE_EXPORT_LINE="export GT_LAST_LOG=\"\$HOME/.local/bin/GenomicTools/LAST_LOG.log\""
        ;;
    "fish")
        PROFILE_FILE="$HOME/.config/fish/config.fish"
        PROFILE_EXPORT_LINE="set -gx GT_LAST_LOG \$HOME/.local/bin/GenomicTools/LAST_LOG.log"
        ;;
    *)
        echo "Unsupported shell: $SHELL_NAME"
        exit 1
        ;;
esac

if [ -n "$PROFILE_FILE" ]; then
    mkdir -p "$(dirname "$PROFILE_FILE")"
    if [ ! -f "$PROFILE_FILE" ]; then
        touch "$PROFILE_FILE"
    fi
fi

# Update PATH in profile
PATH_LINE="export PATH=\"$PKG_BIN:\$PATH\""
echo "Configuring PATH in profile..."
if ! grep -Fxq "$PATH_LINE" "$PROFILE_FILE"; then
    echo "$PATH_LINE" >> "$PROFILE_FILE"
fi
export PATH="$PKG_BIN:$PATH"

# Ensure GT_LAST_LOG is exported for future shells
if [ -n "$PROFILE_EXPORT_LINE" ]; then
    if ! grep -Fxq "$PROFILE_EXPORT_LINE" "$PROFILE_FILE"; then
        echo "$PROFILE_EXPORT_LINE" >> "$PROFILE_FILE"
    fi
    export GT_LAST_LOG="$HOME/.local/bin/GenomicTools/LAST_LOG.log"
fi

# Remove deprecated GT_LAST_LOG helper files
rm -f "$PKG_DIR/GT_LAST_LOG" "$PKG_DIR/GT_LAST_LOG.env" "$HOME/GT_LAST_LOG"

# Prepare log storage
mkdir -p "$PKG_DIR"
LAST_LOG_FILE="$PKG_DIR/LAST_LOG.log"
: > "$LAST_LOG_FILE"

# Find scripts
SCRIPT_NAMES=()
while IFS= read -r script; do
    SCRIPT_NAMES+=("$(basename "$script" .py)")
done < <(find "$SRC_DIR" -maxdepth 1 -name "*.py")

# Drop utils from script names and remove empty entries
mapfile -t SCRIPT_NAMES < <(printf '%s\n' "${SCRIPT_NAMES[@]}" | grep -v '^utils$' | grep -v '^$')

# Create symlinks
echo "Creating symbolic links for scripts..."
for SCRIPT_NAME in "${SCRIPT_NAMES[@]}"; do
    ln -sf "../scripts/dispatch.sh" "$PKG_BIN/GT-$SCRIPT_NAME"
done

# Create or update .pythonrc.py
if [ ! -f "$PYTHONRC_FILE" ]; then
    echo "Creating .pythonrc.py..."
    touch "$PYTHONRC_FILE"
fi
touch "$PKG_DIR/src/__init__.py"

cat > "$PYTHONRC_FILE" << 'EOL'
def initGenomicTools() -> bool:
    import os
    import importlib.util
    import sys
    from typing import Dict
    _GENOMIC_TOOLS_LOADED: Dict[str, bool] = {}
    tools_path = os.path.expanduser("~/.local/bin/GenomicTools/src")
    
    if not os.path.exists(tools_path):
        print(f"Warning: GenomicTools directory not found at {tools_path}", file=sys.stderr)
        return False
    
    # Check if already loaded
    if _GENOMIC_TOOLS_LOADED.get('loaded', False):
        return True
        
    # Get all .py files
    py_files = [f for f in os.listdir(tools_path) 
                if f.endswith('.py') and not f.startswith('__')]
    
    if not py_files:
        print(f"Warning: No Python files found in {tools_path}", file=sys.stderr)
        return False
    
    # Import each .py file
    for file in py_files:
        try:
            module_name = file[:-3]  # Remove .py extension
            file_path = os.path.join(tools_path, file)
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[module_name] = module
        except Exception as e:
            print(f"Warning: Error loading {file}: {str(e)}", file=sys.stderr)
    
    _GENOMIC_TOOLS_LOADED['loaded'] = True
    print("GenomicTools loaded successfully")
EOL

if ! grep -q "PYTHONSTARTUP" "$PROFILE_FILE"; then
    echo 'export PYTHONSTARTUP="$HOME/.pythonrc.py"' >> "$PROFILE_FILE"
fi

chmod 644 "$PYTHONRC_FILE"

# Create the GenomicTools manual page
mkdir -p "$HOME/.local/share/man/man1/"
MAN_FILE="$HOME/.local/share/man/man1/GenomicTools.1"
echo "Creating the manual page..."

# Function to process a script and extract description and usage from Markdown
process_script() {
    local SCRIPT_NAME="$1"
    local MD_FILE="$DOCS_API_DIR/${SCRIPT_NAME}.md"
    local DESCRIPTION
    local USAGE
    if [ -f "$MD_FILE" ]; then
        DESCRIPTION=$(awk '/^## Description$/ {getline; sub(/^[ \t]+/, ""); print; exit}' "$MD_FILE")
        USAGE=$(awk '/^```sh$/ {getline; sub(/^[ \t]+/, ""); print; exit}' "$MD_FILE")
        if [ -z "$DESCRIPTION" ]; then
            DESCRIPTION="No description available."
        fi
        if [ -z "$USAGE" ]; then
            USAGE="No usage information available."
        fi
    else
        DESCRIPTION="No description available."
        USAGE="No usage information available."
    fi
    echo ".TP"
    echo "\\fBGT-$SCRIPT_NAME\\fR"
    echo "$DESCRIPTION"
    echo ".RS"
    echo "Usage: $USAGE"
    echo ".RE"
}

# Generate the manual page
{
    echo ".TH GenomicTools \"$(date +%Y-%m-%d)\" \"GenomicTools Manual\" \"GenomicTools\""
    echo ".SH NAME"
    echo "GenomicTools - A comprehensive toolkit for genomic data processing."
    echo ".SH SYNOPSIS"
    echo "GT-\\fIcommand\\fR [OPTIONS] \\fIarguments\\fR"
    echo ".SH DESCRIPTION"
    echo "GenomicTools is a comprehensive toolkit for genomic data processing and analysis at scale, with speed, and at the convenience of the command line."
    echo ".SH COMMANDS"
    for SCRIPT_NAME in "${SCRIPT_NAMES[@]}"; do
        process_script "$SCRIPT_NAME"
    done
    echo ".SH OPTIONS"
    echo "\\fB-h, --help\\fR"
    echo "    Show the help message and exit."
    echo ""
    echo "\\fB-v, --verbose\\fR \\fILEVEL\\fR"
    echo "    Set verbosity level (INFO or DEBUG)."
    echo ""
    echo "\\fB-l, --log\\fR \\fIFILE\\fR"
    echo "    Direct logs to FILE."
    echo ".SH EXAMPLES"
    echo ".B GT-plotManhattan -i results.hdf5 -o plot.png"
    echo ".br"
    echo ".B GT-saveToHDF5 -i data.csv -o data.hdf5"
    echo ".br"
    echo ".B GT-saveToHDF5 -i input.csv -o output.hdf5 --verbose DEBUG --log analysis.log"
    echo ".SH AUTHOR"
    echo "Valizadeh, A. - Department of Psychiatry, Yale University"
    echo ".SH CITATION"
    echo "If you use this software, please cite it as follows:"
    echo ".br"
    echo "Valizadeh, A. (2024). GenomicTools [Computer software]. Yale University."
    echo ".br" 
    echo "Available from https://git.yale.edu/av746/GenomicTools.git"
    echo ".SH BUGS"
    echo "Use --verbose INFO for short logs or --verbose DEBUG for extensive logs."
    echo ".br"
    echo "Direct logs to a file with --log <file_name>."
    echo ".br"
    echo "Please report bugs at:"
    echo ".br"
    echo "https://git.yale.edu/av746/GenomicTools/issues"
    echo ".SH SEE ALSO"
    echo "man(1), conda(1), R(1)"
    echo ".SH LICENSE"
    echo "Copyright (c) 2024 GenomicTools"
    echo ".SS Permissions"
    echo "Free of charge to use, copy, modify, and publish the Software for academic and non-commercial research purposes only."
    echo ".SS Conditions"
    echo "1. The above copyright notice and this permission notice must be included in all copies or substantial portions of the Software."
    echo "2. Explicit permission must be obtained for any commercial use."
    echo ".SS Limitations"
    echo "Redistribution, sublicensing, and/or selling of the Software is prohibited without explicit permission from the copyright holders."
    echo ".SS Attribution"
    echo "Users must cite GenomicTools in any publications or presentations resulting from the use of the Software."
    echo ".SS Disclaimer"
    echo "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED..."
    echo ".SS Contact"
    echo "For commercial use or redistribution permissions, please contact amir.valizadeh@yale.edu"
} > "$MAN_FILE"


# Update MANPATH
echo "Configuring MANPATH in the profile..."
MANPATH_LINE="export MANPATH=\"\$HOME/.local/share/man:\$MANPATH\""
if ! grep -Fxq "$MANPATH_LINE" "$PROFILE_FILE"; then
    echo "$MANPATH_LINE" >> "$PROFILE_FILE"
fi
export MANPATH="$HOME/.local/share/man:$MANPATH"

# Copy documentation to installation directory
echo "Copying documentation..."
mkdir -p "$PKG_DIR/docs"
mkdir -p "$PKG_DIR/docs/api"
cp -r "$DOCS"/* "$PKG_DIR/docs/"

# Final output
if command_exists tput && [ -t 1 ]; then
    ncolors=$(tput colors)
    if [ -n "$ncolors" ] && [ "$ncolors" -ge 8 ]; then
        green=$(tput setaf 2)
        yellow=$(tput setaf 3)
        reset=$(tput sgr0)
        echo "${green}GenomicTools installed successfully.${reset}"
        if [ "$MAMBA_AVAILABLE" = false ]; then
            echo "${yellow}Note: Installation completed without mamba. Environment setup was skipped.${reset}"
        fi
        echo "${yellow}Please restart your shell to apply changes.${reset}"
    else
        echo "GenomicTools installed successfully."
        if [ "$MAMBA_AVAILABLE" = false ]; then
            echo "Note: Installation completed without mamba. Environment setup was skipped."
        fi
        echo "Please restart your shell to apply changes."
    fi
else
    echo "GenomicTools installed successfully."
    if [ "$MAMBA_AVAILABLE" = false ]; then
        echo "Note: Installation completed without mamba. Environment setup was skipped."
    fi
    echo "Please restart your shell to apply changes."
fi
