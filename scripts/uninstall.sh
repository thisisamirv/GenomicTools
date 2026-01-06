#!/bin/bash
set -euo pipefail

# Determine the directories to remove
PKG_DIR="$HOME/.local/bin/GenomicTools"
PKG_BIN="$PKG_DIR/bin"
MAN_PAGE="$HOME/.local/share/man/man1/GenomicTools.1"

# Ask for confirmation
echo "The following directory will be removed:"
echo "  Installation Directory: $PKG_DIR"
read -r -p "Do you want to proceed with the uninstallation? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "Uninstallation aborted."
    exit 0
fi

# Remove package directory
echo "Removing package directory..."
if [ -d "$PKG_DIR" ]; then
    rm -rf "$PKG_DIR"
    echo "Package directory $PKG_DIR removed."
else
    echo "Package directory $PKG_DIR does not exist. Skipping package directory removal."
fi

# Remove man page
echo "Removing man page..."
if [ -f "$MAN_PAGE" ]; then
    rm -f "$MAN_PAGE"
    echo "Man page $MAN_PAGE removed."
else
    echo "Man page $MAN_PAGE does not exist. Skipping."
fi

# Remove Mamba 'GT' environment and its packages (if mamba is available)
if command -v mamba >/dev/null 2>&1; then
    # Ensure mamba can list environments before grepping
    if mamba env list >/dev/null 2>&1; then
        if mamba env list | grep -E '(^|/)(envs/)?GT([[:space:]]|$)' >/dev/null 2>&1; then
            echo "Removing Mamba environment 'GT' and all its packages..."
            if mamba env remove -n GT -y >/dev/null 2>&1; then
                echo "Mamba environment 'GT' removed."
            else
                echo "Warning: Failed to remove Mamba environment 'GT'. You may need to remove it manually."
            fi
        else
            echo "Mamba is available but environment 'GT' was not found. Skipping."
        fi
    else
        echo "Mamba is available but could not list environments. Skipping GT removal."
    fi
else
    echo "Mamba not found in PATH. Skipping removal of Mamba environment 'GT'."
fi

# Determine the shell
SHELL_NAME=$(basename "$SHELL")
PROFILE_FILE=""
case "$SHELL_NAME" in
    "bash")
        PROFILE_FILE="$HOME/.bashrc"
        ;;
    "zsh")
        PROFILE_FILE="$HOME/.zshrc"
        ;;
    "fish")
        PROFILE_FILE="$HOME/.config/fish/config.fish"
        ;;
    *)
        echo "Unsupported shell: $SHELL_NAME"
        exit 1
        ;;
esac

# Remove BIN_DIR from PATH in the profile file
PATH_LINE="export PATH=\"$PKG_BIN:\$PATH\""
echo "Removing GenomicTools bin directory from PATH in $SHELL_NAME profile..."
if [ -f "$PROFILE_FILE" ] && grep -qF "$PATH_LINE" "$PROFILE_FILE"; then
    sed -i.bak "\|$PATH_LINE|d" "$PROFILE_FILE"
    echo "Removed GenomicTools bin directory from PATH in $SHELL_NAME profile."
else
    echo "GenomicTools bin directory not found in PATH or $SHELL_NAME profile does not exist. Skipping PATH removal."
fi

echo "GenomicTools uninstalled successfully."
