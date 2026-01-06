.PHONY: help install uninstall lint clean build_docs test install-dev

# Directories
SCRIPTS_DIR := ./scripts
SRC := src/
TESTS := tests/

# Tools
BLACK := black
FLAKE8 := flake8
SHELL_LINT := shellcheck -x $(SCRIPTS_DIR)/*.sh
RLINT := Rscript -e 'lintr::lint_dir("src")'


# Default target
.DEFAULT_GOAL := help

# Determine installer flags based on DEV variable
INSTALL_FLAGS := $(if $(DEV),--dev,)

# Show available targets
help:
	@echo "Available targets:"
	@echo "  install		   Run the install script (use DEV=1 for dev deps)"
	@echo "  install-dev	   Install including development dependencies"
	@echo "  uninstall		 Run the uninstall script"
	@echo "  lint			  Run code linters"
	@echo "  clean			 Remove build artifacts"
	@echo "  build_docs		Build the documentation"
	@echo "  test			  Run the tests with coverage"

# Install the project
install:
	@# Ensure install.sh exists (try to rename install.txt -> install.sh if needed)
	@if [ ! -f "$(SCRIPTS_DIR)/install.sh" ]; then \
		if [ -f "$(SCRIPTS_DIR)/install.txt" ]; then \
			echo "Found $(SCRIPTS_DIR)/install.txt -> renaming to install.sh"; \
			mv "$(SCRIPTS_DIR)/install.txt" "$(SCRIPTS_DIR)/install.sh"; \
			chmod +x "$(SCRIPTS_DIR)/install.sh"; \
		else \
			echo "Error: neither $(SCRIPTS_DIR)/install.sh nor $(SCRIPTS_DIR)/install.txt found" >&2; \
			exit 1; \
	  	fi \
	fi
	@rm -f "$(SCRIPTS_DIR)/install.txt"
	@# Also rename uninstall.txt and dispatch.txt to .sh if present (and .sh doesn't already exist)
	@for f in uninstall dispatch; do \
	  	if [ -f "$(SCRIPTS_DIR)/$$f.txt" ] && [ ! -f "$(SCRIPTS_DIR)/$$f.sh" ]; then \
			echo "Found $(SCRIPTS_DIR)/$$f.txt -> renaming to $$f.sh"; \
			mv "$(SCRIPTS_DIR)/$$f.txt" "$(SCRIPTS_DIR)/$$f.sh"; \
			chmod +x "$(SCRIPTS_DIR)/$$f.sh"; \
	  	fi \
		rm -f "$(SCRIPTS_DIR)/$$f.txt"; \
	done
	bash $(SCRIPTS_DIR)/install.sh $(INSTALL_FLAGS)

install-dev: DEV=1
install-dev: install

# Uninstall the project
uninstall:
	bash $(SCRIPTS_DIR)/uninstall.sh

# Linting
lint:
	$(BLACK) $(SRC)
	$(BLACK) $(TESTS)
	$(FLAKE8) $(SRC) || true
	$(FLAKE8) $(TESTS) || true
	$(SHELL_LINT)
	$(RLINT)

# Clean up build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf **/*.pyc .coverage.* tests/output/* **/.pytest_cache .pytest_cache

# Build documentation
.PHONY: build_docs
build_docs:
	mkdocs build --config-file=config/mkdocs.yml
	mkdocs gh-deploy --config-file=config/mkdocs.yml

# Run tests
test:
	pytest $(TESTS) --cov=$(SRC) --cov-report=term-missing
