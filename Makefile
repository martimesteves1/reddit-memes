.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install
	@uv run python -m ipykernel install --user --name=reddit-memes --display-name="Reddit Memes Kernel"

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv sync --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry .

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --doctest-modules

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## clean build artifacts
	@rm -rf dist

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9._%+-]+:.*##/ { split($$1, a, ":"); desc = substr($$0, index($$0, "##") + 2); gsub(/^[ \t]+/, "", desc); printf "  %-20s %s\n", a[1], desc }' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help
