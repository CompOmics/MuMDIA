# Makefile for MuMDIA testing and development

.PHONY: help test test-unit test-integration test-fast test-slow coverage lint format type-check install-test-deps clean diagrams

# Default target
help:
	@echo "MuMDIA Development Commands:"
	@echo ""
	@echo "Testing:"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-fast         Run fast tests only"
	@echo "  test-slow         Run slow tests only"
	@echo "  coverage          Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint              Run linting checks"
	@echo "  format            Format code with black and isort"
	@echo "  type-check        Run mypy type checking"
	@echo ""
	@echo "Documentation:"
	@echo "  diagrams          Generate workflow diagrams"
	@echo ""
	@echo "Setup:"
	@echo "  install-test-deps Install test dependencies"
	@echo "  clean             Clean up test artifacts"

# Test commands
test:
	python tests/run_tests.py all --verbose

test-unit:
	pytest tests/ -m "unit" -v

test-integration:
	pytest tests/ -m "integration" -v

test-fast:
	pytest tests/ -m "fast" -v

test-slow:
	pytest tests/ -m "slow" -v

coverage:
	python tests/run_tests.py all --coverage

# Code quality commands
lint:
	flake8 feature_generators/ data_structures.py utilities/ --max-line-length=88
	@echo "Linting complete!"

format:
	black feature_generators/ data_structures.py utilities/ tests/
	isort feature_generators/ data_structures.py utilities/ tests/
	@echo "Code formatting complete!"

type-check:
	mypy feature_generators/ data_structures.py utilities/
	@echo "Type checking complete!"

# Setup commands
install-test-deps:
	pip install -r test_requirements.txt
	@echo "Test dependencies installed!"

# Cleanup
clean:
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"

# Documentation
diagrams:
	@echo "Generating workflow diagrams..."
	cd workflow_visualization && ./generate_diagrams.sh

# Development workflow
dev-setup: install-test-deps
	@echo "Development environment setup complete!"

check-all: format lint type-check test
	@echo "All checks passed!"
