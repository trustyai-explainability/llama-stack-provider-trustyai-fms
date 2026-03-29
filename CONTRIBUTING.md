# CONTRIBUTING GUIDELINES

Welcome to the llama-stack-provider-trustyai-fms contributing guide! We're excited to have you here and appreciate your contributions. This document provides guidelines and instructions for contributing to this project.

## Table of Contents
* [Development Setup](#development-setup)
* [Development Workflow](#development-workflow)
    * [Pull Request Checklist](#pull-request-checklist)
    * [Pre-commit Hooks](#pre-commit-hooks)
    * [Code Quality Standards](#code-quality-standards)
    * [Running Tests](#running-tests)

## Development Setup
### Prerequisites
* Python 3.12+
* pip (Python package installer)

### Installation

1. Clone the repository:
    ```
    git clone <repository-url>
    cd llama-stack-provider-trustyai-fms
    ```

2. Install the project in development mode:
    ```
    pip install -e ".[dev]"
    ```

## Development Workflow
To contribute your work, follow these steps:

1. **Fork the repository:** Fork the project repository to your GitHub account.
2. **Clone your fork:** Clone your fork to your local machine.
3. **Create a feature branch:** Create a branch from the `main` branch
4. **Develop:** Make your changes locally.
5. **Run tests:** Ensure everything works.
6. **Push changes:** Push your changes to your GitHub fork.
7. **Open a Pull Request (PR):** Create a PR against the main project's `main` branch.

## Pull Request Checklist
Before submitting your Pull Request (PR) on GitHub, please ensure you have completed the following steps. This checklist helps maintain the quality and consistency of the codebase.

### 1. Tests Passing:
Run the project's test suite to make sure all tests pass. Include new tests if you are adding new features or fixing bugs.

Run the tests:
```
pytest
```

### 2. Check Code Quality:
Adhere to the project's coding style guidelines.

Check code quality:

```
pre-commit run --all-files
```

### 3. Documentation:
Ensure that all new code is properly documented. Update the `README.md` and any other relevant documentation if your changes introduce new features or change existing functionality.

## Pre-commit Hooks
This project uses pre-commit hooks to ensure code quality and consistency. The hooks run automatically on every commit and can also be run manually if needed.

### Pre-commit Checks
The pre-commit configuration includes:

* **Ruff linting:** Python code quality checks and formatting
* **Ruff formatting:** Automatic code formatting (similar to Black)
* **YAML validation:** Checks YAML files for syntax errors (supports multi-document YAML)
* **General file checks:** Trailing whitespace, end-of-file, merge conflicts, etc.
* **Unit tests:** Runs pytest to ensure tests pass

### Setup Pre-commit
**Manual Setup:**
```
# Install pre-commit if not already installed
pip install pre-commit

# Install the hooks
pre-commit install
```

### Using Pre-commit
* **Automatic:** Hooks run automatically on every commit
* **Manual run on all files:**
    ```
    pre-commit run --all-files
    ```
* **Manual run on specific files:**
    ```
    pre-commit run --files path/to/file.py
    ```
* **Skip hooks** (use sparingly):
    ```
    git commit --no-verify -m "Emergency fix"
    ```

### Pre-commit CI Integration

This project uses [pre-commit.ci](https://pre-commit.ci/) which automatically runs pre-commit hooks on all pull requests and commits. This means that:
* **No local setup required** for contributors – hooks run automatically in CI
* **Consistent checks** across all contributions
* **Automatic fixes** – if possible, pre-commit.ci will create a commit with fixes
* **Status checks** – PRs will show pre-commit status and block merging if checks fail

**Note:** Even with pre-commit.ci, it is still recommended to set up and run pre-commit locally for faster feedback during development.

### Pre-commit Configuration
The configuration is in `.pre-commit-config.yaml` and includes:
* **Ruff hooks:** Code linting and formatting
* **YAML hygiene:** With multi-document support for Kubernetes manifests
* **File hygiene:** Various file quality checks
* **Local pytest hook:** Ensures tests pass before commit

## Code Quality Standards
### Python Code Style
* **Line length:** 88 characters (Black's default)
* **Formatting:** Automatic with ruff
* **Linting:** Comprehensive checks with ruff
* **Python version:** 3.12+

### YAML Files
* **Multi-document YAML:** Supported for Kubernetes manifests
* **Syntax validation:** Automatic checking on commit
* **Format:** Consistent indentation and structure

## Running Tests
### Test Configuration
Tests are configured in `pytest.ini` and `pyproject.toml`:
* **Test paths:** `tests/` directory
* **Python files:** `test_*.py` pattern
* **Source paths:** `src/` directory
* **Options:** Verbose output enabled

### How to Run Tests
```
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_*.py

# Run with verbose output
pytest -v
```
