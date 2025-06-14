# Contributing to RoboDSL

Thank you for your interest in contributing to RoboDSL! We welcome contributions from the community to help improve the project. This guide will help you get started with contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Environment](#development-environment)
4. [Code Style](#code-style)
5. [Testing](#testing)
6. [Pull Request Process](#pull-request-process)
7. [Reporting Issues](#reporting-issues)
8. [Feature Requests](#feature-requests)
9. [Documentation](#documentation)
10. [License](#license)

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](code_of_conduct.md). Please report any unacceptable behavior to the project maintainers.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
   ```bash
   git clone https://github.com/your-username/robodsl.git
   cd robodsl
   ```
3. Create a **branch** for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes and **commit** them
   ```bash
   git commit -m "Add your commit message"
   ```
5. **Push** your changes to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
6. Open a **pull request** against the main repository

## Development Environment

### Prerequisites

- Python 3.8+
- ROS2 (Humble/Foxy/Galactic)
- CUDA Toolkit 11.0+
- CMake 3.15+
- Git

### Setup

1. Create a Python virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. Install pre-commit hooks
   ```bash
   pre-commit install
   ```

## Code Style

### Python

We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide with the following additional guidelines:

- Maximum line length: 88 characters (enforced by Black)
- Use double quotes for strings
- Use type hints for all function signatures
- Document all public APIs with docstrings

### C++

We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) with the following exceptions:

- Use `#pragma once` instead of include guards
- Maximum line length: 100 characters
- Use C++17 features where appropriate

### Formatting

We use automated tools to enforce code style:

- **Python**: [Black](https://github.com/psf/black), [isort](https://pycqa.github.io/isort/), [flake8](https://flake8.pycqa.org/)
- **C++**: [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
- **Markdown**: [Prettier](https://prettier.io/)

Run the following command to format your code:

```bash
make format
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run tests with coverage
pytest --cov=robodsl tests/
```

### Writing Tests

- Write tests for all new functionality
- Follow the Arrange-Act-Assert pattern
- Use descriptive test names
- Keep tests independent and isolated
- Mock external dependencies

## Pull Request Process

1. Ensure your code passes all tests and linters
2. Update documentation as needed
3. Add or update tests for your changes
4. Keep your pull request focused on a single feature or bug fix
5. Write a clear commit message following the [Conventional Commits](https://www.conventionalcommits.org/) specification:
   ```
   <type>[optional scope]: <description>
   
   [optional body]
   
   [optional footer]
   ```
   
   Example:
   ```
   feat(parser): add support for custom message types
   
   Added support for defining custom message types in the DSL syntax.
   This allows users to create their own message types with custom fields.
   
   Fixes #123
   ```

6. Reference any related issues in your pull request description
7. Request reviews from at least one maintainer

## Reporting Issues

When reporting issues, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Python version, ROS2 version, etc.)
6. Any relevant error messages or logs

## Feature Requests

For feature requests, please:

1. Describe the feature you'd like to see
2. Explain why this feature would be useful
3. Provide any relevant examples or use cases
4. Consider contributing the feature yourself if possible

## Documentation

Good documentation is crucial for the success of the project. When making changes:

1. Update relevant documentation
2. Add docstrings to new functions and classes
3. Include examples where appropriate
4. Keep the README up to date

## License

By contributing to RoboDSL, you agree that your contributions will be licensed under the [MIT License](LICENSE).

## Acknowledgments

Thank you to all the contributors who have helped make RoboDSL better!
