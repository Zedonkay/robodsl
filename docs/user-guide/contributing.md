# Contributing to RoboDSL

We welcome contributions! Here's how to get started.

## Quick Start

```bash
# 1. Fork & clone
$ git clone https://github.com/yourusername/robodsl.git
$ cd robodsl

# 2. Setup env
$ python -m venv venv
$ source venv/bin/activate  # or `venv\Scripts\activate` on Windows
$ pip install -e ".[dev]"
$ pre-commit install

# 3. Make changes
$ git checkout -b feat/your-feature
# ... make changes ...

# 4. Test & submit
$ pytest
$ git commit -m "feat: add new feature"
$ git push origin feat/your-feature
# Create PR from GitHub UI
```

## Development Environment

### Prerequisites

- Python 3.8+
- ROS2 Humble or newer (for ROS2 features)
- CUDA Toolkit 11.0+ (for CUDA features)
- CMake 3.15+
- Git

### Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Testing

```bash
# Run all tests
$ pytest

# With coverage
$ pytest --cov=robodsl

# Specific test file
$ pytest tests/test_feature.py -v
```

Write tests in `tests/` with `test_*.py` naming.

## Pull Requests

1. Keep PRs focused and small
2. Update docs if needed
3. Add tests for new features
4. Ensure all tests pass
5. Follow commit message format:

```
type(scope): description

Optional body

Closes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

## Issues & Features

### Reporting Bugs

1. Check existing issues first
2. Include:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details
   - Error logs

### Feature Requests

1. Check existing issues
2. Explain the use case
3. Propose a solution
4. Include examples if possible

## Documentation

```bash
# Build docs
$ cd docs
$ pip install -r requirements.txt
$ make html
```

- Keep docs in `docs/`
- Use Markdown (.md)
- Follow Google style for docstrings
- Update when adding features

## License

MIT - See [LICENSE](LICENSE)

## Code Style

- **Python**: Black + isort
- **C++**: clang-format
- **Commits**: Conventional Commits
- **Docs**: Google style

Pre-commit handles formatting. Install hooks once:

```bash
$ pre-commit install
```

## Code Review Guidelines

### Reviewers

- Focus on code quality and correctness
- Check for security implications
- Verify test coverage
- Ensure documentation is updated
- Consider performance impact

### Authors

- Keep PRs focused and small
- Address all review comments
- Update documentation and tests
- Be responsive to feedback
- Fix CI failures promptly

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a release tag
4. Build and publish packages
5. Update documentation

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
