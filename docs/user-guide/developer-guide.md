# RoboDSL Developer Guide

Quick reference for RoboDSL development and contribution.

## Project Structure

```
robodsl/
├── src/               # Source code
│   ├── parser/       # Parser implementation
│   ├── generator/    # Code generators
│   └── runtime/      # Runtime components
├── templates/        # Code templates
└── tests/            # Test suite
```

## Getting Started

### Prerequisites
- Python 3.8+
- CMake 3.16+
- C++17 compatible compiler
- CUDA 11.0+ (for GPU support)

### Setup

```bash
# Clone and setup
$ git clone https://github.com/yourusername/robodsl.git
$ cd robodsl
$ python -m venv venv
$ source venv/bin/activate  # or `venv\Scripts\activate` on Windows
$ pip install -e .[dev]     # Install in development mode
```

## Development

### Build & Test

```bash
# Build
$ python setup.py build

# Run tests
$ pytest

# Run linter
$ pylint src/

# Format code
$ black src/
```

### Debugging

```bash
# Run with debug output
$ robodsl --debug <input.dsl>

# Generate debug info
$ robodsl --dump-ast <input.dsl>
```

## Extending RoboDSL

### Adding Node Types

1. Define node in `src/parser/nodes/`
2. Add templates in `templates/`
3. Register in `src/generator/__init__.py`
4. Add tests in `tests/`

### Custom Generators

1. Create `src/generator/<name>.py`
2. Implement required methods
3. Register in `src/generator/__init__.py`

## Performance Tips

- Use `@lru_cache` for expensive operations
- Minimize string operations in hot paths
- Profile with `cProfile`
- Use `__slots__` for data classes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Submit a PR

## Documentation

Build with Sphinx:

```bash
$ cd docs
$ make html
```

## License

MIT - See [LICENSE](LICENSE)
