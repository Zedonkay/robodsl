# RoboDSL Developer Guide

Welcome to the RoboDSL developer guide! This document provides comprehensive information about the project's architecture, code organization, and development workflow to help you understand and contribute effectively.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Module Reference](#module-reference)
4. [Code Organization](#code-organization)
5. [Development Workflow](#development-workflow)
6. [Extending RoboDSL](#extending-robodsl)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [Additional Resources](#additional-resources)

## Project Overview

RoboDSL is a domain-specific language (DSL) and compiler designed to simplify the development of GPU-accelerated robotics applications using ROS2 and CUDA. The project addresses several key challenges:

- **Complex Integration**: Simplifies ROS2 and CUDA integration
- **Boilerplate Reduction**: Automates repetitive code generation
- **Build System**: Handles complex CMake configurations
- **Standardization**: Enforces best practices for ROS2/CUDA development

### Key Concepts

1. **DSL (Domain-Specific Language)**
   - Declarative syntax for defining robotics nodes
   - Supports both C++ and Python node generation
   - Enables CUDA kernel integration

2. **Code Generation**
   - Generates ROS2 node templates
   - Creates build system configurations
   - Produces launch files and parameter configurations

3. **Project Structure**
   - Standardized directory layout
   - Separation of generated and source code
   - Support for nested node organization

## Architecture

### Core Components

1. **CLI Interface** (`cli.py`)
   - Command-line interface using Click
   - Handles project and node lifecycle
   - Coordinates code generation
   - Manages command execution flow

2. **DSL Parser** (`parser.py`)
   - Parses `.robodsl` files
   - Validates syntax and semantics
   - Generates abstract syntax tree (AST)
   - Handles error reporting

3. **Code Generator** (`generator.py`)
   - Transforms AST into target code
   - Manages template rendering
   - Handles file system operations
   - Supports multiple target languages

4. **Templates**
   - Node templates for different languages
   - Launch file templates
   - Configuration templates
   - Build system templates

### Data Flow

1. **Initialization**
   ```
   User Command → CLI → Project Setup → File Generation
   ```

2. **Node Addition**
   ```
   User Command → CLI → Parser → AST → Generator → File Creation
   ```

3. **Build Process**
   ```
   User Command → CLI → Build System Generation → Build Execution
   ```

## Module Reference

### CLI Module (`cli.py`)

The CLI module provides the command-line interface for RoboDSL, handling user interactions and orchestrating the code generation process.

#### Key Functions

1. **`init(project_name, template, output_dir)`**
   - Creates a new RoboDSL project
   - Sets up the directory structure
   - Initializes configuration files

2. **`add_node(node_name, publisher, subscriber, language, project_dir)`**
   - Adds a new node to an existing project
   - Handles file generation for the specified language
   - Updates build configurations

3. **`build(project_dir)`**
   - Generates build files
   - Executes the build process
   - Handles build artifacts

### Parser Module (`parser.py`)

The parser module processes the RoboDSL syntax and generates an abstract syntax tree (AST).

#### Key Components

1. **Lexer**
   - Tokenizes input files
   - Handles syntax highlighting
   - Reports lexical errors

2. **Parser**
   - Validates syntax
   - Builds AST
   - Enforces semantic rules

3. **AST**
   - Represents the program structure
   - Enables code generation
   - Supports transformations

### Generator Module (`generator.py`)

The generator module transforms the AST into target code.

#### Key Features

1. **Template Rendering**
   - Uses Jinja2 for templating
   - Supports multiple output formats
   - Handles indentation and formatting

2. **File Management**
   - Creates directories as needed
   - Handles file overwrites
   - Manages file permissions

3. **Code Generation**
   - C++ node generation
   - Python node generation
   - Build system files
   - Launch configurations

## Code Organization

### Source Code (`src/robodsl/`)

- **`__init__.py`**: Package initialization and version information
- **`cli.py`**: Command-line interface implementation using Click
- **`parser.py`**: DSL parsing and validation logic
- **`generator.py`**: Code generation logic and template handling

### Test Structure (`tests/`)

- **`test_cli.py`**: Tests for command-line interface
- **`test_parser.py`**: Tests for DSL parsing and validation
- **`test_add_node.py`**: Tests for node generation
- **`test_subnodes.py`**: Tests for nested node functionality
- **`fixtures/`**: Test fixtures and helper functions

### Templates (`templates/`)
- **`cpp/`**: C++ node templates
- **`python/`**: Python node templates
- **`launch/`**: ROS2 launch file templates
- **`cmake/`**: CMake configuration templates

### Generated Project Structure

```
my_project/
├── src/                    # Node source files
│   ├── my_node.cpp         # C++ node implementation
│   └── my_node.py          # Python node implementation
│
├── include/               # C++ headers
│   └── my_node/
│       └── my_node.hpp   # Node header
│
├── launch/                # Launch files
│   └── my_node.launch.py
│
├── config/                # Configuration files
│   └── my_node.yaml
│
├── robodsl/               # DSL definitions
│   └── nodes/
│       └── my_node.robodsl
│
├── CMakeLists.txt         # Build configuration
└── package.xml            # ROS2 package definition
```

### Test Structure (`tests/`)

- `test_cli.py`: CLI command tests
- `test_parser.py`: DSL parsing tests
- `test_add_node.py`: Node generation tests
- `test_subnodes.py`: Nested node tests
- `fixtures/`: Test fixtures and utilities

## File-by-File Breakdown

### Root Directory

- **`LICENSE`**: The MIT License file for the project.
- **`PLAN.md`**: The development plan for the project.
- **`README.md`**: The main README file with a project overview, features, and usage instructions.
- **`DEVELOPER_GUIDE.md`**: This file, providing detailed information for developers.
- **`pytest.ini`**: Configuration file for pytest, specifying the test directory.
- **`setup.py`**: The setup script for the project, used for packaging and distribution.

### `config/` Directory

- **`*.yaml`**: Sample YAML configuration files for nodes. These files are used to define node parameters and settings.

### `include/` Directory

- **`*/*_node.h`**: C++ header files for nodes. These files contain the class definitions for the nodes.

### `launch/` Directory

- **`*.launch.py`**: ROS2 launch files for nodes. These files are used to launch the nodes and configure their settings.

### `src/` Directory

- **`robodsl/`**: The main package for the RoboDSL tool.
  - **`__init__.py`**: Initializes the `robodsl` package.
  - **`cli.py`**: The heart of the RoboDSL tool, defining the command-line interface (CLI) using `click`. It handles commands like `init`, `build`, and `add-node`, and orchestrates the creation of node files, launch files, and configuration files.
  - **`parser.py`**: Contains the parsing logic for the RoboDSL language. It is responsible for reading `.robodsl` files and extracting the node definitions, publishers, subscribers, and other information.
- **`robodsl.egg-info/`**: Contains metadata for the `robodsl` package.
- **`*_node.cpp` / `*_node.py`**: Generated source files for nodes. These files contain the main logic for the nodes.

### `tests/` Directory

- **`test_add_node.py`**: Tests for the `add-node` command, ensuring that nodes are created correctly.
- **`test_cli.py`**: General tests for the CLI, covering basic commands and error handling.
- **`test_parser.py`**: Tests for the RoboDSL parser, ensuring that `.robodsl` files are parsed correctly.
- **`test_subnodes.py`**: Tests for the subnode functionality, ensuring that nested nodes are created and configured correctly.

## Core Components

### 1. Command-Line Interface (`cli.py`)

The CLI is the main entry point for RoboDSL, built using the `click` library.

#### Key Commands

```bash
# Initialize a new project
robodsl init my_project

# Add a new node
robodsl add-node my_node --language cpp

# Build the project
robodsl build
```

#### Key Functions

- `init()`: Creates a new project structure
- `add_node()`: Adds a new node to the project
- `build()`: Builds the project
- `_create_node_files()`: Handles file generation
- `_get_node_file_paths()`: Manages file paths

### 2. DSL Parser (`parser.py`)

Parses `.robodsl` files and validates the syntax.

#### Key Features
- Supports node definitions with publishers, subscribers, and parameters
- Validates message types and parameter values
- Generates an abstract syntax tree (AST)

#### Example DSL
```rubynode my_node {
    publisher /output std_msgs/msg/String "Hello"
    subscriber /input std_msgs/msg/String
    parameter rate 10
}
```

### 3. Code Generation

#### C++ Generation
- Creates `.hpp` and `.cpp` files
- Handles class definitions and implementations
- Manages include guards and namespaces

#### Python Generation
- Generates executable Python nodes
- Handles imports and class definitions
- Sets up ROS2 node lifecycle

### 4. Build System
- Generates `CMakeLists.txt`
- Handles dependencies
- Configures CUDA compilation (coming soon)

## Development Workflow

### Setting Up for Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/robodsl.git
   cd robodsl
   ```

2. **Set up a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_add_node.py -v

# Run with coverage
pytest --cov=robodsl tests/
```

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **Mypy** for type checking

```bash
# Format code
black .

# Check style
flake8
# Check types
mypy .
```

## Contributing

We welcome contributions! Here's how to get started:

1. **Find an issue** or create a new one
2. **Fork** the repository
3. Create a feature branch: `git checkout -b feature/amazing-feature`
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a **Pull Request**

### Pull Request Guidelines

- Include tests for new features
- Update documentation
- Follow the existing code style
- Keep PRs focused and small

## Additional Resources

- [ROS2 Documentation](https://docs.ros.org/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Click Documentation](https://click.palletsprojects.com/)
- [Pytest Documentation](https://docs.pytest.org/)

## Acknowledgments

- The ROS2 and CUDA communities
- All contributors who have helped improve RoboDSL
- The Python ecosystem for amazing developer tools
