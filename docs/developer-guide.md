# RoboDSL Developer Guide

Welcome to the RoboDSL developer guide! This document provides comprehensive information about the project's architecture, code organization, and development workflow to help you understand and contribute effectively.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
   - [Core Components](#core-components)
   - [Data Flow](#data-flow)
   - [Build System Integration](#build-system-integration)
3. [Module Reference](#module-reference)
   - [CLI Module](#cli-module)
   - [Parser Module](#parser-module)
   - [Generator Module](#generator-module)
   - [Template System](#template-system)
4. [Code Organization](#code-organization)
   - [Source Code Structure](#source-code-structure)
   - [Build System](#build-system)
   - [Testing Framework](#testing-framework)
5. [Development Workflow](#development-workflow)
   - [Environment Setup](#environment-setup)
   - [Building from Source](#building-from-source)
   - [Running Tests](#running-tests)
   - [Debugging](#debugging)
6. [Extending RoboDSL](#extending-robodsl)
   - [Adding New Node Types](#adding-new-node-types)
   - [Custom Code Generators](#custom-code-generators)
   - [Template Customization](#template-customization)
7. [Performance Optimization](#performance-optimization)
   - [Code Generation](#code-generation-performance)
   - [Runtime Performance](#runtime-performance)
   - [Memory Management](#memory-management)

## Project Overview

RoboDSL is a domain-specific language designed for building GPU-accelerated robotics applications. It provides a high-level, declarative syntax for defining robot behaviors, data processing pipelines, and hardware interfaces.

### Key Features

- **Declarative Syntax**: Define robot behaviors and data flows in a clean, readable format
- **GPU Acceleration**: Seamless integration with CUDA for high-performance computing
- **Modular Architecture**: Easily extendable with custom components and templates
- **Cross-Platform**: Works on Linux, Windows, and macOS with consistent behavior
- **ROS2 Integration**: Native support for ROS2 nodes and communication patterns

## Architecture

### Core Components

1. **Parser**: Converts RoboDSL source code into an abstract syntax tree (AST)
2. **Generator**: Transforms the AST into target code (C++, Python, etc.)
3. **Runtime**: Provides the execution environment for generated code
4. **Standard Library**: Collection of reusable components and utilities

### Data Flow

1. **Source Code**: `.robodsl` files containing the application definition
2. **Parsing**: Source code is parsed into an AST
3. **Validation**: AST is validated against the language specification
4. **Code Generation**: Target code is generated from the validated AST
5. **Compilation**: Generated code is compiled into executable binaries
6. **Execution**: The final application runs with the RoboDSL runtime

## Module Reference

### CLI Module

The Command Line Interface (CLI) module provides a user-friendly way to interact with RoboDSL. It's built using the `click` library and supports various commands for building, testing, and managing RoboDSL projects.

### Parser Module

The Parser module is responsible for converting RoboDSL source code into an Abstract Syntax Tree (AST). It uses a combination of regular expressions and parsing rules to validate and structure the input.

### Generator Module

The Generator module takes the validated AST and produces target code in the specified output language (e.g., C++ with CUDA). It uses a template-based approach for flexibility.

## Code Organization

### Source Code Structure

```
robodsl/
├── src/                    # Source code
│   ├── parser/            # Parser implementation
│   ├── generator/         # Code generators
│   ├── runtime/           # Runtime components
│   └── utils/             # Utility functions
├── templates/             # Code templates
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Development Workflow

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/robodsl.git
   cd robodsl
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Building from Source

```bash
# Install in development mode
pip install -e .

```

### Running Tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_parser.py
```

## Extending RoboDSL

### Adding New Node Types

1. Define the node in the appropriate DSL file
2. Create corresponding template files
3. Register the node type in the generator
4. Add tests for the new node type

### Custom Code Generators

RoboDSL supports custom code generators for different target platforms. To create a new generator:

1. Create a new module in `src/generator/`
2. Implement the required generator interface
3. Register the generator in the main application
4. Update the build system if needed

## Performance Optimization

### Code Generation

- Use efficient string operations
- Minimize template lookups
- Cache generated code when possible

### Runtime Performance

- Optimize hot paths
- Use appropriate data structures
- Profile and optimize memory usage

### Memory Management

- Use RAII for resource management
- Implement move semantics where appropriate
- Minimize allocations in performance-critical code
