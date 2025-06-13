# RoboDSL Developer Guide

Welcome to the developer guide for RoboDSL! This document provides a detailed overview of the project structure, individual files, and core components to help you understand and contribute to the project.

## Project Overview

RoboDSL is a domain-specific language (DSL) and compiler designed to simplify the development of GPU-accelerated robotics applications using ROS2 and CUDA. The project aims to address the complexities of ROS2/CUDA integration, CMake configuration, and GPU profiling by providing a streamlined, high-level workflow.

## Directory Structure

The project is organized into the following directories:

- **`config/`**: Contains sample YAML configuration files for nodes.
- **`include/`**: Holds C++ header files for nodes.
- **`launch/`**: Contains ROS2 launch files for nodes.
- **`src/`**: Contains the source code for the RoboDSL tool and the generated node source files.
- **`tests/`**: Contains all the tests for the project.

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

### Command-Line Interface (`cli.py`)

The CLI is the main entry point for the RoboDSL tool. It is built using the `click` library and provides the following commands:

- **`init`**: Initializes a new RoboDSL project.
- **`build`**: Builds the RoboDSL project (not yet implemented).
- **`add-node`**: Adds a new node to an existing project, creating the necessary files and directories.

### Parser (`parser.py`)

The parser is responsible for reading and interpreting `.robodsl` files. It extracts the node definitions, publishers, subscribers, and other information, which is then used by the CLI to generate the necessary code and configuration files.

### Code Generation

The code generation logic is located within the `cli.py` file. It uses the information from the parser to generate the following files:

- **Node source files** (`.py` or `.cpp`)
- **Launch files** (`.launch.py`)
- **RoboDSL configuration files** (`.robodsl`)

## How to Contribute

We welcome contributions to RoboDSL! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and add tests for them.
4.  Ensure that all tests pass.
5.  Submit a pull request with a detailed description of your changes.
