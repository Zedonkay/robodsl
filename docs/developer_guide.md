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
8. [Troubleshooting](#troubleshooting)
   - [Common Issues](#common-issues)
   - [Debugging Tips](#debugging-tips)
   - [Performance Profiling](#performance-profiling)
9. [Contributing](#contributing)
   - See [Contributing](contributing.md) for detailed contribution guidelines.
   - [Pull Request Process](#pull-request-process)
   - [Code Review Guidelines](#code-review-guidelines)
   - [Release Process](#release-process)
10. [Code of Conduct](#code-of-conduct)
   - Please review our [Code of Conduct](code_of_conduct.md) before contributing.
11. [Additional Resources](#additional-resources)
    - [ROS2 Documentation](https://docs.ros.org/)
    - [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
    - [Python Packaging Guide](https://packaging.python.org/)

## Project Overview

RoboDSL is a domain-specific language (DSL) and compiler designed to simplify the development of GPU-accelerated robotics applications using ROS2 and CUDA. The project addresses several key challenges in robotics software development:

- **Complex Integration**: Streamlines ROS2 and CUDA integration
- **Boilerplate Reduction**: Automates repetitive code generation
- **Build System**: Handles complex CMake configurations
- **Standardization**: Enforces best practices for ROS2/CUDA development
- **Performance**: Optimized code generation for real-time systems

### Key Components

1. **DSL Parser**
   - Parses RoboDSL source files
   - Validates syntax and semantics
   - Generates an Abstract Syntax Tree (AST)

2. **Code Generator**
   - Converts AST into executable code
   - Supports C++17 and CUDA
   - Generates ROS2 nodes with lifecycle support

3. **Template System**
   - Jinja2-based template engine
   - Extensible template architecture
   - Support for custom templates

4. **Build System**
   - CMake integration
   - Cross-platform support
   - Dependency management

### Project Structure

```
robodsl/
├── src/                    # Source code
│   ├── robodsl/            # Core package
│   │   ├── cli.py          # Command-line interface
│   │   ├── parser.py       # DSL parser
│   │   ├── generator.py    # Code generator
│   │   └── templates/      # Code templates
├── examples/               # Example projects
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Implementation Architecture

This section provides an overview of RoboDSL's implementation architecture, focusing on key design patterns and extension points. For the complete system architecture, see the [DSL Specification](./dsl_specification.md#architecture).

### Core Design Patterns

1. **Command Pattern (CLI Interface)**
   ```python
   @click.group()
   def cli():
       """RoboDSL command-line interface."""
       pass
   
   @cli.command()
   @click.argument('project_name')
   def init(project_name):
       """Initialize a new RoboDSL project."""
       # Implementation...
   ```

2. **Interpreter Pattern (DSL Parser)**
   - Lexical analysis with PLY (Python Lex-Yacc)
   - Context-free grammar for syntax validation
   - Abstract Syntax Tree (AST) generation
   - Semantic analysis and type checking

3. **Template Method (Code Generation)**
   - Base generator class with common functionality
   - Specialized generators for different node types
   - Template rendering with Jinja2
   - Post-processing for code formatting

### Extension Points

1. **Custom Node Types**
   ```python
   class CustomNodeGenerator(NodeGenerator):
       def __init__(self, node_config):
           super().__init__(node_config)
           self.template_name = 'custom_node.cpp.jinja2'
   
       def validate(self):
           # Custom validation logic
           pass
   ```

2. **Template Customization**
   - Override default templates in `~/.robodsl/templates/`
   - Use template inheritance for customizations
   - Add custom Jinja2 filters and extensions

3. **Build System Integration**
   - Custom CMake modules
   - Platform-specific build configurations
   - Dependency management
   - Cross-compilation support

### Threading and Concurrency

1. **Thread Safety**
   - Immutable configuration objects
   - Thread-local storage for parser state
   - Lock-free data structures where possible

2. **Parallel Code Generation**
   - Concurrent template rendering
   - Parallel file I/O operations
   - Caching of intermediate results

### Error Handling

1. **Validation Framework**
   - Schema validation with JSON Schema
   - Custom validation rules
   - Detailed error messages with context

2. **Recovery Strategies**
   - Partial parsing and error recovery
   - Auto-correction of common mistakes
   - Fallback mechanisms for optional features

### Data Flow

RoboDSL processes your DSL code through several well-defined stages, each transforming the input into a more refined representation:

```mermaid
flowchart LR
    A[DSL Source] -->|Lexing| B[Tokens]
    B -->|Parsing| C[AST]
    C -->|Validation| D[Semantic Model]
    D -->|Code Generation| E[Source Files]
    E -->|Compilation| F[Executable]
    
    style A fill:#f9f,stroke:#333
    style F fill:#9f9,stroke:#333
```

## Implementing ROS2 Lifecycle Nodes

For detailed specifications about Lifecycle Nodes, see the [DSL Specification](./dsl_specification.md#lifecycle-nodes). This section focuses on implementation patterns and best practices.

### Implementation Patterns

1. **State Transition Handlers**
   - Implement clean separation between state transition logic
   - Use guard clauses for early returns in error cases
   - Log state transitions for debugging

```python
def on_configure_callback() -> CallbackReturn:
    """Handle configure transition."""
    try:
        if not initialize_resources():
            get_logger().error("Failed to initialize resources")
            return CallbackReturn.ERROR
        return CallbackReturn.SUCCESS
    except Exception as e:
        get_logger().error(f"Configuration failed: {str(e)}")
        return CallbackReturn.FAILURE
```

2. **Resource Management**
   - Use RAII for resource management
   - Implement proper cleanup in error cases
   - Consider using smart pointers for dynamic resources

```cpp
class ResourceManager {
public:
    ResourceManager() {
        // Initialize resources
    }
    
    ~ResourceManager() {
        // Cleanup resources
    }
    
    // Delete copy/move to prevent accidental copies
    ResourceManager(const ResourceManager&) = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;
};
```

### Best Practices

1. **Error Handling**
   - Log detailed error messages
   - Implement recovery strategies
   - Use error codes for programmatic error handling

2. **State Validation**
   - Validate state preconditions before transitions
   - Implement state guards for thread safety
   - Use state patterns for complex state machines

3. **Testing**
   - Test all state transitions
   - Simulate error conditions
   - Verify resource cleanup

Example lifecycle node definition:

```python
node sensor_driver {
    lifecycle = true
    
    on_configure = """
    RCLCPP_INFO(this->get_logger(), "Configuring...");
    // Initialize resources
    return CallbackReturnT::SUCCESS;
    """
    
    on_activate = """
    RCLCPP_INFO(this->get_logger(), "Activating...");
    // Start timers, subscribers, etc.
    return CallbackReturnT::SUCCESS;
    """
    
    // Other lifecycle callbacks...
}
```

## QoS Configuration

Quality of Service (QoS) settings in RoboDSL provide fine-grained control over communication reliability, durability, and resource usage. The QoS configuration is specified using a declarative syntax and supports both predefined profiles and custom settings.

### Predefined QoS Profiles

```python
qos_profile "sensor_data" {
    reliability = "best_effort"
    durability = "volatile"
    history = "keep_last"
    depth = 10
    deadline = "100ms"
    lifespan = "1s"

2. **Command and Control**
   - Use `reliable` for critical commands
   - Consider `transient_local` for important state updates
   - Example:
     ```python
     command_qos = {
         reliability = "reliable"
         durability = "transient_local"
         history = { kind: "keep_last", depth: 1 }
     }
     ```

3. **Parameters**
   - Always use `reliable` and `transient_local`
   - Use `keep_all` history for parameters
   - Example:
     ```python
     param_qos = {
         reliability = "reliable"
         durability = "transient_local"
         history = { kind: "keep_all" }
     }
     ```

### Best Practices

1. **Matching QoS Policies**
   - Ensure publishers and subscribers have compatible QoS settings
   - Use `rqt` or `ros2 topic info --verbose` to verify QoS compatibility
   - Test with different QoS settings to find the optimal configuration

2. **Performance Considerations**
   - Higher reliability and durability settings increase resource usage
   - Larger queue depths consume more memory
   - Consider using `best_effort` for non-critical, high-frequency data

3. **Debugging**
   - Enable ROS2 logging for QoS compatibility warnings
   - Monitor system resources when changing QoS settings
   - Use `ros2 topic hz` to verify message rates

### Applying QoS Profiles

```python
node control_node {
    publishers = [
        {
            name = "cmd_vel"
            type = "geometry_msgs/msg/Twist"
            qos = "command"  # Using predefined profile
        }
    ]
    
    subscribers = [
        {
            name = "sensor_data"
            type = "sensor_msgs/msg/Imu"
            qos = {
                reliability = "best_effort"
                history = "keep_last"
                depth = 5
            }  # Inline QoS settings
        }
    ]
}
```

## Namespace and Remapping

RoboDSL provides flexible namespace management and topic/service remapping to support complex deployment scenarios.

### Node Namespacing

```python
# Absolute namespace
node sensor_node {
    namespace = "/sensors"
    # Node name: /sensors/sensor_node
}

# Nested namespaces
with namespace = "/robot1" {
    node navigation_node {
        # Node name: /robot1/navigation_node
    }
    
    with namespace = "sensors" {
        node lidar_node {
            # Node name: /robot1/sensors/lidar_node
        }
    }
}
```

### Topic and Service Remapping

```python
node my_node {
    # Simple string-based remapping
    remappings = [
        "cmd_vel:=base/cmd_vel",
        "odom:=/sensors/odom"
    ]
    
    # Structured remapping with additional options
    remappings = [
        {
            from = "/camera/image_raw"
            to = "/sensors/camera/front/image_raw"
            qos_override = "sensor_data"
        },

## CUDA Offloading

For detailed specifications about CUDA integration, see the [DSL Specification](./dsl_specification.md#gpu-acceleration). This section covers practical implementation patterns and performance considerations.

### Kernel Implementation Patterns

1. **Memory Management**
   - Use RAII wrappers for CUDA memory
   - Implement proper error checking
   - Consider using memory pools for frequent allocations

```cpp
class CudaBuffer {
    float* d_data;
    size_t size;
    
public:
    CudaBuffer(size_t n) : size(n) {
        cudaMalloc(&d_data, n * sizeof(float));
    }
    
    ~CudaBuffer() {
        if (d_data) cudaFree(d_data);
    }
    
    // Delete copy constructor/assignment
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // Move semantics
    CudaBuffer(CudaBuffer&& other) noexcept : d_data(other.d_data), size(other.size) {
        other.d_data = nullptr;
    }
    
    float* data() const { return d_data; }
};

### Performance Optimization

1. **Asynchronous Operations**
   - Overlap computation and data transfer
   - Use CUDA streams for concurrency
   - Example:
     ```cpp
     cudaStream_t stream;
     cudaStreamCreate(&stream);
     
     // Asynchronous memory copy
     cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
     
     // Launch kernel
     vector_add_kernel<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c, n);
     
     // Asynchronous copy back
     cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);
     
     // Synchronize when needed
     cudaStreamSynchronize(stream);
     cudaStreamDestroy(stream);
     ```

2. **Optimization Techniques**
   - Use shared memory for data reuse
   - Optimize memory access patterns
   - Consider using CUDA libraries (cuBLAS, cuFFT, etc.)
   - Profile with Nsight Systems/Compute

### Debugging CUDA Code

1. **Error Checking**
   - Always check CUDA API return values
   - Use `cudaGetLastError()` after kernel launches
   - Implement proper error reporting

2. **Tools**
   - `cuda-memcheck` for memory errors
   - `nvprof` for performance profiling
   - Nsight tools for detailed analysis
   - CUDA-GDB for debugging

### Using CUDA Kernels in Nodes

```python
node image_processor {
    # ... other node configuration ...
    
    cuda_kernels = ["process_image"]
    
    initialize = """
    // Allocate GPU memory
    d_input_.create(cv::Size(640, 480), CV_8UC3);
    d_output_.create(cv::Size(640, 480), CV_8UC3);
    """
    
    process_image_callback = """
    // Upload to GPU
    d_input_.upload(input_image);
    
    // Process on GPU
    process_image(d_input_, params_, d_output_);
    
    // Download result
    d_output_.download(output_image);
    """
}
```

## Conditional Compilation

RoboDSL supports conditional compilation to handle different build configurations and platform-specific code.

### Feature Flags

```python
build_options = {
    "ENABLE_ROS2": true,  # Enable ROS2 integration
    "ENABLE_CUDA": true,  # Enable CUDA support
    "SIMULATION": false   # Simulation mode
}

# In your RoboDSL files
node my_node {
    # ...
    
    # Conditional code blocks
    code = """
    #if ENABLE_CUDA
        // CUDA-specific code
        processOnGpu();
    #else
        // Fallback CPU implementation
        processOnCpu();
    #endif
    
    #if SIMULATION
        // Simulation-specific code
        useSimulatedSensors();
    #endif
    """
}
```

### Platform-Specific Code

```python
code = """
#if defined(__linux__)
    // Linux-specific code
#elif defined(_WIN32)
    // Windows-specific code
#elif defined(__APPLE__)
    // macOS-specific code
#endif
"""
```

#### 1. Source Processing

```mermaid
flowchart LR
    A[.robodsl File] -->|Read| B[Lexer]
    B -->|Token Stream| C[Parser]
    C -->|AST| D[Semantic Analyzer]
    D -->|Validated AST| E[Intermediate Representation]
    
    style A fill:#f9f,stroke:#333
    style E fill:#bbf,stroke:#333
```

- **Lexing**: Converts raw text into tokens
- **Parsing**: Builds an Abstract Syntax Tree (AST)
- **Validation**: Performs semantic checks and type inference
- **IR Generation**: Creates an optimized intermediate representation

#### 2. Code Generation

```mermaid
flowchart LR
    A[IR] -->|Template Selection| B[Jinja2 Templates]
    B -->|Rendering| C[Source Code]
    C -->|Formatting| D[Generated Files]
    
    style A fill:#bbf,stroke:#333
    style D fill:#9f9,stroke:#333
```

- **Template Selection**: Chooses appropriate templates based on node types
- **Rendering**: Fills templates with IR data
- **Formatting**: Applies consistent code style
- **File Writing**: Outputs to the build directory

#### 3. Build Process

```mermaid
flowchart LR
    A[Generated Code] -->|CMake Configure| B[Build System]
    B -->|Compile| C[Object Files]
    C -->|Link| D[Executable/Library]
    D -->|Package| E[ROS2 Package]
    
    style A fill:#9f9,stroke:#333
    style E fill:#f9f,stroke:#333
```

- **CMake Configuration**: Sets up build rules and dependencies
- **Compilation**: Converts source to object files
- **Linking**: Combines objects into final binaries
- **Packaging**: Creates installable ROS2 packages

### Build System Integration

RoboDSL generates comprehensive CMake build files that seamlessly integrate with the ROS2 build system (ament_cmake). The build system is designed to be both powerful and flexible, supporting a wide range of build configurations.

#### Key Features

```mermaid
graph TD
    A[RoboDSL] -->|Generates| B[CMakeLists.txt]
    B --> C[ROS2 Build System]
    C --> D[ament_cmake]
    C --> E[ament_cmake_python]
    C --> F[catkin_make]
    
    style A fill:#f9f,stroke:#333
    style B fill:#9f9,stroke:#333
```

#### 1. Dependency Management

RoboDSL automatically handles dependencies through CMake's `find_package` and `ament` utilities:

```cmake
# Core Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Conditional Dependencies
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ROS2 Components
if(ENABLE_ROS2)
    find_package(rclcpp_components REQUIRED)
    find_package(lifecycle_msgs REQUIRED)
endif()
```

#### 2. Build Configuration

RoboDSL generates optimized build configurations with support for:

- **Compiler Flags**:
  ```cmake
  add_compile_options(
      $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
      $<$<CONFIG:Release>:-O3 -DNDEBUG>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  ```

- **Target Properties**:
  ```cmake
  set_target_properties(${PROJECT_NAME} PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "75;80"  # Turing and Ampere
  )
  ```

#### 3. Installation Rules

RoboDSL generates proper installation rules for ROS2 packages:

```cmake
# Install executables
install(TARGETS ${NODE_TARGETS}
    RUNTIME DESTINATION lib/${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install launch files
install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

# Install parameter files
install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

# Install Python modules
if(PYTHON_INSTALL_DIR)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${PYTHON_INSTALL_DIR}/
        DESTINATION ${PYTHON_INSTALL_DIR}/
    )
endif()
```

#### 4. Testing Infrastructure

RoboDSL sets up a comprehensive testing framework:

```cmake
if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()

    # Add GTest
    find_package(ament_cmake_gtest REQUIRED)
    
    # Add unit tests
    ament_add_gtest(${PROJECT_NAME}_test
        test/test_basic.cpp
    )
    target_link_libraries(${PROJECT_NAME}_test
        ${PROJECT_NAME}
    )
    
    # Add performance tests if CUDA is enabled
    if(ENABLE_CUDA)
        add_executable(${PROJECT_NAME}_benchmark
            benchmark/benchmark.cu
        )
        target_link_libraries(${PROJECT_NAME}_benchmark
            benchmark::benchmark
        )
    endif()
endif()
```

#### 5. Cross-Platform Support

RoboDSL generates platform-agnostic build configurations:

```cmake
# Platform-specific settings
if(WIN32)
    add_compile_definitions(NOMINMAX)
    add_compile_options(/bigobj)
elseif(UNIX AND NOT APPLE)
    add_compile_options(-fPIC)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-march=native)
    endif()
endif()

# Handle different CUDA architectures
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_ARCHITECTURES "75;80")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0")
    endif()
endif()
```

## Module Reference

### CLI Module (`src/robodsl/cli.py`)

#### Key Functions

```python
def init_project(project_name: str, template: str = "default") -> None:
    """Initialize a new RoboDSL project."""
    # Implementation...

def add_node(node_name: str, node_type: str = "basic") -> None:
    """Add a new node to the project."""
    # Implementation...

def generate_code(force: bool = False) -> None:
    """Generate code from RoboDSL files."""
    # Implementation...
```

### Parser Module (`src/robodsl/parser.py`)

#### Key Classes

```python
class ASTNode:
    """Base class for all AST nodes."""
    pass

class Parser:
    """Parses RoboDSL source files into an AST."""
    
    def parse(self, source: str) -> ASTNode:
        """Parse source code into an AST."""
        # Implementation...
```

### Generator Module (`src/robodsl/generator.py`)

#### Key Components

```python
class CodeGenerator:
    """Generates code from an AST using templates."""
    
    def generate(self, ast: ASTNode) -> Dict[str, str]:
        """Generate code files from AST."""
        # Implementation...

class TemplateManager:
    """Manages template loading and rendering."""
    
    def render(self, template_name: str, context: Dict) -> str:
        """Render a template with the given context."""
        # Implementation...
```

## Development Workflow

### Environment Setup

1. **Prerequisites**
   - Python 3.8+
   - ROS2 Humble or newer
   - CUDA Toolkit 11.0+
   - CMake 3.15+

2. **Installation**
   ```bash
   # Clone the repository
   git clone https://github.com/Zedonkay/robodsl.git
   cd robodsl
   
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

### Building from Source

```bash
# Configure the build
mkdir -p build && cd build
cmake ..

# Build the project
cmake --build . --parallel $(nproc)


# Install
cmake --install .
```

### Running Tests

```bash
# Run all tests
pytest tests/


# Run a specific test file
pytest tests/test_parser.py

# Run with coverage report
pytest --cov=robodsl tests/
```

### Debugging

1. **Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Debugging Tests**
   ```bash
   # Run pytest with Python debugger
   pytest --pdb
   ```

## Extending RoboDSL

RoboDSL is designed with extensibility in mind, allowing developers to add new node types, custom generators, and template customizations. This section provides detailed guidance on extending the framework's capabilities.

### Adding New Node Types

1. **Create Node Templates**
   - Create a new directory in `src/robodsl/templates/nodes/`
   - Add template files following the pattern `node_type.node_type`:
     - `node_type.hpp.j2`: Header file template
     - `node_type.cpp.j2`: Implementation template
     - `node_type.launch.py.j2`: Launch file template (optional)
     - `node_type.params.yaml.j2`: Parameters template (optional)

2. **Register the Node Type**
   Update `src/robodsl/generator.py` to include your new node type:
   ```python
   NODE_TYPES = {
       'custom': {
           'description': 'Custom node type',
           'templates': {
               'header': 'nodes/custom/custom.hpp.j2',
               'source': 'nodes/custom/custom.cpp.j2',
           },
           'dependencies': ['rclcpp', 'std_msgs'],
           'cuda_support': True  # Set to True if using CUDA
       }
   }
   ```

3. **Add Validation Rules**
   Extend the parser in `src/robodsl/parser.py` to validate your node's syntax:
   ```python
   def visit_CustomNode(self, node):
       self._validate_required_fields(node, ['name', 'publisher', 'subscriber'])
       # Additional validation logic
   ```

### Custom Code Generators

For advanced use cases, you can create custom code generators:

```python
from robodsl.generator import CodeGenerator
from robodsl.ast import NodeVisitor

class CustomGenerator(CodeGenerator):
    def __init__(self, template_dir=None):
        super().__init__(template_dir)
        self.visitor = CustomNodeVisitor()
    
    def generate(self, ast, output_dir):
        # Custom generation logic
        context = self.visitor.visit(ast)
        self._render_templates(context, output_dir)

class CustomNodeVisitor(NodeVisitor):
    def visit_Node(self, node):
        # Extract node information
        return {
            'node_name': node.name,
            'dependencies': self._collect_dependencies(node)
        }
```

### Template Customization

RoboDSL uses Jinja2 for templating, providing several customization options:

1. **Template Inheritance**
   ```jinja
   {# templates/nodes/custom/custom.hpp.j2 #}
   {% extends "base_node.hpp.j2" %}
   
   {% block class_definition %}
   class {{ node.name|to_pascal_case }} : public rclcpp::Node {
   public:
       {{ node.name|to_pascal_case }}(const rclcpp::NodeOptions& options);
       // Custom methods
   };
   {% endblock %}
   ```

2. **Custom Filters**
   Add custom filters in `src/robodsl/generator.py`:
   ```python
   def setup_template_engine(self):
       env = Environment(loader=FileSystemLoader(self.template_dirs))
       
       # Add custom filters
       env.filters['to_snake_case'] = lambda s: s.replace(' ', '_').lower()
       env.filters['to_camel_case'] = lambda s: ''.join(
           word.capitalize() if i > 0 else word
           for i, word in enumerate(s.replace('_', ' ').split())
       )
       
       return env
   ```

3. **Template Context Processors**
   Add custom context processors to inject additional data into templates:
   ```python
   def get_template_context(self, node):
       context = super().get_template_context(node)
       context['generation_time'] = datetime.now().isoformat()
       context['ros_version'] = self._detect_ros_version()
       return context
   ```

### Plugin System

RoboDSL supports a plugin system for extending functionality:

1. **Create a Plugin**
   ```python
   # my_plugin/__init__.py
   from robodsl.plugins import Plugin
   
   class MyPlugin(Plugin):
       def register(self):
           # Register custom node types
           self.register_node_type('custom', CustomGenerator)
           
           # Add custom template directories
           self.add_template_dir('path/to/templates')
   ```

2. **Register the Plugin**
   Create a `robodsl_plugins.py` in your project root:
   ```python
   def get_plugins():
       from my_plugin import MyPlugin
       return [MyPlugin()]
   ```

3. **Enable the Plugin**
   Add to your project's configuration file:
   ```yaml
   # .robodsl/config.yaml
   plugins:
     - my_plugin
   ```

## Performance Optimization

RoboDSL provides several mechanisms to optimize both the code generation process and the generated code's runtime performance. This section covers best practices and techniques for achieving optimal performance.

### Build-Time Optimization

1. **Compiler Flags**
   - Enable link-time optimization (LTO)
   - Use profile-guided optimization (PGO)
   - Set appropriate architecture flags

   ```cmake
   # In generated CMakeLists.txt
   if(CMAKE_BUILD_TYPE STREQUAL "Release")
       add_compile_options(
           "$<$<CONFIG:RELEASE>:-O3 -march=native -flto -DNDEBUG>"
       )
       set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
   endif()
   ```

2. **Template Caching**
   - Enable template caching to avoid recompiling unchanged templates
   - Use in-memory template caching for repeated generations

   ```python
   # In generator.py
   def __init__(self):
       self.env = Environment(
           loader=FileSystemLoader(template_dirs),
           cache_size=1000,  # Cache up to 1000 templates
           auto_reload=False  # Disable auto-reload in production
       )
   ```

### Runtime Performance

1. **Zero-Copy Data Transfer**
   - Use `std::shared_ptr` for message passing
   - Leverage ROS2's intra-process communication
   - Implement move semantics where appropriate

   ```cpp
   // In generated node implementation
   void process_image(const sensor_msgs::msg::Image::SharedPtr msg) {
       // Process image without copying
       cv::Mat cv_image = cv_bridge::toCvShare(msg, "bgr8")->image;
       // ...
   }
   ```

2. **Memory Pooling**
   - Pre-allocate memory for real-time critical paths
   - Use object pools for frequently allocated/deallocated objects
   - Implement custom allocators for ROS2 messages

   ```cpp
   // Custom allocator example
   using ImageAllocator = rclcpp::message_memory_strategy::MessagePoolAllocator<sensor_msgs::msg::Image>;
   auto image_pool = std::make_shared<rclcpp::message_memory_strategy::MessagePool<sensor_msgs::msg::Image>>(10);
   ```

3. **Threading Model**
   - Configure ROS2 executor for optimal performance
   - Use callback groups to isolate real-time callbacks
   - Consider using the `rclcpp::executors::StaticSingleThreadedExecutor` for low-latency applications

   ```cpp
   // In generated node implementation
   rclcpp::executor::ExecutorArgs args;
   args.context = context;
   auto executor = std::make_shared<rclcpp::executors::StaticSingleThreadedExecutor>(args);
   executor->add_node(node);
   executor->spin();
   ```

### CUDA-Specific Optimizations

1. **Stream Management**
   - Use multiple CUDA streams for concurrent kernel execution
   - Overlap computation and data transfer
   - Implement asynchronous memory operations

   ```cpp
   // In CUDA-accelerated node
   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   
   // Process different data in parallel
   process_kernel1<<<blocks, threads, 0, stream1>>>(d_data1);
   process_kernel2<<<blocks, threads, 0, stream2>>>(d_data2);
   ```

2. **Unified Memory**
   - Use CUDA managed memory for simplified memory management
   - Prefer `cudaMallocManaged` for data accessed by both CPU and GPU
   - Use `cudaMemPrefetchAsync` to optimize data location

   ```cpp
   float* data;
   cudaMallocManaged(&data, size * sizeof(float));
   
   // Prefetch to GPU
   cudaMemPrefetchAsync(data, size * sizeof(float), device_id, stream);
   ```

3. **Kernel Optimization**
   - Optimize block and grid dimensions
   - Use shared memory for frequently accessed data
   - Minimize thread divergence

   ```cpp
   __global__ void optimized_kernel(float* input, float* output, int width) {
       __shared__ float tile[TILE_SIZE][TILE_SIZE];
       // ...
   }
   ```

### Memory Management

1. **RAII for Resource Management**
   - Use smart pointers for automatic resource cleanup
   - Implement custom deleters for CUDA resources
   - Leverage move semantics for efficient resource transfer

   ```cpp
   struct CudaDeleter {
       void operator()(void* ptr) const { cudaFree(ptr); }
   };
   
   std::unique_ptr<float, CudaDeleter> d_data(static_cast<float*>(cuda_malloc(size)));
   ```

2. **Memory Pooling**
   - Implement custom memory pools for ROS2 messages
   - Reuse message objects when possible
   - Monitor memory usage with custom allocators

   ```cpp
   // Custom message pool
   class MessagePool {
   public:
       MessagePool(size_t initial_size) { /* ... */ }
       std::shared_ptr<sensor_msgs::msg::Image> acquire() { /* ... */ }
       void release(const std::shared_ptr<sensor_msgs::msg::Image>& msg) { /* ... */ }
   };
   ```

3. **Memory Analysis**
   - Use tools like Valgrind and AddressSanitizer
   - Monitor memory usage with ROS2's memory tools
   - Implement custom memory tracking for debugging

   ```bash
   # Run with memory checking
   valgrind --tool=massif --stacks=yes ros2 run package node
   ```

### Profiling and Optimization

1. **CPU Profiling**
   - Use `perf` for low-overhead profiling
   - Generate flame graphs for visualization
   - Profile both wall time and CPU time

   ```bash
   perf record -F 99 -g --call-graph dwarf ros2 run package node
   perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
   ```

2. **GPU Profiling**
   - Use NVIDIA Nsight Systems for timeline analysis
   - Profile CUDA kernels with `nvprof`
   - Analyze memory access patterns

   ```bash
   nsys profile --stats=true ros2 run package node
   ```

3. **ROS2-Specific Tools**
   - Use `ros2 topic hz` to monitor message rates
   - Profile with `ros2 trace` for system-wide analysis
   - Use `rqt` for runtime visualization

   ```bash
   ros2 trace --duration 10 -s tracing_session
   ```

### Best Practices

1. **Minimize Dynamic Memory Allocation**
   - Pre-allocate memory during initialization
   - Use fixed-size containers when possible
   - Avoid memory fragmentation

2. **Optimize Message Passing**
   - Use `std::unique_ptr` for exclusive ownership
   - Implement zero-copy where possible
   - Consider using ROS2's intra-process communication

3. **Profile Before Optimizing**
   - Identify actual bottlenecks before optimization
   - Measure performance impact of changes
   - Use A/B testing for optimization validation

## Troubleshooting

### Common Issues

1. **CUDA Compilation Errors**
   - Verify CUDA toolkit installation
   - Check compute capability compatibility
   - Review kernel launch parameters

2. **ROS2 Integration**
   - Source ROS2 setup files
   - Verify package dependencies
   - Check topic and service names

### Debugging Tips

1. **Verbose Output**
   ```bash
   robodsl --verbose generate
   ```

2. **Debug Symbols**
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   ```

### Performance Profiling

1. **CPU Profiling**
   ```bash
   perf record -g ./your_node
   perf report
   ```

2. **GPU Profiling**
   ```bash
   nvprof ./your_node
   ```

## Contributing

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Commit changes with descriptive messages
4. Push to the branch
5. Create a pull request

### Code Review Guidelines

- Follow the existing code style
- Include unit tests
- Update documentation
- Keep commits atomic

### Release Process

1. Update version in `pyproject.toml`
2. Update changelog
3. Create a release tag
4. Build and publish packages

## Additional Resources

- [ROS2 Documentation](https://docs.ros.org/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Python Packaging Guide](https://packaging.python.org/)
- [CMake Documentation](https://cmake.org/documentation/)
- [Jinja2 Template Documentation](https://jinja.palletsprojects.com/)

## Architecture

RoboDSL follows a modular architecture designed for extensibility and maintainability. The system is composed of several core components that work together to provide a seamless development experience for GPU-accelerated robotics applications.

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

#### Practical Examples

This section provides practical examples of RoboDSL usage patterns and best practices. For complete syntax reference, see the [DSL Specification](./dsl_specification.md).

### 1. Basic Node with Lifecycle Support

```python
# my_robot.robodsl

# Define a lifecycle node
node my_robot_node {
    # Configure node as a lifecycle node
    lifecycle = true
    
    # Add parameters
    parameters = [
        { name: "max_speed", type: "double", default: 1.0 },
        { name: "sensor_timeout", type: "double", default: 1.0 }
    ]
    
    # Define publishers and subscribers with QoS profiles
    publishers = [
        {
            topic: "/robot/status",
            type: "std_msgs/msg/String",
            qos: "sensor_data"  # Reference to a QoS profile
        }
    ]
    
    subscribers = [
        {
            topic: "/robot/command",
            type: "std_msgs/msg/String",
            callback: "command_callback",
            qos: "command"
        }
    ]
    
    # Define lifecycle callbacks
    callbacks = [
        { name: "on_configure", impl: "on_configure_callback" },
        { name: "on_activate", impl: "on_activate_callback" },
        { name: "on_deactivate", impl: "on_deactivate_callback" },
        { name: "on_cleanup", impl: "on_cleanup_callback" },
        { name: "on_shutdown", impl: "on_shutdown_callback" }
    ]
}

# Define custom QoS profiles
qos_profile "sensor_data" {
    reliability = "best_effort"
    history = { kind: "keep_last", depth: 10 }
    deadline = "100ms"
}

qos_profile "command" {
    reliability = "reliable"
    durability = "transient_local"
    history = { kind: "keep_last", depth: 1 }
}
```

### 2. CUDA-Accelerated Image Processing

```python
# image_processor.robodsl

# Enable CUDA support
cuda = {
    enabled = true
    compute_capability = "7.5"
}

# Define a CUDA kernel
cuda_kernel image_processor {
    inputs = [
        { name: "input", type: "const cv::cuda::GpuMat&" },
        { name: "params", type: "const ImageParams&" }
    ]
    outputs = [
        { name: "output", type: "cv::cuda::GpuMat&" }
    ]
    
    includes = [
        "opencv2/core/cuda.hpp",
        "opencv2/cudaimgproc.hpp"
    ]
    
    block_size = [32, 32, 1]
    
    code = """
    __global__ void process_image_kernel(
        const cv::cuda::PtrStepSz<uchar3> src,
        cv::cuda::PtrStepSz<uchar3> dst,
        const ImageParams* params) {
        
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x >= src.cols || y >= src.rows) return;
        
        // Process pixel at (x,y)
        uchar3 pixel = src(y, x);
        // ... processing ...
        dst(y, x) = pixel;
    }
    """
}

# Define the node that uses the CUDA kernel
node image_processor_node {
    # Enable CUDA support
    cuda_kernels = ["image_processor"]
    
    # Add necessary dependencies
    dependencies = ["OpenCV"]
    
    # Define parameters
    parameters = [
        { name: "threshold", type: "double", default: 0.5 },
        { name: "blur_size", type: "int", default: 5 }
    ]
    
    # Define publishers and subscribers
    publishers = [
        {
            topic: "/processed_image",
            type: "sensor_msgs/msg/Image",
            qos: "sensor_data"
        }
    ]
    
    subscribers = [
        {
            topic: "/camera/image_raw",
            type: "sensor_msgs/msg/Image",
            callback: "image_callback"
        }
    ]
}
```

### 3. Building and Running

```bash
# Build the project
robodsl build

# Source the setup files
source install/setup.bash

# Run a node
ros2 run my_package my_robot_node

# Or use the launch file
ros2 launch my_package my_robot.launch.py
```

### 4. Debugging and Monitoring

```bash
# View node graph
rqt_graph

# Monitor topics
ros2 topic list
ros2 topic echo /robot/status

# View parameters
ros2 param list
ros2 param get /my_robot_node max_speed

# Profile performance
ros2 run --prefix 'perf record -g' my_package my_robot_node
```

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
