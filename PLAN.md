# RoboDSL Project Plan

## Project Overview
RoboDSL is a Domain-Specific Language (DSL) and compiler designed to simplify the development of GPU-accelerated robotics applications using ROS2 and CUDA. The goal is to abstract away the complexity of ROS2 and CUDA integration, allowing developers to focus on their application logic.

## Goals
1. **Simplify Development**: Provide a clean, declarative syntax for defining robotics applications
2. **Automate Boilerplate**: Generate ROS2 nodes, message definitions, and CUDA kernels
3. **Streamline Build Process**: Handle CMake configuration and build system generation
4. **Improve Performance**: Facilitate efficient GPU-CPU communication patterns

## Current Status
- [x] Project setup and CLI scaffolding
- [x] Basic DSL parser implementation
- [ ] ROS2 node generation
- [ ] CUDA kernel management
- [ ] Build system integration
- [ ] Documentation and examples

## Roadmap

### Phase 1: Core Functionality (MVP)
- [x] Define DSL syntax for nodes and CUDA kernels
- [x] Implement parser for the DSL
- [ ] Generate ROS2 node templates
  - [ ] Basic node structure
  - [ ] Publisher/Subscriber generation
  - [ ] Service and Action server/client support
- [ ] CUDA kernel generation
  - [ ] Kernel template generation
  - [ ] Memory management helpers
  - [ ] CPU-GPU bridge code

### Phase 2: Build System & Integration
- [ ] CMake generator
  - [ ] ROS2 package generation
  - [ ] CUDA compilation flags
  - [ ] Dependency management
- [ ] Project scaffolding
  - [ ] New project creation
  - [ ] Adding components to existing projects

### Phase 3: Advanced Features
- [ ] ROS2 parameter support
- [ ] Launch file generation
- [ ] Debugging and profiling tools
- [ ] ROS2 message generation from DSL

## Usage Examples

### Create a new project
```bash
robodsl new my_robot_project
```

### Add a node to an existing project
```bash
cd existing_project
robodsl add-node image_processor --publisher /processed_image sensor_msgs/msg/Image
```

### Add a CUDA kernel
```bash
robodsl add-kernel image_filter --input Image --output Image
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
MIT
