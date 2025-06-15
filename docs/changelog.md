# Changelog

All notable changes to the RoboDSL project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- ROS2 Lifecycle node support with full lifecycle management
- QoS configuration for publishers and subscribers
- Namespace and remapping support for nodes
- CUDA offloading in action handlers
- Documentation for conditional compilation with ENABLE_ROS2/ENABLE_CUDA
- Comprehensive developer guide
- Expanded DSL specification document
- Enhanced examples with ROS2 and CUDA integration
- Detailed contributing guidelines
- Modernized code of conduct

### Changed
- Updated build system to support conditional compilation
- Improved error handling and validation in code generator
- Enhanced documentation structure and coverage
- Refactored example projects for better clarity
- Updated development dependencies and tooling

### Fixed
- Resolved issues with message type resolution
- Fixed template generation for ROS2 interfaces
- Addressed build warnings and improved cross-platform compatibility

## [0.2.0] - 2025-06-21

### Added
- Support for ROS2 parameters with runtime reconfiguration
- Action server/client generation
- QoS profile configuration for all communication primitives
- Namespace and remapping support
- Comprehensive documentation structure
- Example projects for common robotics patterns
- Developer guide with architecture overview
- Contribution guidelines and code of conduct

## [0.1.0] - 2025-06-14

### Added
- Initial release of RoboDSL
- Basic support for ROS2 node generation
- CUDA kernel generation with Thrust support
- CMake build system integration
- Example projects and documentation

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/yourusername/robodsl/tags).

## Authors

- Your Name - Initial work - [YourUsername](https://github.com/yourusername)

See also the list of [contributors](https://github.com/yourusername/robodsl/contributors) who participated in this project.
