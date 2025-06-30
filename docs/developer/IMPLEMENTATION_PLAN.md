# RoboDSL & CUIF Implementation Plan

## Progress Summary (as of 06-27-2025)
- **Phases 0-4**: ✅ Complete (Foundation, Lark Parser, Enhanced Features, ONNX Integration, Pipeline System)
- **Repository Reorganization**: ✅ Complete - Restructured codebase for better maintainability
- **Grammar & Parser**: ✅ Complete - Fixed all reduce/reduce conflicts and AST builder issues
- **Test Coverage**: 113/114 tests pass successfully
- **VSCode Integration**: ✅ Complete - Full syntax highlighting and language support

### Current Status
- ✅ All legacy parser code removed; only Lark-based AST remains
- ✅ All tests pass, including pipelines and ONNX features
- ✅ CLI and code generation fully functional
- ✅ Repository fully reorganized and up to date
- ✅ Pipeline system is implemented and robust

### Remaining Work
- **Phase 5**: AI-Powered Development Tools (Not started)
- **Phase 6**: Real-time Constraints (Not started)
- **Phase 7**: Training Integration (Not started)
- **Phase 8**: Advanced Features (Not started)

---

## Overview
This document outlines the implementation plan for RoboDSL and CUIF, combining the original RoboDSL roadmap with new CUIF DSL features. The plan maintains all existing functionality while adding powerful new development tools.

## Target State
- Lark-based grammar parser
- C++ method integration within nodes
- ONNX model integration
- Pipeline system with branching
- Real-time constraints
- Enhanced VSCode tooling:
  - Inline completions
  - Hover documentation
  - AI-assisted code generation
  - RAG-based contextual help
- Visualization and documentation generation
- Training integration

## Implementation Phases

### Phase 0: Foundation & VSCode Integration ✅ COMPLETE
**Priority**: 0 (Parallel Track)  
**Estimated Time**: 2-3 weeks  
**Goal**: Establish core RoboDSL functionality with VSCode integration

#### Completed Tasks
- [x] Project setup and single-source CLI (Click)
- [x] Advanced DSL parser implementation with ROS2 features
- [x] ROS2 node generation with lifecycle support
- [x] CUDA kernel management with Thrust integration
- [x] Build system integration with CMake
- [x] Comprehensive documentation
- [x] ROS2 features implementation
- [x] Comprehensive test coverage (113/114 tests pass)
- [x] VSCode extension with syntax highlighting and language support

#### Files Created
- VSCode Extension: `package.json`, `syntaxes/cuif.tmLanguage.json`, `language-configuration.json`, `src/extension.ts`
- Core Functionality: `src/robodsl/grammar/robodsl.lark`, `src/robodsl/parser/ast_builder.py`, `src/robodsl/generators/`, `templates/`

---

### Phase 1: Foundation - Lark Parser Migration ✅ COMPLETE
**Priority**: 1  
**Estimated Time**: 2-3 days  
**Goal**: Replace regex parsing with proper grammar-based parsing

#### Completed Tasks
- [x] Design Core Grammar (Lark) - Basic syntax, structured QoS format, comments, value types
- [x] Migrate Existing Parser - Convert regex-based parsing to Lark grammar
- [x] CUDA Kernel Grammar - Formalize CUDA kernel syntax and structured parameter definitions

#### Files Created/Modified
- `src/robodsl/grammar/robodsl.lark` - Main grammar file
- `src/robodsl/parser/ast_builder.py` - Build AST from Lark parse tree
- `src/robodsl/parser/semantic_analyzer.py` - Validate semantics
- Updated `src/robodsl/parser.py` to use new Lark parser

---

### Phase 2: Enhanced Compiler Features ✅ COMPLETE
**Priority**: 1 (Parallel Track)  
**Estimated Time**: 3-4 weeks  
**Goal**: Build upon the existing Lark-based parser to add essential compiler features

#### Completed Tasks
1. **Semantic Analysis**
   - [x] Symbol table implementation for tracking variables and types
   - [x] Type checking for ROS message types and C++ types
   - [x] Name resolution for node components (publishers, subscribers, etc.)
   - [x] Validation of QoS configurations
   - [x] Cross-reference checking for remapped topics

2. **Code Generation Improvements**
   - [x] Enhanced error handling in generated code
   - [x] Support for C++ method generation in nodes
   - [x] Template specialization for different ROS2 DDS implementations
   - [x] Generation of component lifecycle documentation

3. **Build System Integration**
   - [x] Support for custom CMake modules
   - [x] Dependency tracking for generated code
   - [x] Incremental build support
   - [x] Cross-compilation toolchain configuration

4. **Development Tools**
   - [x] VSCode extension with syntax highlighting, code completion, hover documentation
   - [x] Validation & linting with semantic validation and performance anti-pattern detection

5. **Testing Infrastructure**
   - [x] Unit testing for parser and semantic analysis
   - [x] Integration testing for end-to-end compilation

---

### Phase 3: ONNX Integration ✅ COMPLETE
**Priority**: 3  
**Estimated Time**: 4-5 days  
**Goal**: Seamless ONNX model integration

#### Completed Tasks
1. **ONNX Model Grammar** ✅
   ```lark
   onnx_model: "onnx_model" STRING "{" model_config "}"
   model_config: (input_def | output_def | device | optimization)*
   input_def: "input" ":" STRING "->" TYPE
   output_def: "output" ":" STRING "->" TYPE
   device: "device" ":" ("cpu" | "cuda")
   optimization: "optimization" ":" ("tensorrt" | "openvino")
   ```

2. **Code Generation** ✅
   - [x] Generate ONNX Runtime integration code
   - [x] Handle input/output tensor management
   - [x] Memory management and optimization

3. **Integration with Nodes** ✅
   - [x] Allow nodes to reference ONNX models
   - [x] Generate inference methods automatically

#### Files Created/Modified ✅
- [x] Updated `src/robodsl/grammar/robodsl.lark` to include ONNX grammar
- [x] `src/robodsl/generators/onnx_integration.py` - ONNX code generation
- [x] `src/robodsl/templates/onnx/` - ONNX integration templates
- [x] Updated node generators to handle ONNX models
- [x] `src/robodsl/core/ast.py` - Added ONNX AST nodes
- [x] `tests/test_onnx_integration.py` - Comprehensive ONNX tests

#### Implementation Details
- **AST Nodes Added**: `OnnxModelNode`, `ModelConfigNode`, `InputDefNode`, `OutputDefNode`, `DeviceNode`, `OptimizationNode`
- **Grammar Support**: Full ONNX model definition syntax with input/output specifications
- **Code Generation**: Complete ONNX Runtime integration with proper memory management
- **Template System**: Comprehensive templates for ONNX integration code
- **Testing**: Full test coverage including model parsing, code generation, and integration scenarios

---

### Phase 4: Simple Pipeline System ✅ COMPLETE
**Priority**: 4  
**Estimated Time**: 5-6 days  
**Goal**: Basic data flow pipelines

#### Completed Tasks
1. **Pipeline Grammar**
   ```lark
   pipeline: "pipeline" NAME "{" stage* "}"
   stage: "stage" NAME "{" stage_config "}"
   stage_config: (input | output | method | model | topic)*
   ```

2. **Pipeline Generation**
   - [x] Generate ROS 2 nodes for each stage
   - [x] Create topic connections between stages
   - [x] Handle data flow between stages

3. **Basic Branching**
   - [x] Conditional stages (if/else)
   - [x] Simple routing logic

#### Files Created/Modified
- Updated `src/robodsl/grammar/robodsl.lark` to include pipeline grammar
- `src/robodsl/generators/pipeline.py` - Pipeline generation
- `src/robodsl/templates/pipeline/` - Pipeline templates
- Updated main generator to handle pipelines

---
---

### Phase 5: Cloud-Hosted Development Environment  
**Priority**: 4  
**Estimated Time**: 7–8 days  
**Goal**: Build a cloud-hosted IDE and development platform that enables users to write, test, and deploy RoboDSL projects using CUDA + ROS2 without local hardware or OS constraints.

#### Tasks  
1. **Cloud IDE / Web Dashboard**
   - Develop a web-based IDE (React/Next.js or similar) integrated with:
     - Syntax highlighting and code completion for `.robodsl`
     - Interactive terminals connected to cloud compute nodes
     - File management and versioning
   - Optionally build a VSCode extension that connects to the cloud backend for remote development

2. **Remote Development Backend**
   - Provision GPU-enabled Linux VMs or containers on-demand (NVIDIA GPUs)
   - Preinstall ROS2 + CUDA + RoboDSL tooling in the environment
   - Provide APIs for:
     - File sync & storage (e.g., persistent volumes or cloud buckets)
     - Remote code compilation & pipeline generation
     - Remote build and launch of simulations or CUDA binaries
     - Real-time logs and performance metrics streaming

3. **Authentication & Multi-User Support**
   - User accounts and workspace isolation
   - Role-based access control for teams
   - Secure SSH/websocket tunnels for remote terminals

4. **Resource Management & Scaling**
   - Autoscaling GPU resources based on demand
   - Usage monitoring and billing integration (optional)
   - Container or VM lifecycle management for ephemeral dev environments

5. **Integration with RoboDSL CLI**
   - Extend `cuif` CLI to connect and interact with cloud dev platform
   - Commands for uploading code, triggering builds, running tests remotely
   - Fetching logs, artifacts, and telemetry from cloud sessions

#### Files to Create/Modify
- `src/web/` — Web frontend code for IDE/dashboard  
- `src/backend/` — Backend APIs for remote file management, build, run  
- `src/robodsl/cli/cloud_ext.py` — Extended CLI commands for remote dev  
- Dockerfiles and container images preconfigured with CUDA + ROS2 + RoboDSL tools

#### Deliverables  
- Fully functional cloud IDE or remote VSCode development environment  
- Remote compute infrastructure with pre-installed CUDA + ROS2  
- Transparent developer experience: write and test RoboDSL code without local CUDA/Linux  
- Scalable resource provisioning and user management

#### Success Criteria  
- Users can write RoboDSL code on Mac/Windows browsers without local CUDA or Linux  
- Compile, build, and run pipelines on powerful cloud GPUs remotely  
- Real-time interaction with remote terminals, logs, and outputs  
- Seamless sync between cloud IDE and local RoboDSL CLI tooling  

---


### Phase 5: AI-Powered Development Tools
**Priority**: 5  
**Estimated Time**: 5-6 days  
**Goal**: Integrate AI-assisted development features

#### Tasks
1. **AI Code Generation**
   - Implement `cuif.generateFromPrompt` command
   - Create input box for natural language descriptions
   - Integrate with GPT-4 for code generation
   - Add output preview and insertion

2. **Retrieval-Augmented Generation (RAG)**
   - Set up local vector DB (ChromaDB/FAISS)
   - Create document corpus (grammar, examples, docs)
   - Implement semantic search for relevant documentation
   - Integrate with LLM for contextual help

3. **AI-Powered Features**
   - `cuif explain`: Document generation from code
   - `cuif lint`: Style and structure suggestions
   - `cuif doc`: Auto-generate API documentation

#### Files to Create/Modify
- `src/commands/aiGenerate.ts`
- `src/services/ragService.ts`
- `src/commands/explainCommand.ts`
- `src/commands/lintCommand.ts`
- `src/commands/docCommand.ts`

#### Deliverables
- AI-powered code generation from natural language
- Contextual help using RAG
- Documentation generation tools
- Code quality suggestions

---

### Phase 6: Real-time Constraints
**Priority**: 5  
**Estimated Time**: 4-5 days  
**Goal**: Real-time guarantees and monitoring

#### Tasks
1. **Real-time Grammar**
   ```lark
   realtime_config: "realtime" "{" rt_settings "}"
   rt_settings: (priority | deadline | cpu_affinity | memory_policy)*
   priority: "priority" ":" NUMBER
   deadline: "deadline" ":" TIME
   ```

2. **Code Generation**
   - Generate real-time thread configurations
   - Deadline monitoring code
   - Priority scheduling setup

3. **Runtime Monitoring**
   - Generate performance monitoring code
   - Deadline violation detection
   - Resource usage tracking

#### Files to Create/Modify
- Update `src/robodsl/grammar/robodsl.lark` to include real-time grammar
- `src/robodsl/generators/realtime.py` - Real-time code generation
- `src/robodsl/templates/realtime/` - Real-time templates

#### Deliverables
- Real-time constraint parsing and generation
- Performance monitoring integration
- Example real-time pipeline

#### Success Criteria
- Real-time constraints can be defined
- Generated code includes monitoring
- Performance metrics are tracked

---

### Phase 7: Training Integration
**Priority**: 7  
**Estimated Time**: 3-4 days  
**Goal**: Framework-agnostic training configuration

#### Tasks
1. **Training Grammar**
   ```lark
   training_config: "training" NAME "{" train_settings "}"
   train_settings: (dataset | model | epochs | optimizer | export)*
   ```

2. **Code Generation**
   - Generate training scripts
   - Handle model export to ONNX
   - Integration with existing pipeline

#### Files to Create/Modify
- Update `src/robodsl/grammar/robodsl.lark` to include training grammar
- `src/robodsl/generators/training.py` - Training script generation
- `src/robodsl/templates/training/` - Training templates

#### Deliverables
- Training configuration parsing
- PyTorch training script generation
- ONNX export integration

#### Success Criteria
- Training configurations can be defined
- Training scripts are generated automatically
- Models export to ONNX format

---

### Phase 8: Advanced Features
**Priority**: 8  
**Estimated Time**: 5-7 days  
**Goal**: Enhanced functionality

#### Tasks
1. **Visualization**
   - Parse pipeline structure into graph representation
   - Generate DOT files for Graphviz
   - Create Mermaid diagrams
   - Auto-generate pipeline documentation

2. **Hardware Abstraction**
   - Minimal hardware configuration
   - Device-specific optimizations

3. **Docker Integration**
   - Container generation
   - Multi-stage builds

4. **Advanced Pipeline Features**
   - Loops and complex branching
   - Error handling and recovery
   - Dynamic pipeline modification

#### Files to Create/Modify
- Update grammar for advanced features
- `src/robodsl/generators/visualization.py` - Diagram generation
- `src/robodsl/generators/docker.py` - Docker generation
- `src/robodsl/generators/hardware.py` - Hardware abstraction

#### Deliverables
- Pipeline visualization tools
- Docker container generation
- Hardware abstraction layer
- Advanced pipeline features

#### Success Criteria
- Pipelines can be visualized as diagrams
- Containers can be generated automatically
- Hardware configurations are abstracted
- Complex pipelines work correctly

---

## File Structure

### Current Directory Structure
```
src/robodsl/
├── grammar/
│   └── robodsl.lark          # Main grammar file
├── parser/
│   ├── __init__.py
│   ├── ast_builder.py        # Build AST from Lark parse tree
│   └── semantic_analyzer.py  # Validate semantics
├── generators/
│   ├── base_generator.py     # Base generator class
│   ├── cmake_generator.py    # CMake generation
│   ├── cpp_node_generator.py # C++ node generation
│   ├── cuda_kernel_generator.py # CUDA kernel generation
│   ├── launch_generator.py   # Launch file generation
│   ├── main_generator.py     # Main generation orchestration
│   ├── onnx_integration.py   # ONNX code generation
│   ├── package_generator.py  # Package generation
│   ├── pipeline_generator.py # Pipeline generation
│   └── python_node_generator.py # Python node generation
└── templates/
    ├── cmake/                # CMake templates
    ├── cpp/                  # C++ node templates
    ├── cuda/                 # CUDA kernel templates
    ├── launch/               # Launch file templates
    ├── onnx/                 # ONNX integration templates
    ├── pipeline/             # Pipeline templates
    └── py/                   # Python node templates
```

### Future Additions
```
src/robodsl/
├── generators/
│   ├── realtime.py           # Real-time code generation
│   ├── visualization.py      # Diagram generation
│   ├── training.py           # Training script generation
│   ├── docker.py             # Docker generation
│   └── hardware.py           # Hardware abstraction
└── templates/
    ├── realtime/             # Real-time templates
    ├── visualization/        # Visualization templates
    ├── training/             # Training templates
    └── docker/               # Docker templates
```

## Testing Strategy

### Test Organization
```
tests/
├── conftest.py               # Test configuration
├── test_cli.py               # CLI tests
├── test_grammar.py           # Grammar validation tests
├── test_parser.py            # Parser tests
├── test_generator.py         # Generator tests
├── test_onnx_integration.py  # ONNX integration tests
├── test_pipeline_generator.py # Pipeline tests
└── examples/                 # Example-based tests
```

### Test Coverage
- **Grammar Tests**: Validate syntax parsing
- **Integration Tests**: End-to-end code generation
- **Performance Tests**: Real-time constraint validation
- **Documentation Tests**: Ensure docs are generated correctly

## Documentation Strategy

### Documentation Organization
```
docs/
├── developer/                # Developer documentation
│   ├── code-of-conduct.md
│   ├── contributing.md
│   ├── IMPLEMENTATION_PLAN.md
│   ├── PLAN.md
│   └── REALTIME_GUIDE.md
├── user-guide/               # User documentation
└── website/                  # Web documentation
    ├── css/
    ├── js/
    ├── images/
    └── *.html
```

## Key Decisions Made

### Technology Choices
- **Lark** for parsing (modern, Python-native)
- **ONNX Runtime** for inference (framework-agnostic)
- **Graphviz/Mermaid** for visualization
- **PyTorch** for training (easiest integration)

### Design Principles
1. **No backward compatibility** - Clean break from old syntax
2. **Inline C++ methods** - Methods defined within nodes
3. **ONNX-first approach** - Start with ONNX, expand later
4. **Simple pipelines first** - Add complexity incrementally
5. **Real-time constraints** - Support for real-time systems
6. **Visualization** - Auto-generated diagrams and docs

## Success Metrics

### Overall Success Criteria
- [x] Language supports all planned features (Phases 0-4)
- [x] Documentation is comprehensive
- [x] Tests provide good coverage (113/114 tests pass)
- [x] Examples demonstrate all capabilities
- [ ] Cloud Comuting support
- [ ] AI-powered development tools integrated
- [ ] Real-time constraints implemented
- [ ] Training integration complete
- [ ] Advanced features implemented

## Getting Started

To continue implementation:

1. **Start with Phase 5**: AI-Powered Development Tools
2. **Set up development environment**: Install AI/ML dependencies
3. **Implement incrementally**: One phase at a time
4. **Document as you go**: Keep documentation updated
5. **Test thoroughly**: Add tests for each feature

This plan provides a clear roadmap for completing the remaining phases of RoboDSL development while maintaining the ability to pick up development at any point. 