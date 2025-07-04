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
## Phase 5 — Cloud + Local GPU Development Integration

**Priority**: 5  
**Estimated Time**: 5–7 days  
**Goal**: Provide GPU-enabled development environments via the cloud or local GPU servers, tightly integrated with RoboDSL's CLI and editor tooling.

### 🧩 Motivation

Robotics and CUDA development typically require:
- Linux OS
- NVIDIA GPU
- ROS 2 and CUDA setup

Most developers do not have this locally. This phase makes it possible to:
- Write and test CUDA/ROS2 code without owning a Linux machine or NVIDIA GPU
- Provide a one-click cloud-based developer environment
- Allow local teams (e.g., CMU Racing) to expose their shared GPU machine for remote access


### ✅ Deliverables

- Hosted cloud development environment (via AWS/GCP with GPU instance)
- VSCode extension or Web UI for editing and deploying `.robodsl` files
- Remote execution backend with Docker/Podman and ROS 2
- Local GPU server support with GUI access via browser
- Developer API for compiling, deploying, running, and visualizing output


### 🚀 Cloud Server Architecture (Option 1)

Host development servers on AWS/GCP with GPU-backed VM instances (e.g., `g4dn.xlarge` or `a2-highgpu-1g`):

#### Features
- JupyterLab + VSCode Web UI via reverse proxy (e.g., NGINX)
- Docker containers with preinstalled:
  - ROS 2 Humble/Foxy
  - CUDA Toolkit
  - RoboDSL CLI and templates
- GPU acceleration via `nvidia-docker`
- RoboDSL CLI integrated into notebook + terminal

#### User Workflow
1. `robodsl cloud connect` opens VSCode or browser
2. Developer edits `.robodsl` file
3. RoboDSL auto-generates code
4. User builds and runs via Docker container
5. Simulators and visualizers are streamed via noVNC or WebRTC

#### AWS/GCP Support
- Provide helper script: `robodsl cloud deploy --provider aws`
- Auto-deploy:
  - VM with GPU
  - Docker container
  - Reverse proxy and authentication
- Add `robodsl cloud start/stop/status` to manage VM lifecycle
- Usage-based cost control (e.g., $0.40/hr when running, $0 when off)


### 🖥️ Local GPU Server Support (Option 2)

Allow organizations (e.g., CMU Racing) to expose a Linux GPU server on the local network:

#### Features
- Headless containerized development
- Optional GUI via noVNC (e.g., Gazebo, RViz)
- Developer connects via:
  - `robodsl local connect` → opens local VSCode with remote container
  - Or browser-based GUI
- Sandbox each user session for isolation

#### Deployment
- Install RoboDSL runtime on server: `robodsl host init`
- Start daemon: `robodsl host serve --gui`
- Developers connect via:
  - VSCode Remote SSH
  - Web IDE (e.g., code-server)
- Use `tmate` or `libvncserver` for collaborative development

### 🧪 Tasks

   #### Cloud Support
   - [x] Create `robodsl cloud deploy`, `start`, `stop`, `connect` CLI
   - [x] Terraform + Docker setup for AWS/GCP GPU VMs
   - [x] Install CUDA, ROS2, RoboDSL toolchain on startup
   - [x] Add VSCode Web + Jupyter + Terminal support
   - [x] Add GPU availability check + usage monitoring

   #### Local Server Support
   - [x] `robodsl host init` to set up server
   - [x] Launch `code-server` or VSCode Remote
   - [x] Integrate graphical support via X11/VNC/WebRTC
   - [x] Add auth + sandboxing for multiple users



### ✅ Success Criteria

- [ ] Users can connect to a GPU-hosted ROS2 + CUDA dev environment
- [ ] They can write `.robodsl` files and test nodes in-browser or with VSCode
- [ ] Cloud servers can be started/stopped from the CLI
- [ ] Local GPU servers can run sessions with GUI (e.g., RViz/Gazebo)
- [ ] Sessions are isolated per user

### 🛠️ Future Extensions

- User authentication + persistent storage
- Session recording and debugging tools
- Simulator streaming with GPU-accelerated rendering (WebRTC)
- GPU utilization monitoring and quota enforcement

### 🏁 Summary

This phase transforms RoboDSL into a **cloud-native robotics IDE** that removes friction from CUDA + ROS2 development. Developers can build GPU-accelerated pipelines without worrying about setup, hardware, or dependencies—locally or in the cloud.


---


### Phase 6: AI-Powered Development Tools
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

### Phase 7: Real-time Constraints
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

### Phase 8: Training Integration
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

### Phase 9: Advanced Features
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