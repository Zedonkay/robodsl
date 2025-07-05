"""Main Generator for RoboDSL.

This generator orchestrates all other generators to create a complete ROS2 package.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

from .base_generator import BaseGenerator
from .cpp_node_generator import CppNodeGenerator
from .cuda_kernel_generator import CudaKernelGenerator
from .python_node_generator import PythonNodeGenerator
from .cmake_generator import CMakeGenerator
from .launch_generator import LaunchGenerator
from .package_generator import PackageGenerator
from .onnx_integration import OnnxIntegrationGenerator
from .pipeline_generator import PipelineGenerator
from .config_generator import ConfigGenerator
from .advanced_cpp_generator import AdvancedCppGenerator
from ..core.ast import RoboDSLAST


class MainGenerator(BaseGenerator):
    """Main generator that orchestrates all other generators."""
    
    def __init__(self, output_dir: str = ".", template_dirs: Optional[List[Path]] = None, debug: bool = False):
        """Initialize the main generator.
        
        Args:
            output_dir: Base directory for generated files
            template_dirs: Additional template directories to search
            debug: Whether to enable debug mode
        """
        super().__init__(output_dir, template_dirs)
        self.debug = debug
        
        # Initialize all sub-generators
        self.cpp_generator = CppNodeGenerator(output_dir, template_dirs)
        self.cuda_generator = CudaKernelGenerator(output_dir, template_dirs)
        self.python_generator = PythonNodeGenerator(output_dir, template_dirs)
        self.cmake_generator = CMakeGenerator(output_dir, template_dirs)
        self.launch_generator = LaunchGenerator(output_dir, template_dirs)
        self.package_generator = PackageGenerator(output_dir, template_dirs)
        self.onnx_generator = OnnxIntegrationGenerator(output_dir)
        self.pipeline_generator = PipelineGenerator(output_dir)
        self.config_generator = ConfigGenerator(output_dir, template_dirs)
        self.advanced_cpp_generator = AdvancedCppGenerator(output_dir, template_dirs)
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate all files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for all generated files
        """
        all_generated_files = []
        
        # Generate C++ node files
        print("Generating C++ node files...")
        cpp_files = self.cpp_generator.generate(ast)
        all_generated_files.extend(cpp_files)
        print(f"Generated {len(cpp_files)} C++ files")
        
        # Generate CUDA kernel files
        print("Generating CUDA kernel files...")
        cuda_files = self.cuda_generator.generate(ast)
        all_generated_files.extend(cuda_files)
        print(f"Generated {len(cuda_files)} CUDA files")
        
        # Generate Python node files
        print("Generating Python node files...")
        python_files = self.python_generator.generate(ast)
        all_generated_files.extend(python_files)
        print(f"Generated {len(python_files)} Python files")
        
        # Generate CMake files
        print("Generating CMake files...")
        cmake_files = self.cmake_generator.generate(ast)
        all_generated_files.extend(cmake_files)
        print(f"Generated {len(cmake_files)} CMake files")
        
        # Generate launch files
        print("Generating launch files...")
        launch_files = self.launch_generator.generate(ast)
        all_generated_files.extend(launch_files)
        print(f"Generated {len(launch_files)} launch files")
        
        # Generate package files
        print("Generating package files...")
        package_files = self.package_generator.generate(ast)
        all_generated_files.extend(package_files)
        print(f"Generated {len(package_files)} package files")
        
        # Generate config files
        print("Generating config files...")
        config_files = self.config_generator.generate(ast)
        all_generated_files.extend(config_files)
        print(f"Generated {len(config_files)} config files")
        
        # Generate global raw C++ code files
        print("Generating global raw C++ code files...")
        raw_cpp_files = self._generate_global_raw_cpp_code(ast)
        all_generated_files.extend(raw_cpp_files)
        print(f"Generated {len(raw_cpp_files)} global raw C++ code files")
        
        # Generate advanced C++ features
        print("Generating advanced C++ features...")
        advanced_cpp_files = self._generate_advanced_cpp_features(ast)
        all_generated_files.extend(advanced_cpp_files)
        print(f"Generated {len(advanced_cpp_files)} advanced C++ feature files")
        
        # Generate ONNX integration files
        print("Generating ONNX integration files...")
        onnx_files = self._generate_onnx_integration(ast)
        all_generated_files.extend(onnx_files)
        print(f"Generated {len(onnx_files)} ONNX integration files")
        
        # Generate pipeline files (including per-stage CUDA/ONNX integration)
        print("Generating pipeline files...")
        pipeline_files = []
        if hasattr(ast, 'pipelines') and ast.pipelines:
            for pipeline in ast.pipelines:
                # Use the project name if available, else fallback
                project_name = getattr(ast, 'project_name', 'robodsl_project')
                if self.debug:
                    print(f"DEBUG: Processing pipeline '{pipeline.name}' with project_name '{project_name}'")
                for idx, stage in enumerate(pipeline.content.stages):
                    if self.debug:
                        print(f"DEBUG: Processing stage '{stage.name}' (index {idx})")
                    files = self.pipeline_generator._generate_stage_node(stage, pipeline.name, idx, project_name)
                    if self.debug:
                        print(f"DEBUG: Stage '{stage.name}' generated files: {list(files.keys())}")
                    # Write files to disk and collect their paths
                    for rel_path, content in files.items():
                        abs_path = Path(self.output_dir) / rel_path
                        abs_path.parent.mkdir(parents=True, exist_ok=True)
                        abs_path.write_text(content)
                        pipeline_files.append(abs_path)
        all_generated_files.extend(pipeline_files)
        print(f"Generated {len(pipeline_files)} pipeline files")
        
        # Generate README
        print("Generating README.md...")
        try:
            readme_content = self.render_template('README.md.jinja2', self._prepare_readme_context(ast))
        except Exception as e:
            print(f"Template error for README.md: {e}")
            readme_content = self._generate_fallback_readme(ast)
        readme_path = Path(self.output_dir) / 'README.md'
        readme_path.write_text(readme_content)
        all_generated_files.append(readme_path)
        print("Generated README.md")
        
        print(f"Total generated files: {len(all_generated_files)}")
        return all_generated_files
    
    def _generate_global_raw_cpp_code(self, ast: RoboDSLAST) -> List[Path]:
        """Generate global raw C++ code files that get passed through as-is."""
        generated_files = []
        
        if hasattr(ast, 'raw_cpp_code') and ast.raw_cpp_code:
            # Create src directory if it doesn't exist
            src_dir = Path(self.output_dir) / 'src'
            src_dir.mkdir(parents=True, exist_ok=True)
            
            for i, cpp_block in enumerate(ast.raw_cpp_code):
                if cpp_block.location == "global":
                    # Generate a unique filename for each global C++ code block
                    filename = f"global_cpp_code_{i}.cpp"
                    file_path = src_dir / filename
                    
                    # Write the raw C++ code directly to the file
                    file_path.write_text(cpp_block.code)
                    generated_files.append(file_path)
                    
                    if self.debug:
                        print(f"Generated global C++ code file: {file_path}")
        
        return generated_files
    
    def _generate_advanced_cpp_features(self, ast: RoboDSLAST) -> List[Path]:
        """Generate advanced C++ features files."""
        generated_files = []
        
        if hasattr(ast, 'advanced_cpp_features') and ast.advanced_cpp_features:
            # Create include directory for header files
            include_dir = Path(self.output_dir) / 'include'
            include_dir.mkdir(parents=True, exist_ok=True)
            
            # Create src directory for implementation files
            src_dir = Path(self.output_dir) / 'src'
            src_dir.mkdir(parents=True, exist_ok=True)
            
            # Separate features by type
            templates = []
            static_asserts = []
            global_vars = []
            operator_overloads = []
            constructors = []
            bitfields = []
            preprocessor_directives = []
            function_attributes = []
            concepts = []
            friend_declarations = []
            user_defined_literals = []
            
            for feature in ast.advanced_cpp_features:
                if hasattr(feature, '__class__'):
                    class_name = feature.__class__.__name__
                    if 'Template' in class_name:
                        templates.append(feature)
                    elif 'StaticAssert' in class_name:
                        static_asserts.append(feature)
                    elif 'Global' in class_name:
                        global_vars.append(feature)
                    elif 'Operator' in class_name:
                        operator_overloads.append(feature)
                    elif 'Constructor' in class_name or 'Destructor' in class_name:
                        constructors.append(feature)
                    elif 'Bitfield' in class_name:
                        bitfields.append(feature)
                    elif 'Preprocessor' in class_name:
                        preprocessor_directives.append(feature)
                    elif 'FunctionAttribute' in class_name:
                        function_attributes.append(feature)
                    elif 'Concept' in class_name:
                        concepts.append(feature)
                    elif 'Friend' in class_name:
                        friend_declarations.append(feature)
                    elif 'UserDefinedLiteral' in class_name:
                        user_defined_literals.append(feature)
            
            # Generate header file with templates and declarations
            if templates or static_asserts or global_vars or operator_overloads or concepts or user_defined_literals:
                header_content = self.advanced_cpp_generator.generate_templates(templates)
                header_content += "\n\n" + self.advanced_cpp_generator.generate_static_asserts(static_asserts)
                header_content += "\n\n" + self.advanced_cpp_generator.generate_global_variables(global_vars)
                header_content += "\n\n" + self.advanced_cpp_generator.generate_operator_overloads(operator_overloads)
                header_content += "\n\n" + self.advanced_cpp_generator.generate_concepts(concepts)
                header_content += "\n\n" + self.advanced_cpp_generator.generate_user_defined_literals(user_defined_literals)
                
                header_path = include_dir / 'advanced_features.hpp'
                header_path.write_text(header_content)
                generated_files.append(header_path)
            
            # Generate implementation file with constructors, bitfields, and function attributes
            if constructors or bitfields or function_attributes or friend_declarations:
                impl_content = self.advanced_cpp_generator.generate_constructors(constructors)
                impl_content += "\n\n" + self.advanced_cpp_generator.generate_bitfields(bitfields)
                impl_content += "\n\n" + self.advanced_cpp_generator.generate_function_attributes(function_attributes)
                impl_content += "\n\n" + self.advanced_cpp_generator.generate_friend_declarations(friend_declarations)
                
                impl_path = src_dir / 'advanced_features.cpp'
                impl_path.write_text(impl_content)
                generated_files.append(impl_path)
            
            # Generate preprocessor directives file
            if preprocessor_directives:
                preprocessor_content = self.advanced_cpp_generator.generate_preprocessor_directives(preprocessor_directives)
                preprocessor_path = include_dir / 'preprocessor_directives.hpp'
                preprocessor_path.write_text(preprocessor_content)
                generated_files.append(preprocessor_path)
        
        return generated_files
    
    def _generate_readme(self, ast: RoboDSLAST) -> Path:
        """Generate a README.md file for the package."""
        context = self._prepare_readme_context(ast)
        
        try:
            content = self.render_template('README.md.jinja2', context)
            readme_path = self.get_output_path('README.md')
            return self.write_file(readme_path, content)
        except Exception as e:
            print(f"Template error for README.md: {e}")
            # Fallback to simple README
            content = self._generate_fallback_readme(ast)
            readme_path = self.get_output_path('README.md')
            return self.write_file(readme_path, content)
    
    def _prepare_readme_context(self, ast: RoboDSLAST) -> Dict[str, Any]:
        """Prepare context for README template rendering."""
        # Determine package name
        package_name = getattr(ast, 'package_name', 'robodsl_package')
        
        # Count different types of components
        node_count = len(ast.nodes)
        cuda_kernel_count = 0
        
        if ast.cuda_kernels:
            cuda_kernel_count += len(ast.cuda_kernels.kernels)
        
        for node in ast.nodes:
            if node.content.cuda_kernels:
                cuda_kernel_count += len(node.content.cuda_kernels)
        
        # Prepare node descriptions
        nodes = []
        for node in ast.nodes:
            node_info = {
                'name': node.name,
                'publishers': len(node.content.publishers),
                'subscribers': len(node.content.subscribers),
                'services': len(node.content.services),
                'actions': len(node.content.actions),
                'timers': len(node.content.timers),
                'parameters': len(node.content.parameters),
                'cuda_kernels': len(node.content.cuda_kernels) if node.content.cuda_kernels else 0,
                'is_lifecycle': node.content.lifecycle is not None
            }
            nodes.append(node_info)
        
        # Prepare standalone kernel descriptions
        standalone_kernels = []
        if ast.cuda_kernels:
            for kernel in ast.cuda_kernels.kernels:
                kernel_info = {
                    'name': kernel.name,
                    'parameters': len(kernel.content.parameters) if kernel.content.parameters else 0,
                    'block_size': kernel.content.block_size[0] if kernel.content.block_size else 256,
                    'use_thrust': kernel.content.use_thrust
                }
                standalone_kernels.append(kernel_info)
        
        # Generate file tree
        file_tree = self._generate_file_tree(ast)
        
        return {
            'package_name': package_name,
            'version': '0.1.0',
            'description': 'Generated ROS2 package from RoboDSL specification',
            'node_count': node_count,
            'cuda_kernel_count': cuda_kernel_count,
            'nodes': nodes,
            'standalone_kernels': standalone_kernels,
            'has_cuda': cuda_kernel_count > 0,
            'file_tree': file_tree
        }
    
    def _generate_file_tree(self, ast: RoboDSLAST) -> str:
        """Generate a file tree representation of the package structure."""
        tree_lines = []
        package_name = getattr(ast, 'package_name', 'robodsl_package')
        
        # Root package directory
        tree_lines.append(f"{package_name}/")
        
        # CMake files
        tree_lines.append("├── CMakeLists.txt")
        tree_lines.append("├── package.xml")
        tree_lines.append("├── README.md")
        
        # Source directory
        tree_lines.append("├── src/")
        tree_lines.append("│   └── " + package_name + "/")
        
        # Collect all files to determine proper tree structure
        source_files = []
        
        # Node source files
        for node in ast.nodes:
            source_files.append(f"{node.name}.cpp")
            source_files.append(f"{node.name}.hpp")
            
            # CUDA kernels for this node
            if node.content.cuda_kernels:
                for kernel in node.content.cuda_kernels:
                    source_files.append(f"{kernel.name}.cu")
                    source_files.append(f"{kernel.name}.cuh")
        
        # Standalone CUDA kernels
        if ast.cuda_kernels:
            for kernel in ast.cuda_kernels.kernels:
                source_files.append(f"{kernel.name}.cu")
                source_files.append(f"{kernel.name}.cuh")
        
        # Python nodes
        python_nodes = [node for node in ast.nodes if hasattr(node.content, 'language') and node.content.language == 'python']
        for node in python_nodes:
            source_files.append(f"{node.name}.py")
        
        # Add source files with proper tree structure
        for i, file in enumerate(source_files):
            if i == len(source_files) - 1:
                tree_lines.append(f"│       └── {file}")
            else:
                tree_lines.append(f"│       ├── {file}")
        
        # Launch files
        launch_files = [f"{package_name}_launch.py"]
        for node in ast.nodes:
            launch_files.append(f"{node.name}_launch.py")
        
        tree_lines.append("├── launch/")
        for i, file in enumerate(launch_files):
            if i == len(launch_files) - 1:
                tree_lines.append(f"│   └── {file}")
            else:
                tree_lines.append(f"│   ├── {file}")
        
        # Include directory
        tree_lines.append("├── include/")
        tree_lines.append(f"│   └── {package_name}/")
        for i, node in enumerate(ast.nodes):
            if i == len(ast.nodes) - 1:
                tree_lines.append(f"│       └── {node.name}.hpp")
            else:
                tree_lines.append(f"│       ├── {node.name}.hpp")
        
        # Config directory
        tree_lines.append("├── config/")
        tree_lines.append("│   └── params.yaml")
        
        # Test directory
        tree_lines.append("└── test/")
        tree_lines.append("    └── test_" + package_name + ".py")
        
        return "\n".join(tree_lines)
    
    def _generate_fallback_readme(self, ast: RoboDSLAST) -> str:
        """Generate a fallback README.md if template fails."""
        package_name = getattr(ast, 'package_name', 'robodsl_package')
        
        content = f"""# {package_name}

Generated ROS2 package from RoboDSL specification.

## Overview

This package contains {len(ast.nodes)} ROS2 nodes and {len(ast.cuda_kernels.kernels) if ast.cuda_kernels else 0} standalone CUDA kernels.

## Package Structure

```
{self._generate_file_tree(ast)}
```

## Nodes

"""
        
        for node in ast.nodes:
            content += f"""### {node.name}

- **Type**: {'Lifecycle Node' if node.content.lifecycle else 'Standard Node'}
- **Publishers**: {len(node.content.publishers)}
- **Subscribers**: {len(node.content.subscribers)}
- **Services**: {len(node.content.services)}
- **Actions**: {len(node.content.actions)}
- **Timers**: {len(node.content.timers)}
- **Parameters**: {len(node.content.parameters)}
- **CUDA Kernels**: {len(node.content.cuda_kernels) if node.content.cuda_kernels else 0}

"""
        
        if ast.cuda_kernels:
            content += "## Standalone CUDA Kernels\n\n"
            for kernel in ast.cuda_kernels.kernels:
                content += f"""### {kernel.name}

- **Parameters**: {len(kernel.content.parameters) if kernel.content.parameters else 0}
- **Block Size**: {kernel.content.block_size[0] if kernel.content.block_size else 256}
- **Use Thrust**: {kernel.content.use_thrust}

"""
        
        content += """## Building

```bash
colcon build
```

## Running

```bash
# Source the workspace
source install/setup.bash

# Launch all nodes
ros2 launch {package_name} main_launch.py

# Or launch individual nodes
ros2 launch {package_name} <node_name>_launch.py
```

## License

Apache-2.0
"""
        
        return content 
    
    def _generate_onnx_integration(self, ast: RoboDSLAST) -> List[Path]:
        """Generate ONNX integration files for all models."""
        generated_files = []
        
        # Handle standalone ONNX models
        for model in ast.onnx_models:
            try:
                # Generate ONNX integration files
                onnx_files = self.onnx_generator.generate_onnx_integration(model, model.name)
                
                # Write files to disk
                for file_path, content in onnx_files.items():
                    path = Path(file_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
                    generated_files.append(path)
                
                # Generate CMake integration
                cmake_content = self.onnx_generator.generate_cmake_integration(model, model.name)
                cmake_path = self.get_output_path(f"{model.name}_onnx.cmake")
                generated_files.append(self.write_file(cmake_path, cmake_content))
                
            except Exception as e:
                print(f"Error generating ONNX integration for standalone model {model.name}: {e}")
        
        # Handle ONNX models within nodes
        for node in ast.nodes:
            for model in node.content.onnx_models:
                try:
                    # Generate ONNX integration files for the node
                    onnx_files = self.onnx_generator.generate_onnx_integration(model, node.name)
                    
                    # Write files to disk
                    for file_path, content in onnx_files.items():
                        path = Path(file_path)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(content)
                        generated_files.append(path)
                    
                    # Generate CMake integration for the node
                    cmake_content = self.onnx_generator.generate_cmake_integration(model, node.name)
                    cmake_path = self.get_output_path(f"{node.name}_onnx.cmake")
                    generated_files.append(self.write_file(cmake_path, cmake_content))
                    
                    # Generate node integration code
                    node_integration = self.onnx_generator.generate_node_integration(model, node.name)
                    node_path = self.get_output_path(f"{node.name}_main.cpp")
                    generated_files.append(self.write_file(node_path, node_integration))
                    
                except Exception as e:
                    print(f"Error generating ONNX integration for model {model.name} in node {node.name}: {e}")
        
        return generated_files 

    def _generate_pipelines(self, ast: RoboDSLAST) -> List[Path]:
        """Generate pipeline files for all pipelines."""
        generated_files = []
        
        for pipeline in ast.pipelines:
            try:
                # Generate pipeline files
                pipeline_files = self.pipeline_generator.generate(pipeline, "robodsl_project")
                
                # Write files to disk
                for file_path, content in pipeline_files.items():
                    path = Path(file_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
                    generated_files.append(path)
                
            except Exception as e:
                print(f"Error generating pipeline {pipeline.name}: {e}")
        
        return generated_files 