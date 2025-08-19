"""CMake Generator for RoboDSL.

This generator creates CMakeLists.txt files for ROS2 packages.
"""

from pathlib import Path
from typing import Dict, List, Any

from .base_generator import BaseGenerator
from ..core.ast import RoboDSLAST


class CMakeGenerator(BaseGenerator):
    """Generates CMakeLists.txt files."""
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate CMake files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for generated files
        """
        generated_files = []
        
        # Generate main CMakeLists.txt
        cmake_path = self._generate_cmake_lists(ast)
        generated_files.append(cmake_path)
        
        # Generate build configuration
        build_config_path = self._generate_build_config(ast)
        generated_files.append(build_config_path)
        
        return generated_files
    
    def _generate_cmake_lists(self, ast: RoboDSLAST) -> Path:
        """Generate the main CMakeLists.txt file."""
        context = self._prepare_cmake_context(ast)
        
        try:
            content = self.render_template('cmake/CMakeLists.txt.jinja2', context)
            cmake_path = self.output_dir / 'CMakeLists.txt'
            return self.write_file(cmake_path, content)
        except Exception as e:
            print(f"Template error for CMakeLists.txt: {e}")
            # Fallback to simple CMakeLists.txt
            content = self._generate_fallback_cmake_lists(ast)
            cmake_path = self.output_dir / 'CMakeLists.txt'
            return self.write_file(cmake_path, content)
    
    def _generate_build_config(self, ast: RoboDSLAST) -> Path:
        """Generate build configuration file."""
        context = self._prepare_build_config_context(ast)
        
        try:
            content = self.render_template('cmake/build_config.cmake.jinja2', context)
            config_path = self.output_dir / 'build_config.cmake'
            return self.write_file(config_path, content)
        except Exception as e:
            print(f"Template error for build_config.cmake: {e}")
            # Fallback to simple build config
            content = self._generate_fallback_build_config(ast)
            config_path = self.output_dir / 'build_config.cmake'
            return self.write_file(config_path, content)
    
    def _prepare_cmake_context(self, ast: RoboDSLAST) -> Dict[str, Any]:
        """Prepare context for CMake template."""
        # Get package information
        package_info = getattr(ast, 'package', {})
        package_name = package_info.get('name', 'robodsl_package')
        version = package_info.get('version', '0.1.0')
        
        # Collect all dependencies
        dependencies = set()
        build_dependencies = set()
        
        # Add basic ROS2 dependencies
        build_dependencies.update([
            'ament_cmake',
            'ament_cmake_python'
        ])
        
        dependencies.update([
            'rclcpp',
            'std_msgs',
            'geometry_msgs',
            'sensor_msgs'
        ])
        
        # Add dependencies based on node content
        for node in ast.nodes:
            # Check for lifecycle nodes
            if getattr(node, 'lifecycle', False) or 'lifecycle' in node.name.lower():
                dependencies.add('rclcpp_lifecycle')
            
            # Check for action nodes
            if hasattr(node, 'actions') and node.actions:
                dependencies.add('rclcpp_action')
            
            # Check for service nodes
            if hasattr(node, 'services') and node.services:
                dependencies.add('std_srvs')
            
            # Check for message package usage (only add packages that appear)
            for pub in getattr(node, 'publishers', []):
                if 'nav_msgs' in pub.msg_type:
                    dependencies.add('nav_msgs')
                if 'visualization_msgs' in pub.msg_type:
                    dependencies.add('visualization_msgs')
                if 'tf2_msgs' in pub.msg_type:
                    dependencies.add('tf2_msgs')
                # vision_msgs removed unless explicitly present; examples now use std_msgs
                # Generic: add package for any ROS type string of form package/category/Type
                parts = pub.msg_type.split('/')
                if len(parts) >= 3:
                    dependencies.add(parts[0])
                if 'trajectory_msgs' in pub.msg_type:
                    dependencies.add('trajectory_msgs')
            
            for sub in getattr(node, 'subscribers', []):
                if 'nav_msgs' in sub.msg_type:
                    dependencies.add('nav_msgs')
                if 'visualization_msgs' in sub.msg_type:
                    dependencies.add('visualization_msgs')
                if 'tf2_msgs' in sub.msg_type:
                    dependencies.add('tf2_msgs')
                # vision_msgs removed unless explicitly present
                parts = sub.msg_type.split('/')
                if len(parts) >= 3:
                    dependencies.add(parts[0])
                if 'trajectory_msgs' in sub.msg_type:
                    dependencies.add('trajectory_msgs')

            # Services and actions: add their packages
            for srv in getattr(node, 'services', []):
                parts = srv.srv_type.split('/')
                if len(parts) >= 3:
                    dependencies.add(parts[0])
            for act in getattr(node, 'actions', []):
                parts = act.action_type.split('/')
                if len(parts) >= 3:
                    dependencies.add(parts[0])
        
        # Check for CUDA usage
        has_cuda = hasattr(ast, 'cuda_kernels') and ast.cuda_kernels
        
        # Check for OpenCV usage
        has_opencv = False
        if hasattr(ast, 'global_cpp_code'):
            for code_block in ast.global_cpp_code:
                if 'cv_bridge' in str(code_block) or 'opencv' in str(code_block) or 'cv::' in str(code_block):
                    has_opencv = True
                    break
        
        if has_opencv:
            dependencies.add('cv_bridge')
            dependencies.add('opencv2')
        
        # Check for ONNX usage
        has_onnx = hasattr(ast, 'onnx_models') and ast.onnx_models
        
        # Collect node executables
        executables = []
        for node in ast.nodes:
            subdir = self._get_node_subdirectory(node)
            if subdir:
                source = f'src/nodes/{subdir}/{node.name}_node.cpp'
            else:
                source = f'src/nodes/{node.name}_node.cpp'
            executables.append({
                'name': f'{node.name}_node',
                'source': source
            })

        # Message dependencies (set for clarity; in this template they are merged via dependencies)
        message_dependencies = set()
        for node in ast.nodes:
            for pub in getattr(node, 'publishers', []):
                if 'visualization_msgs' in pub.msg_type:
                    message_dependencies.add('visualization_msgs')
                if 'tf2_msgs' in pub.msg_type:
                    message_dependencies.add('tf2_msgs')
            for sub in getattr(node, 'subscribers', []):
                if 'visualization_msgs' in sub.msg_type:
                    message_dependencies.add('visualization_msgs')
                if 'tf2_msgs' in sub.msg_type:
                    message_dependencies.add('tf2_msgs')
        
        # Collect CUDA source files
        cuda_sources = []
        if has_cuda:
            for kernel in ast.cuda_kernels:
                cuda_sources.append(f'src/cuda/{kernel.name}_kernel.cu')
        
        return {
            'package_name': package_name,
            'version': version,
            'dependencies': sorted(list(dependencies)),
            'message_dependencies': sorted(list(message_dependencies)),
            'build_dependencies': sorted(list(build_dependencies)),
            'executables': executables,
            'cuda_sources': cuda_sources,
            'has_cuda': has_cuda,
            'has_opencv': has_opencv,
            'has_onnx': has_onnx
        }
    
    def _get_node_subdirectory(self, node) -> str:
        """Determine the appropriate subdirectory for a node based on its name and content."""
        # For subnodes with dots, use the existing logic
        if '.' in node.name:
            parts = node.name.split('.')
            if len(parts) > 1:
                return '/'.join(parts[:-1])
        
        # For regular nodes, organize by type/function
        node_name = node.name.lower()
        
        # Main/control nodes
        if 'main' in node_name:
            return 'main'
        # Perception/vision nodes
        elif 'perception' in node_name or 'vision' in node_name or 'camera' in node_name:
            return 'perception'
        # Navigation/movement nodes
        elif 'navigation' in node_name or 'movement' in node_name or 'drive' in node_name:
            return 'navigation'
        # Safety/monitoring nodes
        elif 'safety' in node_name or 'monitor' in node_name or 'emergency' in node_name:
            return 'safety'
        # Default to nodes directory
        else:
            return ''
    
    def _prepare_build_config_context(self, ast: RoboDSLAST) -> Dict[str, Any]:
        """Prepare context for build config template."""
        # Check for CUDA usage
        has_cuda = hasattr(ast, 'cuda_kernels') and ast.cuda_kernels
        
        # Check for OpenCV usage
        has_opencv = False
        if hasattr(ast, 'global_cpp_code'):
            for code_block in ast.global_cpp_code:
                if 'cv_bridge' in str(code_block) or 'opencv' in str(code_block) or 'cv::' in str(code_block):
                    has_opencv = True
                    break
        
        return {
            'has_cuda': has_cuda,
            'has_opencv': has_opencv
        }
    
    def _generate_fallback_cmake_lists(self, ast: RoboDSLAST) -> str:
        """Generate a fallback CMakeLists.txt if template fails."""
        return """cmake_minimum_required(VERSION 3.8)
project(robodsl_package VERSION 0.1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# CUDA support
find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
set(CMAKE_CUDA_STANDARD 17)

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Create library for common code
add_library(${PROJECT_NAME}_lib INTERFACE)
target_include_directories(${PROJECT_NAME}_lib INTERFACE include)
ament_target_dependencies(${PROJECT_NAME}_lib INTERFACE rclcpp)

# CUDA library
add_library(${PROJECT_NAME}_cuda STATIC
  src/cuda/vector_add_kernel.cu
  src/cuda/matrix_multiply_kernel.cu
  src/cuda/image_filter_kernel.cu
)

set_target_properties(${PROJECT_NAME}_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${PROJECT_NAME}_cuda PUBLIC include)
target_link_libraries(${PROJECT_NAME}_cuda ${PROJECT_NAME}_lib)

# Create executables
add_executable(main_node_node src/nodes/main/main_node_node.cpp)
target_link_libraries(main_node_node ${PROJECT_NAME}_lib)
target_link_libraries(main_node_node ${PROJECT_NAME}_cuda)
ament_target_dependencies(main_node_node rclcpp std_msgs geometry_msgs sensor_msgs cv_bridge)

add_executable(perception_node_node src/nodes/perception/perception_node_node.cpp)
target_link_libraries(perception_node_node ${PROJECT_NAME}_lib)
target_link_libraries(perception_node_node ${PROJECT_NAME}_cuda)
ament_target_dependencies(perception_node_node rclcpp std_msgs geometry_msgs sensor_msgs cv_bridge)

add_executable(navigation_node_node src/nodes/navigation/navigation_node_node.cpp)
target_link_libraries(navigation_node_node ${PROJECT_NAME}_lib)
target_link_libraries(navigation_node_node ${PROJECT_NAME}_cuda)
ament_target_dependencies(navigation_node_node rclcpp std_msgs geometry_msgs sensor_msgs)

add_executable(safety_node_node src/nodes/safety/safety_node_node.cpp)
target_link_libraries(safety_node_node ${PROJECT_NAME}_lib)
target_link_libraries(safety_node_node ${PROJECT_NAME}_cuda)
ament_target_dependencies(safety_node_node rclcpp std_msgs geometry_msgs sensor_msgs)

add_executable(robot_cpp_node_node src/nodes/robot_cpp_node_node.cpp)
target_link_libraries(robot_cpp_node_node ${PROJECT_NAME}_lib)
target_link_libraries(robot_cpp_node_node ${PROJECT_NAME}_cuda)
ament_target_dependencies(robot_cpp_node_node rclcpp std_msgs)

# Install targets
install(TARGETS
  main_node_node
  perception_node_node
  navigation_node_node
  safety_node_node
  robot_cpp_node_node
  ${PROJECT_NAME}_lib
  ${PROJECT_NAME}_cuda
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cuh"
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install configuration files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
endif()

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_lib)
ament_export_libraries(${PROJECT_NAME}_cuda)
ament_export_dependencies(rclcpp std_msgs geometry_msgs sensor_msgs cv_bridge)

ament_package()"""
    
    def _generate_fallback_build_config(self, ast: RoboDSLAST) -> str:
        """Generate a fallback build config if template fails."""
        return """# Build configuration for RoboDSL package

# CUDA configuration
if(CUDA_FOUND)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# OpenCV configuration
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC)
endif()"""