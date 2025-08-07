"""Advanced CMake features comprehensive testing.

This module provides extensive test coverage for advanced CMake features including:
- Multi-configuration builds
- Cross-compilation support
- Advanced package management
- Build system optimization
- Dependency management
- Custom build targets
- Advanced compiler flags
- Platform-specific configurations
"""

import pytest
import tempfile
import shutil
import os
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from robodsl.parsers.lark_parser import parse_robodsl
from conftest import skip_if_no_ros2, has_ros2
from robodsl.generators.cmake_generator import CMakeGenerator
from robodsl.core.ast import NodeNode


class TestCMakeAdvancedFeatures:
    """Advanced CMake features test suite."""
    
    @pytest.fixture
    def cmake_config(self):
        """Get CMake configuration."""
        return {
            "cmake_version": self._get_cmake_version(),
            "compiler_info": self._get_compiler_info(),
            "platform_info": self._get_platform_info(),
            "available_packages": self._get_available_packages(),
            "build_types": ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"]
        }
    
    def _get_cmake_version(self):
        """Get CMake version."""
        try:
            result = subprocess.run(['cmake', '--version'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                return version_line.split()[-1]
        except:
            pass
        return "unknown"
    
    def _get_compiler_info(self):
        """Get compiler information."""
        compilers = {}
        
        # Check GCC
        try:
            result = subprocess.run(['gcc', '--version'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                compilers['gcc'] = result.stdout.split('\n')[0].split()[-1]
        except:
            pass
        
        # Check Clang
        try:
            result = subprocess.run(['clang', '--version'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                compilers['clang'] = result.stdout.split('\n')[0].split()[-1]
        except:
            pass
        
        # Check NVCC
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        compilers['nvcc'] = line.split('release')[1].strip().split(',')[0]
                        break
        except:
            pass
        
        return compilers
    
    def _get_platform_info(self):
        """Get platform information."""
        import platform
        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0]
        }
    
    def _get_available_packages(self):
        """Get available packages."""
        return [
            "rclcpp", "std_msgs", "sensor_msgs", "geometry_msgs",
            "cuda", "cudart", "nvinfer", "nvonnxparser",
            "OpenCV", "Eigen3", "Boost", "PCL"
        ]
    
    def test_multi_configuration_builds(self, cmake_config):
        """Test multi-configuration builds."""
        skip_if_no_ros2()
        
        build_types = cmake_config["build_types"]
        
        for build_type in build_types:
            dsl_code = f'''
            package multi_config_package {{
                name: "multi_config_package"
                version: "1.0.0"
                description: "Multi-configuration package"
                
                cpp_node config_node_{build_type.lower()} {{
                    publisher: "topic" -> "std_msgs/String" {{
                        qos: reliable
                        queue_size: 10
                    }}
                }}
                
                build_type: "{build_type}"
                build_configuration: {{
                    "CMAKE_BUILD_TYPE": "{build_type}",
                    "CMAKE_CXX_FLAGS_{build_type.upper()}": "-O3 -DNDEBUG",
                    "CMAKE_C_FLAGS_{build_type.upper()}": "-O3 -DNDEBUG"
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            assert len(ast.packages) == 1
            package = ast.packages[0]
            assert package.name == "multi_config_package"
            assert len(package.cpp_nodes) == 1
    
    def test_cross_compilation_support(self, cmake_config):
        """Test cross-compilation support."""
        skip_if_no_ros2()
        
        platforms = ["x86_64", "aarch64", "armv7", "armv8"]
        
        for platform in platforms:
            dsl_code = f'''
            package cross_compilation_package_{platform} {{
                name: "cross_compilation_package_{platform}"
                version: "1.0.0"
                description: "Cross-compilation package for {platform}"
                
                cpp_node cross_node_{platform} {{
                    publisher: "topic" -> "std_msgs/String" {{
                        qos: reliable
                        queue_size: 10
                    }}
                }}
                
                cross_compilation: true
                target_platform: "{platform}"
                toolchain_file: "toolchain_{platform}.cmake"
                build_configuration: {{
                    "CMAKE_SYSTEM_NAME": "Linux",
                    "CMAKE_SYSTEM_PROCESSOR": "{platform}",
                    "CMAKE_C_COMPILER": "{platform}-linux-gnu-gcc",
                    "CMAKE_CXX_COMPILER": "{platform}-linux-gnu-g++"
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            assert len(ast.packages) == 1
            package = ast.packages[0]
            assert package.name == f"cross_compilation_package_{platform}"
    
    def test_advanced_package_management(self, cmake_config):
        """Test advanced package management."""
        skip_if_no_ros2()
        
        available_packages = cmake_config["available_packages"]
        # Convert Python list to DSL array format
        dsl_packages = "[" + ", ".join(f'"{pkg}"' for pkg in available_packages) + "]"
        
        dsl_code = f'''
        package advanced_package {{
            name: "advanced_package"
            version: "1.0.0"
            description: "Advanced package with comprehensive dependencies"
            
            dependencies: {dsl_packages}
            
            cpp_node advanced_node {{
                publisher: "topic" -> "std_msgs/String" {{
                    qos: reliable
                    queue_size: 10
                }}
            }}
            
            package_management: true
            dependency_resolution: true
            version_constraints: {{
                "rclcpp": ">=2.0.0",
                "cuda": ">=11.0",
                "OpenCV": ">=4.5.0"
            }}
            optional_dependencies: ["PCL", "Boost"]
            system_dependencies: ["libssl-dev", "libcurl4-openssl-dev"]
        }}
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.packages) == 1
        package = ast.packages[0]
        assert package.name == "advanced_package"
        assert len(package.dependencies) == len(available_packages)
    
    def test_build_system_optimization(self, cmake_config):
        """Test build system optimization."""
        skip_if_no_ros2()
        
        dsl_code = '''
        package optimized_package {
            name: "optimized_package"
            version: "1.0.0"
            description: "Optimized build package"
            
            cpp_node optimized_node {
                publisher: "topic" -> "std_msgs/String" {
                    qos: reliable
                    queue_size: 10
                }
            }
            
            build_optimization: true
            parallel_build: true
            build_jobs: 8
            build_cache: true
            ccache: true
            ninja: true
            build_configuration: {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_CXX_FLAGS": "-O3 -march=native -mtune=native",
                "CMAKE_C_FLAGS": "-O3 -march=native -mtune=native",
                "CMAKE_EXE_LINKER_FLAGS": "-Wl,--as-needed",
                "CMAKE_SHARED_LINKER_FLAGS": "-Wl,--as-needed"
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.packages) == 1
        package = ast.packages[0]
        assert package.name == "optimized_package"
    
    def test_dependency_management_advanced(self, cmake_config):
        """Test advanced dependency management."""
        skip_if_no_ros2()
        
        dsl_code = '''
        package dependency_managed_package {
            name: "dependency_managed_package"
            version: "1.0.0"
            description: "Package with advanced dependency management"
            
            cpp_node dependency_node {
                publisher: "topic" -> "std_msgs/String" {
                    qos: reliable
                    queue_size: 10
                }
            }
            
            dependencies: ["rclcpp", "std_msgs", "cuda", "OpenCV"]
            
            dependency_management: true
            dependency_resolution: true
            dependency_caching: true
            dependency_parallel_download: true
            dependency_verification: true
            dependency_licenses: ["MIT", "Apache-2.0", "GPL-3.0"]
            
            dependency_configuration: {
                "rclcpp": {
                    "version": ">=2.0.0",
                    "components": ["rclcpp", "rclcpp_components"],
                    "optional": false
                },
                "cuda": {
                    "version": ">=11.0",
                    "components": ["cuda", "cudart"],
                    "optional": true
                },
                "OpenCV": {
                    "version": ">=4.5.0",
                    "components": ["opencv_core", "opencv_imgproc"],
                    "optional": true
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.packages) == 1
        package = ast.packages[0]
        assert package.name == "dependency_managed_package"
        assert len(package.dependencies) == 4
    
    def test_custom_build_targets(self, cmake_config):
        """Test custom build targets."""
        skip_if_no_ros2()
        
        dsl_code = '''
        package custom_targets_package {
            name: "custom_targets_package"
            version: "1.0.0"
            description: "Package with custom build targets"
            
            cpp_node custom_node {
                publisher: "topic" -> "std_msgs/String" {
                    qos: reliable
                    queue_size: 10
                }
            }
            
            custom_targets: [
                "custom_library",
                "custom_executable",
                "custom_test",
                "custom_benchmark"
            ]
            
            custom_target_configuration: {
                "custom_library": {
                    "type": "shared_library",
                    "sources": ["src/custom_lib.cpp"],
                    "include_directories": ["include"],
                    "dependencies": ["rclcpp"]
                },
                "custom_executable": {
                    "type": "executable",
                    "sources": ["src/custom_main.cpp"],
                    "dependencies": ["custom_library", "rclcpp"]
                },
                "custom_test": {
                    "type": "test",
                    "sources": ["test/custom_test.cpp"],
                    "dependencies": ["custom_library", "gtest"]
                },
                "custom_benchmark": {
                    "type": "benchmark",
                    "sources": ["benchmark/custom_benchmark.cpp"],
                    "dependencies": ["custom_library", "benchmark"]
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.packages) == 1
        package = ast.packages[0]
        assert package.name == "custom_targets_package"
        assert len(package.custom_targets) == 4
    
    def test_advanced_compiler_flags(self, cmake_config):
        """Test advanced compiler flags."""
        skip_if_no_ros2()
        
        compilers = cmake_config["compiler_info"]
        
        for compiler, version in compilers.items():
            dsl_code = f'''
            package compiler_flags_package_{compiler} {{
                name: "compiler_flags_package_{compiler}"
                version: "1.0.0"
                description: "Package with advanced compiler flags for {compiler}"
                
                cpp_node compiler_node_{compiler} {{
                    publisher: "topic" -> "std_msgs/String" {{
                        qos: reliable
                        queue_size: 10
                    }}
                }}
                
                compiler: "{compiler}"
                compiler_version: "{version}"
                
                build_flags: {{
                    "CMAKE_CXX_FLAGS": "-std=c++17",
                    "CMAKE_C_FLAGS": "-std=c11",
                    "CMAKE_CXX_FLAGS_DEBUG": "-g -O0 -DDEBUG",
                    "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG -march=native",
                    "CMAKE_CXX_FLAGS_RELWITHDEBINFO": "-O2 -g -DNDEBUG",
                    "CMAKE_CXX_FLAGS_MINSIZEREL": "-Os -DNDEBUG"
                }}
                
                optimization_flags: {{
                    "vectorization": true,
                    "link_time_optimization": true,
                    "profile_guided_optimization": true,
                    "function_inlining": true
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            assert len(ast.packages) == 1
            package = ast.packages[0]
            assert package.name == f"compiler_flags_package_{compiler}"
    
    def test_platform_specific_configurations(self, cmake_config):
        """Test platform-specific configurations."""
        skip_if_no_ros2()
        
        platform_info = cmake_config["platform_info"]
        
        dsl_code = f'''
        package platform_specific_package {{
            name: "platform_specific_package"
            version: "1.0.0"
            description: "Platform-specific package"
            
            cpp_node platform_node {{
                publisher: "topic" -> "std_msgs/String" {{
                    qos: reliable
                    queue_size: 10
                }}
            }}
            
            platform: "{platform_info['system']}"
            architecture: "{platform_info['architecture']}"
            machine: "{platform_info['machine']}"
            
            platform_configuration: {{
                "linux": {{
                    "CMAKE_SYSTEM_NAME": "Linux",
                    "CMAKE_CXX_FLAGS": "-std=c++17",
                    "CMAKE_C_FLAGS": "-std=c11"
                }},
                "darwin": {{
                    "CMAKE_SYSTEM_NAME": "Darwin",
                    "CMAKE_CXX_FLAGS": "-std=c++17",
                    "CMAKE_C_FLAGS": "-std=c11"
                }},
                "windows": {{
                    "CMAKE_SYSTEM_NAME": "Windows",
                    "CMAKE_CXX_FLAGS": "/std:c++17",
                    "CMAKE_C_FLAGS": "/std:c11"
                }}
            }}
            
            architecture_configuration: {{
                "x86_64": {{
                    "CMAKE_CXX_FLAGS": "${{CMAKE_CXX_FLAGS}} -march=x86-64 -mtune=generic"
                }},
                "aarch64": {{
                    "CMAKE_CXX_FLAGS": "${{CMAKE_CXX_FLAGS}} -march=armv8-a"
                }},
                "armv7": {{
                    "CMAKE_CXX_FLAGS": "${{CMAKE_CXX_FLAGS}} -march=armv7-a -mfpu=neon"
                }}
            }}
        }}
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.packages) == 1
        package = ast.packages[0]
        assert package.name == "platform_specific_package"
    
    def test_build_variants_comprehensive(self, cmake_config):
        """Test comprehensive build variants."""
        skip_if_no_ros2()
        
        build_variants = [
            ("debug", "Debug", "-g -O0 -DDEBUG"),
            ("release", "Release", "-O3 -DNDEBUG"),
            ("relwithdebinfo", "RelWithDebInfo", "-O2 -g -DNDEBUG"),
            ("minsizerel", "MinSizeRel", "-Os -DNDEBUG")
        ]
        
        for variant_name, build_type, flags in build_variants:
            dsl_code = f'''
            package build_variant_package_{variant_name} {{
                name: "build_variant_package_{variant_name}"
                version: "1.0.0"
                description: "Build variant package for {variant_name}"
                
                cpp_node variant_node_{variant_name} {{
                    publisher: "topic" -> "std_msgs/String" {{
                        qos: reliable
                        queue_size: 10
                    }}
                }}
                
                build_variant: "{variant_name}"
                build_type: "{build_type}"
                build_flags: {{
                    "CMAKE_CXX_FLAGS": "{flags}"
                }}
                
                variant_configuration: {{
                    "debug": {{
                        "CMAKE_BUILD_TYPE": "Debug",
                        "CMAKE_CXX_FLAGS": "-g -O0 -DDEBUG",
                        "BUILD_TESTING": "ON"
                    }},
                    "release": {{
                        "CMAKE_BUILD_TYPE": "Release",
                        "CMAKE_CXX_FLAGS": "-O3 -DNDEBUG",
                        "BUILD_TESTING": "OFF"
                    }},
                    "relwithdebinfo": {{
                        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
                        "CMAKE_CXX_FLAGS": "-O2 -g -DNDEBUG",
                        "BUILD_TESTING": "OFF"
                    }},
                    "minsizerel": {{
                        "CMAKE_BUILD_TYPE": "MinSizeRel",
                        "CMAKE_CXX_FLAGS": "-Os -DNDEBUG",
                        "BUILD_TESTING": "OFF"
                    }}
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            assert len(ast.packages) == 1
            package = ast.packages[0]
            assert package.name == f"build_variant_package_{variant_name}"
    
    def test_package_generation_comprehensive(self, cmake_config):
        """Test comprehensive package generation."""
        skip_if_no_ros2()
        
        dsl_code = '''
        package comprehensive_package {
            name: "comprehensive_package"
            version: "1.0.0"
            description: "Comprehensive package with all features"
            maintainer: "maintainer@example.com"
            license: "MIT"
            url: "https://github.com/example/comprehensive_package"
            
            cpp_node comprehensive_node {
                publisher: "topic" -> "std_msgs/String" {
                    qos: reliable
                    queue_size: 10
                }
                
                subscriber: "topic" -> "std_msgs/String" {
                    qos: reliable
                    queue_size: 10
                }
                
                service: "service" -> "std_srvs/Trigger" {
                    qos: services_default
                    callback: "service_callback"
                }
                
                timer: "timer" {
                    period: 1.0
                    callback: "timer_callback"
                }
            }
            
            dependencies: ["rclcpp", "std_msgs", "std_srvs"]
            
            build_configuration: {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_CXX_STANDARD": "17",
                "CMAKE_CXX_STANDARD_REQUIRED": "ON",
                "CMAKE_CXX_EXTENSIONS": "OFF",
                "BUILD_TESTING": "ON",
                "BUILD_BENCHMARKING": "ON"
            }
            
            install_configuration: {
                "install_headers": true,
                "install_libraries": true,
                "install_executables": true,
                "install_config_files": true,
                "install_launch_files": true
            }
            
            test_configuration: {
                "unit_tests": true,
                "integration_tests": true,
                "benchmark_tests": true,
                "coverage": true,
                "valgrind": true
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.packages) == 1
        package = ast.packages[0]
        assert package.name == "comprehensive_package"
        assert len(package.cpp_nodes) == 1
        assert len(package.dependencies) == 3
    
    def test_cmake_generator_integration(self, test_output_dir):
        """Test CMake generator integration."""
        skip_if_no_ros2()
        
        cmake_generator = CMakeGenerator(str(test_output_dir))
        
        # Test package configuration
        package_config = {
            "name": "test_package",
            "version": "1.0.0",
            "description": "Test package",
            "dependencies": ["rclcpp", "std_msgs"],
            "build_type": "Release",
            "compiler": "gcc",
            "optimization": True
        }
        
        # Generate CMakeLists.txt
        cmake_content = cmake_generator.generate_cmake_lists(package_config)
        # Convert Path to string if needed
        if hasattr(cmake_content, 'read_text'):
            cmake_content = cmake_content.read_text()
        assert "find_package(rclcpp REQUIRED)" in cmake_content
        assert "find_package(std_msgs REQUIRED)" in cmake_content
        assert "CMAKE_BUILD_TYPE" in cmake_content
        
        # Generate package.xml
        package_xml_path = cmake_generator.generate_package_xml(package_config)
        # Convert Path to string if needed
        if hasattr(package_xml_path, 'read_text'):
            package_xml = package_xml_path.read_text()
        else:
            package_xml = package_xml_path
        assert "<name>test_package</name>" in package_xml  # The generator uses the actual package name
        assert "<depend>rclcpp</depend>" in package_xml
        assert "<depend>std_msgs</depend>" in package_xml
    
    def test_build_system_performance(self, cmake_config):
        """Test build system performance features."""
        skip_if_no_ros2()
        
        dsl_code = '''
        package performance_package {
            name: "performance_package"
            version: "1.0.0"
            description: "Performance-optimized package"
            
            cpp_node performance_node {
                publisher: "topic" -> "std_msgs/String" {
                    qos: reliable
                    queue_size: 10
                }
            }
            
            build_performance: true
            parallel_build: true
            build_jobs: 16
            build_cache: true
            ccache: true
            ninja: true
            precompiled_headers: true
            unity_build: true
            link_time_optimization: true
            profile_guided_optimization: true
            
            performance_configuration: {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_CXX_FLAGS": "-O3 -march=native -mtune=native -flto",
                "CMAKE_C_FLAGS": "-O3 -march=native -mtune=native -flto",
                "CMAKE_EXE_LINKER_FLAGS": "-Wl,--as-needed -flto",
                "CMAKE_SHARED_LINKER_FLAGS": "-Wl,--as-needed -flto"
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.packages) == 1
        package = ast.packages[0]
        assert package.name == "performance_package"
    
    def test_cross_platform_compatibility(self, cmake_config):
        """Test cross-platform compatibility."""
        skip_if_no_ros2()
        
        platforms = ["linux", "darwin", "windows"]
        
        for platform in platforms:
            dsl_code = f'''
            package cross_platform_package_{platform} {{
                name: "cross_platform_package_{platform}"
                version: "1.0.0"
                description: "Cross-platform package for {platform}"
                
                cpp_node cross_platform_node_{platform} {{
                    publisher: "topic" -> "std_msgs/String" {{
                        qos: reliable
                        queue_size: 10
                    }}
                }}
                
                target_platform: "{platform}"
                cross_platform: true
                
                platform_configuration: {{
                    "linux": {{
                        "CMAKE_SYSTEM_NAME": "Linux",
                        "CMAKE_CXX_FLAGS": "-std=c++17",
                        "CMAKE_C_FLAGS": "-std=c11"
                    }},
                    "darwin": {{
                        "CMAKE_SYSTEM_NAME": "Darwin",
                        "CMAKE_CXX_FLAGS": "-std=c++17",
                        "CMAKE_C_FLAGS": "-std=c11"
                    }},
                    "windows": {{
                        "CMAKE_SYSTEM_NAME": "Windows",
                        "CMAKE_CXX_FLAGS": "/std:c++17",
                        "CMAKE_C_FLAGS": "/std:c11"
                    }}
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            assert len(ast.packages) == 1
            package = ast.packages[0]
            assert package.name == f"cross_platform_package_{platform}" 