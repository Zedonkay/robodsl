"""Advanced Features Test Configuration.

This file defines comprehensive test cases for pipeline, ONNX, and TensorRT validation.
"""

import pytest
from typing import List, Dict, Any

# Test case definitions for advanced features validation

PIPELINE_TEST_CASES = [
    {
        "name": "basic_pipeline",
        "source": """
        pipeline basic_pipeline {
            stage preprocessing {
                input: "raw_data"
                output: "processed_data"
                method: "preprocess"
                topic: /pipeline/preprocess
            }
            
            stage processing {
                input: "processed_data"
                output: "result"
                method: "process"
                topic: /pipeline/process
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "pipeline": 1,
            "performance": 1,
            "practices": 1
        }
    },
    {
        "name": "pipeline_with_cuda",
        "source": """
        cuda_kernels {
            kernel process_kernel {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                input: float data(1000)
                output: float result(1000)
            }
        }
        
        pipeline cuda_pipeline {
            stage cuda_processing {
                input: "input_data"
                output: "output_data"
                method: "process_with_cuda"
                cuda_kernel: "process_kernel"
                topic: /pipeline/cuda_process
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "pipeline": 1,
            "cuda": 1,
            "performance": 1
        }
    },
    {
        "name": "pipeline_with_onnx",
        "source": """
        onnx_model inference_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: gpu
                optimization: tensorrt
            }
        }
        
        pipeline onnx_pipeline {
            stage inference {
                input: "input_data"
                output: "output_data"
                method: "run_inference"
                onnx_model: "inference_model"
                topic: /pipeline/inference
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "pipeline": 1,
            "onnx": 1,
            "tensorrt": 1,
            "performance": 1
        }
    }
]

ONNX_TEST_CASES = [
    {
        "name": "basic_onnx",
        "source": """
        onnx_model basic_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cpu
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "onnx": 1,
            "performance": 1,
            "practices": 1
        }
    },
    {
        "name": "onnx_with_tensorrt",
        "source": """
        onnx_model tensorrt_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
                optimization: fp16
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "onnx": 1,
            "tensorrt": 1,
            "cuda": 1,
            "performance": 1
        }
    },
    {
        "name": "onnx_with_multiple_optimizations",
        "source": """
        onnx_model optimized_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
                optimization: fp16
                optimization: int8
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "onnx": 1,
            "tensorrt": 1,
            "cuda": 1,
            "performance": 1
        }
    }
]

TENSORRT_TEST_CASES = [
    {
        "name": "tensorrt_basic",
        "source": """
        onnx_model tensorrt_basic {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "tensorrt": 1,
            "cuda": 1,
            "performance": 1
        }
    },
    {
        "name": "tensorrt_fp16",
        "source": """
        onnx_model tensorrt_fp16 {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
                optimization: fp16
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "tensorrt": 1,
            "cuda": 1,
            "performance": 1
        }
    },
    {
        "name": "tensorrt_int8",
        "source": """
        onnx_model tensorrt_int8 {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
                optimization: int8
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "tensorrt": 1,
            "cuda": 1,
            "performance": 1
        }
    }
]

COMPREHENSIVE_TEST_CASES = [
    {
        "name": "full_feature_pipeline",
        "source": """
        cuda_kernels {
            kernel preprocess_kernel {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                input: float raw_data(1000)
                output: float processed_data(1000)
            }
            
            kernel postprocess_kernel {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                input: float inference_result(1000)
                output: float final_result(1000)
            }
        }
        
        onnx_model inference_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
                optimization: fp16
            }
        }
        
        pipeline full_feature_pipeline {
            stage preprocessing {
                input: "raw_data"
                output: "processed_data"
                method: "preprocess"
                cuda_kernel: "preprocess_kernel"
                topic: /pipeline/preprocess
            }
            
            stage inference {
                input: "processed_data"
                output: "inference_result"
                method: "run_inference"
                onnx_model: "inference_model"
                topic: /pipeline/inference
            }
            
            stage postprocessing {
                input: "inference_result"
                output: "final_result"
                method: "postprocess"
                cuda_kernel: "postprocess_kernel"
                topic: /pipeline/postprocess
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "pipeline": 1,
            "onnx": 1,
            "tensorrt": 1,
            "cuda": 1,
            "performance": 1
        }
    }
]

EDGE_CASE_TEST_CASES = [
    {
        "name": "minimal_pipeline",
        "source": """
        pipeline minimal_pipeline {
            stage minimal_stage {
                input: "input_data"
                output: "output_data"
                method: "minimal_process"
                topic: /pipeline/minimal
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "pipeline": 1,
            "performance": 1,
            "practices": 1
        }
    },
    {
        "name": "minimal_onnx",
        "source": """
        onnx_model minimal_model {
            config {
                input: "input" -> "float32[1,1,1,1]"
                output: "output" -> "float32[1,1]"
                device: cpu
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "onnx": 1,
            "performance": 1,
            "practices": 1
        }
    }
]

# Test case generators for parameterized testing

def generate_pipeline_test_cases():
    """Generate test cases for pipeline validation."""
    for test_case in PIPELINE_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

def generate_onnx_test_cases():
    """Generate test cases for ONNX validation."""
    for test_case in ONNX_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

def generate_tensorrt_test_cases():
    """Generate test cases for TensorRT validation."""
    for test_case in TENSORRT_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

def generate_comprehensive_test_cases():
    """Generate test cases for comprehensive validation."""
    for test_case in COMPREHENSIVE_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

def generate_edge_case_test_cases():
    """Generate test cases for edge case validation."""
    for test_case in EDGE_CASE_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

# Validation configuration for advanced features

ADVANCED_FEATURES_CONFIG = {
    "syntax_tolerance": 0,      # No syntax errors allowed
    "pipeline_tolerance": 2,    # Allow 2 pipeline suggestions
    "onnx_tolerance": 2,        # Allow 2 ONNX suggestions
    "tensorrt_tolerance": 2,    # Allow 2 TensorRT suggestions
    "cuda_tolerance": 2,        # Allow 2 CUDA suggestions
    "performance_tolerance": 3, # Allow 3 performance suggestions
    "practice_tolerance": 2,    # Allow 2 practice suggestions
    
    "compiler_flags": [
        '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O2',
        '-fno-exceptions', '-fno-rtti', '-DNDEBUG'
    ],
    
    "cuda_flags": [
        '-std=c++17', '-Wall', '-Wextra', '-O2',
        '-arch=sm_60', '-DNDEBUG'
    ],
    
    "timeout_seconds": 30,
    "max_file_size_kb": 200
}

# Test categories for advanced features

ADVANCED_FEATURES_CATEGORIES = {
    "pipeline": {
        "description": "Pipeline generation tests",
        "test_cases": PIPELINE_TEST_CASES,
        "priority": "high"
    },
    "onnx": {
        "description": "ONNX model integration tests",
        "test_cases": ONNX_TEST_CASES,
        "priority": "high"
    },
    "tensorrt": {
        "description": "TensorRT optimization tests",
        "test_cases": TENSORRT_TEST_CASES,
        "priority": "high"
    },
    "comprehensive": {
        "description": "Comprehensive feature integration tests",
        "test_cases": COMPREHENSIVE_TEST_CASES,
        "priority": "medium"
    },
    "edge_cases": {
        "description": "Edge case tests",
        "test_cases": EDGE_CASE_TEST_CASES,
        "priority": "low"
    }
} 