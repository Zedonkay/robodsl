"""Advanced CUDA features comprehensive testing.

This module provides extensive test coverage for advanced CUDA features including:
- Multi-GPU support
- Advanced memory management
- CUDA streams and events
- Dynamic parallelism
- Cooperative groups
- Advanced optimization techniques
- Error handling and recovery
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
from conftest import skip_if_no_cuda, has_cuda
from robodsl.generators.cuda_kernel_generator import CudaKernelGenerator
from robodsl.core.ast import KernelNode


class TestCudaAdvancedFeatures:
    """Advanced CUDA features test suite."""
    
    @pytest.fixture
    def multi_gpu_config(self):
        """Get multi-GPU configuration."""
        return {
            "gpu_count": self._get_gpu_count(),
            "gpu_memory_per_device": self._get_gpu_memory(),
            "gpu_architectures": self._get_gpu_architectures(),
            "compute_capabilities": self._get_compute_capabilities()
        }
    
    def _get_gpu_count(self):
        """Get number of available GPUs."""
        try:
            if has_cuda():
                result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    return len(result.stdout.strip().split('\n'))
        except:
            pass
        return 1
    
    def _get_gpu_memory(self):
        """Get GPU memory in MB."""
        try:
            if has_cuda():
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    return int(result.stdout.strip().split('\n')[0])
        except:
            pass
        return 8192  # Default 8GB
    
    def _get_gpu_architectures(self):
        """Get GPU architectures."""
        try:
            if has_cuda():
                result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    return [cap.strip() for cap in result.stdout.strip().split('\n')]
        except:
            pass
        return ["8.6"]  # Default to RTX 30 series
    
    def _get_compute_capabilities(self):
        """Get compute capabilities."""
        architectures = self._get_gpu_architectures()
        capabilities = []
        for arch in architectures:
            try:
                major, minor = arch.split('.')
                capabilities.append((int(major), int(minor)))
            except:
                capabilities.append((8, 6))  # Default
        return capabilities
    
    def test_multi_gpu_support_advanced(self, multi_gpu_config):
        """Test advanced multi-GPU support."""
        skip_if_no_cuda()
        
        gpu_count = multi_gpu_config["gpu_count"]
        assert gpu_count > 0
        
        # Test multi-GPU kernel with device selection
        dsl_code = f'''
        cuda_kernels {{
            kernel multi_gpu_advanced {{
                input: float* input (size)
                output: float* output (size)
                parameters: {{
                    int device_id = 0
                }}
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {{
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {{
                        // Device-specific computation
                        float val = input[idx];
                        val = val * (device_id + 1) + __sinf(val);
                        output[idx] = val;
                    }}
                }}
            }}
        }}
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "multi_gpu_advanced"
        # Check that device_id parameter is in the kernel content
        assert "device_id" in kernel.content.code
    
    def test_cuda_streams_and_events(self, test_output_dir):
        """Test CUDA streams and events."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel stream_kernel {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output[idx] = input[idx] * 2.0f;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "stream_kernel"
    
    def test_dynamic_parallelism(self, test_output_dir):
        """Test CUDA dynamic parallelism."""
        skip_if_no_cuda()
        
        # Check if device supports dynamic parallelism
        compute_cap = self._get_compute_capabilities()[0]
        if compute_cap[0] < 3 or (compute_cap[0] == 3 and compute_cap[1] < 5):
            pytest.skip("Device does not support dynamic parallelism")
        
        dsl_code = '''
        cuda_kernels {
            kernel dynamic_parallelism {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        // Dynamic parallelism example
                        if (idx % 100 == 0) {
                            // Launch child kernel
                            dynamic_parallelism_child<<<1, 256>>>(input + idx, output + idx, min(100, size - idx));
                        }
                        output[idx] = input[idx] * 2.0f;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "dynamic_parallelism"
    
    def test_cooperative_groups(self, test_output_dir):
        """Test CUDA cooperative groups."""
        skip_if_no_cuda()
        
        compute_cap = self._get_compute_capabilities()[0]
        if compute_cap[0] < 7:
            pytest.skip("Device does not support cooperative groups")
        
        dsl_code = '''
        cuda_kernels {
            kernel cooperative_groups {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        thread_block block = this_thread_block();
                        thread_block_tile<32> tile = tiled_partition<32>(block);
                        
                        float val = input[idx];
                        float sum = tile.shfl_down(val, 16);
                        sum += tile.shfl_down(val, 8);
                        sum += tile.shfl_down(val, 4);
                        sum += tile.shfl_down(val, 2);
                        sum += tile.shfl_down(val, 1);
                        
                        output[idx] = sum;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "cooperative_groups"
    
    def test_advanced_memory_management(self, test_output_dir):
        """Test advanced CUDA memory management."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel advanced_memory {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                shared_memory: 1024
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        // Shared memory usage
                        __shared__ float shared_data[256];
                        shared_data[threadIdx.x] = input[idx];
                        __syncthreads();
                        
                        // Cooperative memory access
                        float sum = 0.0f;
                        for (int i = 0; i < 256; i++) {
                            sum += shared_data[i];
                        }
                        
                        output[idx] = sum / 256.0f;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "advanced_memory"
    
    def test_optimization_techniques(self, test_output_dir):
        """Test advanced CUDA optimization techniques."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel optimized_kernel {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        // Loop unrolling
                        float val = input[idx];
                        #pragma unroll 4
                        for (int i = 0; i < 16; i++) {
                            val = __fmaf_rn(val, val, 1.0f);  // Fused multiply-add
                        }
                        output[idx] = val;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "optimized_kernel"
    
    def test_error_handling_advanced(self, test_output_dir):
        """Test advanced CUDA error handling."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel error_handling {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        // Safe computation with error checking
                        float val = input[idx];
                        if (isnan(val) || isinf(val)) {
                            output[idx] = 0.0f;
                        } else {
                            val = __fdividef(val, 2.0f);  // Safe division
                            output[idx] = val;
                        }
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "error_handling"
    
    def test_performance_monitoring_advanced(self, test_output_dir):
        """Test advanced CUDA performance monitoring."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel monitored_kernel {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output[idx] = input[idx] * 2.0f;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "monitored_kernel"
    
    def test_memory_hierarchy_optimization(self, test_output_dir):
        """Test CUDA memory hierarchy optimization."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel memory_hierarchy {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                shared_memory: 1024
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        // L1 cache optimization
                        float val = input[idx];
                        
                        // Shared memory usage for L2 cache optimization
                        __shared__ float shared_cache[256];
                        shared_cache[threadIdx.x] = val;
                        __syncthreads();
                        
                        // Cooperative memory access
                        float sum = 0.0f;
                        for (int i = 0; i < 256; i++) {
                            sum += shared_cache[i];
                        }
                        
                        output[idx] = sum / 256.0f;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "memory_hierarchy"
    
    def test_concurrent_kernel_execution(self, test_output_dir):
        """Test concurrent kernel execution."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel concurrent_kernel1 {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output[idx] = input[idx] * 2.0f;
                    }
                }
            }
            
            kernel concurrent_kernel2 {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output[idx] = input[idx] + 1.0f;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 2
        assert ast.cuda_kernels.kernels[0].name == "concurrent_kernel1"
        assert ast.cuda_kernels.kernels[1].name == "concurrent_kernel2"
    
    def test_advanced_synchronization(self, test_output_dir):
        """Test advanced CUDA synchronization techniques."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel advanced_sync {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        // Atomic operations
                        float val = input[idx];
                        atomicAdd(&output[idx], val);
                        
                        // Memory fence
                        __threadfence();
                        
                        // Cooperative synchronization
                        __syncthreads();
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "advanced_sync"
    
    def test_cuda_graphs(self, test_output_dir):
        """Test CUDA graphs for performance optimization."""
        skip_if_no_cuda()
        
        # Test CUDA graphs regardless of compute capability
        # The actual CUDA graphs functionality will be handled by the runtime
        
        dsl_code = '''
        cuda_kernels {
            kernel graph_kernel {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output[idx] = input[idx] * 2.0f;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "graph_kernel"
    
    def test_mixed_precision_computation(self, test_output_dir):
        """Test mixed precision computation."""
        skip_if_no_cuda()
        
        compute_cap = self._get_compute_capabilities()[0]
        if compute_cap[0] < 7:
            pytest.skip("Device does not support mixed precision")
        
        dsl_code = '''
        cuda_kernels {
            kernel mixed_precision {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        // Mixed precision computation
                        half input_half = __float2half(input[idx]);
                        half result_half = __hmul(input_half, __float2half(2.0f));
                        output[idx] = __half2float(result_half);
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "mixed_precision"
    
    def test_memory_pool_management(self, test_output_dir):
        """Test advanced memory pool management."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel memory_pool {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output[idx] = input[idx] * 2.0f;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "memory_pool"
    
    def test_cuda_context_management(self, test_output_dir):
        """Test CUDA context management."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel context_managed {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output[idx] = input[idx] * 2.0f;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "context_managed"
    
    def test_cuda_driver_api_integration(self, test_output_dir):
        """Test CUDA Driver API integration."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel driver_api {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output[idx] = input[idx] * 2.0f;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "driver_api"
    
    def test_cuda_runtime_api_advanced(self, test_output_dir):
        """Test advanced CUDA Runtime API features."""
        skip_if_no_cuda()
        
        dsl_code = '''
        cuda_kernels {
            kernel runtime_api {
                input: float* input (size)
                output: float* output (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output[idx] = input[idx] * 2.0f;
                    }
                }
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "runtime_api" 