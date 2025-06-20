# CUDA Acceleration

RoboDSL provides first-class support for CUDA acceleration, allowing you to write high-performance GPU-accelerated code directly in your RoboDSL files. This document covers how to define and use CUDA kernels, manage device memory, and optimize your GPU code.

## Table of Contents
- [Defining CUDA Kernels](#defining-cuda-kernels)
- [Memory Management](#memory-management)
- [Streams and Events](#streams-and-events)
- [Multi-GPU Support](#multi-gpu-support)
- [Thrust Integration](#thrust-integration)
- [Performance Optimization](#performance-optimization)
- [Debugging CUDA Code](#debugging-cuda-code)

## Defining CUDA Kernels

### Basic Kernel Definition

```robodsl
// Define a CUDA kernel
cuda_kernels {
    kernel vector_add {
        // Input parameters
        input float* a
        input float* b
        output float* c
        input int n
        
        // Kernel configuration
        block_size = (256, 1, 1)
        grid_size = ((n + 255) / 256, 1, 1)
        
        // Kernel code
        code {
            __global__ void vector_add_kernel(
                const float* a,
                const float* b,
                float* c,
                int n) {
                
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }
        }
    }
}
```

### Kernel with Shared Memory

```robodsl
cuda_kernels {
    kernel matrix_multiply {
        // Input parameters
        input float* a
        input float* b
        output float* c
        input int width
        
        // Shared memory declaration
        shared: "extern __shared__ float sdata[];"
        
        // Kernel configuration
        block_size = (16, 16, 1)
        grid_size = (width / 16, width / 16, 1)
        
        // Shared memory size (in bytes)
        shared_mem_size: "2 * 16 * 16 * sizeof(float)"
        
        // Kernel code
        code {
            __global__ void matrix_multiply_kernel(
                const float* A,
                const float* B,
                float* C,
                int width) {
                
                // 2D thread and block indices
                int tx = threadIdx.x;
                int ty = threadIdx.y;
                int bx = blockIdx.x;
                int by = blockIdx.y;
                
                // Shared memory for tiles
                extern __shared__ float sdata[];
                float* sA = sdata;
                float* sB = sdata + 16 * 16;
                
                // Calculate row and column for this thread
                int row = by * blockDim.y + ty;
                int col = bx * blockDim.x + tx;
                
                float sum = 0.0f;
                
                // Loop over tiles
                for (int t = 0; t < width / 16; ++t) {
                    // Load tiles into shared memory
                    sA[ty * 16 + tx] = A[row * width + (t * 16 + tx)];
                    sB[ty * 16 + tx] = B[(t * 16 + ty) * width + col];
                    
                    // Synchronize to ensure all threads have loaded their data
                    __syncthreads();
                    
                    // Compute partial product
                    for (int k = 0; k < 16; ++k) {
                        sum += sA[ty * 16 + k] * sB[k * 16 + tx];
                    }
                    
                    // Synchronize before next tile
                    __syncthreads();
                }
                
                // Write result to global memory
                if (row < width && col < width) {
                    C[row * width + col] = sum;
                }
            }
        }
    }
}
```

## Memory Management

### Device Memory Allocation

RoboDSL provides automatic memory management for device memory:

```robodsl
node memory_example {
    // Allocate device memory
    device_memory {
        // Allocate a float array on the device
        float* d_a = cuda_malloc<float>(1024)
        
        // Allocate a 2D array
        float* d_matrix = cuda_malloc_2d<float>(1024, 1024)
        
        // Allocate a 3D array
        float* d_volume = cuda_malloc_3d<float>(256, 256, 256)
        
        // Memory is automatically freed when it goes out of scope
    }
    
    // Or use smart pointers for automatic management
    device_ptr<float> d_data = cuda_make_shared<float>(1024)
    
    // Access the raw pointer when needed
    float* raw_ptr = d_data.get()
}
```

### Memory Transfers

```robodsl
node memory_transfer_example {
    // Host data
    std::vector<float> h_data(1024, 1.0f)
    
    // Allocate device memory
    device_ptr<float> d_data = cuda_make_shared<float>(1024)
    
    // Copy host to device
    cuda_memcpy_host_to_device(d_data.get(), h_data.data(), 1024 * sizeof(float))
    
    // Process data on device...
    
    // Copy device to host
    std::vector<float> h_result(1024)
    cuda_memcpy_device_to_host(h_result.data(), d_data.get(), 1024 * sizeof(float))
    
    // Asynchronous copy with stream
    cuda_stream stream = cuda_stream_create()
    cuda_memcpy_async(
        d_data.get(), 
        h_data.data(), 
        1024 * sizeof(float), 
        cudaMemcpyHostToDevice, 
        stream
    )
    
    // Wait for the copy to complete
    cuda_stream_synchronize(stream)
    cuda_stream_destroy(stream)
}
```

## Streams and Events

### Using Streams for Concurrent Execution

```robodsl
node stream_example {
    // Create streams
    cuda_stream stream1 = cuda_stream_create()
    cuda_stream stream2 = cuda_stream_create()
    
    // Allocate host and device memory
    std::vector<float> h_data1(1024, 1.0f)
    std::vector<float> h_data2(1024, 2.0f)
    device_ptr<float> d_data1 = cuda_make_shared<float>(1024)
    device_ptr<float> d_data2 = cuda_make_shared<float>(1024)
    
    // Asynchronous memory copies in different streams
    cuda_memcpy_async(
        d_data1.get(), 
        h_data1.data(), 
        1024 * sizeof(float), 
        cudaMemcpyHostToDevice, 
        stream1
    )
    
    cuda_memcpy_async(
        d_data2.get(), 
        h_data2.data(), 
        1024 * sizeof(float), 
        cudaMemcpyHostToDevice, 
        stream2
    )
    
    // Launch kernels in different streams
    launch_kernel("vector_add", stream1, d_data1.get(), d_data2.get(), d_data1.get(), 1024)
    launch_kernel("vector_add", stream2, d_data2.get(), d_data1.get(), d_data2.get(), 1024)
    
    // Synchronize streams
    cuda_stream_synchronize(stream1)
    cuda_stream_synchronize(stream2)
    
    // Clean up
    cuda_stream_destroy(stream1)
    cuda_stream_destroy(stream2)
}
```

### Using Events for Timing

```robodsl
node timing_example {
    // Create events for timing
    cuda_event start = cuda_event_create()
    cuda_event stop = cuda_event_create()
    
    // Record start event
    cuda_event_record(start, 0)
    
    // Execute kernel
    launch_kernel("vector_add", 0, d_a, d_b, d_c, 1024)
    
    // Record stop event
    cuda_event_record(stop, 0)
    cuda_event_synchronize(stop)
    
    // Calculate elapsed time
    float elapsed_time = 0.0f
    cuda_event_elapsed_time(&elapsed_time, start, stop)
    
    print("Kernel execution time: ", elapsed_time, " ms")
    
    // Clean up
    cuda_event_destroy(start)
    cuda_event_destroy(stop)
}
```

## Multi-GPU Support

```robodsl
node multi_gpu_example {
    // Get number of available GPUs
    int num_gpus = cuda_get_device_count()
    
    // Process data on each GPU
    for (int i = 0; i < num_gpus; ++i) {
        // Set the current device
        cuda_set_device(i)
        
        // Allocate device memory
        device_ptr<float> d_data = cuda_make_shared<float>(1024 * 1024)
        
        // Process on this GPU
        launch_kernel("process_data", 0, d_data.get(), 1024 * 1024)
        
        // Results will be automatically synchronized
    }
    
    // Reset to default device
    cuda_set_device(0)
}
```

## Thrust Integration

RoboDSL includes built-in support for Thrust, a parallel algorithms library:

```robodsl
node thrust_example {
    // Create host vector
    std::vector<float> h_data(1024)
    
    // Fill with random numbers
    std::generate(h_data.begin(), h_data.end(), rand)
    
    // Copy to device
    device_ptr<float> d_data = cuda_make_shared<float>(1024)
    cuda_memcpy_host_to_device(d_data.get(), h_data.data(), 1024 * sizeof(float))
    
    // Sort on device using Thrust
    thrust_sort(d_data.get(), d_data.get() + 1024)
    
    // Find maximum element
    float max_val = thrust_reduce(
        d_data.get(), 
        d_data.get() + 1024, 
        std::numeric_limits<float>::min(), 
        thrust::maximum<float>()
    )
    
    print("Maximum value: ", max_val)
}
```

## Performance Optimization

### Profiling with Nsight Systems

RoboDSL integrates with Nsight Systems for performance profiling:

```robodsl
node profiling_example {
    // Enable NVTX markers for profiling
    enable_nvtx: true
    
    // Add NVTX ranges
    nvtx_range("Initialization") {
        // Initialization code
    }
    
    // Profile kernel execution
    nvtx_range("Kernel Execution") {
        launch_kernel("compute", 0, d_data, 1024)
    }
}
```

### Optimizing Memory Access

```robodsl
node memory_optimization_example {
    // Use coalesced memory access
    kernel coalesced_access {
        input float* input
        output float* output
        input int width
        input int height
        
        block_size = (16, 16, 1)
        grid_size = (width / 16, height / 16, 1)
        
        code {
            __global__ void coalesced_access_kernel(
                const float* input,
                float* output,
                int width,
                int height) {
                
                // Coalesced access pattern: adjacent threads access adjacent memory locations
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (x < width && y < height) {
                    // Coalesced access: threads in a warp access consecutive memory locations
                    int idx = y * width + x;
                    output[idx] = input[idx] * 2.0f;
                }
            }
        }
    }
}
```

## Debugging CUDA Code

RoboDSL provides several tools for debugging CUDA code:

### Error Checking

```robodsl
node error_checking_example {
    // Check for CUDA errors
    cuda_error_t err = cuda_get_last_error()
    if (err != cudaSuccess) {
        print("CUDA error: ", cuda_get_error_string(err))
        return
    }
    
    // Or use the CHECK_CUDA macro
    CHECK_CUDA(cudaMalloc(&d_data, 1024 * sizeof(float)))
    
    // Check after kernel launch
    launch_kernel("my_kernel", 0, d_data, 1024)
    CHECK_CUDA(cuda_get_last_error())
    
    // Synchronize and check for errors
    CHECK_CUDA(cuda_device_synchronize())
}
```

### Debugging with cuda-gdb

1. Compile with debug symbols:
   ```bash
   robodsl build --debug
   ```

2. Run with cuda-gdb:
   ```bash
   cuda-gdb --args my_robot_node
   ```

3. Set breakpoints and debug as usual

### Debugging with Nsight

1. Launch Nsight from the command line:
   ```bash
   nsight-sys --trace=cuda,nvtx ./my_robot_node
   ```

2. Or use the Nsight VSCode extension for a GUI experience

## Best Practices

1. **Use shared memory** for data reuse within a block
2. **Minimize global memory** access latency by using coalesced access patterns
3. **Avoid thread divergence** within warps
4. **Use constant memory** for read-only data accessed by all threads
5. **Profile your code** to identify bottlenecks
6. **Check for errors** after every CUDA API call
7. **Use streams** for concurrent execution of independent operations
8. **Consider memory transfers** between host and device - they are expensive!
9. **Use the right block size** for your hardware (typically 128-256 threads per block)
10. **Prefer 32-bit data types** when possible for better memory bandwidth utilization
