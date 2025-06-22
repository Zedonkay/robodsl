# CUDA Acceleration

RoboDSL provides seamless CUDA integration for GPU acceleration. Here's a quick reference:

## Basic Usage

```robodsl
// In your .robodsl file
node gpu_processor {
    // Enable CUDA and specify kernels
    cuda_kernels = ["vector_add", "matrix_mult"]
    
    // CUDA source files
    cuda_sources = [
        "src/kernels/vector_ops.cu",
        "src/kernels/matrix_ops.cu"
    ]
    
    // Compilation flags
    cuda_flags = ["-O3"]
}
```

## Kernel Definition

```cuda
// In vector_ops.cu
__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

## Memory Management

```robodsl
// Allocate device memory
device_ptr<float> d_data = cuda_malloc<float>(1024)

// Copy data to device
cuda_memcpy_host_to_device(d_data.get(), h_data.data(), 1024 * sizeof(float))

// Process on GPU...

// Copy back to host
cuda_memcpy_device_to_host(h_result.data(), d_data.get(), 1024 * sizeof(float))
```

## Advanced Features

### Streams

```robodsl
// Create and use streams
cuda_stream stream = cuda_stream_create()

// Launch kernel with stream
cuda_launch_kernel(
    kernel_func,
    grid_size,
    block_size,
    args...,
    stream
)

// Synchronize stream
cuda_stream_synchronize(stream)
cuda_stream_destroy(stream)
```

### Shared Memory

```cuda
// In your CUDA kernel
__global__ void shared_mem_kernel(float* input, float* output) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    // Load into shared memory
    sdata[tid] = input[tid];
    __syncthreads();
    
    // Process...
    
    // Write back
    output[tid] = sdata[tid];
}
```

## Multi-GPU

```robodsl
// Set device
cuda_set_device(device_id)

// Get device count
int count = cuda_get_device_count()

// Get device properties
cuda_device_prop prop = cuda_get_device_properties(device_id)
```

## Performance Tips

1. Use `device_ptr` for automatic memory management
2. Prefer asynchronous operations with streams
3. Minimize host-device transfers
4. Use shared memory for data reuse
5. Set appropriate block/grid sizes

## Debugging

```robodsl
// Check for CUDA errors
cuda_error_t err = cuda_get_last_error()
if (err != cudaSuccess) {
    // Handle error
}

// Synchronize device
cuda_device_synchronize()
```

## Integration with ROS2

```robodsl
node gpu_node {
    // Process ROS2 messages on GPU
    subscribers = [{
        name = "input"
        type = "sensor_msgs/msg/Image"
        callback = "process_image"
    }]
    
    publishers = [{
        name = "output"
        type = "sensor_msgs/msg/Image"
    }]
    
    cuda_kernels = ["image_processing"]
}
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
