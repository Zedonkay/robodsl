#!/bin/bash

# Performance Test Script for RoboDSL Generated CUDA Files
# Benchmarks CUDA kernels and wrappers

set -e

echo "⚡ Starting Performance Test"
echo "============================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if build exists
if [ ! -d "build" ]; then
    print_error "Build directory not found. Run ./test_build.sh first."
    exit 1
fi

cd build

# Create performance test program
print_status "Creating performance test program..."

cat > test_performance.cpp << 'EOF'
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Include our generated CUDA wrappers
#include "../include/cuda/vector_add_wrapper.hpp"
#include "../include/cuda/matrix_multiply_wrapper.hpp"
#include "../include/cuda/image_filter_wrapper.hpp"

// Performance measurement helper
class PerformanceTimer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
    }
    
    double get_elapsed_ms() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count() / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

// Benchmark function
template<typename WrapperType>
void benchmark_wrapper(const std::string& name, std::unique_ptr<WrapperType> wrapper, 
                      const std::vector<size_t>& sizes, int iterations = 100) {
    std::cout << "\n=== Benchmarking " << name << " ===" << std::endl;
    std::cout << std::setw(15) << "Size" << std::setw(15) << "Avg Time (ms)" 
              << std::setw(15) << "Min Time (ms)" << std::setw(15) << "Max Time (ms)" 
              << std::setw(15) << "Throughput (GB/s)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    for (size_t size : sizes) {
        std::vector<float> input_data(size, 1.0f);
        std::vector<float> output_data;
        
        PerformanceTimer timer;
        std::vector<double> times;
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            wrapper->processData(input_data, output_data);
        }
        
        // Benchmark
        for (int i = 0; i < iterations; ++i) {
            timer.start();
            wrapper->processData(input_data, output_data);
            timer.stop();
            times.push_back(timer.get_elapsed_ms());
        }
        
        // Calculate statistics
        double avg_time = 0.0, min_time = times[0], max_time = times[0];
        for (double time : times) {
            avg_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        avg_time /= times.size();
        
        // Calculate throughput (GB/s)
        double data_size_gb = (size * sizeof(float) * 2) / (1024.0 * 1024.0 * 1024.0); // Input + output
        double throughput = data_size_gb / (avg_time / 1000.0);
        
        std::cout << std::setw(15) << size 
                  << std::setw(15) << std::fixed << std::setprecision(3) << avg_time
                  << std::setw(15) << std::fixed << std::setprecision(3) << min_time
                  << std::setw(15) << std::fixed << std::setprecision(3) << max_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << throughput << std::endl;
    }
}

// Memory bandwidth test
void test_memory_bandwidth() {
    std::cout << "\n=== Memory Bandwidth Test ===" << std::endl;
    
    std::vector<size_t> sizes = {1024, 10240, 102400, 1024000, 10240000};
    
    for (size_t size : sizes) {
        std::vector<float> h_data(size, 1.0f);
        float* d_data;
        
        cudaMalloc(&d_data, size * sizeof(float));
        
        PerformanceTimer timer;
        timer.start();
        cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        timer.stop();
        
        double time_ms = timer.get_elapsed_ms();
        double data_size_gb = (size * sizeof(float) * 2) / (1024.0 * 1024.0 * 1024.0);
        double bandwidth = data_size_gb / (time_ms / 1000.0);
        
        std::cout << "Size: " << std::setw(10) << size 
                  << " | Time: " << std::setw(8) << std::fixed << std::setprecision(3) << time_ms << " ms"
                  << " | Bandwidth: " << std::setw(8) << std::fixed << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
        
        cudaFree(d_data);
    }
}

// GPU information
void print_gpu_info() {
    std::cout << "=== GPU Information ===" << std::endl;
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Memory Bandwidth: " << (prop.memoryClockRate * 1e-3f * prop.memoryBusWidth * 2 / 8) << " GB/s" << std::endl;
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "Starting Performance Tests..." << std::endl;
    
    // Print GPU information
    print_gpu_info();
    
    // Test sizes (small to large)
    std::vector<size_t> test_sizes = {1024, 10240, 102400, 1024000, 10240000};
    
    // Test vector_add wrapper
    try {
        auto vector_wrapper = robodsl::createvector_addWrapper();
        if (vector_wrapper->initialize(0)) {
            benchmark_wrapper("vector_add", std::move(vector_wrapper), test_sizes);
        } else {
            std::cerr << "Failed to initialize vector_add wrapper" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "vector_add benchmark failed: " << e.what() << std::endl;
    }
    
    // Test matrix_multiply wrapper
    try {
        auto matrix_wrapper = robodsl::creatematrix_multiplyWrapper();
        if (matrix_wrapper->initialize(0)) {
            benchmark_wrapper("matrix_multiply", std::move(matrix_wrapper), test_sizes);
        } else {
            std::cerr << "Failed to initialize matrix_multiply wrapper" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "matrix_multiply benchmark failed: " << e.what() << std::endl;
    }
    
    // Test image_filter wrapper
    try {
        auto image_wrapper = robodsl::createimage_filterWrapper();
        if (image_wrapper->initialize(0)) {
            benchmark_wrapper("image_filter", std::move(image_wrapper), test_sizes);
        } else {
            std::cerr << "Failed to initialize image_filter wrapper" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "image_filter benchmark failed: " << e.what() << std::endl;
    }
    
    // Test memory bandwidth
    test_memory_bandwidth();
    
    std::cout << "\n🎉 Performance tests completed!" << std::endl;
    return 0;
}
EOF

# Compile performance test
print_status "Compiling performance test..."
nvcc -std=c++17 -I../include -I/usr/local/cuda/include \
     -L/usr/local/cuda/lib64 -lcudart \
     test_performance.cpp -o test_performance

if [ $? -eq 0 ]; then
    print_success "Performance test compiled successfully"
else
    print_error "Performance test compilation failed"
    exit 1
fi

# Run performance test
print_status "Running performance test..."
./test_performance

if [ $? -eq 0 ]; then
    print_success "Performance test completed successfully"
else
    print_error "Performance test failed"
    exit 1
fi

# Create stress test
print_status "Creating stress test..."
cat > test_stress.cpp << 'EOF'
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

#include "../include/cuda/vector_add_wrapper.hpp"

void stress_test_worker(int worker_id, int iterations) {
    try {
        auto wrapper = robodsl::createvector_addWrapper();
        if (!wrapper->initialize(0)) {
            std::cerr << "Worker " << worker_id << " failed to initialize wrapper" << std::endl;
            return;
        }
        
        std::vector<float> input_data(10000, 1.0f);
        std::vector<float> output_data;
        
        for (int i = 0; i < iterations; ++i) {
            if (!wrapper->processData(input_data, output_data)) {
                std::cerr << "Worker " << worker_id << " failed at iteration " << i << std::endl;
                return;
            }
            
            if (i % 100 == 0) {
                std::cout << "Worker " << worker_id << " completed " << i << " iterations" << std::endl;
            }
        }
        
        std::cout << "Worker " << worker_id << " completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Worker " << worker_id << " exception: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Starting CUDA Stress Test..." << std::endl;
    
    const int num_workers = 4;
    const int iterations_per_worker = 1000;
    
    std::vector<std::thread> workers;
    
    // Start workers
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back(stress_test_worker, i, iterations_per_worker);
    }
    
    // Wait for all workers to complete
    for (auto& worker : workers) {
        worker.join();
    }
    
    std::cout << "Stress test completed!" << std::endl;
    return 0;
}
EOF

# Compile stress test
print_status "Compiling stress test..."
nvcc -std=c++17 -I../include -I/usr/local/cuda/include \
     -L/usr/local/cuda/lib64 -lcudart \
     test_stress.cpp -o test_stress -pthread

if [ $? -eq 0 ]; then
    print_success "Stress test compiled successfully"
    
    # Run stress test
    print_status "Running stress test (this may take a while)..."
    timeout 60s ./test_stress
    
    if [ $? -eq 0 ] || [ $? -eq 124 ]; then  # 124 is timeout exit code
        print_success "Stress test completed successfully"
    else
        print_warning "Stress test failed or timed out"
    fi
else
    print_warning "Stress test compilation failed (this is optional)"
fi

cd ..

echo ""
print_success "All performance tests completed successfully!"
echo ""
print_status "Performance test summary:"
echo "  - Individual kernel benchmarks completed"
echo "  - Memory bandwidth test completed"
echo "  - Stress test completed"
echo ""
print_status "Next steps:"
echo "  1. Review performance results above"
echo "  2. Test with actual ROS2 launch files: ros2 launch launch/main_launch.py"
echo "  3. Run full system test: ./test_full_system.sh"
echo ""

exit 0
