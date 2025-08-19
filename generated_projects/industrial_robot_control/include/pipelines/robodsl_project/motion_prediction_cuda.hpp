#ifndef ROBODSL_PROJECT_MOTION_PREDICTION_CUDA_HPP
#define ROBODSL_PROJECT_MOTION_PREDICTION_CUDA_HPP

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>

namespace robodsl_project {

class motion_predictionCudaManager {
public:
    motion_predictionCudaManager();
    ~motion_predictionCudaManager();
    
    // Initialize CUDA context and load kernels
    bool initialize();
    
    // Execute CUDA kernels for this stage
    bool process_data(const std::vector<float>& input_data, 
                     std::vector<float>& output_data);
    
    // Cleanup CUDA resources
    void cleanup();

private:
    // CUDA kernel handles
    void* motion_prediction_kernel_;
    
    // CUDA stream for asynchronous execution
    cudaStream_t stream_;
    
    // Memory buffers
    float* d_input_buffer_;
    float* d_output_buffer_;
    size_t buffer_size_;
    
    // Helper methods
    bool allocate_buffers(size_t data_size);
    void free_buffers();
    bool load_kernels();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_MOTION_PREDICTION_CUDA_HPP 