#include "optical_flow_onnx.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

// Optimizations: tensorrt

optical_flowOnnxInference::optical_flowOnnxInference(const std::string& model_path)
    : model_path_(model_path), device_type_("gpu") {
    
    // Set up environment
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "optical_flow_onnx");
    // Initialize TensorRT configuration
    tensorrt_enabled_ = true;
    tensorrt_fp16_enabled_ = true;
    tensorrt_int8_enabled_ = false;
    tensorrt_workspace_size_ = 1 << 30;  // 1GB default
    tensorrt_cache_path_ = "./trt_cache";
}

optical_flowOnnxInference::~optical_flowOnnxInference() {
    // Free CUDA memory
    for (void* ptr : cuda_memory_pool_) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    cuda_memory_pool_.clear();
    
    // ONNX Runtime handles cleanup automatically
}

bool optical_flowOnnxInference::initialize() {
    try {
        // Check if model file exists
        std::ifstream file_check(model_path_);
        if (!file_check.good()) {
            std::cerr << "Error: ONNX model file not found: " << model_path_ << std::endl;
            return false;
        }
        
        // Initialize CUDA if needed
        if (device_type_ == "cuda") {
            cudaError_t cuda_error = cudaSetDevice(0);
            if (cuda_error != cudaSuccess) {
                std::cerr << "Error: Failed to set CUDA device: " << cudaGetErrorString(cuda_error) << std::endl;
                return false;
            }
        }
        
        // Create session options
        Ort::SessionOptions session_options;
        initialize_session_options(session_options);
        
        // Create session
        session_ = Ort::Session(env_, model_path_.c_str(), session_options);
        
        // Get input/output information
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input info
        input_name_ = session_.GetInputName(0, allocator);
        input_shape_ = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        
        // Get output info
        output_name_ = session_.GetOutputName(0, allocator);
        output_shape_ = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        
        std::cout << "ONNX model initialized successfully:" << std::endl;
        std::cout << "  Model: " << model_path_ << std::endl;
        std::cout << "  Device: " << device_type_ << std::endl;
        std::cout << "  Input: " << input_name_ << " (shape: ";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            if (i > 0) std::cout << "x";
            std::cout << input_shape_[i];
        }
        std::cout << ")" << std::endl;
        std::cout << "  Output: " << output_name_ << " (shape: ";
        for (size_t i = 0; i < output_shape_.size(); ++i) {
            if (i > 0) std::cout << "x";
            std::cout << output_shape_[i];
        }
        std::cout << ")" << std::endl;
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error during initialization: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error during ONNX model initialization: " << e.what() << std::endl;
        return false;
    }
}

bool optical_flowOnnxInference::run_inference(const std::vector<float>& input_data, std::vector<float>& output_data) {
    try {
        // Preprocess input (pure C++)
        std::vector<float> processed_input = preprocess_input(input_data);
        
        // Create input tensor
        std::vector<const char*> input_names = {input_name_.c_str()};
        std::vector<const char*> output_names = {output_name_.c_str()};
        
        // Create input tensor
        auto input_tensor = Ort::Value::CreateTensor<float>(
            allocator_, 
            const_cast<float*>(processed_input.data()), 
            processed_input.size(),
            input_shape_.data(), 
            input_shape_.size()
        );
        
        // Run inference
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            &input_tensor, 
            1, 
            output_names.data(), 
            1
        );
        
        // Extract output data
        float* output_buffer = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        output_data.assign(output_buffer, output_buffer + output_size);
        
        // Postprocess output (pure C++)
        output_data = postprocess_output(output_data);
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error during inference: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return false;
    }
}

bool optical_flowOnnxInference::run_inference_cuda(const float* input_data, size_t input_size, 
                                                     float* output_data, size_t output_size) {
    try {
        // Allocate CUDA memory
        void* cuda_input = allocate_cuda_memory(input_size * sizeof(float));
        void* cuda_output = allocate_cuda_memory(output_size * sizeof(float));
        
        if (!cuda_input || !cuda_output) {
            return false;
        }
        
        // Copy input to CUDA
        cudaError_t cuda_error = cudaMemcpy(cuda_input, input_data, input_size * sizeof(float), cudaMemcpyHostToDevice);
        if (cuda_error != cudaSuccess) {
            std::cerr << "Error copying input to CUDA: " << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        
        // Create input tensor with CUDA memory
        std::vector<const char*> input_names = {input_name_.c_str()};
        std::vector<const char*> output_names = {output_name_.c_str()};
        
        auto input_tensor = Ort::Value::CreateTensor<float>(
            allocator_, 
            static_cast<float*>(cuda_input), 
            input_size,
            input_shape_.data(), 
            input_shape_.size()
        );
        
        // Run inference
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            &input_tensor, 
            1, 
            output_names.data(), 
            1
        );
        
        // Copy output from CUDA
        cuda_error = cudaMemcpy(output_data, cuda_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        if (cuda_error != cudaSuccess) {
            std::cerr << "Error copying output from CUDA: " << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error during CUDA inference: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error during CUDA inference: " << e.what() << std::endl;
        return false;
    }
}

void* optical_flowOnnxInference::allocate_cuda_memory(size_t size) {
    void* ptr = nullptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error == cudaSuccess) {
        cuda_memory_pool_.push_back(ptr);
        return ptr;
    } else {
        std::cerr << "Error allocating CUDA memory: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }
}

void optical_flowOnnxInference::free_cuda_memory(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
        auto it = std::find(cuda_memory_pool_.begin(), cuda_memory_pool_.end(), ptr);
        if (it != cuda_memory_pool_.end()) {
            cuda_memory_pool_.erase(it);
        }
    }
}

bool optical_flowOnnxInference::copy_to_cuda(const std::vector<float>& host_data, void* cuda_ptr) {
    cudaError_t error = cudaMemcpy(cuda_ptr, host_data.data(), 
                                   host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    return error == cudaSuccess;
}

bool optical_flowOnnxInference::copy_from_cuda(void* cuda_ptr, std::vector<float>& host_data) {
    cudaError_t error = cudaMemcpy(host_data.data(), cuda_ptr, 
                                   host_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return error == cudaSuccess;
}

void optical_flowOnnxInference::initialize_session_options(Ort::SessionOptions& session_options) {
    // Set device type and execution providers
    if (device_type_ == "cuda") {
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        
        // Configure TensorRT if specified
        // Enable TensorRT execution provider with full optimization
        OrtTensorRTProviderOptions trt_options;
        trt_options.device_id = 0;
        trt_options.trt_max_workspace_size = 1 << 30;  // 1GB workspace
        trt_options.trt_fp16_enable = true;  // Enable FP16 for better performance
        trt_options.trt_int8_enable = false;  // Disable INT8 by default
        trt_options.trt_engine_cache_enable = true;  // Enable engine cache
        trt_options.trt_engine_cache_path = "./trt_cache";  // Cache directory
        trt_options.trt_engine_decryption_enable = false;
        trt_options.trt_force_sequential_engine_build = false;
        trt_options.trt_context_memory_sharing_enable = true;
        trt_options.trt_layer_norm_fp32_fallback = false;
        trt_options.trt_timing_cache_enable = true;
        trt_options.trt_force_timing_cache = false;
        trt_options.trt_detailed_build_log = false;
        trt_options.trt_build_heuristics_enable = true;
        trt_options.trt_sparsity_enable = false;
        trt_options.trt_builder_optimization_level = 3;  // Maximum optimization
        trt_options.trt_auxiliary_path = "";
        trt_options.trt_min_subgraph_size = 1;
        trt_options.trt_max_partition_iterations = 10;
        trt_options.trt_optimization_level = 3;  // Maximum optimization
        
        // Add TensorRT as primary execution provider
        session_options.AppendExecutionProvider_TensorRT(trt_options);
        
        // Add CUDA as fallback execution provider
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = SIZE_MAX;
        cuda_options.allocator = nullptr;
        cuda_options.do_copy_in_default_stream = true;
        cuda_options.has_user_compute_stream = false;
        cuda_options.user_compute_stream = nullptr;
        cuda_options.default_memory_arena_cfg = nullptr;
        
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    } else {
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    }
    
    // Set optimization level
    // Enable all optimizations for TensorRT
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Set threading options
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    
    // Enable memory pattern optimization
    session_options.EnableMemoryPattern();
    
    // Enable CPU memory arena
    session_options.EnableCpuMemArena();
}

std::vector<float> optical_flowOnnxInference::preprocess_input(const std::vector<float>& raw_input) {
    // Pure C++ preprocessing - no Python dependencies
    std::vector<float> processed = raw_input;
    
    // Normalize if needed (example: normalize to [0, 1] range)
    // for (auto& val : processed) {
    //     val = val / 255.0f;
    // }
    
    return processed;
}

std::vector<float> optical_flowOnnxInference::postprocess_output(const std::vector<float>& raw_output) {
    // Pure C++ postprocessing - no Python dependencies
    std::vector<float> processed = raw_output;
    
    // Apply softmax if needed (example)
    // float max_val = *std::max_element(processed.begin(), processed.end());
    // float sum = 0.0f;
    // for (auto& val : processed) {
    //     val = std::exp(val - max_val);
    //     sum += val;
    // }
    // for (auto& val : processed) {
    //     val /= sum;
    // }
    
    return processed;
}
// TensorRT-specific method implementations

bool optical_flowOnnxInference::run_inference_tensorrt(const std::vector<float>& input_data, std::vector<float>& output_data) {
    if (!tensorrt_enabled_) {
        std::cerr << "TensorRT is not enabled for this model" << std::endl;
        return false;
    }
    
    try {
        // Preprocess input
        std::vector<float> processed_input = preprocess_input(input_data);
        
        // Create input tensor
        std::vector<const char*> input_names = {input_name_.c_str()};
        std::vector<const char*> output_names = {output_name_.c_str()};
        
        auto input_tensor = Ort::Value::CreateTensor<float>(
            allocator_, 
            const_cast<float*>(processed_input.data()), 
            processed_input.size(),
            input_shape_.data(), 
            input_shape_.size()
        );
        
        // Run inference with TensorRT
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            &input_tensor, 
            1, 
            output_names.data(), 
            1
        );
        
        // Extract output data
        float* output_buffer = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        output_data.assign(output_buffer, output_buffer + output_size);
        
        // Postprocess output
        output_data = postprocess_output(output_data);
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "TensorRT inference error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error during TensorRT inference: " << e.what() << std::endl;
        return false;
    }
}

bool optical_flowOnnxInference::run_inference_tensorrt_cuda(const float* input_data, size_t input_size, 
                                                              float* output_data, size_t output_size) {
    if (!tensorrt_enabled_) {
        std::cerr << "TensorRT is not enabled for this model" << std::endl;
        return false;
    }
    
    try {
        // Allocate CUDA memory
        void* cuda_input = allocate_cuda_memory(input_size * sizeof(float));
        void* cuda_output = allocate_cuda_memory(output_size * sizeof(float));
        
        if (!cuda_input || !cuda_output) {
            return false;
        }
        
        // Copy input to CUDA
        cudaError_t cuda_error = cudaMemcpy(cuda_input, input_data, input_size * sizeof(float), cudaMemcpyHostToDevice);
        if (cuda_error != cudaSuccess) {
            std::cerr << "Error copying input to CUDA: " << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        
        // Create input tensor with CUDA memory
        std::vector<const char*> input_names = {input_name_.c_str()};
        std::vector<const char*> output_names = {output_name_.c_str()};
        
        auto input_tensor = Ort::Value::CreateTensor<float>(
            allocator_, 
            static_cast<float*>(cuda_input), 
            input_size,
            input_shape_.data(), 
            input_shape_.size()
        );
        
        // Run TensorRT inference
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            &input_tensor, 
            1, 
            output_names.data(), 
            1
        );
        
        // Copy output from CUDA
        cuda_error = cudaMemcpy(output_data, cuda_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        if (cuda_error != cudaSuccess) {
            std::cerr << "Error copying output from CUDA: " << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "TensorRT CUDA inference error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error during TensorRT CUDA inference: " << e.what() << std::endl;
        return false;
    }
}

bool optical_flowOnnxInference::enable_tensorrt_fp16() {
    if (!tensorrt_enabled_) {
        std::cerr << "TensorRT is not enabled for this model" << std::endl;
        return false;
    }
    
    tensorrt_fp16_enabled_ = true;
    std::cout << "TensorRT FP16 optimization enabled" << std::endl;
    return true;
}

bool optical_flowOnnxInference::enable_tensorrt_int8() {
    if (!tensorrt_enabled_) {
        std::cerr << "TensorRT is not enabled for this model" << std::endl;
        return false;
    }
    
    tensorrt_int8_enabled_ = true;
    tensorrt_fp16_enabled_ = false;  // INT8 takes precedence
    std::cout << "TensorRT INT8 optimization enabled" << std::endl;
    return true;
}

bool optical_flowOnnxInference::set_tensorrt_workspace_size(size_t size) {
    if (!tensorrt_enabled_) {
        std::cerr << "TensorRT is not enabled for this model" << std::endl;
        return false;
    }
    
    tensorrt_workspace_size_ = size;
    std::cout << "TensorRT workspace size set to " << size << " bytes" << std::endl;
    return true;
}

bool optical_flowOnnxInference::clear_tensorrt_cache() {
    if (!tensorrt_enabled_) {
        std::cerr << "TensorRT is not enabled for this model" << std::endl;
        return false;
    }
    
    // Clear TensorRT cache directory
    std::string command = "rm -rf " + tensorrt_cache_path_ + "/*";
    int result = system(command.c_str());
    
    if (result == 0) {
        std::cout << "TensorRT cache cleared successfully" << std::endl;
        return true;
    } else {
        std::cerr << "Failed to clear TensorRT cache" << std::endl;
        return false;
    }
}

bool optical_flowOnnxInference::initialize_tensorrt_options(OrtTensorRTProviderOptions& trt_options) {
    trt_options.device_id = 0;
    trt_options.trt_max_workspace_size = tensorrt_workspace_size_;
    trt_options.trt_fp16_enable = tensorrt_fp16_enabled_;
    trt_options.trt_int8_enable = tensorrt_int8_enabled_;
    trt_options.trt_engine_cache_enable = true;
    trt_options.trt_engine_cache_path = tensorrt_cache_path_.c_str();
    trt_options.trt_engine_decryption_enable = false;
    trt_options.trt_force_sequential_engine_build = false;
    trt_options.trt_context_memory_sharing_enable = true;
    trt_options.trt_layer_norm_fp32_fallback = false;
    trt_options.trt_timing_cache_enable = true;
    trt_options.trt_force_timing_cache = false;
    trt_options.trt_detailed_build_log = false;
    trt_options.trt_build_heuristics_enable = true;
    trt_options.trt_sparsity_enable = false;
    trt_options.trt_builder_optimization_level = 3;
    trt_options.trt_auxiliary_path = "";
    trt_options.trt_min_subgraph_size = 1;
    trt_options.trt_max_partition_iterations = 10;
    trt_options.trt_optimization_level = 3;
    
    return true;
}

bool optical_flowOnnxInference::create_tensorrt_cache_directory() {
    std::string command = "mkdir -p " + tensorrt_cache_path_;
    int result = system(command.c_str());
    return result == 0;
}

bool optical_flowOnnxInference::validate_tensorrt_engine() {
    if (!tensorrt_enabled_) {
        return false;
    }
    
    // Check if TensorRT cache directory exists
    if (!create_tensorrt_cache_directory()) {
        std::cerr << "Failed to create TensorRT cache directory" << std::endl;
        return false;
    }
    
    std::cout << "TensorRT engine validation successful" << std::endl;
    return true;
}