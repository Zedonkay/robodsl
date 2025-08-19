#include "robodsl_project/pose_estimation_onnx.hpp"
#include <iostream>
#include <fstream>

namespace robodsl_project {

pose_estimationOnnxManager::pose_estimationOnnxManager() {
    // Initialize ONNX Runtime environment
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "pose_estimation_onnx");
}

pose_estimationOnnxManager::~pose_estimationOnnxManager() {
    cleanup();
}

bool pose_estimationOnnxManager::initialize() {
    // Set model paths
    pose_estimation_model_path_ = "models/pose_estimation.onnx";
    
    // Load ONNX models
    if (!load_models()) {
        std::cerr << "Failed to load ONNX models" << std::endl;
        return false;
    }
    
    // Setup input/output information
    if (!setup_io_info()) {
        std::cerr << "Failed to setup I/O information" << std::endl;
        return false;
    }
    
    return true;
}

bool pose_estimationOnnxManager::run_inference(const std::vector<float>& input_data, 
                                              std::vector<float>& output_data) {
    if (sessions_.empty()) {
        std::cerr << "No ONNX sessions available" << std::endl;
        return false;
    }
    
    // Prepare input tensors
    std::vector<Ort::Value> input_tensors;
    if (!prepare_input_tensors(input_data, input_tensors)) {
        std::cerr << "Failed to prepare input tensors" << std::endl;
        return false;
    }
    
    // Run inference for each model
    std::vector<Ort::Value> current_inputs = input_tensors;
    std::vector<Ort::Value> current_outputs;
    
    // Run pose_estimation model
    try {
        current_outputs = sessions_[0]->Run(
            Ort::RunOptions{nullptr}, 
            input_names_.data(), 
            current_inputs.data(), 
            input_tensors.size(),
            output_names_.data(), 
            output_names_.size()
        );
        
        // Use outputs as inputs for next model (if any)
        current_inputs = current_outputs;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX inference failed for pose_estimation: " << e.what() << std::endl;
        return false;
    }
    
    // Extract final output data
    if (!extract_output_tensors(current_outputs, output_data)) {
        std::cerr << "Failed to extract output tensors" << std::endl;
        return false;
    }
    
    return true;
}

void pose_estimationOnnxManager::cleanup() {
    sessions_.clear();
}

bool pose_estimationOnnxManager::load_models() {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    // Load pose_estimation model
    try {
        auto session = std::make_unique<Ort::Session>(env_, pose_estimation_model_path_.c_str(), session_options);
        sessions_.push_back(std::move(session));
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load pose_estimation model: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

bool pose_estimationOnnxManager::setup_io_info() {
    if (sessions_.empty()) {
        return false;
    }
    
    // Get input/output information from the first session
    auto& session = sessions_[0];
    
    // Get input names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session->GetInputCount();
    input_names_.resize(num_input_nodes);
    input_shapes_.resize(num_input_nodes);
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        input_names_[i] = input_name.get();
        
        auto type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shapes_[i] = tensor_info.GetShape();
    }
    
    // Get output names
    size_t num_output_nodes = session->GetOutputCount();
    output_names_.resize(num_output_nodes);
    output_shapes_.resize(num_output_nodes);
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        output_names_[i] = output_name.get();
        
        auto type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_shapes_[i] = tensor_info.GetShape();
    }
    
    return true;
}

bool pose_estimationOnnxManager::prepare_input_tensors(const std::vector<float>& input_data,
                                                       std::vector<Ort::Value>& input_tensors) {
    if (input_shapes_.empty()) {
        return false;
    }
    
    input_tensors.clear();
    
    // Create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Calculate total elements from shape
    size_t total_elements = 1;
    for (auto dim : input_shapes_[0]) {
        if (dim > 0) {
            total_elements *= dim;
        }
    }
    
    // Reshape input data if needed
    std::vector<float> reshaped_data = input_data;
    if (reshaped_data.size() < total_elements) {
        reshaped_data.resize(total_elements, 0.0f);
    } else if (reshaped_data.size() > total_elements) {
        reshaped_data.resize(total_elements);
    }
    
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, reshaped_data.data(), reshaped_data.size(),
        input_shapes_[0].data(), input_shapes_[0].size()
    );
    
    input_tensors.push_back(std::move(input_tensor));
    
    return true;
}

bool pose_estimationOnnxManager::extract_output_tensors(const std::vector<Ort::Value>& output_tensors,
                                                        std::vector<float>& output_data) {
    if (output_tensors.empty()) {
        return false;
    }
    
    // Extract data from the first output tensor
    auto& output_tensor = output_tensors[0];
    float* output_buffer = output_tensor.GetTensorMutableData<float>();
    size_t output_size = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    
    output_data.assign(output_buffer, output_buffer + output_size);
    
    return true;
}

} // namespace robodsl_project 