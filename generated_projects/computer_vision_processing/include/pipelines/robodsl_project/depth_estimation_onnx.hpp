#ifndef ROBODSL_PROJECT_DEPTH_ESTIMATION_ONNX_HPP
#define ROBODSL_PROJECT_DEPTH_ESTIMATION_ONNX_HPP

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace robodsl_project {

class depth_estimationOnnxManager {
public:
    depth_estimationOnnxManager();
    ~depth_estimationOnnxManager();
    
    // Initialize ONNX Runtime and load models
    bool initialize();
    
    // Run inference with ONNX models
    bool run_inference(const std::vector<float>& input_data, 
                      std::vector<float>& output_data);
    
    // Cleanup ONNX resources
    void cleanup();

private:
    // ONNX Runtime environment and session
    Ort::Env env_;
    std::vector<std::unique_ptr<Ort::Session>> sessions_;
    
    // Model file paths
    std::string depth_estimation_model_path_;
    
    // Input/output names and shapes
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    // Helper methods
    bool load_models();
    bool setup_io_info();
    bool prepare_input_tensors(const std::vector<float>& input_data,
                              std::vector<Ort::Value>& input_tensors);
    bool extract_output_tensors(const std::vector<Ort::Value>& output_tensors,
                               std::vector<float>& output_data);
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_DEPTH_ESTIMATION_ONNX_HPP 