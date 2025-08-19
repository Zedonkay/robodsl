
# ONNX Runtime integration for yolo_detector (Pure C++/CUDA/TensorRT)

# Find ONNX Runtime
find_package(ONNXRuntime REQUIRED)

# Find OpenCV (for image preprocessing)
find_package(OpenCV REQUIRED)
# Find TensorRT
find_package(TensorRT REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

# Add include directories
target_include_directories(yolo_detector PRIVATE
    ${ONNXRuntime_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(yolo_detector PRIVATE
    ${ONNXRuntime_LIBRARIES}
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Copy ONNX model file to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/models/yolo_detector.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/yolo_detector.onnx
    COPYONLY
)
# Create TensorRT cache directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/trt_cache)

# Set TensorRT cache path
target_compile_definitions(yolo_detector PRIVATE
    ONNX_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/yolo_detector.onnx"
    TENSORRT_CACHE_PATH="${CMAKE_CURRENT_BINARY_DIR}/trt_cache"
)

# Ensure C++17 standard for ONNX Runtime
set_target_properties(yolo_detector PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
# Set CUDA properties
set_target_properties(yolo_detector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86"
)

# Enable CUDA compilation
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/yolo_detector_onnx.cpp PROPERTIES LANGUAGE CUDA)