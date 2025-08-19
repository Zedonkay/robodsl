
# ONNX Runtime integration for face_recognition (Pure C++/CUDA/TensorRT)

# Find ONNX Runtime
find_package(ONNXRuntime REQUIRED)

# Find OpenCV (for image preprocessing)
find_package(OpenCV REQUIRED)
# Find TensorRT
find_package(TensorRT REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

# Add include directories
target_include_directories(face_recognition PRIVATE
    ${ONNXRuntime_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(face_recognition PRIVATE
    ${ONNXRuntime_LIBRARIES}
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Copy ONNX model file to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/models/face_recognition.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/face_recognition.onnx
    COPYONLY
)
# Create TensorRT cache directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/trt_cache)

# Set TensorRT cache path
target_compile_definitions(face_recognition PRIVATE
    ONNX_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/face_recognition.onnx"
    TENSORRT_CACHE_PATH="${CMAKE_CURRENT_BINARY_DIR}/trt_cache"
)

# Ensure C++17 standard for ONNX Runtime
set_target_properties(face_recognition PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
# Set CUDA properties
set_target_properties(face_recognition PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86"
)

# Enable CUDA compilation
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/face_recognition_onnx.cpp PROPERTIES LANGUAGE CUDA)