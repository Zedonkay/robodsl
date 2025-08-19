cmake_minimum_required(VERSION 3.8)
project(robodsl_package VERSION 0.1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Only add warning flags for C++ (not CUDA)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
  add_compile_options(-fPIC)
endif()

# Build configurations
set(CMAKE_CONFIGURATION_TYPES "")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)


# CUDA support
find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
set(CMAKE_CUDA_STANDARD 17)

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()


# Include directories
include_directories(include)

# Create library for common code (INTERFACE for header-only library)
add_library(${PROJECT_NAME}_lib INTERFACE)

target_include_directories(${PROJECT_NAME}_lib INTERFACE include)
ament_target_dependencies(${PROJECT_NAME}_lib INTERFACE rclcpp)

# CUDA library
add_library(${PROJECT_NAME}_cuda STATIC
  src/cuda/image_preprocessing_kernel.cu
  src/cuda/image_resize_kernel.cu
  src/cuda/gaussian_blur_kernel.cu
  src/cuda/edge_detection_kernel.cu
  src/cuda/feature_matching_kernel.cu
)

set_target_properties(${PROJECT_NAME}_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${PROJECT_NAME}_cuda PUBLIC include)
target_link_libraries(${PROJECT_NAME}_cuda ${PROJECT_NAME}_lib)

# Create executables
add_executable(vision_processor_node
  src/nodes/perception/vision_processor_node.cpp
)

target_link_libraries(vision_processor_node ${PROJECT_NAME}_lib)
target_link_libraries(vision_processor_node ${PROJECT_NAME}_cuda)

ament_target_dependencies(vision_processor_node rclcpp)
ament_target_dependencies(vision_processor_node geometry_msgs)
ament_target_dependencies(vision_processor_node rclcpp)
ament_target_dependencies(vision_processor_node rclcpp_lifecycle)
ament_target_dependencies(vision_processor_node sensor_msgs)
ament_target_dependencies(vision_processor_node std_msgs)

add_executable(camera_synchronizer_node
  src/nodes/perception/camera_synchronizer_node.cpp
)

target_link_libraries(camera_synchronizer_node ${PROJECT_NAME}_lib)
target_link_libraries(camera_synchronizer_node ${PROJECT_NAME}_cuda)

ament_target_dependencies(camera_synchronizer_node rclcpp)
ament_target_dependencies(camera_synchronizer_node geometry_msgs)
ament_target_dependencies(camera_synchronizer_node rclcpp)
ament_target_dependencies(camera_synchronizer_node rclcpp_lifecycle)
ament_target_dependencies(camera_synchronizer_node sensor_msgs)
ament_target_dependencies(camera_synchronizer_node std_msgs)


# Python modules (if any)
# python_install_package(${PROJECT_NAME})  # Disabled for C++-only packages

# Install targets
install(TARGETS
  vision_processor_node
  camera_synchronizer_node
  ${PROJECT_NAME}_lib
  ${PROJECT_NAME}_cuda
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cuh"
)

# Install Python modules (if any)
# install(DIRECTORY
#   ${PROJECT_NAME}
#   DESTINATION lib/python3/dist-packages/
# )

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install configuration files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  
  
  # Add tests
  
  # Python tests (disabled for C++-only packages)
  # ament_add_pytest_test(test_${PROJECT_NAME}_python
  #   ${PROJECT_NAME}/test_${PROJECT_NAME}.py
  #   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  # )
endif()

# Documentation
if(BUILD_DOCS)
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(docs ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM
    )
  endif()
endif()

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_lib)
ament_export_libraries(${PROJECT_NAME}_cuda)
ament_export_dependencies(rclcpp)
ament_export_dependencies(geometry_msgs)
ament_export_dependencies(rclcpp)
ament_export_dependencies(rclcpp_lifecycle)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(std_msgs)

ament_package()
