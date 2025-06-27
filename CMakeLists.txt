cmake_minimum_required(VERSION 3.8)
project(robodsl_package VERSION 0.1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD c++14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options(-fPIC)
endif()

# Build configurations
set(CMAKE_CONFIGURATION_TYPES "Release;Debug;RelWithDebInfo;MinSizeRel")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)

find_package(std_msgs REQUIRED)

# CUDA support
find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
set(CMAKE_CUDA_STANDARD c++14)

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

find_package(CUDA REQUIRED)
find_package(cuda_runtime REQUIRED)
find_package(cudart REQUIRED)

# Include directories
include_directories(include)
include_directories(include)

# Create library for common code
add_library(${PROJECT_NAME}_lib STATIC
  include/image_publisher_node.hpp
  include/pipeline_orchestrator_node.hpp
  include/visualization_node_node.hpp
  include/preprocess_image_kernel.cuh
  include/postprocess_detections_kernel.cuh
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC include)
ament_target_dependencies(${PROJECT_NAME}_lib rclcpp)

# CUDA library
add_library(${PROJECT_NAME}_cuda STATIC
  src/preprocess_image_kernel.cu
  src/postprocess_detections_kernel.cu
)

set_target_properties(${PROJECT_NAME}_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${PROJECT_NAME}_cuda PUBLIC include)
target_link_libraries(${PROJECT_NAME}_cuda ${PROJECT_NAME}_lib)

# Create executables
add_executable(image_publisher_node
  src/image_publisher_node.cpp
  src/pipeline_orchestrator_node.cpp
  src/visualization_node_node.cpp
)

target_link_libraries(image_publisher_node ${PROJECT_NAME}_lib)
target_link_libraries(image_publisher_node ${PROJECT_NAME}_cuda)

ament_target_dependencies(image_publisher_node rclcpp)
ament_target_dependencies(image_publisher_node geometry_msgs)
ament_target_dependencies(image_publisher_node rclcpp)
ament_target_dependencies(image_publisher_node rclcpp_lifecycle)
ament_target_dependencies(image_publisher_node std_msgs)
ament_target_dependencies(image_publisher_node std_msgs)

add_executable(pipeline_orchestrator_node
  src/image_publisher_node.cpp
  src/pipeline_orchestrator_node.cpp
  src/visualization_node_node.cpp
)

target_link_libraries(pipeline_orchestrator_node ${PROJECT_NAME}_lib)
target_link_libraries(pipeline_orchestrator_node ${PROJECT_NAME}_cuda)

ament_target_dependencies(pipeline_orchestrator_node rclcpp)
ament_target_dependencies(pipeline_orchestrator_node geometry_msgs)
ament_target_dependencies(pipeline_orchestrator_node rclcpp)
ament_target_dependencies(pipeline_orchestrator_node rclcpp_lifecycle)
ament_target_dependencies(pipeline_orchestrator_node std_msgs)
ament_target_dependencies(pipeline_orchestrator_node std_msgs)

add_executable(visualization_node_node
  src/image_publisher_node.cpp
  src/pipeline_orchestrator_node.cpp
  src/visualization_node_node.cpp
)

target_link_libraries(visualization_node_node ${PROJECT_NAME}_lib)
target_link_libraries(visualization_node_node ${PROJECT_NAME}_cuda)

ament_target_dependencies(visualization_node_node rclcpp)
ament_target_dependencies(visualization_node_node geometry_msgs)
ament_target_dependencies(visualization_node_node rclcpp)
ament_target_dependencies(visualization_node_node rclcpp_lifecycle)
ament_target_dependencies(visualization_node_node std_msgs)
ament_target_dependencies(visualization_node_node std_msgs)


# Python modules
python_install_package(${PROJECT_NAME})

# Install targets
install(TARGETS
  image_publisher_node
  pipeline_orchestrator_node
  visualization_node_node
  ${PROJECT_NAME}_lib
  ${PROJECT_NAME}_cuda
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cuh"
)

# Install Python modules
install(DIRECTORY
  ${PROJECT_NAME}
  DESTINATION lib/python3/dist-packages/
)

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
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_lint_common REQUIRED)
  
  # Add tests
  ament_add_gtest(test_test_image_publisher_node tests/test_image_publisher_node.cpp)
  target_link_libraries(test_test_image_publisher_node ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_test_image_publisher_node rclcpp)
  ament_add_gtest(test_test_pipeline_orchestrator_node tests/test_pipeline_orchestrator_node.cpp)
  target_link_libraries(test_test_pipeline_orchestrator_node ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_test_pipeline_orchestrator_node rclcpp)
  ament_add_gtest(test_test_visualization_node_node tests/test_visualization_node_node.cpp)
  target_link_libraries(test_test_visualization_node_node ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_test_visualization_node_node rclcpp)
  
  # Python tests
  ament_add_pytest_test(test_${PROJECT_NAME}_python
    ${PROJECT_NAME}/test_${PROJECT_NAME}.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
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
ament_export_dependencies(std_msgs)

ament_package()
