cmake_minimum_required(VERSION 3.8)
project(robot_package VERSION 0.1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD c++14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Only add warning flags for C++ (not CUDA)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
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
find_package(std_srvs REQUIRED)

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
  include/main_node_node.hpp
  include/perception_node_node.hpp
  include/navigation_node_node.hpp
  include/safety_node_node.hpp
  include/robot_cpp_node_node.hpp
  include/vector_add_kernel.cuh
  include/matrix_multiply_kernel.cuh
  include/image_filter_kernel.cuh
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC include)
ament_target_dependencies(${PROJECT_NAME}_lib rclcpp)

# CUDA library
add_library(${PROJECT_NAME}_cuda STATIC
  src/vector_add_kernel.cu
  src/matrix_multiply_kernel.cu
  src/image_filter_kernel.cu
)

set_target_properties(${PROJECT_NAME}_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${PROJECT_NAME}_cuda PUBLIC include)
target_link_libraries(${PROJECT_NAME}_cuda ${PROJECT_NAME}_lib)

# Create executables
add_executable(main_node_node
  src/main_node_node.cpp
  src/perception_node_node.cpp
  src/navigation_node_node.cpp
  src/safety_node_node.cpp
  src/robot_cpp_node_node.cpp
)

target_link_libraries(main_node_node ${PROJECT_NAME}_lib)
target_link_libraries(main_node_node ${PROJECT_NAME}_cuda)

ament_target_dependencies(main_node_node rclcpp)
ament_target_dependencies(main_node_node geometry_msgs)
ament_target_dependencies(main_node_node rclcpp)
ament_target_dependencies(main_node_node rclcpp_lifecycle)
ament_target_dependencies(main_node_node std_msgs)
ament_target_dependencies(main_node_node std_msgs)
ament_target_dependencies(main_node_node std_srvs)

add_executable(perception_node_node
  src/main_node_node.cpp
  src/perception_node_node.cpp
  src/navigation_node_node.cpp
  src/safety_node_node.cpp
  src/robot_cpp_node_node.cpp
)

target_link_libraries(perception_node_node ${PROJECT_NAME}_lib)
target_link_libraries(perception_node_node ${PROJECT_NAME}_cuda)

ament_target_dependencies(perception_node_node rclcpp)
ament_target_dependencies(perception_node_node geometry_msgs)
ament_target_dependencies(perception_node_node rclcpp)
ament_target_dependencies(perception_node_node rclcpp_lifecycle)
ament_target_dependencies(perception_node_node std_msgs)
ament_target_dependencies(perception_node_node std_msgs)
ament_target_dependencies(perception_node_node std_srvs)

add_executable(navigation_node_node
  src/main_node_node.cpp
  src/perception_node_node.cpp
  src/navigation_node_node.cpp
  src/safety_node_node.cpp
  src/robot_cpp_node_node.cpp
)

target_link_libraries(navigation_node_node ${PROJECT_NAME}_lib)
target_link_libraries(navigation_node_node ${PROJECT_NAME}_cuda)

ament_target_dependencies(navigation_node_node rclcpp)
ament_target_dependencies(navigation_node_node geometry_msgs)
ament_target_dependencies(navigation_node_node rclcpp)
ament_target_dependencies(navigation_node_node rclcpp_lifecycle)
ament_target_dependencies(navigation_node_node std_msgs)
ament_target_dependencies(navigation_node_node std_msgs)
ament_target_dependencies(navigation_node_node std_srvs)

add_executable(safety_node_node
  src/main_node_node.cpp
  src/perception_node_node.cpp
  src/navigation_node_node.cpp
  src/safety_node_node.cpp
  src/robot_cpp_node_node.cpp
)

target_link_libraries(safety_node_node ${PROJECT_NAME}_lib)
target_link_libraries(safety_node_node ${PROJECT_NAME}_cuda)

ament_target_dependencies(safety_node_node rclcpp)
ament_target_dependencies(safety_node_node geometry_msgs)
ament_target_dependencies(safety_node_node rclcpp)
ament_target_dependencies(safety_node_node rclcpp_lifecycle)
ament_target_dependencies(safety_node_node std_msgs)
ament_target_dependencies(safety_node_node std_msgs)
ament_target_dependencies(safety_node_node std_srvs)

add_executable(robot_cpp_node_node
  src/main_node_node.cpp
  src/perception_node_node.cpp
  src/navigation_node_node.cpp
  src/safety_node_node.cpp
  src/robot_cpp_node_node.cpp
)

target_link_libraries(robot_cpp_node_node ${PROJECT_NAME}_lib)
target_link_libraries(robot_cpp_node_node ${PROJECT_NAME}_cuda)

ament_target_dependencies(robot_cpp_node_node rclcpp)
ament_target_dependencies(robot_cpp_node_node geometry_msgs)
ament_target_dependencies(robot_cpp_node_node rclcpp)
ament_target_dependencies(robot_cpp_node_node rclcpp_lifecycle)
ament_target_dependencies(robot_cpp_node_node std_msgs)
ament_target_dependencies(robot_cpp_node_node std_msgs)
ament_target_dependencies(robot_cpp_node_node std_srvs)


# Python modules
python_install_package(${PROJECT_NAME})

# Install targets
install(TARGETS
  main_node_node
  perception_node_node
  navigation_node_node
  safety_node_node
  robot_cpp_node_node
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
  ament_add_gtest(test_test_main_node_node tests/test_main_node_node.cpp)
  target_link_libraries(test_test_main_node_node ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_test_main_node_node rclcpp)
  ament_add_gtest(test_test_perception_node_node tests/test_perception_node_node.cpp)
  target_link_libraries(test_test_perception_node_node ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_test_perception_node_node rclcpp)
  ament_add_gtest(test_test_navigation_node_node tests/test_navigation_node_node.cpp)
  target_link_libraries(test_test_navigation_node_node ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_test_navigation_node_node rclcpp)
  ament_add_gtest(test_test_safety_node_node tests/test_safety_node_node.cpp)
  target_link_libraries(test_test_safety_node_node ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_test_safety_node_node rclcpp)
  ament_add_gtest(test_test_robot_cpp_node_node tests/test_robot_cpp_node_node.cpp)
  target_link_libraries(test_test_robot_cpp_node_node ${PROJECT_NAME}_lib)
  ament_target_dependencies(test_test_robot_cpp_node_node rclcpp)
  
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
