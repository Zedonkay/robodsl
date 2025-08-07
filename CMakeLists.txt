cmake_minimum_required(VERSION 3.8)
project(robodsl_package VERSION 0.1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD c++14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CPP_WARNING_FLAGS "-Wall;-Wextra;-Wpedantic")
  set(CPP_POSITION_INDEPENDENT_CODE "-fPIC")
endif()

# Build configurations
set(CMAKE_CONFIGURATION_TYPES "Release;Debug;RelWithDebInfo;MinSizeRel")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)


# CUDA support

# Include directories
include_directories(include)
include_directories(include)

# Create library for common code
add_library(${PROJECT_NAME}_lib STATIC
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC include)
ament_target_dependencies(${PROJECT_NAME}_lib rclcpp)

# Only add warning flags for C++ (not CUDA)
if(CPP_WARNING_FLAGS)
  target_compile_options(${PROJECT_NAME}_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CPP_WARNING_FLAGS}>)
endif()

# Always add -fPIC for C++
if(CPP_POSITION_INDEPENDENT_CODE)
  target_compile_options(${PROJECT_NAME}_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CPP_POSITION_INDEPENDENT_CODE}>)
endif()


# Create executables

# Python modules
python_install_package(${PROJECT_NAME})

# Install targets
install(TARGETS
  ${PROJECT_NAME}_lib
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
ament_export_dependencies(rclcpp)
ament_export_dependencies(geometry_msgs)
ament_export_dependencies(rclcpp)
ament_export_dependencies(std_msgs)

ament_package()
