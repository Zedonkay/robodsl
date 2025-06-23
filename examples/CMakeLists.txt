cmake_minimum_required(VERSION 3.8)
project(robodsl_project VERSION 0.1.0 LANGUAGES CXX)

# Default to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)




# Set include directories
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Add library
add_library(${PROJECT_NAME}_lib
  src/robodsl_project.cpp
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Set C++ standard properties
target_compile_features(${PROJECT_NAME}_lib PRIVATE cxx_std_17)
set_target_properties(${PROJECT_NAME}_lib PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

# Link dependencies
target_link_libraries(${PROJECT_NAME}_lib
  ${rclcpp_LIBRARIES}
  ${rclcpp_components_LIBRARIES}
  ${std_msgs_LIBRARIES}
)

# Add node executables
set(NODE_NAMES)
# my_node node
add_executable(${PROJECT_NAME}_my_node_node
  src/my_node_node.cpp
)

target_link_libraries(${PROJECT_NAME}_my_node_node
  ${PROJECT_NAME}_lib
)

# Set node properties
set_target_properties(${PROJECT_NAME}_my_node_node PROPERTIES
  OUTPUT_NAME my_node_node
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${PROJECT_NAME}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Add to node names list
list(APPEND NODE_NAMES ${PROJECT_NAME}_my_node_node)


target_link_libraries(${node.name}_node
  ${PROJECT_NAME}_lib
)
# minimal_node node
add_executable(${PROJECT_NAME}_minimal_node_node
  src/minimal_node_node.cpp
)

target_link_libraries(${PROJECT_NAME}_minimal_node_node
  ${PROJECT_NAME}_lib
)

# Set node properties
set_target_properties(${PROJECT_NAME}_minimal_node_node PROPERTIES
  OUTPUT_NAME minimal_node_node
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${PROJECT_NAME}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Add to node names list
list(APPEND NODE_NAMES ${PROJECT_NAME}_minimal_node_node)


target_link_libraries(${node.name}_node
  ${PROJECT_NAME}_lib
)
# lifecycle_node node
add_executable(${PROJECT_NAME}_lifecycle_node_node
  src/lifecycle_node_node.cpp
)

target_link_libraries(${PROJECT_NAME}_lifecycle_node_node
  ${PROJECT_NAME}_lib
)

# Set node properties
set_target_properties(${PROJECT_NAME}_lifecycle_node_node PROPERTIES
  OUTPUT_NAME lifecycle_node_node
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${PROJECT_NAME}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Add to node names list
list(APPEND NODE_NAMES ${PROJECT_NAME}_lifecycle_node_node)


target_link_libraries(${node.name}_node
  ${PROJECT_NAME}_lib
)
# param_node node
add_executable(${PROJECT_NAME}_param_node_node
  src/param_node_node.cpp
)

target_link_libraries(${PROJECT_NAME}_param_node_node
  ${PROJECT_NAME}_lib
)

# Set node properties
set_target_properties(${PROJECT_NAME}_param_node_node PROPERTIES
  OUTPUT_NAME param_node_node
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${PROJECT_NAME}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Add to node names list
list(APPEND NODE_NAMES ${PROJECT_NAME}_param_node_node)


target_link_libraries(${node.name}_node
  ${PROJECT_NAME}_lib
)
# qos_node node
add_executable(${PROJECT_NAME}_qos_node_node
  src/qos_node_node.cpp
)

target_link_libraries(${PROJECT_NAME}_qos_node_node
  ${PROJECT_NAME}_lib
)

# Set node properties
set_target_properties(${PROJECT_NAME}_qos_node_node PROPERTIES
  OUTPUT_NAME qos_node_node
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${PROJECT_NAME}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Add to node names list
list(APPEND NODE_NAMES ${PROJECT_NAME}_qos_node_node)


target_link_libraries(${node.name}_node
  ${PROJECT_NAME}_lib
)
# timer_node node
add_executable(${PROJECT_NAME}_timer_node_node
  src/timer_node_node.cpp
)

target_link_libraries(${PROJECT_NAME}_timer_node_node
  ${PROJECT_NAME}_lib
)

# Set node properties
set_target_properties(${PROJECT_NAME}_timer_node_node PROPERTIES
  OUTPUT_NAME timer_node_node
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${PROJECT_NAME}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Add to node names list
list(APPEND NODE_NAMES ${PROJECT_NAME}_timer_node_node)


target_link_libraries(${node.name}_node
  ${PROJECT_NAME}_lib
)
# param_types_node node
add_executable(${PROJECT_NAME}_param_types_node_node
  src/param_types_node_node.cpp
)

target_link_libraries(${PROJECT_NAME}_param_types_node_node
  ${PROJECT_NAME}_lib
)

# Set node properties
set_target_properties(${PROJECT_NAME}_param_types_node_node PROPERTIES
  OUTPUT_NAME param_types_node_node
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/${PROJECT_NAME}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Add to node names list
list(APPEND NODE_NAMES ${PROJECT_NAME}_param_types_node_node)


target_link_libraries(${node.name}_node
  ${PROJECT_NAME}_lib
)

# Install node executables
install(TARGETS
  ${NODE_NAMES}
  EXPORT export_${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

# Register components
foreach(NODE_NAME ${NODE_NAMES})
  get_target_property(TARGET_NAME ${NODE_NAME} OUTPUT_NAME)
  if(TARGET_NAME)
    rclcpp_components_register_nodes(${NODE_NAME} "${PROJECT_NAME}::${TARGET_NAME}")
  endif()
  
  # Export node targets
  ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
  
  # Export dependencies
  ament_export_dependencies(
    rclcpp
    rclcpp_components
    std_msgs
  )
endforeach()

# Install include directories
install(
  DIRECTORY include/
  DESTINATION include
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install package.xml
install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

# Install environment hooks
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/
  DESTINATION share/${PROJECT_NAME}/env-hooks
)

# Export targets
ament_export_include_directories(
  include
)

ament_export_libraries(
  ${PROJECT_NAME}_lib
)

# Generate and install the config file
ament_python_install_package(${PROJECT_NAME})

# Generate the package configuration files
ament_package(
  CONFIG_EXTRAS "${PROJECT_NAME}-extras.cmake.in"
)

# Add uninstall target
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake"
  IMMEDIATE @ONLY
)

add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake
  COMMENT "Uninstalling..."
)

# Add format target if clang-format is found
find_program(CLANG_FORMAT "clang-format")
if(CLANG_FORMAT)
  add_custom_target(
    format
    COMMAND find ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i -style=file
    COMMENT "Formatting source files..."
  )
endif()