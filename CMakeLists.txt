cmake_minimum_required(VERSION 3.10)
project(robodsl_project VERSION 0.1.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "native")

# Set CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --default-stream per-thread")

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Add source files
file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
        Threads::Threads
        ${CMAKE_THREAD_LIBS_INIT}
)

# Add Thrust support if needed
option(ENABLE_THRUST "Enable Thrust support" ON)
if(ENABLE_THRUST)
    target_compile_definitions(${PROJECT_NAME} 
        PRIVATE 
            THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
    )
    
    # Add CUDA libraries needed for Thrust
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${PROJECT_NAME} 
        PRIVATE 
            CUDA::cudart
            CUDA::cublas
            CUDA::cufft
    )
endif()

# Install targets
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Add uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()

# Add tests if enabled
option(ENABLE_TESTS "Build tests" ON)
if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
