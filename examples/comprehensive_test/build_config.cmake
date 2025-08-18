# Build configuration for RoboDSL package

# CUDA configuration
if(CUDA_FOUND)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# OpenCV configuration
if(OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC)
endif()