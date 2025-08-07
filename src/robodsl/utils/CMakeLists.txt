cmake_minimum_required(VERSION 3.10)
project(robodsl_typecheck)

set(CMAKE_CXX_STANDARD 11)

add_library(robodsl_typecheck SHARED robodsl_typecheck.cpp)

target_include_directories(robodsl_typecheck PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(robodsl_typecheck PROPERTIES
    OUTPUT_NAME "robodsl_typecheck"
    PREFIX "lib"
    SUFFIX ".dylib"
)

# On Linux, use .so
if (UNIX AND NOT APPLE)
    set_target_properties(robodsl_typecheck PROPERTIES SUFFIX ".so")
endif() 