cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_maxflow CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

# Set CUDA architectures (adjust as needed for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80)

add_compile_options(-G)

# Add executable
add_executable(cuda_maxflow
        maxflow.cu
)

# Include directories
target_include_directories(cuda_maxflow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries(cuda_maxflow PRIVATE ${CUDA_LIBRARIES})

# Set compiler flags
if(MSVC)
    target_compile_options(cuda_maxflow PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(cuda_maxflow PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

# Enable separable compilation
set_target_properties(cuda_maxflow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)