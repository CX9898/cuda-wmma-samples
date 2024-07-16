# Specify the minimum version of cmake
cmake_minimum_required(VERSION 3.26)

# Project Information
project(wmma-sample)

# Add languages needed when building the project
enable_language(CXX)
enable_language(CUDA)

# Sets the C++ standard used by the project
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Specifies which GPU architectures are supported when compiling CUDA code (optional)
set(CMAKE_CUDA_ARCHITECTURES 70)

# Save the folder path in a variable
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

# All source files in src folder are stored in SRC_FILES variable
file(GLOB SRC_FILES "${SRC_DIR}/*.c" "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.cc" "${SRC_DIR}/*.cxx" "${SRC_DIR}/*.cu")

# Output file list information
message(STATUS "Src files: ${SRC_FILES}")

# Add generate target
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Set the installation Path
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}")

# Set installation rules
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

# Add CUDA header directory (locally)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})

#PRIVATE: The linked library is only visible to the target itself.
#PUBLIC: The linked library is visible to the target itself and to all other targets linked to the target.
#INTERFACE: Linked libraries are only visible to other targets linked to that target.

# Add CUDA header directory (globally)
#include_directories(${CUDA_INCLUDE_DIRS})

target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})

# Linked CUDA library
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_CUDART_LIBRARY})

# Linked cuBLAS library
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_cublas_LIBRARY})

# Linked cuFFT library
#target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_cufft_LIBRARY})

# Linked cuSOLVER library
#target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_cusolver_LIBRARY})

# Linked cuSPARSE library
#target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_cusparse_LIBRARY})

# Linked cuRAND library
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_curand_LIBRARY})

# Find OpenMP package
find_package(OpenMP REQUIRED)

#REQUIRED: If the specified package is not found, CMake will throw an error and stop further configuration. This is a key parameter to ensure project dependency.
#QUIET: When set to QUIET, CMake will not print any warnings or errors to the console even if the package is not found. This is typically used for optional dependencies.
#MODULE: This tells CMake to look for CMake module files (.cmake files) instead of package configuration files

# Linked OpenMP library
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
