# 指定cmake最低版本要求
cmake_minimum_required(VERSION 3.26)

# 项目信息, 指定项目名为 Demo
project(Demo)

# 添加构建项目时需要的语言
enable_language(CXX)
enable_language(CUDA)

# 设置 C++ 版本
set(CMAKE_CXX_STANDARD 11)

# 设置 CUDA 架构类型
#set(CMAKE_CUDA_ARCHITECTURES 86)

# 添加 include 文件夹路径到变量INCLUDE_DIR,添加 src 文件夹路径到变量 SRC_DIR
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

# 读取 include 文件夹中的所有头文件并存储在变量 INCLUDE_FILES 中, src 文件夹中的所有文件储存在变量 SRC_FILES 中
file(GLOB INCLUDE_FILES "${INCLUDE_DIR}/*.hpp" "${INCLUDE_DIR}/*.h" "${INCLUDE_DIR}/*.cuh")
file(GLOB SRC_FILES "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.c" "${SRC_DIR}/*.cu")

# 输出文件列表（可选）
message(STATUS "Include files: ${INCLUDE_FILES}")
message(STATUS "Src files: ${SRC_FILES}")

# 指定生成目标
add_executable(${PROJECT_NAME} ${SRC_FILES})

# 查找 CUDA 包
find_package(CUDAToolkit  REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Thrust REQUIRED)

#REQUIRED: 如果指定的包找不到，CMake 将报错并停止进一步的配置过程。这是确保项目依赖性的关键参数。
#QUIET: 当设置为 QUIET 时，即使找不到包，CMake 也不会在控制台输出任何警告或错误信息。这通常用于可选依赖项。
#MODULE: 这个参数告诉 CMake 查找的是 CMake 模块文件（.cmake 文件），而不是包配置文件

# 添加 CUDA 的头文件目录(局部添加)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})

#PRIVATE：链接的库仅对目标自身可见。
#PUBLIC：链接的库对目标自身和链接到该目标的所有其他目标都可见。
#INTERFACE：链接的库仅对链接到该目标的其他目标可见。

# 添加 CUDA 的头文件目录(全局添加)
#include_directories(${CUDA_INCLUDE_DIRS})

# 链接 CUDA 库
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_CUDART_LIBRARY})

# 链接 Thrust 库
#target_link_libraries(${PROJECT_NAME} PRIVATE ${Thrust_LIBRARIES})

# 链接 cublas 库
#target_link_libraries(cublasMatmul PRIVATE ${CUDA_cublas_LIBRARY})

# 链接 openMP 库
#target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
