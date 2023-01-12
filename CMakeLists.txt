cmake_minimum_required(VERSION 3.0.0)
project(image_kernel_processing VERSION 0.1.0 LANGUAGES CXX CUDA)
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# the following line was also necessary

add_executable(image_kernel_processing main.cu image.cpp image.h convolution_gpu.cu convolution_gpu.h kernel.cpp kernel.h convolution_cpu.h convolution_cpu.cpp)
# target_compile_options(kuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
