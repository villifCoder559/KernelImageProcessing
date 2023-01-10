cmake_minimum_required(VERSION 3.0.0)
project(kuda VERSION 0.1.0 LANGUAGES CXX CUDA)
# find_package(OpenMP)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# the following line was also necessary

add_executable(kuda main.cu image.cpp image.h convolution_gpu.cu convolution_gpu.h kernel.cpp kernel.h)
# target_compile_options(kuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
target_compile_options(kuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       >)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
