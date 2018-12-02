cmake_minimum_required(VERSION 3.9.0)
project(pettingzoo LANGUAGES CXX CUDA)
# set(CMAKE_VERBOSE_MAKEFILE ON)

# Locate headers
include_directories(include)
# Workaround for ben
include_directories(/opt/cuda/include/)

# Set march
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Add extra compiler flags
if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

###############################################################
#
#                           Petting zoo
#
###############################################################
# Find libcsfml
find_library(SFML_GRAPHICS_LIB sfml-graphics)
find_library(SFML_WINDOW_LIB sfml-window)
find_library(SFML_SYSTEM_LIB sfml-system)

# Compile into petting zoo
add_executable(
    pettingzoo
    src/gamelogic.cpp
    src/levelgen.cpp
    src/rendering.cpp
    src/main.cpp
)

# Link libraries
target_link_libraries(pettingzoo "${SFML_GRAPHICS_LIB}")
target_link_libraries(pettingzoo "${SFML_WINDOW_LIB}")
target_link_libraries(pettingzoo "${SFML_SYSTEM_LIB}")
target_link_libraries(pettingzoo m)

###############################################################
#
#                       CPU Program
#
###############################################################
# Compile
add_executable(
    trainCPU
    src/trainCPU.cpp
    src/genetic.cpp
    src/chromosome.cpp
    src/neural_network.cpp
    src/gamelogic.cpp
    src/levelgen.cpp
)

# Link libraries
target_link_libraries(trainCPU m)

###############################################################
#
#                       GPU Program
#
###############################################################
# Configure CUDA flags (-x cu tells nvcc to interpret cpp files as having CPU/GPU code)
# set(CUDA_NVCC_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=ca -v -O3 -arch=sm_61 -Xcompiler -O3")

set_source_files_properties(
   src/gamelogic.cpp
   src/neural_network.cpp
   src/trainGPU.cpp
   PROPERTIES LANGUAGE CUDA
)

# Compile
add_executable(
   trainGPU
   src/trainGPU.cpp
   src/gamelogic.cpp
   src/neural_network.cpp
   src/chromosome.cpp
   src/genetic.cpp
   src/levelgen.cpp
)

# Enable separate compilation for CUDA
set_target_properties(trainGPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(trainGPU m)
