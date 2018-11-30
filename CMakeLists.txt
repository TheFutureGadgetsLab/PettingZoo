cmake_minimum_required(VERSION 3.9.0)
project(pettingzoo LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)
# Locate headers
include_directories(include)

# Set march
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Add extra compiler flags
if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS} –x cu -g -f")

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
#                       Train NN
#
###############################################################
# Compile
add_executable(
    trainNN
    src/trainNN.cpp
    src/genetic.cpp
    src/chromosome.cpp
    src/neural_network.cpp
    src/gamelogic.cpp
    src/levelgen.cpp
)

# Link libraries
target_link_libraries(trainNN m)

# Set the location for library installation -- i.e., /usr/lib
# Use "sudo make install" to apply
#install(TARGETS pettingzoo DESTINATION /usr/lib)

###############################################################
#
#                       Debug Program
#
###############################################################
# Configure CUDA flags (-x cu tells nvcc to interpret cpp files as having CPU/GPU code)

set_source_files_properties(
    src/chromosome.cpp
    src/gamelogic.cpp
    src/genetic.cpp
    src/neural_network.cpp
    src/test.cpp
    PROPERTIES LANGUAGE CUDA
)

# Compile
add_executable(
    test_cuda
    src/test.cpp
    src/chromosome.cpp
    src/gamelogic.cpp
    src/genetic.cpp
    src/levelgen.cpp
    src/neural_network.cpp
)

# Enable separate compilation for CUDA
set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(test_cuda m)
