cmake_minimum_required(VERSION 3.9.0)
project(pettingzoo LANGUAGES CXX)

include_directories(include)

###############################################################
#
#                        Packages / Flags
#
###############################################################
find_package(OpenMP)

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
# Find libsfml
find_library(SFML_GRAPHICS_LIB sfml-graphics)
find_library(SFML_WINDOW_LIB sfml-window)
find_library(SFML_SYSTEM_LIB sfml-system)

# Compile into petting zoo
add_executable(
    pettingzoo
    src/neural_network.cpp
    src/gamelogic.cpp
    src/levelgen.cpp
    src/randfuncts.cpp
    src/rendering.cpp
    src/main.cpp
    src/chromosome.cpp
)

# Link libraries
target_link_libraries(pettingzoo "${SFML_GRAPHICS_LIB}")
target_link_libraries(pettingzoo "${SFML_WINDOW_LIB}")
target_link_libraries(pettingzoo "${SFML_SYSTEM_LIB}")
target_link_libraries(pettingzoo m)

###############################################################
#
#                       Training
#
###############################################################
# Compile
add_executable(
    train
    src/train.cpp
    src/genetic.cpp
    src/chromosome.cpp
    src/neural_network.cpp
    src/gamelogic.cpp
    src/levelgen.cpp
    src/randfuncts.cpp
)

# Link libraries
target_link_libraries(train m)
target_link_libraries(train OpenMP::OpenMP_CXX)