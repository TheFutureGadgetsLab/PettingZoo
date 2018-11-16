cmake_minimum_required(VERSION 3.0.0)
project(pettingzoo)
 
# Locate headers
include_directories(include)

# Add extra compiler flags
if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

# Set march
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
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

# Load game sources
set(GAME_SOURCES src/gamelogic.cpp src/levelgen.cpp src/rendering.cpp src/main.cpp)

# Compile into petting zoo
add_executable(pettingzoo ${GAME_SOURCES})

# Link libraries
target_link_libraries(pettingzoo "${SFML_GRAPHICS_LIB}")
target_link_libraries(pettingzoo "${SFML_WINDOW_LIB}")
target_link_libraries(pettingzoo "${SFML_SYSTEM_LIB}")
target_link_libraries(pettingzoo m)

###############################################################
#
#                       Genetic Algorithm
#
###############################################################
# Load sources
set(CHROM_SOURCES src/genetic.cpp src/chromosome.cpp src/neural_network.cpp src/gamelogic.cpp src/levelgen.cpp)

# Compile
add_executable(genetic ${CHROM_SOURCES})

# Link libraries
target_link_libraries(genetic m)

# Set the location for library installation -- i.e., /usr/lib
# Use "sudo make install" to apply
#install(TARGETS pettingzoo DESTINATION /usr/lib)