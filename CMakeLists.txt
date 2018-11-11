cmake_minimum_required(VERSION 3.0.0)
project(pettingzoo)
 
# Locate headers
include_directories(include)

# Add extra compiler flags
if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")
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
set(GAME_SOURCES src/gamelogic.cpp src/levelgen.cpp src/rendering.cpp src/main.cpp src/neural_network.cpp)

# Compile into petting zoo
add_executable(pettingzoo ${GAME_SOURCES})

# Link libraries
target_link_libraries(pettingzoo "${SFML_GRAPHICS_LIB}")
target_link_libraries(pettingzoo "${SFML_WINDOW_LIB}")
target_link_libraries(pettingzoo "${SFML_AUDIO_LIB}")
target_link_libraries(pettingzoo "${SFML_SYSTEM_LIB}")
target_link_libraries(pettingzoo m)

###############################################################
#
#                           Neural Network
#
###############################################################
# Load NN sources
set(NN_SOURCES src/chromosome.cpp src/neural_network.cpp src/gamelogic.cpp src/levelgen.cpp)

# Compile neural network
add_executable(neural_network ${NN_SOURCES})

# Link libraries
target_link_libraries(neural_network m)

# Set the location for library installation -- i.e., /usr/lib
# Use "sudo make install" to apply
#install(TARGETS pettingzoo DESTINATION /usr/lib)