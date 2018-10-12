cmake_minimum_required(VERSION 3.0.0)
project(pettingzoo)
set(CMAKE_BUILD_TYPE Release)
 
#Bring the headers, such as Student.h into the project
include_directories(include)

# Find libcsfml
find_library(SFML_GRAPHICS_LIB sfml-graphics)
find_library(SFML_WINDOW_LIB sfml-window)
find_library(SFML_AUDIO_LIB sfml-audio)
find_library(SFML_SYSTEM_LIB sfml-system)

# Add sources to env var
file(GLOB SOURCES "src/*.cpp")

# Not 100% sure what is going on with this
add_executable(pettingzoo ${SOURCES})

# Set the location for library installation -- i.e., /usr/lib
# Use "sudo make install" to apply
#install(TARGETS pettingzoo DESTINATION /usr/lib)

# Link it
target_link_libraries(pettingzoo "${SFML_GRAPHICS_LIB}")
target_link_libraries(pettingzoo "${SFML_WINDOW_LIB}")
target_link_libraries(pettingzoo "${SFML_AUDIO_LIB}")
target_link_libraries(pettingzoo "${SFML_SYSTEM_LIB}")
target_link_libraries(pettingzoo m)