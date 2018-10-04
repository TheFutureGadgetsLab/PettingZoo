cmake_minimum_required(VERSION 3.0.0)
project(pettingzoo)
set(CMAKE_BUILD_TYPE Release)
 
#Bring the headers, such as Student.h into the project
include_directories(include)

# Find libcsfml
find_library(CSFML_GRAPHICS_LIB csfml-graphics)
find_library(CSFML_WINDOW_LIB csfml-window)
find_library(CSFML_AUDIO_LIB csfml-audio)
find_library(CSFML_SYSTEM_LIB csfml-system)

# Add sources to env var
file(GLOB SOURCES "src/*.c")

# Not 100% sure what is going on with this
add_executable(pettingzoo ${SOURCES})

# Set the location for library installation -- i.e., /usr/lib
# Use "sudo make install" to apply
#install(TARGETS pettingzoo DESTINATION /usr/lib)

# Link it
target_link_libraries(pettingzoo "${CSFML_GRAPHICS_LIB}")
target_link_libraries(pettingzoo "${CSFML_WINDOW_LIB}")
target_link_libraries(pettingzoo "${CSFML_AUDIO_LIB}")
target_link_libraries(pettingzoo "${CSFML_SYSTEM_LIB}")