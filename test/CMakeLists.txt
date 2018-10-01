project(test_project CXX C)

add_library(game ./game.cpp)

add_executable(main ./main.cpp)

target_link_libraries(main game)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
