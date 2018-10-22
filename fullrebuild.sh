#!/bin/bash

# Create build directory if there is not one already
if ! [ -a build ]
    then
        mkdir build;
fi

# Change directory to build and run cmake
cd build;
rm -rf *;
cmake ..;

# Build if no CMake errors
if [ "$?" -eq "0" ]
    then
        make;
    else
        echo "CMAKE ERROR!";
fi
