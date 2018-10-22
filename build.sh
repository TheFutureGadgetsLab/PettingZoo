#!/bin/bash

cd build;
make;
if [ "$?" -eq "0" ]
    then
        echo "BUILT!"
        ./pettingzoo;
fi
