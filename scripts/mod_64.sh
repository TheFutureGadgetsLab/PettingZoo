#!/bin/bash

../build/trainGPU -n 64 -l 1 -c 5000 -g 500 -o 64_1
../build/trainGPU -n 64 -l 2 -c 5000 -g 500 -o 64_2
../build/trainGPU -n 64 -l 3 -c 5000 -g 500 -o 64_3
../build/trainGPU -n 64 -l 4 -c 5000 -g 500 -o 64_4
