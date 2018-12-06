#!/bin/bash

../build/trainGPU -n 128 -l 1 -c 5000 -g 500 -o 128_1
../build/trainGPU -n 128 -l 2 -c 5000 -g 500 -o 128_2
../build/trainGPU -n 128 -l 3 -c 5000 -g 500 -o 128_3
../build/trainGPU -n 128 -l 4 -c 5000 -g 500 -o 128_4
