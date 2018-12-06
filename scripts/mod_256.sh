#!/bin/bash

../build/trainGPU -n 256 -l 1 -c 5000 -g 500 -o 256_1
../build/trainGPU -n 256 -l 2 -c 5000 -g 500 -o 256_2
../build/trainGPU -n 256 -l 3 -c 5000 -g 500 -o 256_3
../build/trainGPU -n 256 -l 4 -c 5000 -g 500 -o 256_4
