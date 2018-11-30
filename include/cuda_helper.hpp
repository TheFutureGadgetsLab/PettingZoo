#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define cudaErrCheck(ans) { cudaErrorCheck((ans), __FILE__, __LINE__); }

inline void cudaErrorCheck(cudaError_t code, const char *file, int line)
{   
    if (code != cudaSuccess) {
        fprintf(stderr,"cudaError: %s in file %s, line %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#endif