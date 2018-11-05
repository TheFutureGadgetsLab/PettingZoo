#include <stdio.h>
#include <stdlib.h>
#include <neural_network.hpp>
#include <stdint.h>

void generate_chromosome(uint8_t input_h, uint8_t input_w, uint8_t hlc, uint16_t npl);

int main()
{
    printf("Running NN with %d by %d inputs, %d outputs, %d hidden layers, and %d nodes per layer.\n\n", 
        IN_H, IN_W, OUTPUT_SIZE, HLC, NPL);

    generate_chromosome(IN_H, IN_W, HLC, NPL);
    return 0;
}

void generate_chromosome(uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl)
{
    size_t total_size;
    
    total_size = 5 + (in_h * in_w) + (in_h * in_w * npl) + (hlc * npl) + (hlc - 1) * (npl * npl) + (npl * 3);

    printf("Generating chromosome with the following paramenters:\n");
    printf("  IN_H:\t%d\n  IN_W:\t%d\n  HLC:\t%d\n  NPL:\t%d\n",
              in_h, in_w, hlc, npl);
    printf("  Total size: %0.2lfKB\n", total_size / 1000.0);
}