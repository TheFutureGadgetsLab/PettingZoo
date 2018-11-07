#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <stdint.h>
#include <time.h>

int main()
{
    uint8_t *chrom = NULL;
    printf("Running NN with %d by %d inputs, %d outputs, %d hidden layers, and %d nodes per layer.\n", 
        IN_H, IN_W, BUTTON_COUNT, HLC, NPL);

    srand(time(NULL));

    chrom = generate_chromosome(IN_H, IN_W, HLC, NPL);
    print_chromosome(chrom);

    free(chrom);

    return 0;
}