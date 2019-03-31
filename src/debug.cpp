#include <stdio.h>
#include <stdint.h>
#include <chromosome.hpp>
#include <gamelogic.hpp>
#include <genetic.hpp>
#include <neural_network.hpp>
#include <levelgen.hpp>
#include <defs.hpp>
#include <time.h>
#include <math.h>

int main(int argc, char **argv)
{
    int n = atoi(argv[1]);
    int array[n];

    for (int i = 0; i < 100; i++) {
        printf("%d\n", array[i]);
    }

    return 0;
}
