#ifndef GENETIC_H
#define GENETIC_H

#include <defs.hpp>
#include <gamelogic.hpp>
#include <cuda_runtime.h>

struct RecordedChromosome {
    struct chromosome *chrom;
    uint8_t buttons[MAX_FRAMES];
    float fitness;
    struct Game *game;
};

__host__ __device__
int run_generation(struct Game games[GEN_SIZE], struct Player players[GEN_SIZE], struct chromosome *generation, struct RecordedChromosome *winner);
__host__ __device__
void select_and_breed(struct Player players[GEN_SIZE], struct chromosome *generation, struct chromosome *new_generation);

#endif
