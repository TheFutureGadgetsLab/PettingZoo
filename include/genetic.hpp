#ifndef GENETIC_H
#define GENETIC_H

#include <defs.hpp>
#include <gamelogic.hpp>
#include <cuda_runtime.h>

struct RecordedChromosome {
    uint8_t *chrom;
    uint8_t buttons[MAX_FRAMES];
    float fitness;
    struct Game *game;
};

__host__ __device__
int run_generation(struct Game games[GEN_SIZE], struct Player players[GEN_SIZE], uint8_t *generation[GEN_SIZE], struct RecordedChromosome *winner);
__host__ __device__
void select_and_breed(struct Player players[GEN_SIZE], uint8_t **generation, uint8_t **new_generation);

#endif