#ifndef GENETIC_H
#define GENETIC_H

#include <defs.hpp>
#include <gamelogic.hpp>

struct RecordedChromosome {
    struct Chromosome *chrom;
    uint8_t buttons[MAX_FRAMES];
    float fitness;
    struct Game *game;
};

__host__ __device__
int run_generation(struct Game games[GEN_SIZE], struct Player players[GEN_SIZE], struct Chromosome generation[GEN_SIZE], struct RecordedChromosome *winner);
__host__
void select_and_breed(struct Player players[GEN_SIZE], struct Chromosome *generation, struct Chromosome *new_generation);

#endif
