#ifndef GENETIC_H
#define GENETIC_H

#include <defs.hpp>
#include <gamelogic.hpp>

struct RecordedChromosome {
    uint8_t *chrom;
    uint8_t buttons[MAX_FRAMES];
    float fitness;
    struct Game *game;
};

int run_generation(struct Game games[GEN_SIZE], struct Player players[GEN_SIZE], uint8_t *generation[GEN_SIZE], struct RecordedChromosome *winner);
void select_and_breed(struct Player players[GEN_SIZE], uint8_t **generation, uint8_t **new_generation);

#endif