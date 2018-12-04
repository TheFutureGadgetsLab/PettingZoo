#ifndef GENETIC_H
#define GENETIC_H

#include <defs.hpp>
#include <gamelogic.hpp>

int run_generation(struct Game *game, struct Player players[GEN_SIZE], struct Chromosome generation[GEN_SIZE]);
void select_and_breed(struct Player players[GEN_SIZE], struct Chromosome *generation, struct Chromosome *new_generation);
void write_out_progress(FILE *fh, struct Player players[GEN_SIZE]);
void get_gen_stats(char *dirname, struct Game *game, struct Player *players, struct Chromosome *chroms, int quiet, int write_winner, int generation);

#endif
