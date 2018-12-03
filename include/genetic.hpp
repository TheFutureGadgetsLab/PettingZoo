#ifndef GENETIC_H
#define GENETIC_H

#include <defs.hpp>
#include <gamelogic.hpp>

__host__ __device__
int run_generation(struct Game games[GEN_SIZE], struct Player players[GEN_SIZE], struct Chromosome generation[GEN_SIZE]);
__host__
void select_and_breed(struct Player players[GEN_SIZE], struct Chromosome *generation, struct Chromosome *new_generation);
__host__
void write_out_progress(FILE *fh, struct Player players[GEN_SIZE]);
__host__
void get_gen_stats(char *dirname, struct Game *games, struct Player *players, struct Chromosome *chroms, int quiet, int write_winner, int generation);


#endif
