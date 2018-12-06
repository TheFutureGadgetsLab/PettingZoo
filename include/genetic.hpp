#ifndef GENETIC_H
#define GENETIC_H

#include <defs.hpp>
#include <gamelogic.hpp>

int run_generation(struct Game *game, struct Player *players, struct Chromosome *generation, struct Params *params);
void select_and_breed(struct Player *players, struct Chromosome *generation, struct Chromosome *new_generation, struct Params *params);
void write_out_progress(FILE *fh, struct Player *players);
void get_gen_stats(char *dirname, struct Game *game, struct Player *players, 
    struct Chromosome *chroms, int quiet, int write_winner, int generation, struct Params *params);
void create_output_dir(char *dirname, unsigned int seed, struct Params *params);

#endif
