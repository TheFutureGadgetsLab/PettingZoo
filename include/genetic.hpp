/**
 * @file genetic.hpp
 * @author Haydn Jones, Benjamin Mastripolito
 * @brief Holds genetic defs
 * @date 2018-12-11
 */
#ifndef GENETIC_H
#define GENETIC_H

#include <defs.hpp>
#include <gamelogic.hpp>

void run_generation(Game& game, Player *players, struct Chromosome *generation, struct Params& params);
void select_and_breed(Player *players, struct Chromosome *generation, struct Chromosome *new_generation, struct Params& params);
void write_out_progress(FILE *fh, Player *players);
void get_gen_stats(char *dirname, Game& game, Player *players, 
    struct Chromosome *chroms, int quiet, int write_winner, int generation, struct Params& params);
void create_output_dir(char *dirname, unsigned int seed, struct Params& params);

#endif
