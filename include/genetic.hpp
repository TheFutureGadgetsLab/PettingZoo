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
#include <chromosome.hpp>
#include <vector>

void run_generation(Game& game, Player* players, std::vector<Chromosome> & generation, Params& params);
void select_and_breed(Player *players, std::vector<Chromosome> & generation, std::vector<Chromosome> & new_generation, Params& params);
void get_gen_stats(char *dirname, Game& game, Player *players, std::vector<Chromosome> & chroms, int quiet, int write_winner, int generation, Params& params);
void create_output_dir(char *dirname, unsigned int seed, Params& params);

#endif
