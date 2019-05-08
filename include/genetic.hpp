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
#include <vector>
#include <string>
#include <FFNN.hpp>

void run_generation(Game& game, std::vector<Player>& players, std::vector<FFNN> & generation, Params& params);
void select_and_breed(std::vector<Player>& players, std::vector<FFNN> & curGen, std::vector<FFNN> & newGen, Params& params);
void get_gen_stats(std::string& dirname, Game& game, std::vector<FFNN> & chroms, int quiet, int write_winner, int generation, Params& params);
void create_output_dir(std::string& dirname, unsigned int seed, Params& params);
void mutateGeneration(std::vector<FFNN>& generation, float mutateRate);

#endif
