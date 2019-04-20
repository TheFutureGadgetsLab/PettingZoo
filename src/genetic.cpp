/**
 * @file genetic.cpp
 * @author Haydn Jones, Benjamin Mastripolito
 * @brief Holds functions that govern the genetic algorithm
 * @date 2018-12-11
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <genetic.hpp>
#include <gamelogic.hpp>
#include <sys/stat.h>
#include <vector>
#include <randfuncts.hpp>
#include <algorithm>
#include <string>
#include <NeuralNetwork.hpp>

/**
 * @brief This function takes a game, players, and chromosomes to be evaluated.
 * 
 * @param game The game object to use in the evaluation
 * @param players A collection of players
 * @param generation A collection of chromosomes
 * @param params The run parameters
 */
void run_generation(Game& game, std::vector<Player>& players, std::vector<NeuralNetwork>& generation, Params& params)
{
    int g;
    int ret;
    int fitness_idle_updates;
    float max_fitness;
    bool playerNeedsUpdate;
    int playerLastTileX, playerLastTileY;

    // Loop over the entire generation
    #pragma omp parallel for private(ret, fitness_idle_updates, max_fitness, playerNeedsUpdate, playerLastTileX, playerLastTileY)
    for (g = 0; g < params.gen_size; g++) {
        fitness_idle_updates = 0;
        max_fitness = -1.0f;

        playerNeedsUpdate = true;
        playerLastTileX = players[g].body.tile_x;
        playerLastTileY = players[g].body.tile_y;

        // Run game loop until player dies
        while (1) {
            if (playerNeedsUpdate) {
                // evaluate_frame(game, players[g], generation[g]);
                generation[g].evaluate(game, players[g]);
            }
    
            ret = game.update(players[g]);

            //Skip simulating chromosomes if tile position of player hasn't changed
            if (playerLastTileX != players[g].body.tile_x || playerLastTileY != players[g].body.tile_y) {
                playerNeedsUpdate = true;       
            } else {
                playerNeedsUpdate = false;
            }
            playerLastTileX = players[g].body.tile_x;
            playerLastTileY = players[g].body.tile_y;
            
            // Check idle time
            if (players[g].fitness > max_fitness) {
                fitness_idle_updates = 0;
                max_fitness = players[g].fitness;
            } else {
                fitness_idle_updates++;
            }
            
            // Kill the player if fitness hasnt changed in 10 seconds
            if (fitness_idle_updates > AGENT_FITNESS_TIMEOUT) {
                ret = PLAYER_TIMEOUT;
                players[g].death_type = PLAYER_TIMEOUT;
            }

            if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT || ret == PLAYER_COMPLETE)
                break;
        }
    }
}

/**
 * @brief This selection function normalizes a chromosomes fitness with the maximum fitness of that generation
 *        and then uses that value as a probability for being selected to breed. Once (GEN_SIZE / 2) parents
 *        have been selected the first is bred with the second, second with the third, and so on (the first is
 *        also bred with the last to ensure each chrom breeds twice).
 * 
 * @param players Player obj that hold the fitnesses
 * @param generation Generation that has been evaluated
 * @param new_generation Where the new chromosomes will be stored
 * @param params Parameters describing the run
 */
void select_and_breed(std::vector<Player>& players, std::vector<NeuralNetwork> & curGen, std::vector<NeuralNetwork> & newGen, Params& params)
{
    int chrom;
    float best, sum;
    NeuralNetwork *survivors[params.gen_size / 2];
    int n_survivors = 0;
    unsigned int seedState = rand();
    
    //Find the worst and best score
    best = players[0].fitness;
    sum = 0;

    for (Player& player : players) {
        sum += player.fitness;
        best = std::max(best, player.fitness);
    }

    //Select survivors
    while (n_survivors < params.gen_size / 2) {
        chrom = rand_r(&seedState) % params.gen_size;
        if (chance(&seedState, players[chrom].fitness / best)) {
            survivors[n_survivors] = &curGen[chrom];
            n_survivors += 1;
        }
    }

    //Generate seeds for breeding
    std::vector<unsigned int> seeds(params.gen_size / 2);
    for (unsigned int& seed : seeds) {
        seed = rand_r(&seedState);
    }

    //Breed
    #pragma omp parallel for
    for (int surv = 0; surv < params.gen_size / 2; surv++) {
        // single_point_breed(*survivors[surv], *survivors[(surv + 1) % (params.gen_size / 2)], newGen[surv * 2], newGen[surv * 2 + 1], params, seeds[surv]);
        breed(*survivors[surv], *survivors[(surv + 1) % (params.gen_size / 2)], newGen[surv * 2], newGen[surv * 2 + 1], seeds[surv]);
    }
}

void mutateGeneration(std::vector<NeuralNetwork>& generation, float mutateRate)
{
    #pragma omp parallel for
    for (int i = 0; i < generation.size(); i++) {
        generation[i].mutate(mutateRate);
    }
}

/**
 * @brief Writes out the statistics of a run
 * 
 * @param dirname The directory to write the files into
 * @param game The game object
 * @param players A collection of players
 * @param chroms A collection of chromosomes
 * @param quiet Function will print if this is 0
 * @param write_winner Will write out the winner chromosome if 1
 * @param generation The generation number
 * @param params The parameters obj
 */
void get_gen_stats(std::string& dirname, Game& game, std::vector<Player>& players, std::vector<NeuralNetwork> & chroms, int quiet, int write_winner, int generation, Params& params)
{
    int completed, timedout, died, best_index, index;
    float max, min, avg;
    char fname[256];
    FILE *run_data;

    completed = timedout = died = 0;
    avg = 0.0f;
    max = min = players[0].fitness;
    best_index = index = 0;
    for (Player& player : players) {
        avg += player.fitness;

        if (player.fitness > max) {
            max = player.fitness;
            best_index = index;
        } 
           
        min = std::min(min, player.fitness);
        
        if (player.death_type == PLAYER_COMPLETE)
            completed++;
        else if (player.death_type == PLAYER_TIMEOUT)
            timedout++;
        else if (player.death_type == PLAYER_DEAD)
            died++;

        if (!quiet)
            printf("Player %d fitness: %0.4lf\n", index, player.fitness);
        
        index++;
    }

    // Write out best chromosome
    if (write_winner) {
        sprintf(fname, "./%s/gen_%04d_%.2lf.bin", dirname.c_str(), generation, max);
        chroms[best_index].writeToFile(fname, game.seed);
    }

    avg /= params.gen_size;

    // Write progress to file
    sprintf(fname, "%s/run_data.txt", dirname.c_str());
    run_data = fopen(fname, "a");
    fprintf(run_data, "%d, %d, %d, %lf, %lf, %lf\n", completed, timedout, died, avg, max, min);
    fclose(run_data);
    
    // Print out progress
    printf("\nDied:        %.2lf%%\nTimed out:   %.2lf%%\nCompleted:   %.2lf%%\nAvg fitness: %.2lf\n",
            (float)died / params.gen_size * 100, (float)timedout / params.gen_size * 100,
            (float)completed / params.gen_size * 100, avg);
    printf("Max fitness: %.2lf\n", max);
    printf("Min fitness: %.2lf\n", min);
}

/**
 * @brief Creates an output directory
 * 
 * @param dirname The directory name
 * @param seed The game seed
 * @param params The parameters obj
 */
void create_output_dir(std::string& dirname, unsigned int seed, Params& params)
{
    FILE* out_file;
    char name[4069];

    // Create run directory
    mkdir(dirname.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

    // Create run data file
    sprintf(name, "%s/run_data.txt", dirname.c_str());
    out_file = fopen(name, "w+");

    // Write out run data header
    fprintf(out_file, "%d, %d, %d, %d, %d, %d, %lf, %u\n", 
        params.in_h, params.in_w, params.hlc, params.npl, params.gen_size, params.generations, params.mutate_rate, seed);
    
    // Close file
    fclose(out_file);
}