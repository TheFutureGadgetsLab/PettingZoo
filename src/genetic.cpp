/**
 * @file genetic.cpp
 * @author Haydn Jones, Benjamin Mastripolito
 * @brief Holds functions that govern the genetic algorithm
 * @date 2018-12-11
 */
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <genetic.hpp>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <gamelogic.hpp>
#include <sys/stat.h>

void split(void *parentA, void *parentB, void *childA, void *childB, size_t length, size_t split);
int chance_gen(float percent);
void single_point_breed(struct Chromosome *parentA, struct Chromosome *parentB, struct Chromosome *childA, struct Chromosome *childB, struct Params *params);
void mutate(float *data, size_t length, float mutate_rate);

/**
 * @brief This function takes a game, players, and chromosomes to be evaluated.
 * 
 * @param game The game object to use in the evaluation
 * @param players A collection of players
 * @param generation A collection of chromosomes
 * @param params The run parameters
 */
void run_generation(struct Game *game, struct Player *players, struct Chromosome *generation, struct Params *params)
{
    int g, ret;
    float *input_tiles;
    float *node_outputs;
    int fitness_idle_updates;
    float max_fitness;
    uint8_t buttons[BUTTON_COUNT];
    bool playerNeedsUpdate;
    int playerLastTileX = 0;
    int playerLastTileY = 0;

    input_tiles = (float *)malloc(sizeof(float) * params->in_w * params->in_h);
    node_outputs = (float *)malloc(sizeof(float) * params->npl * params->hlc);

    // Loop over the entire generation
    for (g = 0; g < params->gen_size; g++) {
        printf("\33[2K\r%d/%d", g, params->gen_size); // Clears line, moves cursor to the beginning
        fflush(stdout);

        fitness_idle_updates = 0;
        max_fitness = -1.0f;

        playerNeedsUpdate = true;
        playerLastTileX = players[g].body.tile_x;
        playerLastTileY = players[g].body.tile_y;

        // Run game loop until player dies
        while (1) {
            if (playerNeedsUpdate) {
                evaluate_frame(game, &players[g], &generation[g], buttons, input_tiles, node_outputs);
            }
    
            ret = game_update(game, &players[g], buttons);

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

            if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT)
                break;
        }
    }

    // Clear line so progress indicator is removed
    printf("\33[2K\r");
    fflush(stdout);

    free(input_tiles);
    free(node_outputs);
}

/**
 * @brief This selection function normalizes a chromosomes fitness with the maximum fitness of that generation
 *        and then uses that value as a probability for being selected to breed. Once (GEN_SIZE / 2) parents
 *        have been selected the first is bred with the second, second with the third, and so on (the first is
 *        also bred with the last to ensure each chrom breeds twice).
 * 
 * @param players Player structures that hold the fitnesses
 * @param generation Generation that has been evaluated
 * @param new_generation Where the new chromosomes will be stored
 * @param params Parameters describing the run
 */
void select_and_breed(struct Player *players, struct Chromosome *generation, struct Chromosome *new_generation, struct Params *params)
{
    int game;
    float best, sum;
    struct Chromosome *survivors[params->gen_size / 2];
    int n_survivors = 0;

    //Find the worst and best score
    best = players[0].fitness;
    sum = 0;
    for (game = 0; game < params->gen_size; game++) {
        sum += players[game].fitness;
        if (players[game].fitness > best) {
            best = players[game].fitness;
        }
    }

    //Select survivors
    while (n_survivors < params->gen_size / 2) {
        game = rand() % params->gen_size;
        if (chance_gen(players[game].fitness / best)) {
            survivors[n_survivors] = &generation[game];
            n_survivors += 1;
        }
    }

    //Breed
    for (game = 0; game < params->gen_size / 2; game++) {
        single_point_breed(survivors[game], survivors[(game + 1) % (params->gen_size / 2)],
            &new_generation[game * 2], &new_generation[game * 2 + 1], params);
    }
}

/**
 * @brief Chooses as single point in each section of the chromosomes and childA is set up as
 *        ParentA | Parent B, while childB is set up as ParentB | ParentA
 * 
 * @param parentA The chromosome to breed with B
 * @param parentB The chromosome to breed with
 * @param childA Pointer to memory where child A will be held
 * @param childB Pointer to memory where child B will be held
 * @param params The run parameters struct
 */
void single_point_breed(struct Chromosome *parentA, struct Chromosome *parentB, struct Chromosome *childA, struct Chromosome *childB, struct Params *params)
{
    int split_loc, hl;

    // Cross input adj layers and mutate
    split_loc = rand() % (parentA->input_adj_size + 1);
    split(parentA->input_adj, parentB->input_adj, childA->input_adj, childB->input_adj, parentA->input_adj_size * sizeof(float), split_loc * sizeof(float));
    mutate(childA->input_adj, parentA->input_adj_size, params->mutate_rate);
    mutate(childB->input_adj, parentA->input_adj_size, params->mutate_rate);

    // Cross hidden layers and mutate
    size_t section_size = parentA->npl * parentA->npl;
    for (hl = 0; hl < parentA->hlc - 1; hl++) {
        split_loc = rand() % (section_size + 1);
        split(parentA->hidden_adj + section_size * hl, parentB->hidden_adj + section_size * hl,
              childA->hidden_adj + section_size * hl, childB->hidden_adj + section_size * hl, section_size * sizeof(float), split_loc * sizeof(float));
        mutate(childA->hidden_adj + section_size * hl, section_size, params->mutate_rate);
        mutate(childB->hidden_adj + section_size * hl, section_size, params->mutate_rate);
    }

    // Cross output adj layer and mutate
    split_loc = rand() % (parentA->out_adj_size + 1);
    split(parentA->out_adj, parentB->out_adj, childA->out_adj, childB->out_adj, parentA->out_adj_size * sizeof(float), split_loc * sizeof(float));
    mutate(childA->out_adj, parentA->out_adj_size, params->mutate_rate);
    mutate(childB->out_adj, parentA->out_adj_size, params->mutate_rate);
}

/**
 * @brief Copies 'split' bytes into childA from parentA, then after that copies the rest of the section
 *        (length - split) into childA from parentB. This is then done with childB, but the copy order
 *        is reversed (parentB first then parentA).
 * 
 * @param parentA Pointer to beginning of parentA data
 * @param parentB Pointer to beginning of parentB data
 * @param childA Pointer to beginning of childA data
 * @param childB Pointer to beginning of parentB data
 * @param length Length in bytes to be copied
 * @param split location the split occurs
 */
void split(void *parentA, void *parentB, void *childA, void *childB, size_t length, size_t split)
{
    // Must cast for pointer arithmetic
    memcpy(childA, parentA, split);
    memcpy((uint8_t *)childA + split, (uint8_t *)parentB + split, length - split);

    memcpy(childB, parentB, split);
    memcpy((uint8_t *)childB + split, (uint8_t *)parentA + split, length - split);
}

/**
 * @brief Return 1 if random number is <= percent, otherwise 0
 * 
 * @param percent Percent chance between 0.0 and 100.0
 * @return int 1 or 0
 */
int chance_gen(float percent)
{
	return ((float)rand() / (float)RAND_MAX) < (percent / 100.0f);
}

/**
 * @brief Randomly mutate this floating point data with a given mutate probability
 *        Data can mutate in the range of 0-200%
 * 
 * @param data Float array to be mutated
 * @param length Length of the array
 * @param mutate_rate Probability of mutation
 */
void mutate(float *data, size_t length, float mutate_rate)
{
    if (mutate_rate == 0.0f)
        return;

    size_t i;
    for (i = 0; i < length; i++) {
        if (chance_gen(mutate_rate))
            data[i] *= ((float)rand() / (float)RAND_MAX) * 2.0;
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
 * @param params The parameters struct
 */
void get_gen_stats(char *dirname, struct Game *game, struct Player *players, 
    struct Chromosome *chroms, int quiet, int write_winner, int generation, struct Params *params)
{
    int g, completed, timedout, died, best_index;
    float max, min, avg;
    char fname[256];
    FILE *run_data;

    completed = 0;
    timedout = 0;
    died = 0;
    avg = 0.0f;
    max = players[0].fitness;
    min = players[0].fitness;
    best_index = 0;
    for(g = 0; g < params->gen_size; g++) {
        avg += players[g].fitness;

        if (players[g].fitness > max) {
            max = players[g].fitness;
            best_index = g;
        } else if (players[g].fitness < min) {
            min = players[g].fitness;
        }

        if (players[g].death_type == PLAYER_COMPLETE)
            completed++;
        else if (players[g].death_type == PLAYER_TIMEOUT)
            timedout++;
        else if (players[g].death_type == PLAYER_DEAD)
            died++;

        if (!quiet)
            printf("Player %d fitness: %0.4lf\n", g, players[g].fitness);
    }

    // Write out best chromosome
    if (write_winner) {
        sprintf(fname, "./%s/gen_%d_%.2lf.bin", dirname, generation, max);
        write_out_chromosome(fname, &chroms[best_index], game->seed);
    }

    avg /= params->gen_size;

    // Write progress to file
    sprintf(fname, "%s/run_data.txt", dirname);
    run_data = fopen(fname, "a");
    fprintf(run_data, "%d, %d, %d, %lf, %lf, %lf\n", completed, timedout, died, avg, max, min);
    fclose(run_data);
    
    // Print out progress
    printf("\nDied:        %.2lf%%\nTimed out:   %.2lf%%\nCompleted:   %.2lf%%\nAvg fitness: %.2lf\n",
            (float)died / params->gen_size * 100, (float)timedout / params->gen_size * 100,
            (float)completed / params->gen_size * 100, avg);
    printf("Max fitness: %.2lf\n", max);
    printf("Min fitness: %.2lf\n", min);
}

/**
 * @brief Creates an output directory
 * 
 * @param dirname The directory name
 * @param seed The game seed
 * @param params The parameters structure
 */
void create_output_dir(char *dirname, unsigned int seed, struct Params *params)
{
    FILE* out_file;
    char name[4069];

    // Create run directory
    mkdir(dirname, S_IRWXU | S_IRWXG | S_IRWXO);

    // Create run data file
    sprintf(name, "%s/run_data.txt", dirname);
    out_file = fopen(name, "w+");

    // Write out run data header
    fprintf(out_file, "%d, %d, %d, %d, %d, %d, %lf, %u\n", 
        params->in_h, params->in_w, params->hlc, params->npl, params->gen_size, params->generations, params->mutate_rate, seed);
    
    // Close file
    fclose(out_file);
}