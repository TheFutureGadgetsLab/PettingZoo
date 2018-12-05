#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <genetic.hpp>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <gamelogic.hpp>

void split(void *parentA, void *parentB, void *childA, void *childB, size_t length, size_t split);
int chance_gen(float percent);
void single_point_breed(struct Chromosome *parentA, struct Chromosome *parentB, struct Chromosome *childA, struct Chromosome *childB);
void mutate(float *data, size_t length);

/*
 * This function takes an array of games, players, and chromosomes to be evaluated.
 * The fitnesses are written out into the float fitnesses array.
 */
int run_generation(struct Game *game, struct Player players[GEN_SIZE], struct Chromosome generation[GEN_SIZE])
{
    int g, ret;
    float *input_tiles;
    float *node_outputs;
    uint8_t buttons;
    int fitness_idle_updates;
    float max_fitness;

    input_tiles = (float *)malloc(sizeof(float) * IN_W * IN_H);
    node_outputs = (float *)malloc(sizeof(float) * NPL * HLC);

    // Loop over the entire generation
    for (g = 0; g < GEN_SIZE; g++) {
        printf("\33[2K\r%d/%d", g, GEN_SIZE); // Clears line, moves cursor to the beginning
        fflush(stdout);

        fitness_idle_updates = 0;
        max_fitness = -1.0f;

        // Run game loop until player dies
        while (1) {
            ret = evaluate_frame(game, &players[g], &generation[g], &buttons, input_tiles, node_outputs);
            
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

    return 0;
}

/*
 * This selection function normalizes a chromosomes fitness with the maximum fitness of that generation
 * and then uses that value as a probability for being selected to breed. Once (GEN_SIZE / 2) parents
 * have been selected the first is bred with the second, second with the third, and so on (the first is
 * also bred with the last to ensure each chrom breeds twice).
 */
void select_and_breed(struct Player players[GEN_SIZE], struct Chromosome *generation, struct Chromosome *new_generation)
{
    int game;
    float best, sum;
    struct Chromosome *survivors[GEN_SIZE / 2];
    int n_survivors = 0;

    //Find the worst and best score
    best = players[0].fitness;
    sum = 0;
    for (game = 0; game < GEN_SIZE; game++) {
        sum += players[game].fitness;
        if (players[game].fitness > best) {
            best = players[game].fitness;
        }
    }

    //Select survivors
    while (n_survivors < GEN_SIZE / 2) {
        game = rand() % GEN_SIZE;
        if (chance_gen(players[game].fitness / best)) {
            survivors[n_survivors] = &generation[game];
            n_survivors += 1;
        }
    }

    //Breed
    for (game = 0; game < GEN_SIZE / 2; game++) {
        single_point_breed(survivors[game], survivors[(game + 1) % (GEN_SIZE / 2)],
            &new_generation[game * 2], &new_generation[game * 2 + 1]);
    }
}

/*
 * Chooses as single point in each section of the chromosomes and childA is set up as
 * ParentA | Parent B, while childB is set up as ParentB | ParentA
 */
void single_point_breed(struct Chromosome *parentA, struct Chromosome *parentB, struct Chromosome *childA, struct Chromosome *childB)
{
    int split_loc, hl;

    // Cross input adj layers and mutate
    split_loc = rand() % (parentA->input_adj_size + 1);
    split(parentA->input_adj, parentB->input_adj, childA->input_adj, childB->input_adj, parentA->input_adj_size * sizeof(float), split_loc * sizeof(float));
    mutate(childA->input_adj, parentA->input_adj_size);
    mutate(childB->input_adj, parentA->input_adj_size);

    // Cross hidden layers and mutate
    size_t section_size = parentA->npl * parentA->npl;
    for (hl = 0; hl < parentA->hlc - 1; hl++) {
        split_loc = rand() % (section_size + 1);
        split(parentA->hidden_adj + section_size * hl, parentB->hidden_adj + section_size * hl,
              childA->hidden_adj + section_size * hl, childB->hidden_adj + section_size * hl, section_size * sizeof(float), split_loc * sizeof(float));
        mutate(childA->hidden_adj + section_size * hl, section_size);
        mutate(childB->hidden_adj + section_size * hl, section_size);
    }

    // Cross output adj layer and mutate
    split_loc = rand() % (parentA->out_adj_size + 1);
    split(parentA->out_adj, parentB->out_adj, childA->out_adj, childB->out_adj, parentA->out_adj_size * sizeof(float), split_loc * sizeof(float));
    mutate(childA->out_adj, parentA->out_adj_size);
    mutate(childB->out_adj, parentA->out_adj_size);
}

/*
 * Copies 'split' bytes into childA from parentA, then after that copies the rest of the section
 * (length - split) into childA from parentB. This is then done with childB, but the copy order
 * is reversed (parentB first then parentA).
 */
void split(void *parentA, void *parentB, void *childA, void *childB, size_t length, size_t split)
{
    // Must cast for pointer arithmetic
    memcpy(childA, parentA, split);
    memcpy((uint8_t *)childA + split, (uint8_t *)parentB + split, length - split);

    memcpy(childB, parentB, split);
    memcpy((uint8_t *)childB + split, (uint8_t *)parentA + split, length - split);
}

// Return 1 if random number is <= percent, otherwise 0
int chance_gen(float percent)
{
	return ((float)rand() / (float)RAND_MAX) < (percent / 100.0f);
}

void mutate(float *data, size_t length)
{
    if (MUTATE_RATE == 0.0f)
        return;

    size_t i;
    for (i = 0; i < length; i++) {
        if (chance_gen(MUTATE_RATE))
            data[i] *= ((float)rand() / (float)RAND_MAX) * 2.0;
    }
}

void get_gen_stats(char *dirname, struct Game *game, struct Player *players, struct Chromosome *chroms, int quiet, int write_winner, int generation)
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
    for(g = 0; g < GEN_SIZE; g++) {
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

    avg /= GEN_SIZE;

    // Write progress to file
    sprintf(fname, "%s/run_data.txt", dirname);
    run_data = fopen(fname, "w+");
    fprintf(run_data, "%d, %d, %d, %lf, %lf, %lf\n", completed, timedout, died, avg, max, min);
    fclose(run_data);
    
    // Print out progress
    printf("\nDied:        %.2lf%%\nTimed out:   %.2lf%%\nCompleted:   %.2lf%%\nAvg fitness: %.2lf\n",
            (float)died / GEN_SIZE * 100, (float)timedout / GEN_SIZE * 100,
            (float)completed / GEN_SIZE * 100, avg);
    printf("Max fitness: %.2lf\n", max);
    printf("Min fitness: %.2lf\n", min);
}