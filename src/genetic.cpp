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
void mutate_u(uint8_t *data, size_t length);
void mutate_f(float *data, size_t length);

/*
 * This function takes an array of games, players, and chromosomes to be evaluated.
 * The fitnesses are written out into the float fitnesses array.
 */
int run_generation(struct Game games[GEN_SIZE], struct Player players[GEN_SIZE], struct Chromosome generation[GEN_SIZE], struct RecordedChromosome *winner)
{
    int game, buttons_index, ret;
    float *input_tiles;
    float *node_outputs;
    uint8_t buttons[MAX_FRAMES];

    /*
     * input_tiles holds the tiles around the player the NN sees.
     * node_outputs holds the outputs of the nodes. Currently only
     * one is allocated per generation because everything runs in
     * serial. This will need to change when running on the GPU
     */
    input_tiles = (float *)malloc(sizeof(float) * IN_W * IN_H);
    node_outputs = (float *)malloc(sizeof(float) * NPL * HLC);

    // Loop over the entire generation
    for (game = 0; game < GEN_SIZE; game++) {
        printf("\33[2K\r%d/%d", game, GEN_SIZE); // Clears line, moves cursor to the beginning
        fflush(stdout);

        buttons_index = 0;

        // Run game loop until player dies
        while (1) {
            ret = evaluate_frame(&games[game], &players[game], &generation[game],
                input_tiles, node_outputs, buttons + buttons_index);
            buttons_index++;

            if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT)
                break;
        }

        // Save run details if chrom is new best
        if (players[game].fitness > winner->fitness) {
            winner->chrom = &generation[game];
            memcpy(winner->buttons, buttons, MAX_FRAMES);
            winner->fitness = players[game].fitness;
            winner->game = &games[game];
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

    // Cross input layers and mutate
    split_loc = rand() % (parentA->input_act_size + 1);
    split(parentA->input_act, parentB->input_act, childA->input_act, childB->input_act, parentA->input_act_size, split_loc);
    mutate_u(childA->input_act, parentA->input_act_size);
    mutate_u(childB->input_act, parentA->input_act_size);

    // Cross input adj layers and mutate
    split_loc = rand() % (parentA->input_adj_size + 1);
    split(parentA->input_adj, parentB->input_adj, childA->input_adj, childB->input_adj, parentA->input_adj_size * sizeof(float), split_loc * sizeof(float));
    mutate_f(childA->input_adj, parentA->input_adj_size);
    mutate_f(childB->input_adj, parentA->input_adj_size);

    // Cross hidden act layer and mutate
    split_loc = rand() % (parentA->hidden_act_size + 1);
    split(parentA->hidden_act, parentB->hidden_act, childA->hidden_act, childB->hidden_act, parentA->hidden_act_size, split_loc);
    mutate_u(childA->hidden_act, parentA->hidden_act_size);
    mutate_u(childB->hidden_act, parentA->hidden_act_size);

    // Cross hidden layers and mutate
    size_t section_size = parentA->npl * parentA->npl;
    for (hl = 0; hl < parentA->hlc - 1; hl++) {
        split_loc = rand() % (section_size + 1);
        split(parentA->hidden_adj + section_size * hl, parentB->hidden_adj + section_size * hl,
              childA->hidden_adj + section_size * hl, childB->hidden_adj + section_size * hl, section_size * sizeof(float), split_loc * sizeof(float));
        mutate_f(childA->hidden_adj + section_size * hl, section_size);
        mutate_f(childB->hidden_adj + section_size * hl, section_size);
    }

    // Cross output adj layer and mutate
    split_loc = rand() % (parentA->out_adj_size + 1);
    split(parentA->out_adj, parentB->out_adj, childA->out_adj, childB->out_adj, parentA->out_adj_size * sizeof(float), split_loc * sizeof(float));
    mutate_f(childA->out_adj, parentA->out_adj_size);
    mutate_f(childB->out_adj, parentA->out_adj_size);
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

// Mutation
void mutate_u(uint8_t *data, size_t length)
{
    if (MUTATE_RATE == 0.0f)
        return;

    size_t i;
    for (i = 0; i < length; i++) {
        if (chance_gen(MUTATE_RATE))
            data[i] = !data[i];
    }
}

void mutate_f(float *data, size_t length)
{
    if (MUTATE_RATE == 0.0f)
        return;

    size_t i;
    for (i = 0; i < length; i++) {
        if (chance_gen(MUTATE_RATE))
            data[i] *= ((float)rand() / (float)RAND_MAX) * 2.0;
    }
}
