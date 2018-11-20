#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <genetic.hpp>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <gamelogic.hpp>
#include <levelgen.hpp>

void split(void *parentA, void *parentB, void *childA, void *childB, size_t length, size_t split);
int run_generation(struct Game games[GEN_SIZE], struct Player players[GEN_SIZE], uint8_t *generation[GEN_SIZE],
    float fitnesses[GEN_SIZE], struct RecordedChromosome *winner);
int chance_gen(unsigned int *seedp, float percent);
void select_and_breed(uint8_t **generation, float *fitnesses, uint8_t **new_generation, unsigned int seed);
void single_point_breed(uint8_t *parentA, uint8_t *parentB, uint8_t *childA, uint8_t *childB, unsigned int *seed_state);
void mutate_u(uint8_t *data, size_t length, unsigned int *seed_state);
void mutate_f(float *data, size_t length, unsigned int *seed_state);

struct RecordedChromosome {
    uint8_t *chrom;
    uint8_t *buttons;
    float fitness;
    struct Game *game;
};

int main()
{
    struct Game *games;
    struct Player *players;
    uint8_t *genA[GEN_SIZE], *genB[GEN_SIZE];
    float fitnesses[GEN_SIZE];
    float avg_fitness, max, min;
    int completed, timedout, died, gen;
    uint8_t **cur_gen, **next_gen, **tmp;
    unsigned int seed, level_seed, game;
    uint8_t buttons[MAX_FRAMES];
    struct RecordedChromosome winner;
    winner.buttons = buttons;
    winner.fitness = 0;

    seed = (unsigned int)time(NULL);
    srand(seed);

    // Arrays for game and player structures
    games = (struct Game *)malloc(sizeof(struct Game) * GEN_SIZE);
    players = (struct Player *)malloc(sizeof(struct Player) * GEN_SIZE);

    printf("Generating %d chromosomes and games...\n", GEN_SIZE);

    // Allocate space for genA and genB chromosomes
    for (game = 0; game < GEN_SIZE; game++) {
        genA[game] = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
        genB[game] = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
        // Generate chromosome and give it a random seed
        generate_chromosome(genA[game], IN_H, IN_W, HLC, NPL, rand());
    }

    // Initially point current gen to genA, then swap next gen
    cur_gen = genA;
    next_gen = genB;
    for (gen = 0; gen < GENERATIONS; gen++) {
        puts("----------------------------");
        printf("Running generation %d\n", gen);

        avg_fitness = 0;
        timedout = 0;
        completed = 0;
        died = 0;

        // Generate seed for this gens levels and generate them
        level_seed = rand();
        for (game = 0; game < GEN_SIZE; game++) {
            game_setup(&games[game], &players[game], level_seed);
        }

        run_generation(games, players, cur_gen, fitnesses, &winner);

        // Get stats from run
        max = fitnesses[0];
        min = fitnesses[0];
        for(game = 0; game < GEN_SIZE; game++) {
            avg_fitness += fitnesses[game];

            if (fitnesses[game] > max)
                max = fitnesses[game];
            else if (fitnesses[game] < min)
                min = fitnesses[game];

            if (players[game].death_type == PLAYER_COMPLETE)
                completed++;
            else if (players[game].death_type == PLAYER_TIMEOUT)
                timedout++;
            else if (players[game].death_type == PLAYER_DEAD)
                died++;

            printf("Player %d fitness: %0.2lf\n", game, fitnesses[game]);
        }

        avg_fitness /= GEN_SIZE;

        printf("\nDied:        %.2lf%%\nTimed out:   %.2lf%%\nCompleted:   %.2lf%%\nAvg fitness: %.2lf\n",
                (float)died / (float)GEN_SIZE * 100, (float)timedout / (float)GEN_SIZE * 100,
                (float)completed / (float)GEN_SIZE * 100, avg_fitness);
        printf("Max fitness: %.2lf\n", max);
        printf("Min fitness: %.2lf\n", min);

        // Usher in the new generation
        select_and_breed(cur_gen, fitnesses, next_gen, rand());

        // Point current gen to new chromosome and next gen to old
        tmp = cur_gen;
        cur_gen = next_gen;
        next_gen = tmp;
    }
    puts("----------------------------");

    printf("fitness: %f, seed: %u\n", winner.fitness, winner.game->seed);
    write_out(winner.buttons, MAX_FRAMES, winner.chrom, winner.game->seed);

    for (game = 0; game < GEN_SIZE; game++) {
        free(genA[game]);
        free(genB[game]);
    }

    free(games);
    free(players);

    return 0;
}

/*
 * This function takes an array of games, players, and chromosomes to be evaluated.
 * The fitnesses are written out into the float fitnesses array.
 */
int run_generation(struct Game games[GEN_SIZE], struct Player players[GEN_SIZE], uint8_t *generation[GEN_SIZE],
    float fitnesses[GEN_SIZE], struct RecordedChromosome *winner)
{
    int game, buttons_index, ret;
    uint8_t *input_tiles;
    float *node_outputs;
    uint8_t buttons[MAX_FRAMES];

    /*
     * input_tiles holds the tiles around the player the NN sees.
     * node_outputs holds the outputs of the nodes. Currently only
     * one is allocated per generation because everything runs in
     * serial. This will need to change when running on the GPU
     */
    input_tiles = (uint8_t *)malloc(sizeof(uint8_t) * IN_W * IN_H);
    node_outputs = (float *)malloc(sizeof(float) * NPL * HLC);

    // Loop over the entire generation
    for (game = 0; game < GEN_SIZE; game++) {
        printf("\33[2K\r"); // Clears line, moves cursor to the beginning
        printf("%d/%d", game, GEN_SIZE);
        fflush(stdout);

        buttons_index = 0;

        // Run game loop until player dies
        while (1) {
            ret = evaluate_frame(&games[game], &players[game], generation[game], 
                input_tiles, node_outputs, buttons + buttons_index);
            buttons_index++;

            if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT)
                break;
        }
        fitnesses[game] = players[game].fitness;

        //Is best?
        if (fitnesses[game] > winner->fitness) {
            winner->chrom = generation[game];
            memcpy(winner->buttons, buttons, MAX_FRAMES);
            winner->buttons = buttons;
            winner->fitness = fitnesses[game];
            winner->game = &games[game];
        }
    }    

    printf("\33[2K\r");
    fflush(stdout);

    free(input_tiles);
    free(node_outputs);

    return 0;
}

/*
 * This selection function normalizes a chromosomes fitness with the maximum fitness of that generation
 * (so all values are in the range [0, 1]) and then uses that value as a probability for being selected
 * to breed. Once (GEN_SIZE / 2) parents have been selected the first is bred with the second, second
 * with the third, and so on (the first is also bred with the last to ensure each chrom breeds twice).
 */
void select_and_breed(uint8_t **generation, float *fitnesses, uint8_t **new_generation, unsigned int seed)
{
    int i;
    float best;
    uint8_t *survivors[GEN_SIZE / 2];
    int n_survivors = 0;
    uint seed_state = seed;

    //Find the worst and best score
    best = fitnesses[0];
    for (i = 0; i < GEN_SIZE; i++) {
        if (fitnesses[i] > best) {
            best = fitnesses[i];
        }
    }

    //Select survivors
    while (n_survivors < GEN_SIZE / 2) {
        i = rand_r(&seed_state) % GEN_SIZE;
        if (chance_gen(&seed_state, fitnesses[i] / best)) {
            survivors[n_survivors] = generation[i];
            n_survivors += 1;
        }
    }

    //Breed
    for (i = 0; i < GEN_SIZE / 2; i++) {
        single_point_breed(survivors[i], survivors[(i + 1) % (GEN_SIZE / 2)],
            new_generation[i * 2], new_generation[i * 2 + 1], &seed_state);
    }
}

/*
 * Chooses as single point in each section of the chromosomes and childA is set up as
 * ParentA | Parent B, while childB is set up as ParentB | ParentA
 */
void single_point_breed(uint8_t *parentA, uint8_t *parentB, uint8_t *childA, uint8_t *childB, unsigned int *seed_state)
{
    size_t section_size;
    int split_loc, hl;
    struct params parentA_p, parentB_p, childA_p, childB_p;

    // Set chrom headers
    memcpy(childA, parentA, HEADER_SIZE);
    memcpy(childB, parentA, HEADER_SIZE);

    get_params(parentA, &parentA_p);
    get_params(parentB, &parentB_p);
    get_params(childA, &childA_p);
    get_params(childB, &childB_p);

    // Cross input layers
    section_size = (uint8_t *)parentA_p.input_adj - parentA_p.input_act;
    split_loc = rand_r(seed_state) % (section_size + 1);
    split(parentA_p.input_act, parentB_p.input_act, childA_p.input_act, childB_p.input_act, section_size, split_loc);
    mutate_u(childA_p.input_act, section_size, seed_state);
    mutate_u(childB_p.input_act, section_size, seed_state);

    // Cross input adj layers
    section_size = parentA_p.hidden_act - (uint8_t *)parentA_p.input_adj;
    split_loc = rand_r(seed_state) % ((section_size + 1) / 4);
    split(parentA_p.input_adj, parentB_p.input_adj, childA_p.input_adj, childB_p.input_adj, section_size, split_loc * sizeof(float));
    mutate_f(childA_p.input_adj, section_size / 4, seed_state);
    mutate_f(childB_p.input_adj, section_size / 4, seed_state);

    // Cross hidden act layer
    section_size = (uint8_t *)parentA_p.hidden_adj - parentA_p.hidden_act;
    split_loc = rand_r(seed_state) % (section_size + 1);
    split(parentA_p.hidden_act, parentB_p.hidden_act, childA_p.hidden_act, childB_p.hidden_act, section_size, split_loc);
    mutate_u(childA_p.hidden_act, section_size, seed_state);
    mutate_u(childB_p.hidden_act, section_size, seed_state);

    // Cross hidden layers
    section_size = parentA_p.npl * parentA_p.npl;
    for (hl = 0; hl < parentA_p.hlc - 1; hl++) {
        split_loc = rand_r(seed_state) % (section_size + 1);
        split(parentA_p.hidden_adj + section_size * hl, parentB_p.hidden_adj + section_size * hl,
              childA_p.hidden_adj + section_size * hl, childB_p.hidden_adj + section_size * hl, section_size * sizeof(float), split_loc * sizeof(float));
        mutate_f(childA_p.input_adj + section_size * hl, section_size, seed_state);
        mutate_f(childB_p.input_adj + section_size * hl, section_size, seed_state);
    }

    // Cross output adj layer
    section_size = parentA_p.size - ((uint8_t *)parentA_p.out_adj - parentA);
    split_loc = rand_r(seed_state) % ((section_size + 1) / 4);
    split(parentA_p.out_adj, parentB_p.out_adj, childA_p.out_adj, childB_p.out_adj, section_size, split_loc * sizeof(float));
    mutate_f(childA_p.out_adj, section_size / 4, seed_state);
    mutate_f(childB_p.out_adj, section_size / 4, seed_state);
}

/* 
 * Copies 'split' bytes into childA from parentA, then after that copies the rest of the section
 * into childA from parentB. This is then done with childB, but the copy order is reversed
 * (parentB first then parentA).
 */
void split(void *parentA, void *parentB, void *childA, void *childB, size_t length, size_t split)
{
    memcpy(childA, parentA, split);
    memcpy((uint8_t *)childA + split, (uint8_t *)parentB + split, length - split);

    memcpy(childB, parentB, split);
    memcpy((uint8_t *)childB + split, (uint8_t *)parentA + split, length - split);
}

// Return 1 if random number is <= percent, otherwise 0
int chance_gen(unsigned int *seedp, float percent)
{
	return ((float)rand_r(seedp) / (float)RAND_MAX) <= (percent / 100.0f);
}

// Mutation
void mutate_u(uint8_t *data, size_t length, unsigned int *seed_state)
{
    size_t i;
    for (i = 0; i < length; i++) {
        if (chance_gen(seed_state, MUTATE_CHANCE))
            data[i] = !data[i];
    }
}

void mutate_f(float *data, size_t length, unsigned int *seed_state)
{
    size_t i;
    for (i = 0; i < length; i++) {
        if (chance_gen(seed_state, MUTATE_CHANCE))
            data[i] *= ((double)rand_r(seed_state) / (double)RAND_MAX) * 2.0;
    }
}
