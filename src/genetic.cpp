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
int run_generation(struct Game games[GEN_SIZE], struct Player players[GEN_SIZE], uint8_t *generation[GEN_SIZE], float fitnesses[GEN_SIZE]);
int chance_gen(unsigned int *seedp, double percent);
void select_and_breed(uint8_t **generation, float *fitnesses, uint8_t **new_generation, unsigned int seed);
void single_point_breed(uint8_t *parentA, uint8_t *parentB, uint8_t *childA, uint8_t *childB, unsigned int *seed_state);

int main()
{
    struct Game *games;
    struct Player *players;
    uint8_t *genA[GEN_SIZE], *genB[GEN_SIZE];
    float fitnesses[GEN_SIZE];
    float avg_fitness, max, min;
    int completed, timedout, died;
    uint8_t **cur_gen, **next_gen, **tmp;
    unsigned int seed, level_seed, game;
        
    seed = (unsigned int)time(NULL);

    srand(seed);
        
    games = (struct Game *)malloc(sizeof(struct Game) * GEN_SIZE);
    players = (struct Player *)malloc(sizeof(struct Player) * GEN_SIZE);

    printf("Generating %d chromosomes and games...\n", GEN_SIZE);    
    for (game = 0; game < GEN_SIZE; game++) {
        genA[game] = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
        genB[game] = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
        generate_chromosome(genA[game], IN_H, IN_W, HLC, NPL, rand());
    }

    cur_gen = genA;
    next_gen = genB;
    for (int gen = 0; gen < GENERATIONS; gen++) {
        puts("----------------------------");
        printf("Running generation %d\n", gen);

        avg_fitness = 0;
        timedout = 0;
        completed = 0;
        died = 0;

        level_seed = rand();

        for (game = 0; game < GEN_SIZE; game++) {
            game_setup(&games[game], &players[game], level_seed);
        }
    
        run_generation(games, players, cur_gen, fitnesses);

        max = fitnesses[0];
        min = fitnesses[0];
        for(game = 0; game < GEN_SIZE; game++) {
            avg_fitness += fitnesses[game];

            if (fitnesses[game] > max)
                max = fitnesses[game];
            else if (fitnesses[game] < max)
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

        select_and_breed(cur_gen, fitnesses, next_gen, rand());

        tmp = cur_gen;
        cur_gen = next_gen;    
        next_gen = tmp;
    }
    puts("----------------------------");
    
    return 0;   
}

int run_generation(struct Game games[GEN_SIZE], struct Player players[GEN_SIZE], uint8_t *generation[GEN_SIZE], 
    float fitnesses[GEN_SIZE])
{
    int game, buttons_index, ret;
    uint8_t *input_tiles;
    float *node_outputs;
    uint8_t buttons[MAX_FRAMES];
    
    input_tiles = (uint8_t *)malloc(sizeof(uint8_t) * IN_W * IN_H);
    node_outputs = (float *)malloc(sizeof(float) * NPL * HLC);

    for (game = 0; game < GEN_SIZE; game++) {                
        buttons_index = 0;
        while (1) {
            ret = evaluate_frame(&games[game], &players[game], generation[game], input_tiles, node_outputs, buttons + buttons_index);
            buttons_index++;

            if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT)
                break;
        }
        fitnesses[game] = players[game].fitness;
    }

    return 0;
}
// Select the best from a generation, then breed them into a new generation
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

// Chooses as single point in each section of the chromosomes and childA is set up as
// ParentA | Parent B, while childB is set up as ParentB | ParentA
void single_point_breed(uint8_t *parentA, uint8_t *parentB, uint8_t *childA, uint8_t *childB, unsigned int *seed_state)
{
    size_t section_size;
    int split_loc, hl, npl;
    struct params parentA_p, parentB_p, childA_p, childB_p;

    // Set chrom headers
    childA[0] = parentA[0];
    childB[0] = parentA[0];
    childA[1] = parentA[1];
    childB[1] = parentA[1];
    *((uint16_t *)childA + 1) = *((uint16_t *)parentA + 1);
    *((uint16_t *)childB + 1) = *((uint16_t *)parentA + 1);
    childA[4] = parentA[4];
    childB[4] = parentA[4];

    get_params(parentA, &parentA_p);
    get_params(parentB, &parentB_p);
    get_params(childA, &childA_p);
    get_params(childB, &childB_p);

    npl = parentA_p.npl;

    // Cross input layers
    section_size = (uint8_t *)parentA_p.input_adj - parentA_p.input_act;
    split_loc = rand_r(seed_state) % (section_size + 1);
    split(parentA_p.input_act, parentB_p.input_act, childA_p.input_act, childB_p.input_act, section_size, split_loc);

    // Cross input adj layers
    section_size = parentA_p.hidden_act - (uint8_t *)parentA_p.input_adj;
    split_loc = rand_r(seed_state) % ((section_size + 1) / 4);
    split(parentA_p.input_adj, parentB_p.input_adj, childA_p.input_adj, childB_p.input_adj, section_size, split_loc * sizeof(float));

    // Cross hidden act layer
    section_size = (uint8_t *)parentA_p.hidden_adj - parentA_p.hidden_act;
    split_loc = rand_r(seed_state) % (section_size + 1);
    split(parentA_p.hidden_act, parentB_p.hidden_act, childA_p.hidden_act, childB_p.hidden_act, section_size, split_loc);
    
    // Cross hidden layers
    section_size = npl * npl;
    for (hl = 0; hl < parentA_p.hlc - 1; hl++) {
        split_loc = rand_r(seed_state) % (section_size + 1);
        split(parentA_p.hidden_adj + section_size * hl, parentB_p.hidden_adj + section_size * hl, 
              childA_p.hidden_adj + section_size * hl, childB_p.hidden_adj + section_size * hl, section_size * sizeof(float), split_loc * sizeof(float));
    }

    // Cross output adj layer
    section_size = parentA_p.size - ((uint8_t *)parentA_p.out_adj - parentA);
    split_loc = rand_r(seed_state) % ((section_size + 1) / 4);
    split(parentA_p.out_adj, parentB_p.out_adj, childA_p.out_adj, childB_p.out_adj, section_size, split_loc * sizeof(float));
}

void split(void *parentA, void *parentB, void *childA, void *childB, size_t length, size_t split)
{
    memcpy(childA, parentA, split);
    memcpy((uint8_t *)childA + split, (uint8_t *)parentB + split, length - split);

    memcpy(childB, parentB, split);
    memcpy((uint8_t *)childB + split, (uint8_t *)parentA + split, length - split);
}

// Return 0 or 1 probabilistically
int chance_gen(unsigned int *seedp, double percent)
{
	return ((double)rand_r(seedp) / (double)RAND_MAX) <= (percent / 100.0);
}
