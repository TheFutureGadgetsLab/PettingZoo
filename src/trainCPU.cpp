#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>
#include <genetic.hpp>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <gamelogic.hpp>
#include <levelgen.hpp>

void print_gen_stats(struct Player players[GEN_SIZE], int quiet);
void write_out_progress(FILE *fh, struct Player players[GEN_SIZE]);
FILE* create_output_dir(char *dirname, unsigned int seed);

int main()
{
    struct Game *games;
    struct Player *players;
    struct Chromosome genA[GEN_SIZE], genB[GEN_SIZE];
    struct Chromosome *cur_gen, *next_gen, *tmp;
    int gen, game;
    unsigned int seed, level_seed;
    char dir_name[4096];

    seed = (unsigned int)time(NULL);
    seed = 1543861639;
    srand(seed);

    sprintf(dir_name, "./%u", seed);
    create_output_dir(dir_name, seed);

    level_seed = rand();

    // Arrays for game and player structures
    games = (struct Game *)malloc(sizeof(struct Game) * GEN_SIZE);
    players = (struct Player *)malloc(sizeof(struct Player) * GEN_SIZE);

    printf("Running with %d chromosomes for %d generations\n", GEN_SIZE, GENERATIONS);
    printf("Chromosome stats:\n");
    printf("  IN_H: %d\n  IN_W: %d\n  HLC: %d\n  NPL: %d\n", IN_H, IN_W, HLC, NPL);
    printf("Level seed: %u\n", level_seed);
    printf("srand seed: %u\n", seed);

    // Level seed for entire run
    // Allocate space for genA and genB chromosomes
    for (game = 0; game < GEN_SIZE; game++) {
        initialize_chromosome(&genA[game], IN_H, IN_W, HLC, NPL);
        initialize_chromosome(&genB[game], IN_H, IN_W, HLC, NPL);
        
        // Generate random chromosomes
        generate_chromosome(&genA[game], rand());
    }

    // Initially point current gen to genA, then swap next gen
    cur_gen = genA;
    next_gen = genB;
    for (gen = 0; gen < GENERATIONS; gen++) {
        puts("----------------------------");
        printf("Running generation %d/%d\n", gen + 1, GENERATIONS);

        // Generate seed for this gens levels and generate them
        for (game = 0; game < GEN_SIZE; game++) {
            game_setup(&games[game], &players[game], level_seed);
        }

        run_generation(games, players, cur_gen);

        // Write out and/or print stats
        get_gen_stats(dir_name, games, players, cur_gen, 1, 1, gen);

        // Usher in the new generation
        select_and_breed(players, cur_gen, next_gen);

        // Point current gen to new chromosomes and next gen to old
        tmp = cur_gen;
        cur_gen = next_gen;
        next_gen = tmp;
    }
    puts("----------------------------\n");


    for (game = 0; game < GEN_SIZE; game++) {
        free_chromosome(&genA[game]);
        free_chromosome(&genB[game]);
    }

    free(games);
    free(players);

    return 0;
}

FILE* create_output_dir(char *dirname, unsigned int seed)
{
    FILE* out_file;
    char name[4069];

    sprintf(name, "%s", dirname);

    mkdir(name, S_IRWXU | S_IRWXG | S_IRWXO);

    sprintf(name, "%s/run_data.txt", dirname);
    out_file = fopen(name, "w+");

    // Write out header data
    fprintf(out_file, "%d, %d, %d, %d, %d, %d, %lf, %u\n", IN_H, IN_W, HLC, NPL, GEN_SIZE, GENERATIONS, MUTATE_RATE, seed);
    fflush(out_file);

    return out_file;
}
