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

FILE* create_output_dir(char *dirname, unsigned int seed);

int main()
{
    struct Game game;
    struct Player *players;
    struct Chromosome genA[GEN_SIZE], genB[GEN_SIZE];
    struct Chromosome *cur_gen, *next_gen, *tmp;
    int gen, g;
    unsigned int seed, level_seed;
    char dir_name[4096];

    seed = (unsigned int)time(NULL);
    seed = 10;
    srand(seed);

    sprintf(dir_name, "./%u", seed);
    create_output_dir(dir_name, seed);

    level_seed = rand();

    // Arrays for game and player structures
    players = (struct Player *)malloc(sizeof(struct Player) * GEN_SIZE);

    printf("Running with %d chromosomes for %d generations\n", GEN_SIZE, GENERATIONS);
    printf("Chromosome stats:\n");
    printf("  IN_H: %d\n  IN_W: %d\n  HLC: %d\n  NPL: %d\n", IN_H, IN_W, HLC, NPL);
    printf("Level seed: %u\n", level_seed);
    printf("srand seed: %u\n", seed);

    // Level seed for entire run
    // Allocate space for genA and genB chromosomes
    for (g = 0; g < GEN_SIZE; g++) {
        initialize_chromosome(&genA[g], IN_H, IN_W, HLC, NPL);
        initialize_chromosome(&genB[g], IN_H, IN_W, HLC, NPL);
        
        // Generate random chromosomes
        generate_chromosome(&genA[g], rand());
    }

    // Initially point current gen to genA, then swap next gen
    cur_gen = genA;
    next_gen = genB;
    for (gen = 0; gen < GENERATIONS; gen++) {
        puts("----------------------------");
        printf("Running generation %d/%d\n", gen + 1, GENERATIONS);

        // Generate seed for this gens levels and generate them
        game_setup(&game, level_seed);
        for (g = 0; g < GEN_SIZE; g++) {
            player_setup(&players[g]);
        }

        run_generation(&game, players, cur_gen);

        // Write out and/or print stats
        get_gen_stats(dir_name, &game, players, cur_gen, 0, 1, gen);

        // Usher in the new generation
        select_and_breed(players, cur_gen, next_gen);

        // Point current gen to new chromosomes and next gen to old
        tmp = cur_gen;
        cur_gen = next_gen;
        next_gen = tmp;
    }
    puts("----------------------------\n");


    for (g = 0; g < GEN_SIZE; g++) {
        free_chromosome(&genA[g]);
        free_chromosome(&genB[g]);
    }

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
