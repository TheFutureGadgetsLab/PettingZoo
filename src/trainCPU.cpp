#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <genetic.hpp>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <gamelogic.hpp>
#include <levelgen.hpp>
#include <unistd.h>

int main(int argc, char **argv)
{
    // Default global parameters
    struct Params params = {IN_H, IN_W, HLC, NPL, GEN_SIZE, GENERATIONS, MUTATE_RATE};
    int opt;
    char *dir_name = NULL;
 
    while ((opt = getopt(argc, argv, "hi:l:n:c:g:m:o:")) != -1) {
        switch (opt) {
        // Output dir name
        case 'o':
            dir_name = optarg;
            break;
        // NN input size
        case 'i':
            params.in_h = atoi(optarg);
            params.in_w = atoi(optarg);
            break;
        // HLC
        case 'l':
            params.hlc = atoi(optarg);
            break;
        // NPL
        case 'n':
            params.npl = atoi(optarg);
            break;
        // Chromosome count
        case 'c':
            params.gen_size = atoi(optarg);
            break;
        // Generations
        case 'g':
            params.generations = atoi(optarg);
            break;
        // Mutate rate
        case 'm':
            params.mutate_rate = atof(optarg);
            break;
        case 'h':
        default: /* '?' */
            printf("Usage: %s -o OUTPUT_DIR [-i INPUT_SIZE] [-l HLC] [-n NPL] [-c GEN_SIZE] [-g GENERATIONS] [-m MUTATE_RATE]\n", argv[0]);
            printf("  -i    Size (in tiles) of the input area to the chromosomes (default %d)\n", IN_H);
            printf("  -l    Number of hidden layers in the neural networks (default %d)\n", HLC);
            printf("  -n    Nodes in each hidden layer (default %d)\n", NPL);
            printf("  -c    Number of chromosomes in each generation (default %d)\n", GEN_SIZE);
            printf("  -g    Number of generations to run (default %d)\n", GENERATIONS);            
            printf("  -m    Percent chance of mutation (float from 0 - 100, default %lf)\n", MUTATE_RATE);
            printf("  -o    Output directory name\n");
            return 0;
        }
    }

    struct Game game;
    struct Player *players;
    struct Chromosome genA[params.gen_size], genB[params.gen_size];
    struct Chromosome *cur_gen, *next_gen, *tmp;
    int gen, g;
    unsigned int seed, level_seed;

    seed = (unsigned int)time(NULL);
    srand(seed);
    level_seed = rand();

    if (dir_name == NULL) {
        printf("Output directory is required!\n");
        exit(EXIT_FAILURE);
    }
    create_output_dir(dir_name, seed, &params);

    // Arrays for game and player structures
    players = (struct Player *)malloc(sizeof(struct Player) * params.gen_size);

    printf("Running with %d chromosomes for %d generations\n", params.gen_size, params.generations);
    printf("Chromosome stats:\n");
    printf("  IN_H: %d\n  IN_W: %d\n  HLC: %d\n  NPL: %d\n", params.in_h, params.in_w, params.hlc, params.npl);
    printf("Level seed: %u\n", level_seed);
    printf("srand seed: %u\n", seed);

    // Level seed for entire run
    // Allocate space for genA and genB chromosomes
    for (g = 0; g < params.gen_size; g++) {
        initialize_chromosome(&genA[g], params.in_h, params.in_w, params.hlc, params.npl);
        initialize_chromosome(&genB[g], params.in_h, params.in_w, params.hlc, params.npl);
        
        // Generate random chromosomes
        generate_chromosome(&genA[g], rand());
    }

    // Initially point current gen to genA, then swap next gen
    cur_gen = genA;
    next_gen = genB;
    for (gen = 0; gen < params.generations; gen++) {
        puts("----------------------------");
        printf("Running generation %d/%d\n", gen + 1, params.generations);

        // Generate seed for this gens levels and generate them
        game_setup(&game, level_seed);
        for (g = 0; g < params.gen_size; g++) {
            player_setup(&players[g]);
        }

        run_generation(&game, players, cur_gen, &params);

        // Write out and/or print stats
        get_gen_stats(dir_name, &game, players, cur_gen, 0, 1, gen, &params);

        // Usher in the new generation
        select_and_breed(players, cur_gen, next_gen, &params);

        // Point current gen to new chromosomes and next gen to old
        tmp = cur_gen;
        cur_gen = next_gen;
        next_gen = tmp;
    }
    puts("----------------------------\n");


    for (g = 0; g < params.gen_size; g++) {
        free_chromosome(&genA[g]);
        free_chromosome(&genB[g]);
    }

    free(players);

    return 0;
}