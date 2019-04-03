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
#include <vector>

int main(int argc, char **argv)
{
    // Default global parameters
    Params params;
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
            printf("  -i    Size (in tiles) of the input area to the chromosomes (default %d)\n", params.in_h);
            printf("  -l    Number of hidden layers in the neural networks (default %d)\n",  params.hlc);
            printf("  -n    Nodes in each hidden layer (default %d)\n", params.npl);
            printf("  -c    Number of chromosomes in each generation (default %d)\n", params.gen_size);
            printf("  -g    Number of generations to run (default %d)\n", params.generations);            
            printf("  -m    Percent chance of mutation (float from 0 - 100, default %lf)\n", params.mutate_rate);
            printf("  -o    Output directory name\n");
            return 0;
        }
    }
    unsigned int seed, level_seed;
    // seed = (unsigned int)time(NULL);
    seed = 10;
    srand(seed);
    level_seed = rand();

    if (dir_name == NULL) {
        printf("Output directory is required!\n");
        exit(EXIT_FAILURE);
    }
    create_output_dir(dir_name, seed, params);

    Game game;
    Player *players;
    std::vector<Chromosome> genA(params.gen_size, Chromosome(params));
    std::vector<Chromosome> genB(params.gen_size, Chromosome(params));
    players = new Player[params.gen_size];
    std::vector<unsigned int> chrom_seeds(params.gen_size);

    printf("Running with %d chromosomes for %d generations\n", params.gen_size, params.generations);
    printf("Chromosome stats:\n");
    printf("  IN_H: %d\n  IN_W: %d\n  HLC: %d\n  NPL: %d\n", params.in_h, params.in_w, params.hlc, params.npl);
    printf("Level seed: %u\n", level_seed);
    printf("srand seed: %u\n", seed);
    fflush(stdout);

    // Populate list of seeds for parallel generation
    for (int chrom = 0; chrom < params.gen_size; chrom++) {
        chrom_seeds[chrom] = rand();
    }

    // Generate chromosomes in parallel
    #pragma omp parallel for
    for (int g = 0; g < params.gen_size; g++) {
        genA[g].generate(chrom_seeds[g]);
    }

    for (int gen = 0; gen < params.generations; gen++) {
        puts("----------------------------");
        printf("Running generation %d/%d\n", gen + 1, params.generations);

        // Generate map for generation
        game.genMap(level_seed);

        for (int player = 0; player < params.gen_size; player++) {
            players[player].reset();
        }

        run_generation(game, players, genA, params);

        // Write out and/or print stats
        get_gen_stats(dir_name, game, players, genA, 0, 1, gen, params);

        if (gen != params.generations - 1) {
            printf("\nBreeding generation %d/%d\n", gen + 2, params.generations);

            // Usher in the new generation
            select_and_breed(players, genA, genB, params);

            // Swap generations
            genA.swap(genB);
        }
    }
    puts("----------------------------\n");

    delete [] players;

    return 0;
}