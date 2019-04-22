#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <genetic.hpp>
#include <NeuralNetwork.hpp>
#include <gamelogic.hpp>
#include <unistd.h>
#include <vector>
#include <string>   

int getArgs(int argc, char **argv, std::string& dir_name, Params &params);

int main(int argc, char **argv)
{
    Params params;
    std::string dir_name;
    if (getArgs(argc, argv, dir_name, params) == -1) {
        return 0;
    }

    unsigned int seed, level_seed;
    // seed = (unsigned int)time(NULL);
    seed = 10;
    srand(seed);
    level_seed = rand();

    if (dir_name.empty() == true) {
        printf("Output directory is required!\n");
        exit(EXIT_FAILURE);
    }
    create_output_dir(dir_name, seed, params);

    Game game;
    std::vector<NeuralNetwork> genA(params.gen_size, NeuralNetwork(params));
    std::vector<NeuralNetwork> genB(params.gen_size, NeuralNetwork(params));
    std::vector<Player> players(params.gen_size);
    std::vector<unsigned int> chrom_seeds(params.gen_size);

    printf("Running with %d chromosomes for %d generations\n", params.gen_size, params.generations);
    printf("Chromosome stats:\n");
    printf("  IN_H: %d\n  IN_W: %d\n  HLC: %d\n  NPL: %d\n", params.in_h, params.in_w, params.hlc, params.npl);
    printf("Level seed: %u\n", level_seed);
    printf("srand seed: %u\n", seed);
    fflush(stdout);

    // Populate list of seeds for parallel generation
    for (unsigned int& seed : chrom_seeds) {
        seed = rand();
    }

    // Generate chromosomes in parallel
    #pragma omp parallel for
    for (int g = 0; g < params.gen_size; g++) {
        genA[g].seed(chrom_seeds[g]);
        genA[g].generate();
    }

    for (int gen = 0; gen < params.generations; gen++) {
        puts("----------------------------");
        printf("Running generation %d/%d\n", gen + 1, params.generations);

        // Generate map for generation
        game.genMap(level_seed);

        for (Player &player : players) {
            player.reset();
        }

        run_generation(game, players, genA, params);

        // Write out and/or print stats
        get_gen_stats(dir_name, game, genA, 1, 1, gen, params);

        if (gen != params.generations - 1) {
            printf("\nBreeding generation %d/%d\n", gen + 2, params.generations);

            // Breed new generation
            select_and_breed(players, genA, genB, params);
            // Mutate new generation
            mutateGeneration(genB, params.mutate_rate);

            // Swap generations
            genA.swap(genB);
        }
    }
    puts("----------------------------\n");

    return 0;
}

int getArgs(int argc, char **argv, std::string& dir_name, Params &params)
{
    // Default global parameters
    int opt;
 
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
            return -1;
        }
    }

    return 0;
}