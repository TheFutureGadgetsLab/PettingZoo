#include <stdio.h>
#include <stdint.h>
#include <chromosome.hpp>
#include <gamelogic.hpp>
#include <genetic.hpp>
#include <neural_network.hpp>
#include <cuda_helper.hpp>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#define BLOCK_SIZE 32

__global__
void trainGeneration(struct Game *game, struct Player *players, struct Chromosome *gen, struct Params params, float *node_outputs, float *input_tiles);
void initialize_chromosome_gpu(struct Chromosome *chrom, struct Params *params);
void free_chromosome_gpu(struct Chromosome *chrom);

int main(int argc, char **argv)
{
    // Default global parameters
    struct Params params = {IN_H, IN_W, HLC, NPL, GEN_SIZE, GENERATIONS, MUTATE_RATE};
    int opt;
    char *dir_name = NULL;
 
    while ((opt = getopt(argc, argv, "i:l:n:c:g:m:o:")) != -1) {
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
        default: /* '?' */
            printf("Usage: ./%s -o OUTPUT_DIR [-i INPUT_SIZE] [-l HLC] [-n NPL] [-c GEN_SIZE] [-g GENERATIONS] [-m MUTATE_RATE]\n", argv[0]);
            printf(" -i    Size (in tiles) of the input area to the chromosomes (default %d)\n", IN_H);
            printf(" -l    Number of hidden layers in the neural networks (default %d)\n", HLC);
            printf(" -n    Nodes in each hidden layer (default %d)\n", NPL);
            printf(" -c    Number of chromosomes in each generation (default %d)\n", GEN_SIZE);
            printf(" -g    Number of generations to run (default %d)\n", GENERATIONS);            
            printf(" -m    Percent chance of mutation rate (float from 0 - 100, default %lf)\n", MUTATE_RATE);
            printf(" -o    Output directory name\n");
            exit(EXIT_FAILURE);
        }
    }

    struct Game *game;
    struct Player *players;
    struct Chromosome *genA, *genB, *cur_gen, *next_gen, *tmp;
    unsigned int member, seed, level_seed;
    int gen, grid_size;
    float *input_tiles;
    float *node_outputs;

    grid_size = ceil(params.gen_size / (float)BLOCK_SIZE); 

    seed = (unsigned int)time(NULL);
    srand(seed);
    level_seed = rand();

    if (dir_name == NULL) {
        printf("Output directory is required!\n");
        exit(EXIT_FAILURE);
    }
    create_output_dir(dir_name, seed, &params);

    printf("Running with %d chromosomes for %d generations\n", params.gen_size, params.generations);
    printf("Chromosome stats:\n");
    printf("  IN_H: %d\n  IN_W: %d\n  HLC: %d\n  NPL: %d\n", params.in_h, params.in_w, params.hlc, params.npl);
    printf("Level seed: %u\n", level_seed);
    printf("srand seed: %u\n", seed);

    // Allocate on host and device with unified memory
    cudaErrCheck( cudaMallocManaged((void **)&game, sizeof(struct Game)) );
    cudaErrCheck( cudaMallocManaged((void **)&players, sizeof(struct Player) * params.gen_size) );
    cudaErrCheck( cudaMallocManaged((void **)&genA, sizeof(struct Chromosome) * params.gen_size) );
    cudaErrCheck( cudaMallocManaged((void **)&genB, sizeof(struct Chromosome) * params.gen_size) );
    cudaErrCheck( cudaMalloc((void **)&input_tiles, sizeof(float) * params.gen_size * params.in_w * params.in_h) );
    cudaErrCheck( cudaMalloc((void **)&node_outputs, sizeof(float) * params.gen_size * params.npl * params.hlc) );

    // Initialize chromosomes
    for (member = 0; member < params.gen_size; member++) {
        initialize_chromosome_gpu(&genA[member], &params);
        initialize_chromosome_gpu(&genB[member], &params);
        generate_chromosome(&genA[member], rand());
    }

    cur_gen = genA;
    next_gen = genB;
    for (gen = 0; gen < params.generations; gen++) {
        puts("----------------------------");
        printf("Running generation %d/%d\n", gen + 1, params.generations);

        // Regen levels and reset players
        game_setup(game, level_seed);
        for (member = 0; member < params.gen_size; member++) {
            player_setup(&players[member]);
        }

        trainGeneration <<< grid_size, BLOCK_SIZE >>> (game, players, cur_gen, params, node_outputs, input_tiles);
        cudaErrCheck( cudaDeviceSynchronize() );

        // Get stats from run (1 tells function to not print each players fitness)
        get_gen_stats(dir_name, game, players, cur_gen, 1, 1, gen, &params);

        // Usher in the new generation
        if (gen != (params.generations - 1)) {
            select_and_breed(players, cur_gen, next_gen, &params);
        }

        tmp = cur_gen;
        cur_gen = next_gen;
        next_gen = tmp;
    }

    for (member = 0; member < params.gen_size; member++) {
        free_chromosome_gpu(&genA[member]);
        free_chromosome_gpu(&genB[member]);
    }

    cudaErrCheck( cudaFree(game) );
    cudaErrCheck( cudaFree(players) );
    cudaErrCheck( cudaFree(genA) );
    cudaErrCheck( cudaFree(genB) );

    return 0;
}

void initialize_chromosome_gpu(struct Chromosome *chrom, struct Params *params)
{
    chrom->in_h = params->in_h;
    chrom->in_w = params->in_w;
    chrom->hlc = params->hlc;
    chrom->npl = params->npl;

    chrom->input_adj_size = (params->in_h * params->in_w) * params->npl;
    chrom->hidden_adj_size = (params->hlc - 1) * (params->npl * params->npl);
    chrom->out_adj_size = BUTTON_COUNT * params->npl;

    cudaErrCheck( cudaMallocManaged((void **)&chrom->input_adj, sizeof(float) * chrom->input_adj_size) );
    if (params->hlc > 1)
        cudaErrCheck( cudaMallocManaged((void **)&chrom->hidden_adj, sizeof(float) * chrom->hidden_adj_size) );
    cudaErrCheck( cudaMallocManaged((void **)&chrom->out_adj, sizeof(float) * chrom->out_adj_size) );
}

__global__
void trainGeneration(struct Game *game, struct Player *players, struct Chromosome *gen, struct Params params, float *node_outputs, float *input_tiles)
{
    int member = blockIdx.x * blockDim.x + threadIdx.x;
    int ret;
    uint8_t buttons;
    int fitness_idle_updates = 0;
    float max_fitness = -1.0f;
    float *my_tiles, *my_nodes;
    
    if (member < params.gen_size) {
        my_tiles = input_tiles + member * params.in_w * params.in_h;
        my_nodes = node_outputs + member * params.npl * params.hlc;
        while (1) {
            ret = evaluate_frame(game, &players[member], &gen[member], &buttons, my_tiles, my_nodes);

            // Check idle time
            if (players[member].fitness > max_fitness) {
                fitness_idle_updates = 0;
                max_fitness = players[member].fitness;
            } else {
                fitness_idle_updates++;
            }
            
            // Kill the player if fitness hasnt changed in 5 seconds
            if (fitness_idle_updates > AGENT_FITNESS_TIMEOUT) {
                ret = PLAYER_TIMEOUT;
                players[member].death_type = PLAYER_TIMEOUT;
            }

            if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT) {
                break;
            }
        }
    }
}

void free_chromosome_gpu(struct Chromosome *chrom)
{
    cudaFree(chrom->input_adj);
    cudaFree(chrom->hidden_adj);
    cudaFree(chrom->out_adj);
}