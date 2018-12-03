#include <stdio.h>
#include <stdint.h>
#include <chromosome.hpp>
#include <gamelogic.hpp>
#include <genetic.hpp>
#include <neural_network.hpp>
#include <levelgen.hpp>
#include <cuda_helper.hpp>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>

__device__ 
void runChromosome(struct Game *game, struct Player *player, struct Chromosome *chrom);
__global__
void trainGeneration(struct Game *games, struct Player *players, struct Chromosome *gen);
void initialize_chromosome_gpu(struct Chromosome *chrom, uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl);
FILE* create_output_dir(char *dirname, unsigned int seed);

int main()
{
    struct Game *games;
    struct Player *players;
    struct Chromosome *genA, *genB, *cur_gen, *next_gen, *tmp;
    unsigned int member, seed, level_seed;
    int block_size, grid_size;
    FILE *run_data;
    char dir_name[4096];

    seed = (unsigned int)time(NULL);
    srand(seed);

    sprintf(dir_name, "./%u", seed);
    run_data = create_output_dir(dir_name, seed);

    level_seed = rand();

    printf("Running with %d chromosomes for %d generations\n", GEN_SIZE, GENERATIONS);
    printf("Chromosome stats:\n");
    printf("  IN_H: %d\n  IN_W: %d\n  HLC: %d\n  NPL: %d\n", IN_H, IN_W, HLC, NPL);
    printf("Level seed: %u\n", level_seed);
    printf("srand seed: %u\n", seed);

    // Allocate on host and device with unified memory
    cudaErrCheck( cudaMallocManaged((void **)&games, sizeof(struct Game) * GEN_SIZE) );
    cudaErrCheck( cudaMallocManaged((void **)&players, sizeof(struct Player) * GEN_SIZE) );
    cudaErrCheck( cudaMallocManaged((void **)&genA, sizeof(struct Chromosome) * GEN_SIZE) );
    cudaErrCheck( cudaMallocManaged((void **)&genB, sizeof(struct Chromosome) * GEN_SIZE) );

    // Initialize chromosomes
    for (member = 0; member < GEN_SIZE; member++) {
        initialize_chromosome_gpu(&genA[member], IN_H, IN_W, HLC, NPL);
        initialize_chromosome_gpu(&genB[member], IN_H, IN_W, HLC, NPL);
        generate_chromosome(&genA[member], rand());
    }

    // Calc grid/block size
    block_size = 32;
    grid_size = ceil((float)GEN_SIZE / (float)block_size); 

    cur_gen = genA;
    next_gen = genB;
    for (int gen = 0; gen < GENERATIONS; gen++) {
        puts("----------------------------");
        printf("Running generation %d/%d\n", gen + 1, GENERATIONS);

        // Regen levels and reset players
        for (member = 0; member < GEN_SIZE; member++) {
            game_setup(&games[member], &players[member], level_seed);
        }

        // Launch kernel
        trainGeneration <<< grid_size, block_size >>> (games, players, cur_gen);
        cudaErrCheck( cudaDeviceSynchronize() );

        // Get stats from run (1 tells function to not print each players fitness)
        get_gen_stats(run_data, dir_name, games, players, cur_gen, 1, 1, gen);

        // Usher in the new generation
        select_and_breed(players, cur_gen, next_gen);

        // Point current gen to new chromosomes and next gen to old
        tmp = cur_gen;
        cur_gen = next_gen;
        next_gen = tmp;
    }

    return 0;
}

void initialize_chromosome_gpu(struct Chromosome *chrom, uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl)
{
    chrom->in_h = in_h;
    chrom->in_w = in_w;
    chrom->hlc = hlc;
    chrom->npl = npl;

    chrom->input_adj_size = (in_h * in_w) * npl;
    chrom->hidden_adj_size = (hlc - 1) * (npl * npl);
    chrom->out_adj_size = BUTTON_COUNT * npl;

    cudaErrCheck( cudaMallocManaged((void **)&chrom->input_adj, sizeof(float) * chrom->input_adj_size) );
    cudaErrCheck( cudaMallocManaged((void **)&chrom->hidden_adj, sizeof(float) * chrom->hidden_adj_size) );
    cudaErrCheck( cudaMallocManaged((void **)&chrom->out_adj, sizeof(float) * chrom->out_adj_size) );
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

__device__ 
void runChromosome(struct Game *game, struct Player *player, struct Chromosome *chrom)
{
    int ret;
    
    float input_tiles[IN_W * IN_H];
    float node_outputs[NPL * HLC];
    uint8_t buttons;

    // Run game loop until player dies
    while (1) {
        ret = evaluate_frame(game, player, chrom, &buttons, input_tiles, node_outputs);

        if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT) {
            break;
        }
    }
}

__global__
void trainGeneration(struct Game *games, struct Player *players, struct Chromosome *gen)
{
    int member = blockIdx.x * blockDim.x + threadIdx.x;

    if (member < GEN_SIZE) {
        runChromosome(&games[member], &players[member], &gen[member]);
    }
}