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
#include <sys/stat.h>

#define BLOCK_SIZE 32

//nvprof --analysis-metrics --export-profile out_profile.prof --dependency-analysis --track-memory-allocations on --unified-memory-profiling per-process-device --cpu-profiling on ./trainGPU
__global__
void trainGeneration(struct Game *game, struct Player *players, struct Chromosome *gen);
void initialize_chromosome_gpu(struct Chromosome *chrom, uint8_t in_h, uint8_t in_w, uint8_t hlc, uint16_t npl);
void create_output_dir(char *dirname, unsigned int seed);
void free_chromosome_gpu(struct Chromosome *chrom);

int main()
{
    struct Game *game;
    struct Player *players;
    struct Chromosome *genA, *genB, *cur_gen, *next_gen, *tmp;
    unsigned int member, seed, level_seed;
    int gen;
    char dir_name[4096];
    int grid_size; 

    grid_size = ceil(GEN_SIZE / (float)BLOCK_SIZE); 

    seed = (unsigned int)time(NULL);
    seed = 1543861639;
    srand(seed);

    sprintf(dir_name, "./%u", seed);
    create_output_dir(dir_name, seed);

    level_seed = rand();

    printf("Running with %d chromosomes for %d generations\n", GEN_SIZE, GENERATIONS);
    printf("Chromosome stats:\n");
    printf("  IN_H: %d\n  IN_W: %d\n  HLC: %d\n  NPL: %d\n", IN_H, IN_W, HLC, NPL);
    printf("Level seed: %u\n", level_seed);
    printf("srand seed: %u\n", seed);

    // Allocate on host and device with unified memory
    cudaErrCheck( cudaMallocManaged((void **)&game, sizeof(struct Game)) );
    cudaErrCheck( cudaMallocManaged((void **)&players, sizeof(struct Player) * GEN_SIZE) );
    cudaErrCheck( cudaMallocManaged((void **)&genA, sizeof(struct Chromosome) * GEN_SIZE) );
    cudaErrCheck( cudaMallocManaged((void **)&genB, sizeof(struct Chromosome) * GEN_SIZE) );

    // Initialize chromosomes
    for (member = 0; member < GEN_SIZE; member++) {
        initialize_chromosome_gpu(&genA[member], IN_H, IN_W, HLC, NPL);
        initialize_chromosome_gpu(&genB[member], IN_H, IN_W, HLC, NPL);
        generate_chromosome(&genA[member], rand());
    }

    cur_gen = genA;
    next_gen = genB;
    for (gen = 0; gen < GENERATIONS; gen++) {
        puts("----------------------------");
        printf("Running generation %d/%d\n", gen + 1, GENERATIONS);

        // Regen levels and reset players
        game_setup(game, level_seed);
        for (member = 0; member < GEN_SIZE; member++) {
            player_setup(&players[member]);
        }
        
        trainGeneration <<< grid_size, BLOCK_SIZE >>> (game, players, cur_gen);
        cudaErrCheck( cudaDeviceSynchronize() );

        // Get stats from run (1 tells function to not print each players fitness)
        get_gen_stats(dir_name, game, players, cur_gen, 1, 1, gen);

        // Usher in the new generation
        if (gen != (GENERATIONS - 1)) {
            select_and_breed(players, cur_gen, next_gen);
        }

        tmp = cur_gen;
        cur_gen = next_gen;
        next_gen = tmp;
    }

    for (member = 0; member < GEN_SIZE; member++) {
        free_chromosome_gpu(&genA[member]);
        free_chromosome_gpu(&genB[member]);
    }

    cudaErrCheck( cudaFree(game) );
    cudaErrCheck( cudaFree(players) );
    cudaErrCheck( cudaFree(genA) );
    cudaErrCheck( cudaFree(genB) );

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

void create_output_dir(char *dirname, unsigned int seed)
{
    FILE* out_file;
    char name[4069];

    // Create run directory
    mkdir(dirname, S_IRWXU | S_IRWXG | S_IRWXO);

    // Create run data file
    sprintf(name, "%s/run_data.txt", dirname);
    out_file = fopen(name, "w+");

    // Write out run data header
    fprintf(out_file, "%d, %d, %d, %d, %d, %d, %lf, %u\n", IN_H, IN_W, HLC, NPL, GEN_SIZE, GENERATIONS, MUTATE_RATE, seed);
    
    // Close file
    fclose(out_file);
}

__global__
void trainGeneration(struct Game *game, struct Player *players, struct Chromosome *gen)
{
    int member = blockIdx.x * blockDim.x + threadIdx.x;
    int ret;
    float input_tiles[IN_W * IN_H];
    float node_outputs[NPL * HLC];
    uint8_t buttons;
    
    if (member < GEN_SIZE) {
        while (1) {
            ret = evaluate_frame(game, &players[member], &gen[member], &buttons, input_tiles, node_outputs);

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
