#include <stdio.h>
#include <stdint.h>
#include <chromosome.hpp>
#include <gamelogic.hpp>
#include <genetic.hpp>
#include <neural_network.hpp>
#include <levelgen.hpp>
#include <cuda_helper.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

// 2954.31 / 10000
void initialize_chromosome_gpu(struct Chromosome *chromD, struct Chromosome chromH);
void print_gen_stats(struct Player players[GEN_SIZE], int quiet);

__device__ 
void runChromosome(struct Game *game, struct Player *player, struct Chromosome *chrom)
{
    int buttons_index, ret;
    uint8_t buttons[MAX_FRAMES];
    float input_tiles[IN_W * IN_H];
    float node_outputs[NPL * HLC];

    buttons_index = 0;

    // Run game loop until player dies
    while (1) {
        ret = evaluate_frame(game, player, chrom, input_tiles, node_outputs, buttons + buttons_index);
        buttons_index++;

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

int main()
{
    struct Game *gameH, *gameD;
    struct Player *playerH, *playerD;
    struct Chromosome *chromH, *chromHD, *chromD;
    unsigned int member, seed, level_seed;
    int block_size, grid_size;

    seed = (unsigned int)time(NULL);
    seed = 10;
    srand(seed);
    level_seed = rand();

    printf("Running with %d chromosomes for %d generations\n", GEN_SIZE, GENERATIONS);
    printf("Chromosome stats:\n");
    printf("  IN_H: %d\n  IN_W: %d\n  HLC: %d\n  NPL: %d\n", IN_H, IN_W, HLC, NPL);
    printf("Level seed: %u\n", level_seed);
    printf("srand seed: %u\n", seed);
    puts("----------------------------");

    gameH = (struct Game *)malloc(sizeof(struct Game) * GEN_SIZE);
    playerH = (struct Player *)malloc(sizeof(struct Player) * GEN_SIZE);

    chromH = (struct Chromosome *)malloc(sizeof(struct Chromosome) * GEN_SIZE);
    chromHD = (struct Chromosome *)malloc(sizeof(struct Chromosome) * GEN_SIZE);

    cudaErrCheck( cudaMalloc((void **)&gameD, sizeof(struct Game) * GEN_SIZE) );
    cudaErrCheck( cudaMalloc((void **)&playerD, sizeof(struct Player) * GEN_SIZE) );
    cudaErrCheck( cudaMalloc((void **)&chromD, sizeof(struct Chromosome) * GEN_SIZE) );

    for (member = 0; member < GEN_SIZE; member++) {
        // Generate level on host
        levelgen_gen_map(&gameH[member], level_seed);
        
        // Set up player -> allocate player on device -> copy to device
        game_setup(&playerH[member]);

        initialize_chromosome(&chromH[member], IN_H, IN_W, HLC, NPL);
        generate_chromosome(&chromH[member], rand());
        initialize_chromosome_gpu(&chromHD[member], chromH[member]);
    }

    cudaErrCheck( cudaMemcpy(gameD, gameH, sizeof(struct Game) * GEN_SIZE, cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(playerD, playerH, sizeof(struct Player) * GEN_SIZE, cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(chromD, chromHD, sizeof(struct Chromosome) * GEN_SIZE, cudaMemcpyHostToDevice) );

    block_size = 128;
    grid_size = ceil(GEN_SIZE / (float)block_size);
    trainGeneration <<< grid_size, block_size >>> (gameD, playerD, chromD);

    cudaErrCheck( cudaMemcpy(playerH, playerD, sizeof(struct Player) * GEN_SIZE, cudaMemcpyDeviceToHost) );

    print_gen_stats(playerH, 0);

    return 0;
}

void initialize_chromosome_gpu(struct Chromosome *chromD, struct Chromosome chromH)
{
    chromD->input_act_size = chromH.input_act_size;
    chromD->input_adj_size = chromH.input_adj_size;
    chromD->hidden_act_size = chromH.hidden_act_size;
    chromD->hidden_adj_size = chromH.hidden_adj_size;
    chromD->out_adj_size = chromH.out_adj_size;

    chromD->npl = chromH.npl;
    chromD->in_w = chromH.in_w;
    chromD->in_h = chromH.in_h;
    chromD->hlc = chromH.hlc;

    cudaErrCheck( cudaMalloc((void **)&chromD->input_act, sizeof(uint8_t) * chromH.input_act_size) );
    cudaErrCheck( cudaMalloc((void **)&chromD->input_adj, sizeof(float) * chromH.input_adj_size) );
    cudaErrCheck( cudaMalloc((void **)&chromD->hidden_act, sizeof(uint8_t) * chromH.hidden_act_size) );
    cudaErrCheck( cudaMalloc((void **)&chromD->hidden_adj, sizeof(float) * chromH.hidden_adj_size) );
    cudaErrCheck( cudaMalloc((void **)&chromD->out_adj, sizeof(float) * chromH.out_adj_size) );

    cudaErrCheck( cudaMemcpy(chromD->input_act, chromH.input_act, 
        sizeof(uint8_t) * chromH.input_act_size, cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(chromD->input_adj, chromH.input_adj, 
        sizeof(float) * chromH.input_adj_size, cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(chromD->hidden_act,  chromH.hidden_act, 
        sizeof(uint8_t) * chromH.hidden_act_size, cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(chromD->hidden_adj, chromH.hidden_adj, 
        sizeof(float) * chromH.hidden_adj_size, cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(chromD->out_adj, chromH.out_adj, 
        sizeof(float) * chromH.out_adj_size, cudaMemcpyHostToDevice) );
}

void print_gen_stats(struct Player players[GEN_SIZE], int quiet)
{
    int game, completed, timedout, died;
    float max, min, avg;

    completed = 0;
    timedout = 0;
    died = 0;
    avg = 0.0f;
    max = players[0].fitness;
    min = players[0].fitness;
    
    for(game = 0; game < GEN_SIZE; game++) {
        avg += players[game].fitness;

        if (players[game].fitness > max)
            max = players[game].fitness;
        else if (players[game].fitness < min)
            min = players[game].fitness;

        if (players[game].death_type == PLAYER_COMPLETE)
            completed++;
        else if (players[game].death_type == PLAYER_TIMEOUT)
            timedout++;
        else if (players[game].death_type == PLAYER_DEAD)
            died++;

        if (!quiet)
            printf("Player %d fitness: %0.2lf\n", game, players[game].fitness);
    }

    avg /= GEN_SIZE;

    printf("\nDied:        %.2lf%%\nTimed out:   %.2lf%%\nCompleted:   %.2lf%%\nAvg fitness: %.2lf\n",
            (float)died / GEN_SIZE * 100, (float)timedout / GEN_SIZE * 100,
            (float)completed / GEN_SIZE * 100, avg);
    printf("Max fitness: %.2lf\n", max);
    printf("Min fitness: %.2lf\n", min);
}