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

void initialize_chromosome_gpu(struct chromosome *chromD, struct chromosome chromH);

__device__ 
void runChromosome(struct Game *game, struct Player *player, struct chromosome *chrom)
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
            player->fitness = 1000;
            break;
        }
    }
}

__global__
void trainGeneration(struct Game *games, struct Player *players, struct chromosome *generation)
{
    int member = blockIdx.x * blockDim.x + threadIdx.x;

    if (member < GEN_SIZE)
        runChromosome(&games[member], &players[member], &generation[member]);
    
    players[0].fitness = 10;
}

int main() {
    struct chromosome *chromH, chromD[GEN_SIZE];
    struct Game *gamesH, *gamesD;
    struct Player *playersH, *playersD;
    int member;

    gamesH = (struct Game *)malloc(sizeof(struct Game) * GEN_SIZE);
    cudaErrCheck( cudaMalloc((void **)&gamesD, sizeof(struct Game) * GEN_SIZE) );

    playersH = (struct Player *)malloc(sizeof(struct Player) * GEN_SIZE);
    cudaErrCheck( cudaMalloc((void **)&playersD, sizeof(struct Player) * GEN_SIZE) );

    chromH = (struct chromosome *)malloc(sizeof(struct chromosome) * GEN_SIZE);
    cudaErrCheck( cudaMalloc((void **)&chromD, sizeof(struct chromosome) * GEN_SIZE) );

    // Allocate chromosome on host and device, generate
    for (member = 0; member < GEN_SIZE; member++) {
        initialize_chromosome(&chromH[member], IN_H, IN_W, HLC, NPL);
        generate_chromosome(&chromH[member], 144);
        initialize_chromosome_gpu(&chromD[member], chromH[member]);

        game_setup(&playersH[member]);
        levelgen_gen_map(&gamesH[member], 144);
    }

    cudaErrCheck( cudaMemcpy(gamesD, gamesH, sizeof(struct Game) * GEN_SIZE, cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(playersD, playersH, sizeof(struct Player) * GEN_SIZE, cudaMemcpyHostToDevice) );

    // Launch kernel
    trainGeneration <<< 1, 1024 >>> (gamesD, playersD, chromD);
    cudaErrCheck( cudaGetLastError() );

    cudaMemcpy(playersH, playersD, sizeof(struct Player) * GEN_SIZE, cudaMemcpyDeviceToHost);

    for (member = 0; member < GEN_SIZE; member++) {
        printf("Fitness: %lf\n", playersH[member].fitness);
    }

    // printf("Fitness: %lf\n", playerH->fitness);

    // //Free everything
    // free_chromosome(&chromH);
    // // cudaErrCheck( cudaFree(chromD) );
    // cudaErrCheck( cudaFree(gameD) );
    // free(playerH);

    return 0;
}

void initialize_chromosome_gpu(struct chromosome *chromD, struct chromosome chromH)
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
