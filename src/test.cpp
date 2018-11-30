#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chromosome.hpp>
#include <gamelogic.hpp>
#include <genetic.hpp>
#include <neural_network.hpp>
#include <levelgen.hpp>
#include <cuda_helper.hpp>

__global__ 
void runChromosome(struct Game *game, struct Player *player, struct chromosome *chrom) {
    int buttons_index, ret;
    uint8_t buttons[MAX_FRAMES];
    float input_tiles[IN_W * IN_H];
    float node_outputs[NPL * HLC];

    buttons_index = 0;

    // Run game loop until player dies
    while (1) {
        ret = evaluate_frame(game, player, chrom, input_tiles, node_outputs, buttons + buttons_index);
        buttons_index++;

        if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT)
            break;
    }

    /* ret = evaluate_frame(&games[game], &players[game], generation[game], 
    input_tiles, node_outputs, buttons + buttons_index);
    buttons_index++;

    if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT)
        break; */
}

int main() {
    size_t chrom_size;
    struct chromosome chromH;
    struct chromosome chromD;
    struct Game gameH;
    struct Game *gameD = NULL;
    struct Player playerH;
    struct Player *playerD = NULL;
    
    // Allocate chromosome on host and device, generate
    initialize_chromosome(&chromH, IN_H, IN_W, HLC, NPL);
    generate_chromosome(&chromH, 10);
    cudaErrCheck( cudaMalloc((void **)&chromD, chrom_size) );

    game_setup(&playerH);
    levelgen_gen_map(&gameH, 10);
    
    // Allocate Player, Game on device
    cudaErrCheck( cudaMalloc((void **)&gameD, sizeof(struct Game)) );
    cudaErrCheck( cudaMalloc((void **)&playerD, sizeof(struct Player)) );

    // Copy everything over
    cudaErrCheck( cudaMemcpy(chromD, chromH, chrom_size, cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(gameD, &gameH, sizeof(struct Game), cudaMemcpyHostToDevice) );
    cudaErrCheck( cudaMemcpy(playerD, &playerH, sizeof(struct Player), cudaMemcpyHostToDevice) );

    // Launch kernel
    runChromosome <<< 1, 1 >>> (gameD, playerD, chromD);

    // Copy back
    cudaErrCheck( cudaMemcpy(chromH, chromD, chrom_size, cudaMemcpyDeviceToHost) );
    cudaErrCheck( cudaMemcpy(&playerH, playerD, sizeof(struct Player), cudaMemcpyDeviceToHost) );

    printf("Fitness: %lf\n", playerH.fitness);

    free_chromosome(&chromH);
    cudaErrCheck( cudaFree(chromD) );
    cudaErrCheck( cudaFree(gameD) );
    cudaErrCheck( cudaFree(playerD) );

    return 0;
}
