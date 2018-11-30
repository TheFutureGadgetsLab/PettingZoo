#include <stdio.h>
#include <stdlib.h>
#include <chromosome.hpp>
#include <neural_network.hpp>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <gamelogic.hpp>
#include <sys/stat.h>
#include <defs.hpp>
#include <levelgen.hpp>

int main()
{
    uint8_t *chrom = NULL;
    float *tiles = NULL;
    float *node_outputs = NULL;
    struct Game game;
    struct Player player;
    uint8_t buttons[MAX_FRAMES];
    int buttons_index, ret;
    unsigned int seed;

    tiles = (float *)malloc(sizeof(float) * IN_W * IN_H);
    node_outputs = (float *)malloc(sizeof(float) * NPL * HLC);
    chrom = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
    
    seed = time(NULL);
    printf("Seed = %u\n", seed);

    game_setup(&player);
    levelgen_gen_map(&game, seed);
    generate_chromosome(chrom, IN_H, IN_W, HLC, NPL, seed);
    
    buttons_index = 0;
    while (1) {
        ret = evaluate_frame(&game, &player, chrom, tiles, node_outputs, buttons + buttons_index);
        buttons_index++;

        // for (int row = 0; row < IN_H; row++) {
        //     for (int col = 0; col < IN_W; col++) {
        //         if (tiles[row * IN_W + col] == 0) {
        //             printf("  ");
        //         } else if (tiles[row * IN_W + col] == 0.25f) {
        //             printf("# ");
        //         } else if (tiles[row * IN_W + col] == 0.5f) {
        //             printf("C ");
        //         } else if (tiles[row * IN_W + col] == 0.75f) {
        //             printf("T ");
        //         } else if (tiles[row * IN_W + col] == 1.0f) {
        //             printf("B ");
        //         } 
        //     }
        //     puts("");
        // }
        // puts("");

        for (int row = 0; row < NPL; row++) {
            for (int col = 0; col < HLC; col++) {
                printf("% 05.4lf ", node_outputs[row * HLC + col]);
            }
            puts("");
        }
        puts("");

        if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT)
            break;
    }
    
    if (ret == PLAYER_DEAD)
        printf("Player died\n");
    else
        printf("Player timed out\n");
    printf("Fitness: %lf\n", player.fitness);

    free(chrom);
    free(tiles);
    free(node_outputs);

    return 0;
}