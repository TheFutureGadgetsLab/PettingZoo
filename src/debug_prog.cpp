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

int main()
{
    uint8_t *chrom = NULL;
    uint8_t *tiles = NULL;
    float *node_outputs = NULL;
    struct Game game;
    struct Player player;
    uint8_t buttons[MAX_FRAMES];
    int buttons_index, ret;
    unsigned int seed;

    tiles = (uint8_t *)malloc(sizeof(uint8_t) * IN_W * IN_H);
    node_outputs = (float *)malloc(sizeof(float) * NPL * HLC);
    chrom = (uint8_t *)malloc(sizeof(uint8_t) * get_chromosome_size_params(IN_H, IN_W, HLC, NPL));
    
    seed = time(NULL);
    seed = 10;
    printf("Seed = %u\n", seed);

    game_setup(&game, &player, seed);
    generate_chromosome(chrom, IN_H, IN_W, HLC, NPL, seed);
    
    buttons_index = 0;
    while (1) {
        ret = evaluate_frame(&game, &player, chrom, tiles, node_outputs, buttons + buttons_index);
        buttons_index++;

        for (int row = 0; row < NPL; row++) {
            for (int col = 0; col < HLC; col++) {
                printf("%lf ", node_outputs[row * HLC + col]);
            }
            puts("");
        }

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