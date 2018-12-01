#include <stdio.h>
#include <stdint.h>
#include <chromosome.hpp>
#include <gamelogic.hpp>
#include <genetic.hpp>
#include <neural_network.hpp>
#include <levelgen.hpp>

void runChromosome(struct Game *game, struct Player *player, struct chromosome chrom)
{
    int buttons_index, ret;
    uint8_t buttons[MAX_FRAMES];
    float input_tiles[IN_W * IN_H];
    float node_outputs[NPL * HLC];

    buttons_index = 0;

    // Run game loop until player dies
    while (1) {
        ret = evaluate_frame(game, player, &chrom, input_tiles, node_outputs, buttons + buttons_index);
        buttons_index++;

        if (ret == PLAYER_DEAD || ret == PLAYER_TIMEOUT)
            break;
    }
    printf("Fitness: %lf\n", player->fitness);
}

int main() {
    struct chromosome chrom;
    struct Game game;
    struct Player player;
    
    // Allocate chromosome on host and device, generate
    initialize_chromosome(&chrom, IN_H, IN_W, HLC, NPL);
    generate_chromosome(&chrom, 144);
    game_setup(&player);
    levelgen_gen_map(&game, 144);
    
    runChromosome(&game, &player, chrom);

    //Free everything
    free_chromosome(&chrom);

    return 0;
}