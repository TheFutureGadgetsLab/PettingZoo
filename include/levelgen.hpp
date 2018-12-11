/**
 * @file levelgen.cpp
 * @author Haydn Jones, Benjamin Mastripolito
 * @brief Defs for levelgen
 * @date 2018-12-11
 */
#ifndef LEVELGEN_H
#define LEVELGEN_H

#include <gamelogic.hpp>

// Level generation parameters
#define GROUND_HEIGHT (LEVEL_HEIGHT - 4)
#define START_PLATLEN 5
#define SPAWN_X 0
#define SPAWN_Y (GROUND_HEIGHT - 1)

// Sprite parameters (flat index into spritesheet)
#define EMPTY         0
#define PIPE_BOTTOM   1
#define PIPE_MIDDLE   2
#define PIPE_TOP      3
#define GRASS         4
#define DIRT          5
#define BRICKS        6
#define SPIKES_TOP    7
#define SPIKES_BOTTOM 8
#define COIN          9
#define FLAG          10

void levelgen_gen_map(struct Game *game, unsigned int seed);
int chance(unsigned int *seedp, float percent);
void levelgen_clear_level(struct Game *game);

#endif
