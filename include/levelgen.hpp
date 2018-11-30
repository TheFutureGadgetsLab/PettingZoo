#ifndef LEVELGEN_H
#define LEVELGEN_H

#include <gamelogic.hpp>

// Level generation parameters
#define GROUND_HEIGHT (LEVEL_HEIGHT - 4)
#define START_PLATLEN 5
#define SPAWN_X 0
#define SPAWN_Y (GROUND_HEIGHT - 1)

void levelgen_gen_map(struct Game *game, unsigned int seed);
int chance(unsigned int *seedp, float percent);

#endif
