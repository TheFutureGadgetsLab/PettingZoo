#ifndef LEVELGEN_H
#define LEVELGEN_H

#include <gamelogic.hpp>

void levelgen_gen_map(struct Game *game, unsigned int seed);
void levelgen_clear_level(struct Game *game);
int chance(unsigned int *seedp, double percent);

#endif
