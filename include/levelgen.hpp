#ifndef LEVELGEN_H
#define LEVELGEN_H

#include <gamelogic.hpp>

void levelgen_gen_map(struct Game *game, unsigned time);
void levelgen_clear_level(struct Game *game);
int chance(double percent);

#endif
