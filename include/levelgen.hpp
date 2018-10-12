#ifndef LEVELGEN_H
#define LEVELGEN_H

#include <gamelogic.hpp>

void levelgen_gen_map(Game &game);
int randint(int max);
int randrange(int min, int max);
int chance(double percent);
void set_tile(Game &game, int x, int y, unsigned char val);

#endif
