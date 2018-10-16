#ifndef LEVELGEN_H
#define LEVELGEN_H

#include <gamelogic.hpp>

void levelgen_gen_map(struct Game *game);
int randint(int max);
int randrange(int min, int max);
int chance(double percent);
void set_tile(struct Game *game, int x, int y, unsigned char val);
void create_hole(struct Game *game, int origin, int width);
void levelgen_clear_level(struct Game *game);
void create_stair_gap(struct Game *game, int origin, int height, int width);
void create_pipe(struct Game *game, int origin, int width, int height);


#endif
