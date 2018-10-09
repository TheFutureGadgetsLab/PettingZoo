#ifndef LEVELGEN_H
#define LEVELGEN_H

void levelgen_gen_map(struct game_obj *game, int *seed);
int randint(int max);
int randrange(int min, int max);
int chance(double percent);
void set_tile(struct game_obj *game, int x, int y, unsigned char val);

#endif
