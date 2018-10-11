#include <game.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <defs.h>
#include <time.h>
#include <levelgen.h>

void levelgen_gen_map(struct game_obj *game, int *seed) {
	int x, y, i, w, h, ground, val;
    
	*seed = (unsigned)time(NULL);
	srand(*seed);
	ground = GROUND_HEIGHT;

	for (x = 0; x < LEVEL_WIDTH; x++) {
		if (chance(7)) {
			ground -= randrange(-3, 3);
		}
		if (chance(5)) {
			h = randrange(8, ground - 2);
			w = randrange(2, 8);
			for (i = 0; i < w; i++) {
				set_tile(game, x + i, h, T_BRICKS);
			}
		}
		for (y = 0; y < LEVEL_HEIGHT; y++) {
			val = T_EMPTY;
			if (y == ground)
				val = T_GRASS;
			else if (y > ground)
				val = T_DIRT;
			if (val)
				set_tile(game, x, y, val);
		}
	}
}

int randint(int max) {
	return random() % max;
}

int randrange(int min, int max) {
	return min + (random() % (max - min));
}

int chance(double percent) {
	return ((double)random() / (double)RAND_MAX) < (percent / 100.0);
}

void set_tile(struct game_obj *game, int x, int y, unsigned char val) {
	if (x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT) {
		return;
	}
	
	game->tiles[y * LEVEL_WIDTH + x] = val;
}