#include <stdlib.h>
#include <math.h>
#include <defs.hpp>
#include <time.h>
#include <levelgen.hpp>
#include <gamelogic.hpp>

void levelgen_gen_map(struct Game *game) {
	int x, y, i, w, h, ground, val;
    
	game->seed = (unsigned)time(NULL);
	srand(game->seed);
	ground = GROUND_HEIGHT;

	for (x = 0; x < LEVEL_WIDTH; x++) {
		if (chance(15)) {
			ground -= randrange(-3, 3);
			if (ground >= LEVEL_HEIGHT)
				ground = GROUND_HEIGHT;
		}

		if (chance(5)) {
			h = randrange(8, ground - 2);
			w = randrange(2, 8);
			for (i = 0; i < w; i++) {
				set_tile(game, x + i, h, BRICKS);
			}
		}
		
		for (y = 0; y < LEVEL_HEIGHT; y++) {
			val = EMPTY;
			if (y == ground)
				val = GRASS;
			else if (y > ground)
				val = DIRT;
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

void set_tile(struct Game *game, int x, int y, unsigned char val) {
	if (x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT) {
		return;
	}
	
	game->tiles[y * LEVEL_WIDTH + x] = val;
}