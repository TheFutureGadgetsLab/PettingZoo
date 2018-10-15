#include <stdlib.h>
#include <math.h>
#include <defs.hpp>
#include <time.h>
#include <levelgen.hpp>
#include <gamelogic.hpp>
#include <stdio.h>

int ground_heights[LEVEL_WIDTH];

void levelgen_gen_map(struct Game *game) {
	int x, y, i, w, h, ground, val;
    
	game->seed = (unsigned)time(NULL);
	srand(game->seed);
	ground = GROUND_HEIGHT;
	for (x = 0; x < LEVEL_WIDTH; x++) {
		// 15% chance to change ground height in range -3 to 3
		// must be sufficiently far from spawn
		if (chance(15) && x > SPAWN_X + 1) {
			ground -= randrange(-3, 3);
			// Ensure the ground doesnt go below level or above half
			if (ground >= LEVEL_HEIGHT)
				ground = GROUND_HEIGHT;
			if (ground < LEVEL_HEIGHT / 2)
				ground = LEVEL_HEIGHT / 2;
		}

		ground_heights[x] = ground;

		// 5% chance for brick platform
		if (chance(5)) {
			h = randrange(8, ground - 2);
			w = randrange(2, 8);
			for (i = 0; i < w; i++) {
				set_tile(game, x + i, h, BRICKS);
			}
		}
		
		// Insert ground and dirt
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

	// Insert holes
	// Loop over tile space
	for (x = SPAWN_X + 10; x < LEVEL_WIDTH - 5; x++) {
		if (chance(HOLE_CHANCE)) {
			int hole_width = randrange(2, 5);
			create_hole(game, x, hole_width);
			x += hole_width + 1;
		}
	}
}

// Return an integer where 0 <= x < max
int randint(int max) {
	return random() % (max + 1);
}

// Return an integer where min <= x =< max
int randrange(int min, int max) {
	if (min == max)
		return max;
	return min + (random() % (abs(max - min) + 1));
}

// Return 0 or 1 probabilistically
int chance(double percent) {
	return ((double)random() / (double)RAND_MAX) <= (percent / 100.0);
}

// Set tile to given type at (x, y)
void set_tile(struct Game *game, int x, int y, unsigned char val) {
	if (x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT) {
		return;
	}
	
	game->tiles[y * LEVEL_WIDTH + x] = val;
}

// Insert hole at origin with width width
void create_hole(struct Game *game, int origin, int width) {
	int x, y;
	for (x = origin; x < origin + width; x++) {
		for (y = ground_heights[origin]; y < LEVEL_HEIGHT; y++) {
			set_tile(game, x, y, EMPTY);
		}
	}
}

void levelgen_clear_level(struct Game *game) {
	int x, y;
	for (x = 0; x < LEVEL_WIDTH; x++) {
		for (y = 0; y < LEVEL_HEIGHT; y++) {
			set_tile(game, x, y, EMPTY);
		}
	}
}