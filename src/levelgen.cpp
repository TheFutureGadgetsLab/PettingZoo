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
	int static_ground = 0;

	game->seed = (unsigned)time(NULL);
	srand(game->seed);
	ground = GROUND_HEIGHT;
	for (x = 0; x < LEVEL_WIDTH; x++) {
		// Decrement how long ground needs to remain the same
		if (static_ground)
			static_ground -= 1;

		/* 15% chance to change ground height in range -3 to 3.
		   Must be sufficiently far from spawn and ground level
		   change allowed
		*/
		if (chance(15) && x > SPAWN_X + 1 && !static_ground) {
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
			// Platform can be 2 to 4 tile above the ground
			h = randrange(2, 4);
			// Can be length 2 to 6
			w = randrange(2, 6);
			for (i = 0; i < w; i++) {
				set_tile(game, x + i, ground - h, BRICKS);
			}

			// Prevent ground from changing under platform
			static_ground = w;
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

	// Insert holes. Loop over tile space
	for (x = SPAWN_X + 10; x < LEVEL_WIDTH - 15; x++) {
		if (chance(HOLE_CHANCE)) {
			int hole_width = randrange(2, 8);
			create_hole(game, x, hole_width);
			// Prevent holes from overlapping
			x += hole_width + 10;
		}
	}

	// Insert stair gap. Loop over tile space
	for (x = SPAWN_X + 10; x < LEVEL_WIDTH - 15; x++) {
		if (chance(5)) {
			int width = randrange(3, 7);
			int height = randrange(3, 5);

			create_stair_gap(game, x, height, width);
			// Prevent stairgaps from overlapping
			x += 15;
		}
	}

	for (x = SPAWN_X + 10; x < LEVEL_WIDTH - 15; x++) {
		if (chance(10)) {
			int hole_width = randrange(2, 4);
			int hole_height = randrange(1, 4);
			create_pipe(game, x, hole_width, ground_heights[x] - hole_height);

			// Prevent holes from overlapping
			x += hole_width + 10;
		}
	}
}

// Insert hole at origin with width width
void create_hole(struct Game *game, int origin, int width) {
	int x, y;
	for (x = origin; x < origin + width; x++) {
		for (y = ground_heights[x]; y < LEVEL_HEIGHT; y++) {
			set_tile(game, x, y, EMPTY);
		}
	}
}

// Create a pipe at origin, height tiles from the ground with a hole of
// width tiles
void create_pipe(struct Game *game, int origin, int width, int height) {
	int y;
	height += 1; // Not sure why this needs to be
	for (y = 0; y < LEVEL_HEIGHT; y++) {
		if ((y > height - width) && width > 0) {
			width--;
			set_tile(game, origin, y, EMPTY);
		} else if (y == height - width) {
			set_tile(game, origin, y, PIPE_BOTTOM);
		} else if (width == 0) {
			width = -1;
			set_tile(game, origin, y, PIPE_TOP);
		} else {
			set_tile(game, origin, y, PIPE_MIDDLE);
		}
	}
}

// Create a stair gap. Height describes the height of the stairs and
// width describes the width of the gap
void create_stair_gap(struct Game *game, int origin, int height, int width) {
	int x, y;
	int ground = ground_heights[origin];

	// Clear area
	for (x = origin; x < origin + height * 2 + width; x++) {
		for (y = ground - height - 1; y < LEVEL_HEIGHT; y++) {
			set_tile(game, x, y, EMPTY);
		}
	}

	// Lay foundation
	for (x = origin; x < origin + height * 2 + width; x++) {
		for (y = ground; y < LEVEL_HEIGHT; y++) {
			set_tile(game, x, y, DIRT);
		}
	}

	// Insert first stair
	for (x = 0; x < height; x++) {
		for (y = 1; y <= x + 1; y++) {
			if (y == x + 1)
				set_tile(game, x + origin, ground - y, GRASS);
			else
				set_tile(game, x + origin, ground - y, DIRT);
		}
	}
	origin += height; // Shift origin over for next section

	// Insert hole
	for (x = origin; x < origin + width; x++) {
		for (y = ground - height; y < LEVEL_HEIGHT; y++) {
			set_tile(game, x, y, EMPTY);
		}
	}
	origin += width; // Shift origin over for next section

	// Insert last stair
	for (x = 0; x < height; x++) {
		for (y = height - x; y >= 0; y--) {
			if (y == height - x)
				set_tile(game, x + origin, ground - y, GRASS);
			else
				set_tile(game, x + origin, ground - y, DIRT);
		}
	}
}

// Set tile to given type at (x, y)
void set_tile(struct Game *game, int x, int y, unsigned char val) {
	if (x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT) {
		return;
	}
	game->tiles[y * LEVEL_WIDTH + x] = val;
}

// Zero out level
void levelgen_clear_level(struct Game *game) {
	int x, y;
	for (x = 0; x < LEVEL_WIDTH; x++) {
		for (y = 0; y < LEVEL_HEIGHT; y++) {
			set_tile(game, x, y, EMPTY);
		}
	}
}

// Return an integer where 0 <= x <= max
int randint(int max) {
	return random() % (max + 1);
}

// Return an integer where min <= x <= max
int randrange(int min, int max) {
	if (min == max)
		return max;
	return min + (random() % (abs(max - min) + 1));
}

// Return 0 or 1 probabilistically
int chance(double percent) {
	return ((double)random() / (double)RAND_MAX) <= (percent / 100.0);
}