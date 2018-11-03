#include <stdlib.h>
#include <math.h>
#include <defs.hpp>
#include <time.h>
#include <levelgen.hpp>
#include <gamelogic.hpp>
#include <stdio.h>
#include <cstdarg>
#include <list>

int randint(int max);
int randrange(int min, int max);
int chance(double percent);
int choose(int nargs...);
void set_tile(struct Game *game, int x, int y, unsigned char val);
void create_hole(struct Game *game, int origin, int width);
void create_pipe(struct Game *game, int origin, int width, int height);
void create_stair_gap(struct Game *game, int origin, int height, int width, int do_pipe);
void generate_flat_region(struct Game *game, int origin, int length);
void insert_floor(struct Game *game, int origin, int ground, int length);
void insert_platform(struct Game *game, int origin, int height, int length, int type, bool append);
void insert_tee(struct Game *game, int origin, int height, int length);
void insert_enemy(struct Game *game, int x, int y, int type);
int generate_obstacle(struct Game *game, int origin);

struct platform {
	int origin;
	int height;
	int length;
	int type;
};

std::list<struct platform> plats;

void levelgen_gen_map(struct Game *game) {
	printf("------------------------------------------------\n");
	plats.clear();
	int x;
	bool flat_region;

	game->seed = (unsigned)time(NULL);
	srand(game->seed);
	printf("Seed: %d\n", game->seed);

	// Insert ground
	insert_floor(game, 0, GROUND_HEIGHT, LEVEL_WIDTH);

	flat_region = true;
	for (x = START_PLATLEN; x < LEVEL_WIDTH - 20; x++) {
		if (flat_region) {
			int length = randrange(20, 50);

			if (x + length >= LEVEL_WIDTH - START_PLATLEN)
				length = (LEVEL_WIDTH - START_PLATLEN) - x;

			generate_flat_region(game, x, length);

			if (chance(25)) {
				insert_enemy(game, x + (length / 2), GROUND_HEIGHT - 4, ENEMY);
			}

			x += length;
			flat_region = false;
		} else {
			int length = generate_obstacle(game, x);
			x += length;
			flat_region = true;
		}
	}

	// Iterate over plats to place coins
	for (auto const& i : plats) {
    	if (chance(50)) {
			set_tile(game, i.origin + i.length / 2, i.height - 1, COIN);
		}
	}

	// Ending flag
	set_tile(game, LEVEL_WIDTH - 4, GROUND_HEIGHT - 1, FLAG);
}

// Generate a flat region beginning at origin for length tiles
void generate_flat_region(struct Game *game, int origin, int length) {
	int x, plat_len, height, stack_offset;
	int base_plat = 0;
	bool allow_hole = false;
	int type;

	plat_len = 0;
	stack_offset = 0;
	height = 0;
	type = 0;
	for (x = origin; x < origin + length; x++) {
		// Generate platform with 15% chance
		if (chance(15)) {
			plat_len = randrange(3, 8);
			// Ensure platform doesnt extend past region
			if (x + plat_len >= origin + length)
				plat_len = origin + length - x - 1;

			// Choose plat type with equal probability then set
			// height coords based on type
			type = BRICKS + randrange(0, 2); // if rand returns 0 then bricks, 1 top spikes, 2 bottom spikes
			if (type == BRICKS || type == SPIKES_TOP)
				height = randrange(base_plat + 2, base_plat + 3);
			else // Bottom spikes can be higher
				height = randrange(base_plat + 2, base_plat + 4);

			// Only insert plat if length > 0
			if (plat_len > 0) {
				if (base_plat == 0 && type == BRICKS && chance(75)) {
					int t_platlen = plat_len;
					if (t_platlen % 2 == 0)
						t_platlen++;

					if (t_platlen < 0) {
						insert_platform(game, x, height, plat_len, type, true);
						allow_hole = true;
					} else {
						int tee_height = height - base_plat - 1;
						if (tee_height > 3)
							tee_height = 3;
						insert_tee(game, x, tee_height, t_platlen);
						allow_hole = true;
						plat_len = t_platlen;
						height = tee_height;
					}
				} else {
					insert_platform(game, x, height, plat_len, type, true);
					allow_hole = true;
				}

				// If the plat is not a top spike and height allows
				// Then allow stacking
				if (type != SPIKES_TOP && height - base_plat < 3) {
					base_plat = height;
					stack_offset = plat_len;
				} else {
					base_plat = 0;
				}
			}

			// Allows non-stacking plats to overlap
			if (base_plat != 0) {
				x += 2;
			} else {
				x += plat_len + 2;
			}
		// Everytime a plat is not generated from an x cord
		// the length of plat_len will keep track of
		// how far you are from the edge of the previous plat
		// Once far enough set base_plat to 0 to prevent a new
		// plat spawning too high
		} else {
			if (stack_offset > -2)
				stack_offset--;
			else
				base_plat = 0;
		}

		// If height of prev. plat allows, or base_plat is not 0, and the plat is long enough
		// Insert a hole
		if ((height < 4 || base_plat != 0) && plat_len > 3 && allow_hole && (type != SPIKES_BOTTOM || base_plat != 0)) {
			int hole_len = randrange(2, 5);
			int hole_origin = x + randrange(0, plat_len - hole_len + 3);

			if (hole_origin + hole_len >= origin + length)
				hole_len = origin + length - hole_origin - 1;

			create_hole(game, hole_origin, hole_len);
			// Prevent multiple holes under a plat
			allow_hole = false;
		}
	}
}

// Generates an obstacle and returns the length of the obstacle
int generate_obstacle(struct Game *game, int origin) {
	int width, height, do_pipe;

	width = randrange(3, 7);
	height = randrange(3, 5);
	do_pipe = chance(50);

	create_stair_gap(game, origin, height, width, do_pipe);

	return height * 2 + width;
}

// Insert a floor to the bottom of the level beginning at 'origin' with
// at y level 'ground' and of length 'length'
void insert_floor(struct Game *game, int origin, int ground, int length) {
	int x, y, val;
	for (x = origin; x < origin + length; x++) {
		for (y = 0; y < LEVEL_HEIGHT; y++) {
			val = EMPTY;
			if (y == GROUND_HEIGHT)
				val = GRASS;
			else if (y > ground)
				val = DIRT;
			if (val)
				set_tile(game, x, y, val);
		}
	}
}

// Insert a platform at 'origin', 'height' tiles above the ground, 'length' tiles long and of type 'type'
void insert_platform(struct Game *game, int origin, int height, int length, int type, bool append) {
	int x, base;

	base = GROUND_HEIGHT - height - 1;

	for (x = origin; x < origin + length; x++) {
		set_tile(game, x, base, type);
	}

	if (append && type != SPIKES_TOP) {
		struct platform tmp;
		tmp.height = base;
		tmp.length = length;
		tmp.origin = origin;
		tmp.type = type;
		plats.push_back(tmp);
	}
}

// Insert Tee
void insert_tee(struct Game *game, int origin, int height, int length) {
	printf("Tee:\tat %d\theight %d\tlength %d\n", origin, length, height);
	int y, top;
	top = GROUND_HEIGHT - height;

	insert_platform(game, origin, height, length, BRICKS, false);

	for (y = top; y < GROUND_HEIGHT; y++) {
		set_tile(game, origin + (length / 2), y, BRICKS);
	}
}

// Insert enemy at tile position
void insert_enemy(struct Game *game, int x, int y, int type) {
	struct Enemy enemy;
	enemy.body.px = x * TILE_SIZE;
	enemy.body.py = y * TILE_SIZE;
	enemy.type = type;
	enemy.dead = false;
	enemy.body.immune = true;
	enemy.direction = 3;
	game->enemies[game->n_enemies] = enemy;
	game->n_enemies++;
}

// Insert hole in the ground at 'origin' with width 'width'
void create_hole(struct Game *game, int origin, int width) {
	int x, y;
	for (x = origin; x < origin + width; x++) {
		for (y = GROUND_HEIGHT; y < LEVEL_HEIGHT; y++) {
			set_tile(game, x, y, EMPTY);
		}
	}
}

// Create a pipe at 'origin', opens at 'height' tiles from the ground with a gap of 'width' tiles
void create_pipe(struct Game *game, int origin, int width, int height) {
	int y;
	printf("Pipe:\tat %d\theight %d\twidth %d\n", origin, height, width);

	for (y = 0; y < LEVEL_HEIGHT; y++) {
		// Middle sections
		if (y < GROUND_HEIGHT - height - width - 1 || y > GROUND_HEIGHT - height) {
			set_tile(game, origin, y, PIPE_MIDDLE);
		// Bottom of pipe
		} else if (y == GROUND_HEIGHT - height - width - 1) {
			set_tile(game, origin, y, PIPE_TOP);
		// Gap
		} else if (y > GROUND_HEIGHT - height - width - 1 && width > 0) {
			width--;
			set_tile(game, origin, y, EMPTY);
		// Top of pipe
		} else if (width == 0) {
			width--;
			set_tile(game, origin, y, PIPE_BOTTOM);
		}
	}
}

// Create a stair gap. 'height' describes the # of steps in the stairs and
// 'width' describes the width of the gap
// If 'do_pipe' is true, 'width' will be set to next greatest odd if even
void create_stair_gap(struct Game *game, int origin, int height, int width, int do_pipe) {
	int x, y;
	printf("Gap:\tat %d\theight %d\twidth %d\tdopipe %d\n", origin, height, width, do_pipe);
	if (do_pipe && width % 2 == 0) {
		width++;
	}

	// Insert first stair
	for (x = 0; x < height; x++) {
		for (y = 0; y <= x + 1; y++) {
			if (y == x + 1)
				set_tile(game, x + origin, GROUND_HEIGHT - y, GRASS);
			else
				set_tile(game, x + origin, GROUND_HEIGHT - y, DIRT);
		}
	}
	origin += height; // Shift origin over for next section

	// Insert hole
	create_hole(game, origin, width);

	if (do_pipe) {
		int pipe_width = randrange(2, 4);
		int pipe_height = height + choose((7), 0, 1, 1, 2, 2, 2, 3, 3, 3, 3) * choose((2), 1, -1);

		create_pipe(game, origin + floor(width / 2), pipe_width, pipe_height);
	}
	origin += width; // Shift origin over for next section

	// Insert last stair
	for (x = 0; x < height; x++) {
		for (y = height - x; y >= 0; y--) {
			if (y == height - x)
				set_tile(game, x + origin, GROUND_HEIGHT - y, GRASS);
			else
				set_tile(game, x + origin, GROUND_HEIGHT - y, DIRT);
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
	game->n_enemies = 0;
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

// Returns a random integer in list of integers
int choose(int nargs...) {
	va_list args;
	va_start(args, nargs);
	int array[nargs];
	int i;
	for (i = 0; i < nargs; i++) {
		array[i] = va_arg(args, int);
	}
	return array[randint(nargs - 1)];
}
