/**
 * @file levelgen.cpp
 * @author Haydn Jones, Benjamin Mastripolito
 * @brief Functions for generating levels
 * @date 2018-12-11
 */
#include <stdlib.h>
#include <math.h>
#include <defs.hpp>
#include <levelgen.hpp>
#include <gamelogic.hpp>
#include <randfuncts.hpp>

// void game.setTileAt(game, int x, int y, unsigned char val);
void create_hole(Game& game, int origin, int width);
void create_pipe(Game& game, int origin, int width, int height);
void create_stair_gap(Game& game, int origin, int height, int width, int do_pipe);
void generate_flat_region(Game& game, int origin, int length);
void insert_floor(Game& game, int origin, int ground, int length);
void insert_platform(Game& game, int origin, int height, int length, int type);
void insert_tee(Game& game, int origin, int height, int length);
int generate_obstacle(Game& game, int origin);

/**
 * @brief Generate a new map from given seed
 * 
 * @param game The game object to write the new tiles to
 * @param seed The game seed to generate the level with
 */
void levelgen_gen_map(Game& game, unsigned int seed)
{
	int x, flat_region;

	game.seed = seed;
	game.seed_state = seed;

	// Insert ground
	insert_floor(game, 0, GROUND_HEIGHT, LEVEL_WIDTH);

	flat_region = 1;
	for (x = START_PLATLEN; x < LEVEL_WIDTH - 20; x++) {
		if (flat_region) {
			int length = randrange(&game.seed_state, 20, 50);

			if (x + length >= LEVEL_WIDTH - START_PLATLEN)
				length = (LEVEL_WIDTH - START_PLATLEN) - x;

			generate_flat_region(game, x, length);

			x += length;
			flat_region = 0;
		} else {
			int length = generate_obstacle(game, x);
			x += length;
			flat_region = 1;
		}
	}

	// Ending flag
	game.setTileAt(LEVEL_WIDTH - 4, GROUND_HEIGHT - 1, FLAG);
	game.seed_state = seed;
}

/**
 * @brief Generate a flat region 'length' tiles long and beginning at 'origin'
 * 
 * @param game Game to generate level in
 * @param origin Starting location (x tile coord)
 * @param length Length of flat region
 */
void generate_flat_region(Game& game, int origin, int length)
{
	int x, plat_len, height, stack_offset;
	int base_plat = 0;
	int allow_hole = 0;
	int type;

	plat_len = 0;
	stack_offset = 0;
	height = 0;
	type = 0;
	for (x = origin; x < origin + length; x++) {
		// Generate platform with 15% chance
		if (chance(&game.seed_state, 15)) {
			plat_len = randrange(&game.seed_state, 3, 8);
			// Ensure platform doesnt extend past region
			if (x + plat_len >= origin + length)
				plat_len = origin + length - x - 1;

			// Choose plat type with equal probability then set
			// height coords based on type
			type = BRICKS + randrange(&game.seed_state, 0, 2); // if rand returns 0 then bricks, 1 top spikes, 2 bottom spikes
			if (type == BRICKS || type == SPIKES_TOP)
				height = randrange(&game.seed_state, base_plat + 2, base_plat + 3);
			else // Bottom spikes can be higher
				height = randrange(&game.seed_state, base_plat + 2, base_plat + 4);

			// Only insert plat if length > 0
			if (plat_len > 0) {
				if (base_plat == 0 && type == BRICKS && chance(&game.seed_state, 75)) {
					int t_platlen = plat_len;
					if (t_platlen % 2 == 0)
						t_platlen++;

					if (t_platlen < 0) {
						insert_platform(game, x, height, plat_len, type);
						allow_hole = 1;
					} else {
						int tee_height = height - base_plat - 1;
						if (tee_height > 3)
							tee_height = 3;
						insert_tee(game, x, tee_height, t_platlen);
						allow_hole = 1;
						plat_len = t_platlen;
						height = tee_height;
					}
				} else {
					insert_platform(game, x, height, plat_len, type);
					allow_hole = 1;
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
			int hole_len = randrange(&game.seed_state, 2, 5);
			int hole_origin = x + randrange(&game.seed_state, 0, plat_len - hole_len + 3);

			if (hole_origin + hole_len >= origin + length)
				hole_len = origin + length - hole_origin - 1;

			create_hole(game, hole_origin, hole_len);
			// Prevent multiple holes under a plat
			allow_hole = 0;
		}
	}
}

/**
 * @brief Generates an obstacle and returns the length of the obstacle
 * 
 * @param game Game to generate obstacle in
 * @param origin Beginning location of obstacle
 * @return int Obstacle width
 */
int generate_obstacle(Game& game, int origin)
{
	int width, height, do_pipe;

	width = randrange(&game.seed_state, 3, 7);
	height = randrange(&game.seed_state, 3, 5);
	do_pipe = chance(&game.seed_state, 50);

	create_stair_gap(game, origin, height, width, do_pipe);

	return height * 2 + width;
}

/**
 * @brief Insert a floor to the bottom of the level beginning at 'origin' with
 *        at y level 'ground' and of length 'length'
 * 
 * @param game The game object to operate on
 * @param origin The x tile coord to start the floor at
 * @param ground The y coord to start the floor at
 * @param length length of floor
 */
void insert_floor(Game& game, int origin, int ground, int length)
{
	int x, y, val;
	for (x = origin; x < origin + length; x++) {
		for (y = 0; y < LEVEL_HEIGHT; y++) {
			val = EMPTY;
			if (y == GROUND_HEIGHT)
				val = GRASS;
			else if (y > ground)
				val = DIRT;
			if (val)
				game.setTileAt(x, y, val);
		}
	}
}

/**
 * @brief Insert a platform at 'origin', 'height' tiles above the ground, 'length' tiles long and of type 'type'
 * 
 * @param game Game obj to insert platform in
 * @param origin X tile location to insert platform
 * @param height Height above ground platform should be inserted
 * @param length Length of platform
 * @param type Tile type of platform
 */
void insert_platform(Game& game, int origin, int height, int length, int type)
{
	int x, base;

	base = GROUND_HEIGHT - height - 1;

	for (x = origin; x < origin + length; x++) {
		game.setTileAt(x, base, type);
	}
}

/**
 * @brief Insert a platform at 'origin', 'height' tiles above the ground, 'length' tiles long and of type 'type'
 * 
 * @param game Game obj to insert platform in
 * @param origin X tile location to insert tee
 * @param height Height above ground platform should be inserted
 * @param length Length of platform
 */
void insert_tee(Game& game, int origin, int height, int length)
{
	int y, top;
	top = GROUND_HEIGHT - height;

	insert_platform(game, origin, height, length, BRICKS);

	for (y = top; y < GROUND_HEIGHT; y++) {
		game.setTileAt(origin + (length / 2), y, BRICKS);
	}
}

/**
 * @brief Create hole at origin
 * 
 * @param game Game to insert hole on
 * @param origin X tile coordinate
 * @param width Width of hole
 */
void create_hole(Game& game, int origin, int width)
{
	int x, y;
	for (x = origin; x < origin + width; x++) {
		for (y = GROUND_HEIGHT; y < LEVEL_HEIGHT; y++) {
			game.setTileAt(x, y, EMPTY);
		}
	}
}

/**
 * @brief Create a pipe at 'origin', opens at 'height' tiles from the ground with a gap of 'width' tiles
 * 
 * @param game Game to insert pipe in
 * @param origin X tile location of pipe
 * @param width Width of gap in pipe
 * @param height height of hole opening above the ground
 */
void create_pipe(Game& game, int origin, int width, int height)
{
	int y;

	for (y = 0; y < LEVEL_HEIGHT; y++) {
		// Middle sections
		if (y < GROUND_HEIGHT - height - width - 1 || y > GROUND_HEIGHT - height) {
			game.setTileAt(origin, y, PIPE_MIDDLE);
		// Bottom of pipe
		} else if (y == GROUND_HEIGHT - height - width - 1) {
			game.setTileAt(origin, y, PIPE_TOP);
		// Gap
		} else if (y > GROUND_HEIGHT - height - width - 1 && width > 0) {
			width--;
			game.setTileAt(origin, y, EMPTY);
		// Top of pipe
		} else if (width == 0) {
			width--;
			game.setTileAt(origin, y, PIPE_BOTTOM);
		}
	}
}

//

/**
 * @brief Create a stair gap. 'height' describes the # of steps in the stairs and 
 *        'width' describes the width of the gap
 *        If 'do_pipe' is true, 'width' will be set to next greatest odd if even
 * 
 * @param game Game to insert stair gap in
 * @param origin X tile cooordinate of gap
 * @param height Height of stairs
 * @param width Width of gap
 * @param do_pipe Insert pipe on true 
 */
void create_stair_gap(Game& game, int origin, int height, int width, int do_pipe)
{
	int x, y;

	if (do_pipe && width % 2 == 0) {
		width++;
	}

	// Insert first stair
	for (x = 0; x < height; x++) {
		for (y = 0; y <= x + 1; y++) {
			if (y == x + 1)
				game.setTileAt(x + origin, GROUND_HEIGHT - y, GRASS);
			else
				game.setTileAt(x + origin, GROUND_HEIGHT - y, DIRT);
		}
	}
	origin += height; // Shift origin over for next section

	// Insert hole
	create_hole(game, origin, width);

	if (do_pipe) {
		int pipe_width = randrange(&game.seed_state, 2, 4);
		int pipe_height = height + choose(&game.seed_state, (7), 0, 1, 1, 2, 2, 2, 3, 3, 3, 3) * choose(&game.seed_state, (2), 1, -1);

		create_pipe(game, origin + floor(width / 2), pipe_width, pipe_height);
	}
	origin += width; // Shift origin over for next section

	// Insert last stair
	for (x = 0; x < height; x++) {
		for (y = height - x; y >= 0; y--) {
			if (y == height - x)
				game.setTileAt(x + origin, GROUND_HEIGHT - y, GRASS);
			else
				game.setTileAt(x + origin, GROUND_HEIGHT - y, DIRT);
		}
	}
}

/**
 * @brief Set the tile at (x, y)
 * 
 * @param game Game to set tile in
 * @param x X tile coord
 * @param y Y tile coord
 * @param val Value to set tile to
 */
// void game.setTileAt(game, int x, int y, unsigned char val)
// {
// 	if (x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT) {
// 		return;
// 	}
// 	game.tiles[y * LEVEL_WIDTH + x] = val;
// }

/**
 * @brief Clear all tiles from level
 * 
 * @param game Game to clear
 */
void levelgen_clear_level(Game& game)
{
	int x, y;
	for (x = 0; x < LEVEL_WIDTH; x++) {
		for (y = 0; y < LEVEL_HEIGHT; y++) {
			game.setTileAt(x, y, EMPTY);
		}
	}
}
