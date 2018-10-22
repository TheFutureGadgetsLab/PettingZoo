#include <gamelogic.hpp>
#include <levelgen.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct Player player;
struct Game game;

int tile_at(int x, int y);
int tile_solid(int x, int y);
void game_set_tile(struct Game *game, int x, int y, unsigned char val);

void game_setup() {
	levelgen_clear_level(&game);
	levelgen_gen_map(&game);
	player.position_x = SPAWN_X * TILE_SIZE;
	player.position_y = SPAWN_Y * TILE_SIZE;
	player.velocity_x = 0;
	player.velocity_y = 0;
	player.canjump = false;
	player.standing = false;
	player.score = 0;
	player.fitness = 0;
	player.time = 0;
	player.buttonpresses = 0;
}

int game_update(int input[BUTTON_COUNT]) {
	// Estimate of time
	player.time += 1 / UPDATES_PS;

	player.velocity_x += (V_X - player.velocity_x) * input[BUTTON_RIGHT];
	player.velocity_x += (-V_X - player.velocity_x) * input[BUTTON_LEFT];
	if (input[BUTTON_JUMP] && player.canjump) {
		player.isjump = true;
		player.canjump = false;
		if (!player.standing)
			player.velocity_y = -V_JUMP;
	}
	if (!input[BUTTON_JUMP] && player.isjump)
		player.isjump = false;
	if (player.isjump) {
		player.velocity_y -= 1.5;
		if (player.velocity_y <= -V_JUMP) {
			player.isjump = false;
			player.velocity_y = -V_JUMP;
		}
	}

	//Button presses
	player.buttonpresses += input[BUTTON_JUMP] + input[BUTTON_LEFT] + input[BUTTON_RIGHT];

	//Player physics
	int tile_x = floor((player.position_x + player.velocity_x + 16) / TILE_SIZE);
	int tile_y = floor((player.position_y + player.velocity_y + 16) / TILE_SIZE);
	int feet_y = floor((player.position_y + player.velocity_y + 33) / TILE_SIZE);
	int top_y = floor((player.position_y + player.velocity_y - 1) / TILE_SIZE);
	int right_x = floor((player.position_x + player.velocity_x + PLAYER_RIGHT + 1) / TILE_SIZE);
	int left_x = floor((player.position_x + player.velocity_x + PLAYER_LEFT - 1) / TILE_SIZE);

	player.tile_x = tile_x;
	player.tile_y = tile_y;

	player.velocity_y += GRAVITY;
	player.velocity_x /= INTERTA;

	//Right collision
	if (tile_solid(right_x, tile_y) || right_x >= LEVEL_WIDTH) {
		player.velocity_x = 0;
		player.position_x = (right_x - 1) * TILE_SIZE + PLAYER_MARGIN - 2;
	}

	//Left collision
	if (tile_solid(left_x, tile_y) || left_x < 0) {
		player.velocity_x = 0;
		player.position_x = (left_x + 1) * TILE_SIZE - PLAYER_MARGIN + 2;
	}

	int tile_xr = floor((player.position_x + PLAYER_RIGHT) / TILE_SIZE);
	int tile_xl = floor((player.position_x + PLAYER_LEFT) / TILE_SIZE);

	//Collision on bottom
	player.standing = false;
	if (tile_solid(tile_xl, feet_y) > 0 || tile_solid(tile_xr, feet_y) > 0) {
		if (player.velocity_y >= 0) {
			player.velocity_y = 0;
			player.canjump = true;
			player.standing = true;
			if (tile_at(tile_xl, feet_y) == SPIKES_TOP || tile_at(tile_xr, feet_y) == SPIKES_TOP) {
				printf("PLAYER DEAD\n\tSCORE: %d\n\tFITNESS: %d\n", player.score, player.fitness);
				return -1;
			}
		}
		player.position_y = (feet_y - 1) * TILE_SIZE;
	}

	//Collision on top
	if (tile_solid(tile_xl, top_y) > 0 || tile_solid(tile_xr, top_y) > 0) {
		if (player.velocity_y < 0) {
			player.velocity_y = 0;
			if (tile_at(tile_xl, top_y) == SPIKES_BOTTOM || tile_at(tile_xr, top_y) == SPIKES_BOTTOM) {
			printf("PLAYER DEAD\n SCORE: %d\n FITNESS: %d\n", player.score, player.fitness);
				return -1;
			}
		}
		player.position_y = (top_y + 1) * TILE_SIZE;
	}

	//Collisions with coin
	if (tile_at(tile_x, tile_y) == COIN) {
		game_set_tile(&game, tile_x, tile_y, EMPTY);
		player.score += COIN_VALUE;
		return REDRAW;
	}

	//Apply player velocity
	player.position_x += player.velocity_x;
	player.position_y += player.velocity_y;

	//Lower bound
	if (player.position_y > LEVEL_PIXEL_HEIGHT) {
		printf("PLAYER DEAD\n SCORE: %d\n FITNESS: %d\n", player.score, player.fitness);
		return PLAYER_DEAD;
	}

	//Fitness / score
	float fitness;
	fitness = 100 + player.score + player.position_x;
	fitness -= player.time * FIT_TIME_WEIGHT;
	fitness -= player.buttonpresses * FIT_BUTTONS_WEIGHT;
	player.fitness = fitness > player.fitness ? fitness : player.fitness;

	//End of level
	if (player.position_x + PLAYER_RIGHT >= (LEVEL_WIDTH - 4) * TILE_SIZE) {
		return player.fitness;
	}

	//Time limit
	if (player.time >= 0.25 * LEVEL_WIDTH) {
		return PLAYER_DEAD;
	}

	return 0;
}

int tile_at(int x, int y) {
	if (x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT)
		return 0;
	return game.tiles[y * LEVEL_WIDTH + x];
}

int tile_solid(int x, int y) {
	int tile = tile_at(x, y);
	switch(tile) {
		case EMPTY:
		case COIN:
		case FLAG:
			return false;
		default:
			break;
	}
	return true;
}

// Set tile to given type at (x, y)
void game_set_tile(struct Game *game, int x, int y, unsigned char val) {
	if (x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT) {
		return;
	}
	game->tiles[y * LEVEL_WIDTH + x] = val;
}