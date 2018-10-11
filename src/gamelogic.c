#include <gamelogic.h>
#include <levelgen.h>
#include <stdlib.h>
#include <stdio.h>

struct player_obj player;
struct game_obj game;

void game_setup() {
	levelgen_gen_map(&game, &game.seed);
	player.position_x = SPAWN_X * TILE_WIDTH;
	player.position_y = SPAWN_Y * TILE_HEIGHT;
}

void game_update(int input[BUTTON_COUNT]) {
	//Player input
	if (input[BUTTON_RIGHT])
		player.velocity_x = V_X;
	if (input[BUTTON_LEFT])
		player.velocity_x = -V_X;
	if (input[BUTTON_JUMP] && player.canjump) {
		player.velocity_y = -V_JUMP;
		player.canjump = 0;
	}

	//Player physics
	int tile_x = (player.position_x + 16) / TILE_WIDTH;
	int tile_y = (player.position_y + 16) / TILE_HEIGHT;
	int feet_y = (player.position_y + 33) / TILE_HEIGHT;
	int right_x = (player.position_x + 33) / TILE_WIDTH;
	int left_x = (player.position_x - 1) / TILE_WIDTH;

	player.tile_x = tile_x;
	player.tile_y = tile_y;

	//Gravity
	player.velocity_y += GRAVITY;

	//Horizontal inertia
	player.velocity_x /= INTERTA;

	//Collision on bottom
	if (tile_at(tile_x, feet_y) > 0) {
		if (player.velocity_y > 0)
			player.velocity_y = 0;
		player.position_y = (feet_y - 1) * TILE_HEIGHT;
		player.canjump = 1;
	}

	//Right collision
	if (tile_at(right_x, tile_y) || right_x >= LEVEL_WIDTH) {
		if (player.velocity_x > 0)
			player.velocity_x = 0;
		player.position_x = (right_x - 1) * TILE_WIDTH;
	}

	//Left collision
	if (tile_at(left_x, tile_y) || left_x < 0) {
		if (player.velocity_x < 0)
			player.velocity_x = 0;
		player.position_x = (left_x + 1) * TILE_WIDTH;
	}

	//Apply player velocity
	player.position_x += player.velocity_x;
	player.position_y += player.velocity_y;

	//Lower bound
	if (player.position_y > LEVEL_PIXEL_HEIGHT) {
		player.position_y = 0;
		//TODO: Death
	}
}

int tile_at(int x, int y) {
	if (x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT)
		return 0;
	return game.tiles[y * LEVEL_WIDTH + x];
}
