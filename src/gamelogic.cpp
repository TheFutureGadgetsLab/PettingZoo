#include <gamelogic.hpp>
#include <levelgen.hpp>
#include <stdlib.h>
#include <math.h>

struct Player player;
struct Game game;

void game_setup() {
	levelgen_gen_map(&game);
	player.position_x = SPAWN_X * TILE_SIZE;
	player.position_y = SPAWN_Y * TILE_SIZE;
	player.canjump = 1;
	player.score = 0;
	player.time = 0;
}

void game_reset_map() {
	levelgen_clear_level(&game);
	game_setup();
}

void game_update(int input[BUTTON_COUNT]) {
	// Estimate of time
	player.time += 1 / UPDATES_PS;
	// Branchless player input
	float tmp_yvel = player.velocity_y;
	player.velocity_x += (V_X - player.velocity_x) * input[BUTTON_RIGHT];
	player.velocity_x += (-V_X - player.velocity_x) * input[BUTTON_LEFT];
	player.velocity_y += (-V_JUMP - player.velocity_y) * input[BUTTON_JUMP] * player.canjump;
	player.canjump = player.canjump - !(tmp_yvel == player.velocity_y);

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
	if (tile_at(right_x, tile_y) || right_x >= LEVEL_WIDTH) {
		player.velocity_x = 0;
		player.position_x = (right_x - 1) * TILE_SIZE + PLAYER_MARGIN - 2;
	}

	//Left collision
	if (tile_at(left_x, tile_y) || left_x < 0) {
		player.velocity_x = 0;
		player.position_x = (left_x + 1) * TILE_SIZE - PLAYER_MARGIN + 2;
	}

	int tile_xr = floor((player.position_x + PLAYER_RIGHT) / TILE_SIZE);
	int tile_xl = floor((player.position_x + PLAYER_LEFT) / TILE_SIZE);

	//Collision on bottom
	if (tile_at(tile_xl, feet_y) > 0 || tile_at(tile_xr, feet_y) > 0) {
		if (player.velocity_y >= 0) {
			player.velocity_y = 0;
			player.canjump = 1;
		}
		player.position_y = (feet_y - 1) * TILE_SIZE;
	}

	//Collision on top
	if (tile_at(tile_xl, top_y) > 0 || tile_at(tile_xr, top_y) > 0) {
		if (player.velocity_y < 0)
			player.velocity_y = 0;
		player.position_y = (top_y + 1) * TILE_SIZE;
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
