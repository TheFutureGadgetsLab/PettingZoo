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
uint physics_sim(struct Body* body, bool jump);
float dist(float x1, float y1, float x2, float y2);


void game_setup() {
	levelgen_clear_level(&game);
	levelgen_gen_map(&game);
	player.body.px = SPAWN_X * TILE_SIZE;
	player.body.py = SPAWN_Y * TILE_SIZE;
	player.body.vx = 0;
	player.body.vy = 0;
	player.body.canjump = false;
	player.body.standing = false;
	player.score = 0;
	player.fitness = 0;
	player.time = 0;
	player.buttonpresses = 0;
	player.body.immune = false;
}

int game_update(int input[BUTTON_COUNT]) {
	int return_value = 0;

	// Estimate of time
	player.time += 1 / UPDATES_PS;

	//Left and right button press
	player.body.vx += (V_X - player.body.vx) * input[BUTTON_RIGHT];
	player.body.vx += (-V_X - player.body.vx) * input[BUTTON_LEFT];

	//Button presses
	player.buttonpresses += input[BUTTON_JUMP] + input[BUTTON_LEFT] + input[BUTTON_RIGHT];

	//Physics sim for player
	return_value = physics_sim(&player.body, input[BUTTON_JUMP]);

	//Collisions with coin
	if (tile_at(player.body.tile_x, player.body.tile_y) == COIN) {
		game_set_tile(&game, player.body.tile_x, player.body.tile_y, EMPTY);
		player.score += COIN_VALUE;
		return_value = REDRAW;
	}

	//Lower bound
	if (player.body.py > LEVEL_PIXEL_HEIGHT) {
		printf("PLAYER DEAD\n SCORE: %d\n FITNESS: %d\n", player.score, player.fitness);
		return PLAYER_DEAD;
	}

	//Enemies
	int ret;
	uint i;
	for (i = 0; i < game.n_enemies; i++) {
		if (!game.enemies[i].dead) {
			game.enemies[i].body.vx = game.enemies[i].direction;
			ret = physics_sim(&game.enemies[i].body, chance(50));
			if (ret == PLAYER_DEAD) {
				game.enemies[i].dead = true;
			}
			if (ret == COL_RIGHT && game.enemies[i].direction > 0) {
				game.enemies[i].direction = -game.enemies[i].direction;
			} else if (ret == COL_LEFT && game.enemies[i].direction < 0) {
				game.enemies[i].direction = -game.enemies[i].direction;
			}
			if (dist(player.body.px, player.body.py, game.enemies[i].body.px, game.enemies[i].body.py) < 32) {
				return PLAYER_DEAD;
			}
		}
	}
	
	//Fitness / score
	float fitness;
	fitness = 100 + player.score + player.body.px;
	fitness -= player.time * FIT_TIME_WEIGHT;
	fitness -= player.buttonpresses * FIT_BUTTONS_WEIGHT;
	player.fitness = fitness > player.fitness ? fitness : player.fitness;

	//End of level
	if (player.body.px + PLAYER_RIGHT >= (LEVEL_WIDTH - 4) * TILE_SIZE) {
		return PLAYER_DEAD;
	}

	//Time limit
	if (player.time >= 0.25 * LEVEL_WIDTH) {
		return PLAYER_DEAD;
	}

	return return_value;
}

uint physics_sim(struct Body* body, bool jump) {
	uint return_value = 0;

	//Jumping
	if (jump && body->canjump) {
		body->isjump = true;
		body->canjump = false;
		if (!body->standing)
			body->vy = -V_JUMP;
	}
	if (!jump && body->isjump)
		body->isjump = false;
	if (body->isjump) {
		body->vy -= 1.5;
		if (body->vy <= -V_JUMP) {
			body->isjump = false;
			body->vy = -V_JUMP;
		}
	}

	//Player physics
	int tile_x = (body->px + body->vx + 16) / TILE_SIZE;
	int tile_y = (body->py + body->vy + 16) / TILE_SIZE;
	int feet_y = (body->py + body->vy + 33) / TILE_SIZE;
	int top_y = (body->py + body->vy - 1) / TILE_SIZE;
	int right_x = (body->px + body->vx + PLAYER_RIGHT + 1) / TILE_SIZE;
	int left_x = (body->px + body->vx + PLAYER_LEFT - 1) / TILE_SIZE;

	body->tile_x = tile_x;
	body->tile_y = tile_y;

	body->vy += GRAVITY;
	body->vx /= INTERTA;

	//Right collision
	if (tile_solid(right_x, tile_y) || right_x >= LEVEL_WIDTH) {
		body->vx = 0;
		body->px = (right_x - 1) * TILE_SIZE + PLAYER_MARGIN - 2;
		return_value = COL_RIGHT;
	}

	//Left collision
	if (tile_solid(left_x, tile_y) || left_x < 0) {
		body->vx = 0;
		body->px = (left_x + 1) * TILE_SIZE - PLAYER_MARGIN + 2;
		return_value = COL_LEFT;
	}

	int tile_xr = (body->px + PLAYER_RIGHT) / TILE_SIZE;
	int tile_xl = (body->px + PLAYER_LEFT) / TILE_SIZE;

	//Collision on bottom
	body->standing = false;
	if (tile_solid(tile_xl, feet_y) > 0 || tile_solid(tile_xr, feet_y) > 0) {
		if (body->vy >= 0) {
			body->vy = 0;
			body->canjump = true;
			body->standing = true;
			if (!body->immune && (tile_at(tile_xl, feet_y) == SPIKES_TOP || tile_at(tile_xr, feet_y) == SPIKES_TOP)) {
				return PLAYER_DEAD;
			}
		}
		body->py = (feet_y - 1) * TILE_SIZE;
	}

	//Collision on top
	if (tile_solid(tile_xl, top_y) > 0 || tile_solid(tile_xr, top_y) > 0) {
		if (body->vy < 0) {
			body->vy = 0;
			body->isjump = false;
			if (!body->immune && (tile_at(tile_xl, top_y) == SPIKES_BOTTOM || tile_at(tile_xr, top_y) == SPIKES_BOTTOM)) {
				return PLAYER_DEAD;
			}
		}
		body->py = (top_y + 1) * TILE_SIZE;
	}

	//Apply body->velocity
	body->px = round(body->px + body->vx);
	body->py = round(body->py + body->vy);
	
	return return_value;
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

float dist(float x1, float y1, float x2, float y2) {
	return sqrt(pow(x2 - x1, 2.0) + pow(y2 - y1, 2.0));
}
