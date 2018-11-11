#include <gamelogic.hpp>
#include <levelgen.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

int tile_at(struct Game *game, int x, int y);
int tile_solid(struct Game *game, int x, int y);
void game_set_tile(struct Game *game, int x, int y, unsigned char val);
uint physics_sim(struct Game *game, struct Body* body, bool jump);
float dist(float x1, float y1, float x2, float y2);

//Setup for a new game, full reset
void game_setup(struct Game *game, struct Player *player) {
	levelgen_clear_level(game);
	levelgen_gen_map(game, (unsigned)time(NULL));
	player->body.px = SPAWN_X * TILE_SIZE;
	player->body.py = SPAWN_Y * TILE_SIZE;
	player->body.vx = 0;
	player->body.vy = 0;
	player->body.canjump = false;
	player->body.standing = false;
	player->score = 0;
	player->fitness = 0;
	player->time = 0;
	player->buttonpresses = 0;
	player->body.immune = false;
}

//Called every frame
int game_update(struct Game *game, struct Player *player, int input[BUTTON_COUNT]) {
	int return_value = 0;

	// Estimate of time
	player->time += 1 / UPDATES_PS;

	//Left and right button press
	player->body.vx += (V_X - player->body.vx) * input[BUTTON_RIGHT];
	player->body.vx += (-V_X - player->body.vx) * input[BUTTON_LEFT];

	//Button presses
	player->buttonpresses += input[BUTTON_JUMP] + input[BUTTON_LEFT] + input[BUTTON_RIGHT];

	//Physics sim for player
	return_value = physics_sim(game, &player->body, input[BUTTON_JUMP]);

	//Collisions with coin
	if (tile_at(game, player->body.tile_x, player->body.tile_y) == COIN) {
		game_set_tile(game, player->body.tile_x, player->body.tile_y, EMPTY);
		player->score += COIN_VALUE;
		return_value = REDRAW;
	}

	//Lower bound
	if (player->body.py > LEVEL_PIXEL_HEIGHT) {
		printf("PLAYER DEAD\n SCORE: %d\n FITNESS: %d\n", player->score, player->fitness);
		return PLAYER_DEAD;
	}

	//Enemies
	int ret;
	uint i, y;
	bool empty_below;
	struct Enemy *enemy;
	for (i = 0; i < game->n_enemies; i++) {
		enemy = &game->enemies[i];
		empty_below = true;
		if (!enemy->dead) {
			//Enemy physics simulation
			enemy->body.vx = enemy->direction;
			ret = physics_sim(game, &(enemy->body), JUMPING_ENEMIES ? chance(75) : false);
			if (ret == PLAYER_DEAD) {
				enemy->dead = true;
			}

			//Check if empty below
			for (y = enemy->body.tile_y; y < LEVEL_HEIGHT; y++) {
				if (tile_solid(game, enemy->body.tile_x, y))
					empty_below = false;
			}

			//Determine if we need to change direction
			if (empty_below) {
				enemy->direction = -enemy->direction;
			} else if (ret == COL_RIGHT && enemy->direction > 0) {
				enemy->direction = -enemy->direction;
			} else if (ret == COL_LEFT && enemy->direction < 0) {
				enemy->direction = -enemy->direction;
			}

			//Kill player
			if (dist(player->body.px, player->body.py, enemy->body.px, enemy->body.py) < 32) {
				return PLAYER_DEAD;
			}
		}
	}
	
	//Fitness / score
	float fitness;
	fitness = 100 + player->score + player->body.px;
	fitness -= player->time * FIT_TIME_WEIGHT;
	fitness -= player->buttonpresses * FIT_BUTTONS_WEIGHT;
	player->fitness = fitness > player->fitness ? fitness : player->fitness;

	//End of level
	if (player->body.px + PLAYER_RIGHT >= (LEVEL_WIDTH - 4) * TILE_SIZE) {
		return PLAYER_DEAD;
	}

	//Time limit
	if (player->time >= 0.25 * LEVEL_WIDTH) {
		return PLAYER_DEAD;
	}

	return return_value;
}

//Physics simulation for any body
uint physics_sim(struct Game *game, struct Body* body, bool jump) {
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
	if (tile_solid(game, right_x, tile_y) || right_x >= LEVEL_WIDTH) {
		body->vx = 0;
		body->px = (right_x - 1) * TILE_SIZE + PLAYER_MARGIN - 2;
		return_value = COL_RIGHT;
	}

	//Left collision
	if (tile_solid(game, left_x, tile_y) || left_x < 0) {
		body->vx = 0;
		body->px = (left_x + 1) * TILE_SIZE - PLAYER_MARGIN + 2;
		return_value = COL_LEFT;
	}

	int tile_xr = (body->px + PLAYER_RIGHT) / TILE_SIZE;
	int tile_xl = (body->px + PLAYER_LEFT) / TILE_SIZE;

	//Collision on bottom
	body->standing = false;
	if (tile_solid(game, tile_xl, feet_y) > 0 || tile_solid(game, tile_xr, feet_y) > 0) {
		if (body->vy >= 0) {
			body->vy = 0;
			body->canjump = true;
			body->standing = true;
			if (!body->immune && (tile_at(game, tile_xl, feet_y) == SPIKES_TOP || tile_at(game, tile_xr, feet_y) == SPIKES_TOP)) {
				return PLAYER_DEAD;
			}
		}
		body->py = (feet_y - 1) * TILE_SIZE;
	}

	//Collision on top
	if (tile_solid(game, tile_xl, top_y) > 0 || tile_solid(game, tile_xr, top_y) > 0) {
		if (body->vy < 0) {
			body->vy = 0;
			body->isjump = false;
			if (!body->immune && (tile_at(game, tile_xl, top_y) == SPIKES_BOTTOM || tile_at(game, tile_xr, top_y) == SPIKES_BOTTOM)) {
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

//Return the tile at given tile position
int tile_at(struct Game *game, int x, int y) {
	if (x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT)
		return 0;
	return game->tiles[y * LEVEL_WIDTH + x];
}

//Return if the tile at the given tile position is solid
int tile_solid(struct Game *game, int x, int y) {
	int tile = tile_at(game, x, y);
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

//Basic distance function
float dist(float x1, float y1, float x2, float y2) {
	return sqrt(pow(x2 - x1, 2.0) + pow(y2 - y1, 2.0));
}

void get_input_tiles(struct Game *game, struct Player *player, uint8_t *tiles, uint8_t in_h, uint8_t in_w)
{
	int tile_x1, tile_y1;
	int tile_x2, tile_y2;
	int x, y;
	uint8_t *tmp;

	tmp = tiles;

	//Calculate bounds for drawing tiles
	tile_x1 = player->body.tile_x - in_w / 2;
	tile_x2 = player->body.tile_x + in_w / 2;
	tile_y1 = player->body.tile_y - in_h / 2;
	tile_y2 = player->body.tile_y + in_h / 2;

	//Loop over tiles and draw them
	for (y = tile_y1; y < tile_y2; y++) {
		for (x = tile_x1; x < tile_x2; x++) {
			// Report walls on left and right side of level
			if (x < 0 || x >= LEVEL_WIDTH)
				*tmp = BRICKS;
			else if (y < 0 || y >= LEVEL_HEIGHT)
				*tmp = EMPTY;
			else 
				*tmp = game->tiles[y * LEVEL_WIDTH + x];
	
			tmp++;
		}
	}
}