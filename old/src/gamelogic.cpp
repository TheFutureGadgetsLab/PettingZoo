/**
 * @file gamelogic.cpp
 * @author Benjamin Mastripolito, Haydn Jones
 * @brief Functions for interfacing with chromosomes
 * @date 2018-12-06
 */

#include <gamelogic.hpp>
#include <levelgen.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>

///////////////////////////////////////////////////////////////
//
//                  Player Class
//
///////////////////////////////////////////////////////////////
Player::Player()
{
	reset();
}

void Player::reset()
{
	body.reset();

	left = 0;
	right = 0;
	jump = 0;

	score = 0;
	fitness = 0;
	time = 0;
	buttonpresses = 0;
}

///////////////////////////////////////////////////////////////
//
//                  Body Class
//
///////////////////////////////////////////////////////////////
Body::Body()
{
	reset();
}

void Body::reset()
{
	px = SPAWN_X * TILE_SIZE;
	py = SPAWN_Y * TILE_SIZE;

	vx = 0;
	vy = 0;
	
	tile_x = px / TILE_SIZE;
	tile_y = py / TILE_SIZE;
	
	canjump = false;
	isjump = false;
	standing = false;
}

///////////////////////////////////////////////////////////////
//
//                  Game Class
//
///////////////////////////////////////////////////////////////

void Game::setTileAt(int x, int y, int value)
{
	if (!inBounds(x, y))
		return;

	tiles[y * LEVEL_WIDTH + x] = value;
}

int Game::getTileAt(int x, int y)
{
	if (!inBounds(x, y))
		return 0;

	return tiles[y * LEVEL_WIDTH + x];
}

bool Game::inBounds(int x, int y)
{
	return !(x < 0 || x >= LEVEL_WIDTH || y < 0 || y >= LEVEL_HEIGHT);
}

/**
 * @brief Setup a game, full reset
 * 
 * @param game The game to reset
 * @param seed The seed to generate the new level from
 */
void Game::genMap(unsigned int seed)
{
	levelgen_clear_level(*this);
	levelgen_gen_map(*this, seed);
}

/**
 * @brief Update a single frame of the game, simulating physics
 * 
 * @param game The game obj we are operating on
 * @param player The player playing the level
 * @param input Button controls for the player
 * @return int PLAYER_DEAD or PLAYER_COMPLETE
 */
int Game::update(Player& player)
{
	int return_value = 0;

	// Estimate of time
	player.time += 1.0f / (float)UPDATES_PS;
	
	// Time limit
	if (player.time >= MAX_TIME - 1.0f / (float)UPDATES_PS) {
		player.death_type = PLAYER_TIMEOUT;
		return PLAYER_TIMEOUT;
	}
	
	// Left and right button press
	player.body.vx += (V_X - player.body.vx) * player.right;
	player.body.vx += (-V_X - player.body.vx) * player.left;

	// Button presses
	player.buttonpresses += player.jump + player.left + player.right;

	// Physics sim for player
	return_value = physicsSim(player.body, player.jump);
	if (return_value == PLAYER_DEAD) {
		player.death_type = PLAYER_DEAD;
		return PLAYER_DEAD;
	}

	// Lower bound
	if (player.body.py > LEVEL_PIXEL_HEIGHT) {
		player.death_type = PLAYER_DEAD;
		return PLAYER_DEAD;
	}
	
	// Fitness
	float fitness;
	fitness = 100 + player.score + player.body.px;
	fitness -= player.time * FIT_TIME_WEIGHT;
	fitness -= player.buttonpresses * FIT_BUTTONS_WEIGHT;
	// Only increase fitness, never decrease.
	player.fitness = fitness > player.fitness ? fitness : player.fitness;

	// Player completed level
	if (player.body.px + PLAYER_RIGHT >= (LEVEL_WIDTH - 4) * TILE_SIZE) {
		// Reward for finishing
		player.fitness += 2000;
		player.death_type = PLAYER_COMPLETE;
		return PLAYER_COMPLETE;
	}

	return return_value;
}

/**
 * @brief Runs physics simulation for a given body
 * 
 * @param game Game the body is in
 * @param body Body to apply physics to
 * @param jump Whether or not the body is attempting to jump
 * @return int If player died or collided with something on the left/right
 */
int Game::physicsSim(Body& body, bool jump)
{
	int return_value = 0;

	// Jumping
	if (jump && body.canjump) {
		body.isjump = true;
		body.canjump = false;
		if (!body.standing)
			body.vy = -V_JUMP;
	}
	if (!jump && body.isjump)
		body.isjump = false;
	if (body.isjump) {
		body.vy -= 1.5f;
		if (body.vy <= -V_JUMP) {
			body.isjump = false;
			body.vy = -V_JUMP;
		}
	}

	// Player physics
	int tile_x = (body.px + body.vx + 16) / TILE_SIZE;
	int tile_y = (body.py + body.vy + 16) / TILE_SIZE;
	int feet_y = (body.py + body.vy + 33) / TILE_SIZE;
	int top_y = (body.py + body.vy - 1) / TILE_SIZE;
	int right_x = (body.px + body.vx + PLAYER_RIGHT + 1) / TILE_SIZE;
	int left_x = (body.px + body.vx + PLAYER_LEFT - 1) / TILE_SIZE;

	body.tile_x = tile_x;
	body.tile_y = tile_y;

	body.vy += GRAVITY;
	body.vx /= INTERTA;

	// Right collision
	if (tileSolid(right_x, tile_y) || right_x >= LEVEL_WIDTH) {
		body.vx = 0;
		body.px = (right_x - 1) * TILE_SIZE + PLAYER_MARGIN - 2;
		return_value = COL_RIGHT;
	}

	// Left collision
	if (tileSolid(left_x, tile_y) || left_x < 0) {
		body.vx = 0;
		body.px = (left_x + 1) * TILE_SIZE - PLAYER_MARGIN + 2;
		return_value = COL_LEFT;
	}

	int tile_xr = (body.px + PLAYER_RIGHT) / TILE_SIZE;
	int tile_xl = (body.px + PLAYER_LEFT) / TILE_SIZE;

	// Collision on bottom
	body.standing = false;
	if (tileSolid(tile_xl, feet_y) > 0 || tileSolid(tile_xr, feet_y) > 0) {
		if (body.vy >= 0) {
			body.vy = 0;
			body.canjump = true;
			body.standing = true;
			if (getTileAt(tile_xl, feet_y) == SPIKES_TOP || getTileAt(tile_xr, feet_y) == SPIKES_TOP) {
				return PLAYER_DEAD;
			}
		}
		body.py = (feet_y - 1) * TILE_SIZE;
	}

	// Collision on top
	if (tileSolid(tile_xl, top_y) > 0 || tileSolid(tile_xr, top_y) > 0) {
		if (body.vy < 0) {
			body.vy = 0;
			body.isjump = false;
			if (getTileAt(tile_xl, top_y) == SPIKES_BOTTOM || getTileAt(tile_xr, top_y) == SPIKES_BOTTOM) {
				return PLAYER_DEAD;
			}
		}
		body.py = (top_y + 1) * TILE_SIZE;
	}

	// Apply body.velocity
	body.px = round(body.px + body.vx);
	body.py = round(body.py + body.vy);

	// Update tile position
	body.tile_x = (body.px + 16) / TILE_SIZE;
	body.tile_y = (body.py + 16) / TILE_SIZE;
	
	return return_value;
}

/**
 * @brief Exposes the tiles around the player to the neural network
 * 
 * @param game The game object to get tiles from
 * @param player The player object
 * @param tiles The array of tiles to reference
 * @param in_h Height of the chromsome input matrix
 * @param in_w Width of the chromsome input matrix
 */

bool Game::tileSolid(int x, int y)
{
	int tile = getTileAt(x, y);
	switch(tile) {
		case EMPTY:
		case FLAG:
			return false;
		default:
			break;
	}
	return true;
}

void Game::getInputTiles(Player& player, std::vector<float>& out, int in_h, int in_w)
{
	int tile_x1, tile_y1;
	int tile_x2, tile_y2;
	int x, y, i;
	int tile;

	//Calculate bounds for drawing tiles
	tile_x1 = player.body.tile_x - in_w / 2;
	tile_x2 = player.body.tile_x + in_w / 2;
	tile_y1 = player.body.tile_y - in_h / 2;
	tile_y2 = player.body.tile_y + in_h / 2;

	i = 0;
	for (y = tile_y1; y < tile_y2; y++) {
		for (x = tile_x1; x < tile_x2; x++) {
			// Report walls on left and right side of level
			if (x < 0 || x >= LEVEL_WIDTH)
				tile = BRICKS;
			else if (y < 0 || y >= LEVEL_HEIGHT)
				tile = EMPTY;
			else
				tile = tiles[y * LEVEL_WIDTH + x];

			if (tile == EMPTY || tile == FLAG)
				out[i] = 0.0f;
			else if (tile == PIPE_BOTTOM || tile == PIPE_MIDDLE || tile == PIPE_TOP || tile == GRASS || tile == DIRT || tile == BRICKS)
				out[i] = (1.0f / 3.0f);
			else if (tile == SPIKES_TOP)
				out[i] = (2.0f / 3.0f);
			else if (tile == SPIKES_BOTTOM)
				out[i] = 1.0f;
			else
				exit(-1);

			i++;
		}
	}
}