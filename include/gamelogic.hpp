/**
 * @file gamelogic.hpp
 * @author Benjamin Mastripolito, Haydn Jones
 * @brief Functions for interfacing with chromosomes
 * @date 2018-12-06
 */

#ifndef GAMELOGIC_H
#define GAMELOGIC_H

#include <defs.hpp>
#include <stdint.h>

// Fitness measurement parameters
#define FIT_TIME_WEIGHT 2.0f
#define FIT_BUTTONS_WEIGHT 0.2f
#define COIN_VALUE 1000

// Misc return values for game update / physics
#define PLAYER_COMPLETE -3
#define PLAYER_TIMEOUT  -2
#define PLAYER_DEAD     -1
#define REDRAW           1
#define COL_LEFT         2
#define COL_RIGHT        4

// Player physics parameters
#define V_X 6
#define V_JUMP 8
#define INTERTA 1.4f
#define GRAVITY 0.3f
#define PLAYER_WIDTH 24
#define PLAYER_HALFW (PLAYER_WIDTH / 2)
#define PLAYER_MARGIN ((TILE_SIZE - PLAYER_WIDTH) / 2)
#define PLAYER_RIGHT (TILE_SIZE - PLAYER_MARGIN)
#define PLAYER_LEFT (PLAYER_MARGIN / 2)
#define UPDATES_PS 60
#define MAX_TIME 60
#define MAX_FRAMES (MAX_TIME * UPDATES_PS)

// Body that physics can be applied to
struct Body {
	float px;
	float py;
	float vx;
	float vy;
	int tile_x;
	int tile_y;
	bool canjump;
	bool isjump;
	bool standing;
};

// Player structure
struct Player {
	struct Body body;
	float time;
	float fitness;
	int score;
	int buttonpresses;
	int death_type;
};

// Game structure
struct Game {
	unsigned int seed;
	unsigned int seed_state;
	uint8_t tiles[LEVEL_SIZE];
};

void game_setup(struct Game *game, unsigned int seed);
void player_setup(struct Player *player);
int game_update(struct Game *game, struct Player *player, uint8_t input[BUTTON_COUNT]);
void get_input_tiles(struct Game *game, struct Player *player, float *tiles, uint8_t in_h, uint8_t in_w);

#endif