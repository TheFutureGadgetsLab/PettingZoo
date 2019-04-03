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
#include <vector>

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
class Body {
	public:
	float px, py;       // Position X, Y coord
	float vx, vy;       // Velocity X, Y val
	int tile_x, tile_y; // Tile X, Y coord
	bool canjump, isjump, standing;

	Body();
	void reset();
};

// Player class
class Player {
	public:
	Body body;
	uint8_t left, right, jump;
	float time, fitness;
	int score, buttonpresses, death_type;

	Player();
	void reset();
};

// Game class
class Game {
	public:
	uint8_t tiles[LEVEL_SIZE];
	unsigned int seed, seed_state;

	void setTileAt(int, int, uint8_t);
	uint8_t getTileAt(int, int);
	void getInputTiles(Player&, std::vector<float>&, uint8_t, uint8_t);
	bool inBounds(int, int);
	void genMap(unsigned int seed);
	int update(Player& player);
	int physicsSim(Body& body, bool jump);
	bool tileSolid(int, int);
};

void game_setup(Game& game, unsigned int seed);
int game_update(Game& game, Player& player);


#endif