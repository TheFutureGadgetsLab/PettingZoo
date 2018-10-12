#ifndef GAMELOGIC_H
#define GAMELOGIC_H

#include <defs.hpp>

class Player {
	public:
	float position_x;
	float position_y;
	float velocity_x;
	float velocity_y;
	int left;
	int right;
	int jump;
	int canjump;
	int tile_x;
	int tile_y;
};

class Enemy {
	int init_x;
	int init_y;
	unsigned char type;
};

class Game {
	public:
	unsigned char tiles[LEVEL_SIZE];
	Enemy enemies[MAX_ENEMIES];
	unsigned int seed;
};

void game_setup();
void game_update(int input[BUTTON_COUNT]);
int tile_at(int x, int y);

#endif