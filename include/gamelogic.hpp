#ifndef GAMELOGIC_H
#define GAMELOGIC_H

#include <defs.hpp>

struct Body {
	float px;
	float py;
	float vx;
	float vy;
	bool immune;
	bool canjump;
	bool isjump;
	bool standing;
	int tile_x;
	int tile_y;
};

struct Player {
	struct Body body;
	float time;
	int left;
	int right;
	int jump;
	int score;
	int buttonpresses;
	int fitness;
};

struct Enemy {
	struct Body body;
	bool dead;
	float speed;
	unsigned char type;
};

struct Game {
	unsigned char tiles[LEVEL_SIZE];
	struct Enemy enemies[MAX_ENEMIES];
	unsigned int n_enemies;
	unsigned int seed;
};

void game_setup();
int game_update(int input[BUTTON_COUNT]);

#endif