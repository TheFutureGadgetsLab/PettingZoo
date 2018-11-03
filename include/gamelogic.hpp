#ifndef GAMELOGIC_H
#define GAMELOGIC_H

#include <defs.hpp>

struct Body {
	float px;
	float py;
	float vx;
	float vy;
	int tile_x;
	int tile_y;
	bool immune;
	bool canjump;
	bool isjump;
	bool standing;
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
	float speed;
	float direction;
	bool dead;
	unsigned char type;
};

struct Game {
	struct Enemy enemies[MAX_ENEMIES];
	unsigned int n_enemies;
	unsigned int seed;
	unsigned char tiles[LEVEL_SIZE];
};

void game_setup();
int game_update(int input[BUTTON_COUNT]);

#endif