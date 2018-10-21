#ifndef GAMELOGIC_H
#define GAMELOGIC_H

#include <defs.hpp>

struct Player {
	float position_x;
	float position_y;
	float velocity_x;
	float velocity_y;
	float time;
	int left;
	int right;
	int jump;
	int canjump;
	int tile_x;
	int tile_y;
	int score;
	int buttonpresses;
	int fitness;
};

struct Enemy {
	int init_x;
	int init_y;
	unsigned char type;
};

struct Game {
	unsigned char tiles[LEVEL_SIZE];
	struct Enemy enemies[MAX_ENEMIES];
	unsigned int seed;
};

void game_setup();
void game_reset_map();
int game_update(int input[BUTTON_COUNT]);
void game_player_death();
int tile_at(int x, int y);

#endif