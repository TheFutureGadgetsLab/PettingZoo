#ifndef GAMELOGIC_H
#define GAMELOGIC_H

#include <defs.hpp>
#include <stdint.h>

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
	uint8_t tiles[LEVEL_SIZE];
};

void game_setup(struct Game *game, struct Player *player, unsigned seed);
int game_update(struct Game *game, struct Player *player, int input[BUTTON_COUNT]);
void get_input_tiles(struct Game *game, struct Player *player, uint8_t *tiles, uint8_t in_h, uint8_t in_w);


#endif