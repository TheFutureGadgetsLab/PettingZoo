#ifndef DEFS_H
#define DEFS_H

// Button parameters
#define BUTTON_COUNT 3
#define BUTTON_LEFT 0
#define BUTTON_RIGHT 1
#define BUTTON_JUMP 2

// Camera parameters
#define CAMERA_INTERP 0.1

// Level parameters
#define LEVEL_HEIGHT 32
#define LEVEL_WIDTH 256
#define TILE_SIZE 32
#define LEVEL_PIXEL_WIDTH (LEVEL_WIDTH * TILE_SIZE)
#define LEVEL_PIXEL_HEIGHT (LEVEL_HEIGHT * TILE_SIZE)
#define LEVEL_SIZE (LEVEL_HEIGHT * LEVEL_WIDTH)
#define VIEW_SIZE_Y (LEVEL_PIXEL_HEIGHT + 2 * TILE_SIZE)
#define MAX_ENEMIES 32
#define COIN_VALUE 1000
#define ENABLE_ENEMIES false
#define JUMPING_ENEMIES false

// Level generation parameters
#define GROUND_HEIGHT (LEVEL_HEIGHT - 4)
#define START_PLATLEN 5
#define SPAWN_X 0
#define SPAWN_Y (GROUND_HEIGHT - 1)

// Sprite parameters (flat index into spritesheet)
#define EMPTY         0
#define PIPE_BOTTOM   1
#define PIPE_MIDDLE   2
#define PIPE_TOP      3
#define GRASS         4
#define DIRT          5
#define BRICKS        6
#define SPIKES_TOP    7
#define SPIKES_BOTTOM 8
#define COIN          9
#define FLAG          10

#define ENEMY         0
#define BG            1
#define GRID          2
#define LAMP          3

#define INPUT_SIZE 16

// Player physics parameters
#define V_X 6
#define V_JUMP 8
#define INTERTA 1.4
#define GRAVITY 0.3
#define PLAYER_WIDTH 24
#define PLAYER_HALFW (PLAYER_WIDTH / 2)
#define PLAYER_MARGIN ((TILE_SIZE - PLAYER_WIDTH) / 2)
#define PLAYER_RIGHT (TILE_SIZE - PLAYER_MARGIN)
#define PLAYER_LEFT (PLAYER_MARGIN / 2)
#define UPDATES_PS 60
#define MAX_TIME (LEVEL_WIDTH / 4)
#define MAX_FRAMES (MAX_TIME * UPDATES_PS)

// Fitness measurement parameters
#define FIT_TIME_WEIGHT 2.0
#define FIT_BUTTONS_WEIGHT 0.2

#endif