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
#define LEVEL_WIDTH 128
#define TILE_SIZE 32
#define LEVEL_PIXEL_WIDTH (LEVEL_WIDTH * TILE_SIZE)
#define LEVEL_PIXEL_HEIGHT (LEVEL_HEIGHT * TILE_SIZE)
#define LEVEL_SIZE (LEVEL_HEIGHT * LEVEL_WIDTH)
#define VIEW_SIZE_Y (LEVEL_PIXEL_HEIGHT + 2 * TILE_SIZE)
#define MAX_ENEMIES 32

// Level generation parameters
#define GROUND_HEIGHT (LEVEL_HEIGHT - LEVEL_HEIGHT / 16)
#define SPAWN_X 2
#define SPAWN_Y (GROUND_HEIGHT - 1)
#define HOLE_CHANCE 9

// Sprite parameters
#define EMPTY 0
#define BRICKS 1
#define DIRT 2
#define GRASS 3
#define LAMP 4
#define PIPE_BOTTOM 5
#define PIPE_MIDDLE 6
#define PIPE_TOP 7
#define SPIKES 8
#define ENEMY 9
#define GRID 10
#define BG 11

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
#define UPDATES_PS 60.0

#endif