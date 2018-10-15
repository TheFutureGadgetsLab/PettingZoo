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
#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define LEVEL_PIXEL_WIDTH (LEVEL_WIDTH * TILE_WIDTH)
#define LEVEL_PIXEL_HEIGHT (LEVEL_HEIGHT * TILE_HEIGHT)
#define LEVEL_SIZE (LEVEL_HEIGHT * LEVEL_WIDTH)
#define VIEW_SIZE_Y (LEVEL_PIXEL_HEIGHT + 2 * TILE_HEIGHT)
#define MAX_ENEMIES 32

// Level generation parameters
#define GROUND_HEIGHT (LEVEL_HEIGHT - LEVEL_HEIGHT / 8)
#define SPAWN_X 2
#define SPAWN_Y (GROUND_HEIGHT - 1)
#define HOLE_CHANCE 9

// Sprite parameters
#define EMPTY 0
#define GRASS 1
#define DIRT 2
#define BRICKS 3
#define SPIKES 4
#define LAMP 5
#define GRID 6
#define BG 7

// Player physics parameters 
#define V_X 6
#define V_JUMP 8
#define INTERTA 1.4
#define GRAVITY 0.3
#define PLAYER_WIDTH 24
#define PLAYER_HALFW (PLAYER_WIDTH / 2)
#define PLAYER_MARGIN ((TILE_WIDTH - PLAYER_WIDTH) / 2)
#define PLAYER_RIGHT (TILE_WIDTH - PLAYER_MARGIN)
#define PLAYER_LEFT (PLAYER_MARGIN / 2)
#define UPDATES_PS 60.0

#endif