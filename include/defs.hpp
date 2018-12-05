#ifndef DEFS_H
#define DEFS_H

// AI parameters
#define IN_H        12
#define IN_W        12
#define HLC         2
#define NPL         128
#define GEN_SIZE    4096
#define GENERATIONS 1
#define MUTATE_RATE 0.001f

// Enemies
#define ENABLE_ENEMIES  false
#define JUMPING_ENEMIES false
#define MAX_ENEMIES 0

// Button parameters
#define BUTTON_COUNT 3
#define BUTTON_LEFT  0
#define BUTTON_RIGHT 1
#define BUTTON_JUMP  2

// Level parameters
#define LEVEL_HEIGHT 32
#define LEVEL_WIDTH  256
#define TILE_SIZE    32
#define LEVEL_PIXEL_WIDTH (LEVEL_WIDTH * TILE_SIZE)
#define LEVEL_PIXEL_HEIGHT (LEVEL_HEIGHT * TILE_SIZE)
#define LEVEL_SIZE (LEVEL_HEIGHT * LEVEL_WIDTH)

#define ENEMY         0
#define BG            1
#define GRID          2
#define LAMP          3

#define INPUT_SIZE    16

#endif
