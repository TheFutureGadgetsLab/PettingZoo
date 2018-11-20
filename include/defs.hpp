#ifndef DEFS_H
#define DEFS_H

// AI parameters
#define IN_H         12
#define IN_W         12
#define HLC          2
#define NPL          256
#define GEN_SIZE     100
#define GENERATIONS  10

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

#define INPUT_SIZE    16

#endif