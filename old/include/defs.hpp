#ifndef DEFS_H
#define DEFS_H

// Genetic/NN run parameters
class Params {
    public:
    int in_h = 12;
    int in_w = 12;
    int hlc = 2;
    int npl = 128;
    int gen_size = 100;
    int generations = 1;
    int breedType = 0; // 0: Intra, 1: On, 2: Interp
    float mutate_rate = 0.001f;
};

// Time in seconds until NN is considered timedout
// based on fitness change
#define AGENT_FITNESS_TIMEOUT (UPDATES_PS * 6)

// Button parameters
#define BUTTON_COUNT 3
#define LEFT  0
#define RIGHT 1
#define JUMP  2

// Level parameters
#define LEVEL_HEIGHT 32
#define LEVEL_WIDTH  256
#define TILE_SIZE    32
#define LEVEL_PIXEL_WIDTH (LEVEL_WIDTH * TILE_SIZE)
#define LEVEL_PIXEL_HEIGHT (LEVEL_HEIGHT * TILE_SIZE)
#define LEVEL_SIZE (LEVEL_HEIGHT * LEVEL_WIDTH)

#define BG            0
#define GRID          1
#define LAMP          2

#define INPUT_SIZE    16

#endif
