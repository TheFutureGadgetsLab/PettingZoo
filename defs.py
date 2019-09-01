from sfml.sf import Vector2

HALF_TILE = Vector2(16, 16)

# Tile ID's, 0 is empty
EMPTY     = 0
PIPE_BOT  = 1
PIPE_MID  = 2
PIPE_TOP  = 3
GRASS     = 4
DIRT      = 5
COBBLE    = 6
SPIKE_TOP = 7
SPIKE_BOT = 8
COIN      = 9
FLAG      = 10
GRID      = 11
LAMP      = 12
BG        = 13

# Button Indices / ID's
LEFT  = 0
RIGHT = 1
JUMP  = 2

# Fitness measurement parameters
FIT_TIME_WEIGHT    =  2.0
FIT_BUTTONS_WEIGHT = 0.2
COIN_VALUE         = 1000

# Misc
TILE_SIZE    = 32
SPRITE_SIZE  = 32
UPDATES_PS   = 60
MAX_TIME     = 60
GROUND_LEVEL = 4

# Misc return values for game update / physics
PLAYER_COMPLETE = -3
PLAYER_TIMEOUT  = -2
PLAYER_DEAD     = -1