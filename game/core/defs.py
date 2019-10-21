##################################
#  Tiles
##################################
EMPTY      = 0
PIPE_BOT   = 1
PIPE_MID   = 2
PIPE_TOP   = 3
GRASS      = 4
DIRT       = 5
COBBLE     = 6
SPIKE_TOP  = 7
SPIKE_BOT  = 8
COIN       = 9
FLAG       = 10
FINISH_BOT = 11
FINISH_TOP = 12
LAMP       = 13
CAT        = 14
BACKGROUND = 15
SQUARE     = 16

TILES       = [EMPTY, PIPE_BOT, PIPE_MID, PIPE_TOP, GRASS, DIRT, COBBLE, SPIKE_TOP, SPIKE_BOT, COIN, FLAG, FINISH_BOT, FINISH_TOP]
SOLID_TILES = [PIPE_BOT, PIPE_MID, PIPE_TOP, GRASS, DIRT, COBBLE, SPIKE_TOP, SPIKE_BOT, FINISH_BOT, FINISH_TOP]

##################################
#  Buttons
##################################
LEFT  = 0
RIGHT = 1
JUMP  = 2

NUM_BUTTONS = 3

##################################
#  Fitness measurement parameters
##################################
FIT_TIME_WEIGHT = 100

##################################
#  Misc
##################################
TILE_SIZE    = 32
SPRITE_SIZE  = 32
UPDATES_PS   = 60
MAX_TIME     = 60
GROUND_LEVEL = 4