import numpy as np
import game.core.defs as pz
from math import floor, ceil
from sfml.sf import Vector2
from game.core.levelgen import LevelGenerator

# Player physics parameters
V_X     = 6
V_JUMP  = 8.5
INTERTA = 1.4
GRAVITY = 0.3

class Body():
    def __init__(self):
        self.vel  = Vector2(0, 0)
        self.tile = Vector2(0, 0)
        self.pos  = Vector2(0, 0)
        self.size = Vector2(22, 23)
        self.half = self.size / 2

        self.can_jump = True
        self.is_jump  = False
        self.standing = True

    def reset(self):
        self.vel  = Vector2(0, 0)
        self.tile = Vector2(0, 0)
        self.pos  = Vector2(0, 0)

        self.can_jump = True
        self.is_jump  = False
        self.standing = True

class Player(Body):
    def __init__(self):
        super().__init__()

        self.time    = 0
        self.fitness = 0
        self.presses = 0

    def reset(self):
        super().reset()

        self.time    = 0
        self.fitness = 0
        self.presses = 0

class Game():
    def __init__(self, num_chunks, seed):
        self.num_chunks = num_chunks
        self.tiles = None
        self.solid_tiles = None

        self.map_seed = seed

        self.player = Player()
        self.level_generator = LevelGenerator()

        self.width = None
        self.height = None

        self.game_over      = False
        self.game_over_type = None

        self.setup_game()

    def setup_game(self):
        self.tiles, spawn_point = self.level_generator.generate_level(self.num_chunks, self.map_seed)
        self.height, self.width = self.tiles.shape

        self.solid_tiles = np.isin(self.tiles, pz.SOLID_TILES)

        self.player.tile = Vector2(1, spawn_point)
        self.player.pos  = self.player.tile * pz.TILE_SIZE

    def update(self, keys):
        if self.game_over:
            print("STOP CALLING UPDATE! THE GAME IS OVER DUMMY")
            return

        # Estimate of time
        self.player.time += 1.0 / pz.UPDATES_PS

        # Time limit
        if self.player.time > pz.MAX_TIME:
            self.game_over_type = pz.PLAYER_TIMEOUT
            self.game_over = True
            return 

        # Left and right button press
        if keys[pz.RIGHT]:
            self.player.vel.x = V_X
        if keys[pz.LEFT]:
            self.player.vel.x = -V_X

        # Button presses
        self.player.presses += sum(keys)

        # Physics sim for player
        ret = self.physicsSim(self.player, keys[pz.JUMP])
        if ret == pz.PLAYER_DEAD:
            self.game_over_type = pz.PLAYER_DEAD
            self.game_over = True
            return

        # Lower bound
        if self.player.pos.y >= self.height * pz.TILE_SIZE:
            self.game_over_type = pz.PLAYER_DEAD
            self.game_over = True
            return

        # Player completed level
        if ret == pz.PLAYER_COMPLETE:
            # Reward for finishing
            self.player.fitness += 2000
            self.game_over_type = pz.PLAYER_COMPLETE
            self.game_over = True

            return

        # Fitness
        self.player.fitness += self.player.vel.x

    def physicsSim(self, body, jump):
        # Jumping
        if jump and body.can_jump:
            body.can_jump = False
            body.is_jump  = True
            if not body.standing:
                body.vel.y = -V_JUMP

        if not jump and body.is_jump:
            body.is_jump = False

        if body.is_jump:
            body.vel.y -= 1.5
            if body.vel.y <= -V_JUMP:
                body.is_jump = False
                body.vel.y = -V_JUMP

        # Player physics
        body.vel.y += GRAVITY
        body.vel.x /= INTERTA

        body.tile = floor_vec((body.pos + body.vel) / pz.TILE_SIZE)
        feet_tile  = int((body.pos.y + body.vel.y + body.half.y + 1) // pz.TILE_SIZE)
        head_tile  = int((body.pos.y + body.vel.y - body.half.y - 1) // pz.TILE_SIZE)
        right_tile = int((body.pos.x + body.vel.x + body.half.x + 1) // pz.TILE_SIZE)
        left_tile  = int((body.pos.x + body.vel.x - body.half.x - 1) // pz.TILE_SIZE)

        tile_yu = int((body.pos.y - body.half.y) / pz.TILE_SIZE)
        tile_yd = int((body.pos.y + body.half.y) / pz.TILE_SIZE)

        # Right collision
        if self.tile_solid(tile_yd, right_tile) or self.tile_solid(tile_yu, right_tile):
            body.vel.x = 0
            body.pos.x = right_tile * pz.TILE_SIZE - body.half.x - 1

        # Left collision
        if self.tile_solid(tile_yd, left_tile) or self.tile_solid(tile_yu, left_tile):
            body.vel.x = 0
            body.pos.x = (left_tile + 1) * pz.TILE_SIZE + body.half.x

        tile_xr = int((body.pos.x + body.half.x) / pz.TILE_SIZE)
        tile_xl = int((body.pos.x - body.half.x) / pz.TILE_SIZE)

        # Collision on bottom
        body.standing = False
        if self.tile_solid(feet_tile, tile_xl) or self.tile_solid(feet_tile, tile_xr):
            body.vel.y = 0
            body.can_jump = True
            body.standing = True

            if pz.SPIKE_TOP in [self.tiles[feet_tile, tile_xl], self.tiles[feet_tile, tile_xr]]:
                return pz.PLAYER_DEAD

            if pz.FINISH_TOP in [self.tiles[feet_tile, tile_xl]]:
                return pz.PLAYER_COMPLETE

            body.pos.y = feet_tile * pz.TILE_SIZE - body.half.y

        # Collision on top
        if self.tile_solid(head_tile, tile_xl) or self.tile_solid(head_tile, tile_xr):
            if body.vel.y < 0:
                body.vel.y = 0
                body.is_jump = False

                if pz.SPIKE_BOT in [self.tiles[head_tile, tile_xl], self.tiles[head_tile, tile_xr]]:
                    return pz.PLAYER_DEAD

            body.pos.y = (head_tile + 1) * pz.TILE_SIZE + body.half.y

        # Apply body.velocity
        body.pos = round_vec(body.pos + body.vel)

        # Update tile position
        body.tile = floor_vec(body.pos / pz.TILE_SIZE)

    def tile_solid(self, row, col):
        if col >= self.width or col < 0:
            return True
        if row >= self.height:
            return False

        return self.solid_tiles[int(row), int(col)]
    
    def get_player_view(self, in_w, in_h):
        """ Returns a numpy matrix of size (in_h, in_w) of tiles around the player\n
            Player is considered to be in the 'center'
        """
        if (in_w * in_h) % 2 == 0:
            print("BOTH DIMENSIONS OF THE INPUT AROUND THE PLAYER MUST BE ODD!")
            exit(-1)

        out = np.empty((in_h, in_w), dtype=np.int32)

        lbound = self.player.tile.x - in_w // 2
        rbound = self.player.tile.x + in_w // 2
        ubound = self.player.tile.y - in_h // 2
        bbound = self.player.tile.y + in_h // 2

        for y in range(ubound, bbound + 1):
            for x in range(lbound, rbound + 1):
                if y < 0:
                    out[y - ubound, x - lbound] = pz.SPIKE_BOT
                elif y >= self.tiles.shape[0]:
                    out[y - ubound, x - lbound] = pz.SPIKE_TOP
                elif x < 0 or x >= self.tiles.shape[1]:
                    out[y - ubound, x - lbound] = pz.COBBLE
                else:
                    out[y - ubound, x - lbound] = self.tiles[y, x]

        return out

def floor_vec(vec):
    return Vector2(floor(vec.x), floor(vec.y))

def round_vec(vec):
    return Vector2(round(vec.x), round(vec.y))
