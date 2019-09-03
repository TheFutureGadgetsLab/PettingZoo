import numpy as np
import defs as pz
import pysnooper
from math import floor
from sfml.sf import Vector2
from levelgen import LevelGenerator

# Player physics parameters
V_X     = 6
V_JUMP  = 8
INTERTA = 1.4
GRAVITY = 0.3

PLAYER_WIDTH  = 24
PLAYER_HEIGHT = 26
PLAYER_HALFW  = (PLAYER_WIDTH / 2)
PLAYER_MARGIN = ((pz.TILE_SIZE - PLAYER_WIDTH) / 2)
PLAYER_RIGHT  = (pz.TILE_SIZE - PLAYER_MARGIN)
PLAYER_LEFT   = (PLAYER_MARGIN / 2)

class Body():
    def __init__(self):
        self.vel  = Vector2(0, 0)
        self.tile = Vector2(0, 0)
        self.pos  = Vector2(0, 0)
        self.size = Vector2(24, 32)
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

class Game():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.tiles = np.zeros(shape=(height, width), dtype=int)

        self.map_seed = 0

        self.player = Player()
        self.level_generator = LevelGenerator()

    def setup_game(self, seed=0):
        """ Sets up / restarts game, if ``seed`` is none the current level seed is used,
            otherwise a new map is generated
        """
        self.player.reset()

        self.tiles, spawn_point = self.level_generator.generate_level(self.width, self.height, seed)

        self.player.tile = Vector2(1, spawn_point)
        self.player.pos  = self.player.tile * pz.TILE_SIZE

    def update(self, keys):
        # Estimate of time
        self.player.time += 1.0 / pz.UPDATES_PS

        # Time limit
        if self.player.time > pz.MAX_TIME:
            self.player.death_type = pz.PLAYER_TIMEOUT
            return pz.PLAYER_TIMEOUT

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
            self.player.death_type = pz.PLAYER_DEAD
            return pz.PLAYER_DEAD

        # Lower bound
        if self.player.pos.y >= self.height * pz.TILE_SIZE:
            self.player.death_type = pz.PLAYER_DEAD
            return pz.PLAYER_DEAD

        # Fitness
        fitness  = 100 + self.player.fitness + self.player.pos.x
        fitness -= self.player.time * pz.FIT_TIME_WEIGHT
        fitness -= self.player.presses * pz.FIT_BUTTONS_WEIGHT

        # Only increase fitness, never decrease
        if self.player.fitness < fitness:
            self.player.fitness = fitness

        # Player completed level
        if self.player.pos.x + PLAYER_RIGHT >= (self.width - 4) * pz.TILE_SIZE:
            # Reward for finishing
            self.player.fitness += 2000
            self.player.death_type = pz.PLAYER_COMPLETE
            return pz.PLAYER_COMPLETE

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
        body.tile = floor_vec((body.pos + body.vel) / pz.TILE_SIZE)
        feet_tile  = int((body.pos.y + body.vel.y + body.half.y + 1) // pz.TILE_SIZE)
        head_tile  = int((body.pos.y + body.vel.y - body.half.y - 1) // pz.TILE_SIZE)
        right_tile = int((body.pos.x + body.vel.x + body.half.x + 1) // pz.TILE_SIZE)
        left_tile  = int((body.pos.x + body.vel.x + body.half.x - 1) // pz.TILE_SIZE)

        body.vel.y += GRAVITY
        body.vel.x /= INTERTA

        # Right collision
        if self.tile_solid(body.tile.y, right_tile):
            body.vel.x = 0
            body.pos.x = right_tile * pz.TILE_SIZE - body.half.x - 2

        # Left collision
        if self.tile_solid(body.tile.y, left_tile):
            body.vel.x = 0
            body.pos.x = left_tile * pz.TILE_SIZE + body.half.x + 2

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

        if self.tiles[int(row), int(col)] in pz.SOLID_TILES:
            return True

        return False

def floor_vec(vec):
    return Vector2(floor(vec.x), floor(vec.y))

def round_vec(vec):
    return Vector2(round(vec.x), round(vec.y))
    