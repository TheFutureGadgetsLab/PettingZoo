import numpy as np
import defs as pz

# Misc return values for game update / physics
PLAYER_COMPLETE = -3
PLAYER_TIMEOUT  = -2
PLAYER_DEAD     = -1

# Player physics parameters
V_X     = 6
V_JUMP  = 8
INTERTA = 1.4
GRAVITY = 0.3

PLAYER_WIDTH  = 24
PLAYER_HALFW  = (PLAYER_WIDTH / 2)
PLAYER_MARGIN = ((pz.TILE_SIZE - PLAYER_WIDTH) / 2)
PLAYER_RIGHT  = (pz.TILE_SIZE - PLAYER_MARGIN)
PLAYER_LEFT   = (PLAYER_MARGIN / 2)

class Body():
    def __init__(self):
        self.px = 0
        self.py = 0

        self.vx = 0
        self.vy = 0

        self.tile_x = 0
        self.tile_y = 0

        self.can_jump = True
        self.is_jump  = False
        self.standing = True
    
    def reset(self):
        self.px = 0
        self.py = 0

        self.vx = 0
        self.vy = 0

        self.tile_x = 0
        self.tile_y = 0

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

    def setup_game(self, seed=None):
        """ Sets up / restarts game, if ``seed`` is none the current level seed is used,
            otherwise a new map is generated
        """
        self.player.reset()
        self.tiles[:3, :] = pz.DIRT
    
    def print_map(self):
        print(np.flipud(self.tiles))
    
    def get_map(self):
        return np.flipud(self.tiles)
    
    def update(self, keys):
        # Estimate of time
        self.player.time += 1.0 / pz.UPDATES_PS
        
        # Time limit
        if self.player.time > pz.MAX_TIME:
            self.player.death_type = PLAYER_TIMEOUT
            return PLAYER_TIMEOUT
        
        # Left and right button press
        self.player.vx += ( V_X - self.player.vx) * keys[pz.RIGHT]
        self.player.vx += (-V_X - self.player.vx) * keys[pz.LEFT]

        # Button presses
        self.player.presses += sum(keys)

        # Physics sim for player
        ret = self.physicsSim(self.player, keys[pz.JUMP])
        if ret == PLAYER_DEAD:
            self.player.death_type = PLAYER_DEAD
            return PLAYER_DEAD

        # Lower bound
        if self.player.py > self.width * pz.TILE_SIZE:
            self.player.death_type = PLAYER_DEAD
            return PLAYER_DEAD
        
        # Fitness
        fitness  = 100 + self.player.fitness + self.player.px
        fitness -= self.player.time * pz.FIT_TIME_WEIGHT
        fitness -= self.player.presses * pz.FIT_BUTTONS_WEIGHT
        
        # Only increase fitness, never decrease
        if self.player.fitness < fitness:
            self.player.fitness = fitness

        # Player completed level
        if self.player.px + PLAYER_RIGHT >= (self.width - 4) * pz.TILE_SIZE:
            # Reward for finishing
            self.player.fitness += 2000
            self.player.death_type = PLAYER_COMPLETE
            return PLAYER_COMPLETE

        return 0
    
    def physicsSim(self, body, jump):
        # Jumping
        if jump & body.can_jump:
            body.can_jump = False
            if not body.standing:
                body.vy = -V_JUMP

        if not jump & body.is_jump:
            body.is_jump = False

        if body.is_jump:
            body.vy -= 1.5
            if body.vy <= -V_JUMP:
                body.is_jump = False
                body.vy = -V_JUMP

        # Player physics
        tile_x  = int((body.px + body.vx + 16) / pz.TILE_SIZE)
        tile_y  = int((body.py + body.vy + 16) / pz.TILE_SIZE)
        feet_y  = int((body.py + body.vy + 33) / pz.TILE_SIZE)
        top_y   = int((body.py + body.vy - 1) / pz.TILE_SIZE)
        right_x = int((body.px + body.vx + PLAYER_RIGHT + 1) / pz.TILE_SIZE)
        left_x  = int((body.px + body.vx + PLAYER_LEFT - 1) / pz.TILE_SIZE)

        body.tile_x = tile_x
        body.tile_y = tile_y

        body.vy += GRAVITY
        body.vx /= INTERTA

        # Right collision
        if self.tileSolid(right_x, tile_y) or right_x >= self.width:
            body.vx = 0
            body.px = (right_x - 1) * pz.TILE_SIZE + PLAYER_MARGIN - 2

        # Left collision
        if self.tileSolid(left_x, tile_y) or left_x < 0:
            body.vx = 0
            body.px = (left_x + 1) * pz.TILE_SIZE - PLAYER_MARGIN + 2

        tile_xr = int((body.px + PLAYER_RIGHT) / pz.TILE_SIZE)
        tile_xl = int((body.px + PLAYER_LEFT) / pz.TILE_SIZE)

        # Collision on bottom
        body.standing = False
        if self.tileSolid(tile_xl, feet_y) > 0 or self.tileSolid(tile_xr, feet_y) > 0:
            if body.vy >= 0:
                body.vy = 0
                body.can_jump = True
                body.standing = True
                
                if pz.SPIKE_TOP in [self.tiles[tile_xl, feet_y], self.tiles[tile_xr, feet_y]]:
                    return PLAYER_DEAD

            body.py = (feet_y - 1) * pz.TILE_SIZE

        # Collision on top
        if self.tileSolid(tile_xl, top_y) > 0 or self.tileSolid(tile_xr, top_y) > 0:
            if body.vy < 0:
                body.vy = 0
                body.is_jump = False
                
                if pz.SPIKE_BOT in [self.tiles[tile_xl, top_y], self.tiles[tile_xr, top_y]]:
                    return PLAYER_DEAD
            
            body.py = (top_y + 1) * pz.TILE_SIZE

        # Apply body.velocity
        body.px = round(body.px + body.vx)
        body.py = round(body.py + body.vy)

        # Update tile position
        body.tile_x = (body.px + 16) / pz.TILE_SIZE
        body.tile_y = (body.py + 16) / pz.TILE_SIZE
    
    def tileSolid(self, x, y):
        if self.tiles[int(x), int(y)] in [pz.COBBLE, pz.DIRT, pz.GRASS, pz.PIPE_BOT, pz.PIPE_MID, pz.PIPE_TOP, pz.SPIKE_BOT, pz.SPIKE_TOP]:
            return True
        
        return False