import game.defs as pz
from game.levelgen import Level
from pygame import Vector2
from random import randint

# Player physics parameters
V_X     = 6
V_JUMP  = 8.5
INTERTA = 1.4
GRAVITY = 0.3

class Game():
    # Misc return values for game update / physics
    PLAYER_COMPLETE = -3
    PLAYER_TIMEOUT  = -2
    PLAYER_DEAD     = -1

    def __init__(self, num_chunks, seed=None, view_size=None):
        if seed == None:
            seed = randint(0, 1_000_000)

        self.level = Level(num_chunks, seed, view_size)

        self.player = Player()

        self.game_over      = False
        self.game_over_type = None

        self.setup_game()

    def setup_game(self):
        self.player.tile = self.level.spawn_point
        self.player.pos  = self.player.tile * pz.TILE_SIZE

    def update(self, keys):
        if self.game_over:
            print("STOP CALLING UPDATE! THE GAME IS OVER DUMMY")
            return

        # Estimate of time
        self.player.time += 1.0 / pz.UPDATES_PS

        # Time limit
        if self.player.time > pz.MAX_TIME:
            self.game_over_type = Game.PLAYER_TIMEOUT
            self.game_over = True
            return 

        # Left and right button press
        if keys[pz.RIGHT]:
            self.player.vel.x = V_X
        if keys[pz.LEFT]:
            self.player.vel.x = -V_X

        # Button presses
        self.player.presses += keys[0] + keys[1] + keys[2]

        # Physics sim for player
        ret = self.physicsSim(self.player, keys[pz.JUMP])
        if ret == Game.PLAYER_DEAD:
            self.game_over_type = Game.PLAYER_DEAD
            self.game_over = True

            return

        # Lower bound
        if self.player.pos.y >= self.level.size.y * pz.TILE_SIZE:
            self.game_over_type = Game.PLAYER_DEAD
            self.game_over = True

            return

        # Player completed level
        if ret == Game.PLAYER_COMPLETE:
            # Reward for finishing
            self.player.fitness += 2000
            self.game_over_type = Game.PLAYER_COMPLETE
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

        body.tile = (body.pos + body.vel) // pz.TILE_SIZE
        feet_tile  = int((body.pos.y + body.vel.y + body.half.y + 1) // pz.TILE_SIZE)
        head_tile  = int((body.pos.y + body.vel.y - body.half.y - 1) // pz.TILE_SIZE)
        right_tile = int((body.pos.x + body.vel.x + body.half.x + 1) // pz.TILE_SIZE)
        left_tile  = int((body.pos.x + body.vel.x - body.half.x - 1) // pz.TILE_SIZE)

        tile_yu = int((body.pos.y - body.half.y) / pz.TILE_SIZE)
        tile_yd = int((body.pos.y + body.half.y) / pz.TILE_SIZE)

        # Right collision
        if self.level.tile_solid(tile_yd, right_tile) or self.level.tile_solid(tile_yu, right_tile):
            body.vel.x = 0
            body.pos.x = right_tile * pz.TILE_SIZE - body.half.x - 1

        # Left collision
        if self.level.tile_solid(tile_yd, left_tile) or self.level.tile_solid(tile_yu, left_tile):
            body.vel.x = 0
            body.pos.x = (left_tile + 1) * pz.TILE_SIZE + body.half.x

        tile_xr = int((body.pos.x + body.half.x) / pz.TILE_SIZE)
        tile_xl = int((body.pos.x - body.half.x) / pz.TILE_SIZE)

        # Collision on bottom
        body.standing = False
        if self.level.tile_solid(feet_tile, tile_xl) or self.level.tile_solid(feet_tile, tile_xr):
            body.vel.y = 0
            body.can_jump = True
            body.standing = True

            if pz.SPIKE_TOP == self.level.tiles[feet_tile, tile_xl] or pz.SPIKE_TOP == self.level.tiles[feet_tile, tile_xr]:
                return Game.PLAYER_DEAD

            if pz.FINISH_TOP == self.level.tiles[feet_tile, tile_xl]:
                return Game.PLAYER_COMPLETE

            body.pos.y = feet_tile * pz.TILE_SIZE - body.half.y

        # Collision on top
        if self.level.tile_solid(head_tile, tile_xl) or self.level.tile_solid(head_tile, tile_xr):
            if body.vel.y < 0:
                body.vel.y = 0
                body.is_jump = False

                if pz.SPIKE_BOT == self.level.tiles[head_tile, tile_xl] or pz.SPIKE_BOT == self.level.tiles[head_tile, tile_xr]:
                    return Game.PLAYER_DEAD

            body.pos.y = (head_tile + 1) * pz.TILE_SIZE + body.half.y

        # Apply body.velocity
        body.pos.update(body.pos + body.vel)
        body.pos.x = round(body.pos.x)
        body.pos.y = round(body.pos.y)

        # Update tile position
        body.tile.update(body.pos // pz.TILE_SIZE)
    
    def get_player_view(self):
        """ Returns a numpy matrix of size (in_h, in_w) of tiles around the player\n
            Player is considered to be in the 'center'
        """

        return self.level.get_player_view(self.player.tile)

class Body():
    def __init__(self):
        self.vel  = Vector2(0.0, 0.0)
        self.tile = Vector2(0.0, 0.0)
        self.pos  = Vector2(0.0, 0.0)
        self.size = Vector2(22, 23)
        self.half = self.size / 2

        self.can_jump = True
        self.is_jump  = False
        self.standing = True

class Player(Body):
    def __init__(self):
        super().__init__()

        self.time    = 0
        self.fitness = 0
        self.presses = 0