import numpy as np
import game.defs as pz
from pymunk.vec2d import Vec2d
from game.obstacles import *

CHUNK_SIZE = 32

class Level():
	def __init__(self, num_chunks, seed, view_size=None):
		self.tiles      = None
		self.num_chunks = num_chunks

		self.padded_tiles = None

		self.size = Vec2d(CHUNK_SIZE * num_chunks, CHUNK_SIZE)
		self.spawn_point = None

		self.seed = seed
		self.rng = np.random.Generator(np.random.SFC64(seed))

		self.view_size = view_size 
		self.setup()

	def setup(self):
		self.generate()

		if self.view_size != None:
			self.padded_tiles = pad_tiles(self.tiles, self.view_size)
		
	def generate(self):
		n = 10
		string = genString(n)

		sections = []
		for s in string:
			section = s()
			sections.append(section.tiles)
		
		self.tiles = np.hstack(sections)
		self.spawn_point = Vec2d(2, 32-6)

	def tile_solid(self, row, col):
		if col < 0 or col >= self.size.x:
			return True
		if row >= self.size.y:
			return False

		return self.tiles[row, col] in pz.SOLID_TILES
	
	def get_player_view(self, loc):
		""" Returns a numpy matrix of size (in_h, in_w) of tiles around the player\n
			Player is considered to be in the 'center'
		"""

		lbound = int(loc.x)
		rbound = int(loc.x + 2 * (self.view_size.y // 2) + 1)
		ubound = int(loc.y)
		bbound = int(loc.y + 2 * (self.view_size.x // 2) + 1)

		view = self.padded_tiles[ubound:bbound, lbound:rbound]

		return view

def pad_tiles(arr, view_size):
    tile_r, tile_c = arr.shape

    p_c = view_size.x // 2
    p_r = view_size.y // 2

    pad_shape = (tile_r + 2 * p_r, tile_c + 2 * p_c)
    pad = np.zeros(shape=pad_shape, dtype=np.uint8)

    pad[:, :p_c]  = pz.COBBLE # Left
    pad[:, -p_c:] = pz.COBBLE # Right
    pad[:p_c, :]  = pz.SPIKE_BOT # Top
    pad[-p_c:, :] = pz.SPIKE_TOP # Bottom

    pad[p_r: p_r + tile_r, p_c: p_c + tile_c] = arr
    
    return pad