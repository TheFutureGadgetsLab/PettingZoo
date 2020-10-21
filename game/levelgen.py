import numpy as np
import game.defs as pz
from pymunk.vec2d import Vec2d

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
		self.chunks = [0] * self.num_chunks

		# Place start and stop chunks
		self.chunks[0]  = StartChunk(self.rng)
		self.chunks[-1] = StopChunk(self.rng)

		# Initializes all other chunks
		for i in range(1, self.num_chunks - 1):
			self.chunks[i] = Chunk(self.rng)

		# Generate floors
		for chunk in self.chunks:
			chunk.generate_floor()

		# Generate gaps
		for chunk in self.chunks:
			chunk.generate_gaps()

		# Generate plats
		for chunk in self.chunks:
			chunk.generate_platforms()

		self.tiles = np.hstack([chunk.tiles for chunk in self.chunks])
		self.tiles = np.flipud(self.tiles)

		self.spawn_point = Vec2d(1, self.tiles.shape[0] - self.chunks[0].ground_height - 1)

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

class Chunk():
	def __init__(self, rng):
		self.tiles = np.zeros(shape=(CHUNK_SIZE, CHUNK_SIZE), dtype=np.uint8)
		self.rng   = rng

		self.ground_height = None
		self.gaps = []
		self.platforms = []
		self.obstacles = []

	def generate_floor(self):
		self.ground_height = self.rng.integers(low=2, high=7, dtype=np.int32)

		self.tiles[:self.ground_height - 1, :] = pz.DIRT
		self.tiles[self.ground_height - 1, :] = pz.GRASS

	def generate_gaps(self, prob=0.15, start=0, stop=CHUNK_SIZE):
		x = start
		while x < stop:
			if self.rng.random(dtype=np.float32) >= prob:
				x += 1
				continue

			gap_width = self.rng.integers(low=2, high=7, dtype=np.int32)
			if x + gap_width >= stop:
				break

			self.gaps.append( (x, gap_width) )
			self.tiles[:self.ground_height, x:x+gap_width] = pz.EMPTY

			x += gap_width + 1

	def generate_platforms(self, prob=0.15, start=0):
		x = start
		while x < CHUNK_SIZE:
			if self.rng.random(dtype=np.float32) >= prob:
				x += 1
				continue

			plat_width  = self.rng.integers(low=5, high=10, dtype=np.int32)
			plat_height = self.rng.integers(low=1, high=6, dtype=np.int32) # Height from ground height
			plat_type   = self.rng.choice([pz.SPIKE_BOT, pz.SPIKE_TOP, pz.COBBLE])

			if x + plat_width >= CHUNK_SIZE:
				plat_width = CHUNK_SIZE - x - 1

			# Insert plat if
			self.platforms.append( (x, plat_width, plat_height) )
			self.tiles[self.ground_height + plat_height, x:x+plat_width] = plat_type

			x += plat_width + 1

class StartChunk(Chunk):
	start_plat_len = 5

	def __init__(self, rng):
		super().__init__(rng)

	def generate_gaps(self, start=start_plat_len, **kwargs):
		super().generate_gaps(**kwargs, start=self.start_plat_len)

class StopChunk(Chunk):
	stop_plat_len = 5

	def __init__(self, rng):
		super().__init__(rng)

	def generate_floor(self):
		super().generate_floor()

		self.tiles[:self.ground_height - 1, -self.stop_plat_len:] = pz.FINISH_BOT
		self.tiles[self.ground_height - 1, -self.stop_plat_len:] = pz.FINISH_TOP

	def generate_gaps(self, stop=stop_plat_len, **kwargs):
		super().generate_gaps(**kwargs, stop=CHUNK_SIZE - self.stop_plat_len)

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