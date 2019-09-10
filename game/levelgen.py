import numpy as np
from math import ceil
from sfml.sf import Vector2
import game.defs as pz

CHUNK_SIZE      = 32

class LevelGenerator():
	def __init__(self):
		self.tiles  = None
		self.chunks = None
		self.width  = None
		self.height = None

	def generate_level(self, width, seed):
		""" Width describes the number of chunks in a level, not the number of tiles.\
			Each chunk is 32x32 so multiply width by that to get umber of tiles.\n
			Returns (tiles, spawn_height)
		"""
		np.random.seed(seed)

		self.chunks = [0] * width

		# Place start and stop chunks
		self.chunks[0]  = StartChunk()
		self.chunks[-1] = StopChunk()

		# Initializes all other chunks
		for i in range(1, width - 1):
			self.chunks[i] = Chunk()

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

		return np.flipud(self.tiles), self.tiles.shape[0] - self.chunks[0].ground_height - 1

class Chunk():
	def __init__(self):
		self.tiles = np.zeros(shape=(CHUNK_SIZE, CHUNK_SIZE), dtype=np.int32)

		self.ground_height = None
		self.gaps = []
		self.platforms = []
		self.obstacles = []

	def generate_floor(self):
		self.ground_height = np.random.randint(2, 7)

		self.tiles[:self.ground_height - 1, :] = pz.DIRT
		self.tiles[self.ground_height - 1, :] = pz.GRASS

	def generate_gaps(self, prob=0.15, start=0, stop=CHUNK_SIZE):
		x = start
		while x < stop:
			gap_width = np.random.randint(2, 7)
			if x + gap_width >= stop:
				break

			# Insert gap
			if np.random.ranf() < prob:
				self.gaps.append( (x, gap_width) )
				self.tiles[:self.ground_height, x:x+gap_width] = pz.EMPTY
				x += gap_width

			x += 1
		
	def generate_platforms(self, prob=0.15, start=0):
		x = start
		while x < CHUNK_SIZE:
			plat_width  = np.random.randint(5, 10)
			plat_height = np.random.randint(1, 6) # Height from ground height
			plat_type   = np.random.choice([pz.SPIKE_BOT, pz.SPIKE_TOP, pz.COBBLE])

			if x + plat_width >= CHUNK_SIZE:
				plat_width = CHUNK_SIZE - x - 1

			# Insert plat if 
			if np.random.ranf() < prob:
				self.platforms.append( (x, plat_width, plat_height) )
				self.tiles[self.ground_height + plat_height, x:x+plat_width] = plat_type
				x += plat_width

			x += 1

class StartChunk(Chunk):
	start_plat_len = 5
	def __init__(self):
		super().__init__()

	def generate_gaps(self, start=start_plat_len, **kwargs):
		super().generate_gaps(**kwargs, start=self.start_plat_len)

class StopChunk(Chunk):
	stop_plat_len = 5

	def __init__(self):
		super().__init__()

	def generate_floor(self):
		super().generate_floor()
		self.tiles[:self.ground_height - 1, -self.stop_plat_len:] = pz.FINISH_BOT
		self.tiles[self.ground_height - 1, -self.stop_plat_len:] = pz.FINISH_TOP

	def generate_gaps(self, stop=stop_plat_len, **kwargs):
		super().generate_gaps(**kwargs, stop=CHUNK_SIZE - self.stop_plat_len)
