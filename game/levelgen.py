import numpy as np
from math import ceil
from sfml.sf import Vector2
import game.defs as pz

CHUNK_SIZE = 32

class LevelGenerator():
	def __init__(self):
		self.tiles  = None
		self.chunks = None
		# TODO: NEED TO SPECIFY IN REQUIREMENTS.txt THAT NUMPY MUST BE NEW ENOUGH TO HAVE A GENERATOR
		self.rng    = None
	
	def generate_level(self, num_chunks, seed):
		""" Each chunk is 32x32 so multiply width by that to get umber of tiles.\n
			Returns (tiles, spawn_height)
		"""
		self.rng = np.random.Generator(np.random.SFC64(seed))

		self.chunks = [0] * num_chunks

		# Place start and stop chunks
		self.chunks[0]  = StartChunk(self.rng)
		self.chunks[-1] = StopChunk(self.rng)

		# Initializes all other chunks
		for i in range(1, num_chunks - 1):
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

		return np.flipud(self.tiles), self.tiles.shape[0] - self.chunks[0].ground_height - 1

class Chunk():
	def __init__(self, rng):
		self.tiles = np.zeros(shape=(CHUNK_SIZE, CHUNK_SIZE), dtype=np.int32)
		self.rng   = rng

		self.ground_height = None
		self.gaps = []
		self.platforms = []
		self.obstacles = []

	def generate_floor(self):
		self.ground_height = self.rng.integers(low=2, high=7)

		self.tiles[:self.ground_height - 1, :] = pz.DIRT
		self.tiles[self.ground_height - 1, :] = pz.GRASS

	def generate_gaps(self, prob=0.15, start=0, stop=CHUNK_SIZE):
		x = start
		while x < stop:
			gap_width = self.rng.integers(low=2, high=7)
			if x + gap_width >= stop:
				break

			# Insert gap
			if self.rng.random() < prob:
				self.gaps.append( (x, gap_width) )
				self.tiles[:self.ground_height, x:x+gap_width] = pz.EMPTY
				x += gap_width

			x += 1
		
	def generate_platforms(self, prob=0.15, start=0):
		x = start
		while x < CHUNK_SIZE:
			plat_width  = self.rng.integers(low=5, high=10)
			plat_height = self.rng.integers(low=1, high=6) # Height from ground height
			plat_type   = self.rng.choice([pz.SPIKE_BOT, pz.SPIKE_TOP, pz.COBBLE])

			if x + plat_width >= CHUNK_SIZE:
				plat_width = CHUNK_SIZE - x - 1

			# Insert plat if 
			if self.rng.random() < prob:
				self.platforms.append( (x, plat_width, plat_height) )
				self.tiles[self.ground_height + plat_height, x:x+plat_width] = plat_type
				x += plat_width

			x += 1

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
