import numpy as np
from math import ceil
from sfml.sf import Vector2
import defs as pz

CHUNK_SIZE      = 32
START_PLAT_LEN  = 5
FINISH_PLAT_LEN = 5

class LevelGenerator():
    def __init__(self):
        self.tiles = None

    def generate_level(self, width, height, seed):
        np.random.seed(seed)

        num_chunks = ceil(width / CHUNK_SIZE)
        width = num_chunks * CHUNK_SIZE + START_PLAT_LEN + FINISH_PLAT_LEN

        self.tiles = np.zeros(shape=(height, width), dtype=np.int32)

        ground_height = self.set_start_chunk()
        spawn_height = ground_height + 1

        return np.flipud(self.tiles), height - spawn_height

    def set_start_chunk(self):
        ground_height = np.random.randint(2, 7)
        
        # Place starting platform
        self.tiles[0:ground_height, 0:START_PLAT_LEN] = pz.DIRT

        return ground_height
    
    def set_flat_chunk(self, start):
        pass
