import random
import game.defs as pz
import numpy as np
from random import choice

MAX_HEIGHT = 32

class Start():
	def __init__(self):
		self.tiles = np.zeros((MAX_HEIGHT, 5))

		self.tiles[:5, :] = pz.DIRT
		self.tiles[5, :]  = pz.GRASS

		self.tiles = np.flipud(self.tiles)
	
class Flat():
	def __init__(self):
		self.tiles = np.zeros((MAX_HEIGHT, 10))

		self.tiles[:5, :] = pz.DIRT
		self.tiles[5, :]  = pz.GRASS

		self.tiles = np.flipud(self.tiles)
	
class Gap():
	def __init__(self):
		self.tiles = np.zeros((MAX_HEIGHT, 7))
	
class Pipe():
	def __init__(self):
		self.tiles = np.zeros((MAX_HEIGHT, 13))

		self.tiles[:5, :]   = pz.DIRT
		self.tiles[5,  :]   = pz.GRASS
		self.tiles[5, 1:-1] = pz.DIRT

		self.tiles[6, 1:-1] = pz.GRASS
		self.tiles[6, 2:-2] = pz.DIRT

		self.tiles[7, 2:-2] = pz.GRASS
		self.tiles[7, 3:-3] = pz.DIRT

		self.tiles[8, 3:-3] = pz.GRASS
		self.tiles[8, 4:-4] = pz.DIRT
		
		self.tiles[:, 4:9]  = pz.EMPTY

		self.tiles[:, 6]    = pz.PIPE_MID

		offset = random.randint(-3, 4)
		width  = random.randint(2, 3)

		self.tiles[8+offset-1:8+offset+width+1, 6] = pz.EMPTY

		self.tiles[8+offset-1, 6] = pz.PIPE_BOT
		self.tiles[8+offset+width, 6] = pz.PIPE_TOP

		self.tiles = np.flipud(self.tiles)

grammar = {
	Start: [Flat],
	Flat: [Flat, Gap, Pipe],
	Gap:  [Flat, Pipe],
	Pipe: [Flat, Gap]
}

def genString(n):
	string = [Start,]
	for i in range(n):
		prev = string[i]
		choices = grammar[prev]
		next_ = choice(choices)
		
		string.append(next_)

	return string