from pymunk import Vec2d
import numpy as np

def array_split(arr, n):
    Neach_section, extras = divmod(len(arr), n)
    sizes = [0] + extras*[Neach_section+1] + (n-extras)*[Neach_section]
    ranges = np.cumsum(sizes)

    return [arr[ranges[i]:ranges[i+1]] for i in range(len(ranges)-1)]

class IdleDetector():
    def __init__(self, let_idle):
        self.let_idle = let_idle

        self.time_not_moved = 0
        self.last_tile_pos = Vec2d(-1, -1)

        self.timeout_time = 60 * 6

    def update(self, tile_pos):
        if self.let_idle == True:
            return False

        if tile_pos != self.last_tile_pos:
            self.last_tile_pos = tile_pos
            self.time_not_moved = 0

            return False

        self.time_not_moved += 1

        if self.time_not_moved > self.timeout_time:
            return True