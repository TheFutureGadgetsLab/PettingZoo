from sfml import sf
import defs as pz
import pysnooper
import numpy as np
import time

asset_files = {
    pz.COBBLE   : "/home/supa/lin_storage/pettingzoo/assets/cobble.png",
    pz.COIN     : "/home/supa/lin_storage/pettingzoo/assets/coin.png",
    pz.DIRT     : "/home/supa/lin_storage/pettingzoo/assets/dirt.png",
    pz.GRASS    : "/home/supa/lin_storage/pettingzoo/assets/grass.png",
    pz.GRID     : "/home/supa/lin_storage/pettingzoo/assets/grid.png",
    pz.LAMP     : "/home/supa/lin_storage/pettingzoo/assets/lamp.png",
    pz.PIPE_BOT : "/home/supa/lin_storage/pettingzoo/assets/pipe_bottom.png",
    pz.PIPE_MID : "/home/supa/lin_storage/pettingzoo/assets/pipe_middle.png",
    pz.PIPE_TOP : "/home/supa/lin_storage/pettingzoo/assets/pipe_top.png",
    pz.SPIKE_BOT: "/home/supa/lin_storage/pettingzoo/assets/spike_bottom.png",
    pz.SPIKE_TOP: "/home/supa/lin_storage/pettingzoo/assets/spike_top.png",
    pz.LAMP     : "/home/supa/lin_storage/pettingzoo/assets/lamp.png",
    pz.BG       : "/home/supa/lin_storage/pettingzoo/assets/bg.png",
    pz.GRID     : "/home/supa/lin_storage/pettingzoo/assets/grid.png",
}

class TileMap(sf.Drawable):
    def __init__(self, game):
        super().__init__()
        
        self.m_tileset  = sf.Texture.from_file("/home/supa/lin_storage/pettingzoo/assets/spritesheet.png")
        self.m_vertices = sf.VertexArray(sf.PrimitiveType.QUADS, game.width * game.height * 4)

        for i in range(game.width):
            for j in range(game.height):
                # get the current tile number
                id = game.tiles[j, i]

                # find its position in the tileset texture
                tu = int(id % (self.m_tileset.width / (pz.TILE_SIZE + 2)))
                tv = int(id / (self.m_tileset.width / (pz.TILE_SIZE + 2)))

                # get a pointer to the current tile's quad
                index = (i + j * game.width) * 4
                # define its 4 corners
                self.m_vertices[index + 0].position = (i * pz.TILE_SIZE, j * pz.TILE_SIZE)
                self.m_vertices[index + 1].position = ((i + 1) * pz.TILE_SIZE, j * pz.TILE_SIZE)
                self.m_vertices[index + 2].position = ((i + 1) * pz.TILE_SIZE, (j + 1) * pz.TILE_SIZE)
                self.m_vertices[index + 3].position = (i * pz.TILE_SIZE, (j + 1) * pz.TILE_SIZE)

                # define its 4 texture coordinates
                self.m_vertices[index + 0].tex_coords = (tu * (pz.TILE_SIZE + 2) + 1, tv * (pz.TILE_SIZE + 2) + 1);
                self.m_vertices[index + 1].tex_coords = ((tu + 1) * (pz.TILE_SIZE + 2) - 1, tv * (pz.TILE_SIZE + 2) + 1);
                self.m_vertices[index + 2].tex_coords = ((tu + 1) * (pz.TILE_SIZE + 2) - 1, (tv + 1) * (pz.TILE_SIZE + 2) - 1);
                self.m_vertices[index + 3].tex_coords = (tu * (pz.TILE_SIZE + 2) + 1, (tv + 1) * (pz.TILE_SIZE + 2) - 1);

    def draw(self, target, states):
        states.texture = self.m_tileset
        target.draw(self.m_vertices, states)

class Renderer():
    def __init__(self, width=800, height=600):
        self.window_size = sf.Vector2(width, height)
        self.window = sf.RenderWindow(sf.VideoMode(*self.window_size), "PettingZoo")

        self.textures = {}
        self.load_assets()

        self.player = sf.Sprite(self.textures[pz.LAMP])
        self.player.position = (0, 0)
        self.running = True

        self.tilemap = None

        self.font = sf.Font.from_file("/home/supa/lin_storage/pettingzoo/assets/Vera.ttf")
        self.pos_text  = sf.Text(font=self.font)
        self.vel_text  = sf.Text(font=self.font)
        self.tile_text = sf.Text(font=self.font)
        self.vel_text.move((0, 25))
        self.tile_text.move((0, 50))

        self.keys = [0, 0, 0]

    def load_assets(self):
        for id, path in asset_files.items():
            self.textures[id] = sf.Texture.from_file(asset_files[id])
        
    def handle_input(self):
        # for event in self.window.events:
        while True:
            event = self.window.poll_event()
            if not event:
                break
            if event == sf.Event.CLOSED:
                self.running = False
                
            if event == sf.Event.KEY_PRESSED:
                print("Event")
                if sf.Keyboard.is_key_pressed(sf.Keyboard.ESCAPE):
                    self.running = False
                if self.check_key_list([sf.Keyboard.RIGHT, sf.Keyboard.D]):
                    self.keys[pz.RIGHT] = 1
                if self.check_key_list([sf.Keyboard.LEFT, sf.Keyboard.A]):
                    self.keys[pz.LEFT] = 1
                if self.check_key_list([sf.Keyboard.UP, sf.Keyboard.W, sf.Keyboard.SPACE, sf.Keyboard.W]):
                    self.keys[pz.JUMP] = 1

    def check_key_list(self, keys):
        return True in [sf.Keyboard.is_key_pressed(key) for key in keys]

    def draw_grid(self, game):
        for row in range(game.tiles.shape[0]):
            for col in range(game.tiles.shape[1]):
                x = col * 32
                y = row * 32

                self.draw_tile(pz.GRID, x, y)
    
    def draw_tile(self, id, x, y):
        if id == 0:
            return

        sprite = sf.Sprite(self.textures[id])
        sprite.position = (x, y)

        self.window.draw(sprite)

    def draw_overlay(self, game):
        self.pos_text.string  = f"Lamp pos: {self.player.position}"
        self.vel_text.string  = f"Lamp vel: {game.player.vx:.2f}, {game.player.vy:.2f}"
        self.tile_text.string = f"Tile:     {game.player.tile_x, game.player.tile_y}"

        self.window.draw(self.pos_text)
        self.window.draw(self.vel_text)
        self.window.draw(self.tile_text)

    # @profile
    def run(self, game):
        """ Begins rendering game and advances gameloop
        """
        self.tilemap = TileMap(game)

        self.view = sf.View(sf.Rect((0, 0), (pz.TILE_SIZE * game.width, pz.TILE_SIZE * game.height)))
        
        last_draw = time.perf_counter()
        while self.running:
            self.keys = [0, 0, 0]
            self.handle_input()
            while (time.perf_counter() - last_draw) < (1.0 / pz.UPDATES_PS):
                self.handle_input()

            print(self.keys)
            ret = game.update(self.keys)

            if ret in [pz.PLAYER_DEAD, pz.PLAYER_TIMEOUT, pz.PLAYER_COMPLETE]:
                game.setup_game()
                continue

            self.player.position = (game.player.px, game.player.py)

            self.window.view = self.view
            self.window.clear(sf.Color(135, 206, 235))
            
            self.window.draw(self.tilemap)
            self.window.draw(self.player)
            self.draw_overlay(game)

            self.window.display()
            last_draw = time.perf_counter()

            