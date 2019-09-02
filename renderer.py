from sfml import sf
from sfml.sf import Vector2
import defs as pz
import pysnooper
import numpy as np

asset_files = {
    pz.COIN     : "/home/supa/lin_storage/pettingzoo/assets/coin.png",
    pz.LAMP     : "/home/supa/lin_storage/pettingzoo/assets/lamp.png",
    pz.GRID     : "/home/supa/lin_storage/pettingzoo/assets/grid.png",
}

class TileMap(sf.Drawable):
    def __init__(self, game):
        super().__init__()
        
        self.m_tileset  = sf.Texture.from_file("/home/supa/lin_storage/pettingzoo/assets/spritesheet.png")
        self.m_tileset.smooth = True
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
        self.window.key_repeat_enabled = False
        self.window.framerate_limit = 60

        self.player  = None
        self.tilemap = None
        self.textures = {}
        self.font = None
        self.debug_hud_text  = None
        self.keys = [0, 0, 0]

        self.load_assets()

        self.running = True

    def load_assets(self):
        # Textures not in spritesheet
        for id, path in asset_files.items():
            self.textures[id] = sf.Texture.from_file(asset_files[id])

        # Text/Font
        self.font = sf.Font.from_file("/home/supa/lin_storage/pettingzoo/assets/Vera.ttf")
        self.debug_hud_text = sf.Text(font=self.font)
        self.debug_hud_text.color = sf.Color.BLACK

        self.player = sf.Sprite(self.textures[pz.LAMP])

    def handle_input(self):
        for event in self.window.events:
            if event == sf.Event.CLOSED:
                self.running = False
                
            if event in [sf.Event.KEY_PRESSED, sf.Event.KEY_RELEASED]:
                pressed = event == sf.Event.KEY_PRESSED

                if event['code'] in [sf.Keyboard.ESCAPE]:
                    self.running = False
                if event['code'] in [sf.Keyboard.RIGHT, sf.Keyboard.D]:
                    self.keys[pz.RIGHT] = pressed
                if event['code'] in [sf.Keyboard.LEFT, sf.Keyboard.A]:
                    self.keys[pz.LEFT] = pressed
                if event['code'] in [sf.Keyboard.UP, sf.Keyboard.W, sf.Keyboard.SPACE, sf.Keyboard.W]:
                    self.keys[pz.JUMP] = pressed

    def draw_grid(self, game):
        sprite = sf.Sprite(self.textures[pz.GRID])
        for row in range(game.tiles.shape[0]):
            for col in range(game.tiles.shape[1]):
                sprite.position = Vector2(col, row) * pz.TILE_SIZE

                self.window.draw(sprite)

    def draw_overlay(self, game):
        self.debug_hud_text.string = f"Lamp pos: {game.player.pos}\nLamp vel: {game.player.vel} \nTile: {game.player.tile}"
        self.debug_hud_text.position = self.window.view.center - self.window.view.size / 2
        self.window.draw(self.debug_hud_text)

    @profile
    def run(self, game):
        """ Begins rendering game and advances gameloop
        """
        self.tilemap = TileMap(game)

        while self.running:
            self.handle_input()

            ret = game.update(self.keys)

            if ret in [pz.PLAYER_DEAD, pz.PLAYER_TIMEOUT, pz.PLAYER_COMPLETE]:
                game.setup_game()
                continue

            self.player.position = game.player.pos
            self.window.view.center = self.player.position

            self.window.clear(sf.Color(135, 206, 235))
            
            self.window.draw(self.tilemap)
            # self.draw_grid(game)
            self.window.draw(self.player)
            self.draw_overlay(game)

            self.window.display()