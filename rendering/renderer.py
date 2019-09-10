from sfml import sf
from sfml.sf import Vector2
import game.defs as pz
import numpy as np
from math import ceil
import time

asset_files = {
    pz.COIN: "assets/coin.png",
    pz.CAT: "assets/cat.png",
    pz.LAMP: "assets/lamp.png",
}

class Renderer():
    def __init__(self, width=800, height=600):
        self.window_size = sf.Vector2(width, height)
        self.window = sf.RenderWindow(sf.VideoMode(*self.window_size), "PettingZoo")
        self.window.key_repeat_enabled = False
        self.window.framerate_limit = 60

        self.player  = None
        self.level_tilemap = None
        self.grid_tilemap = None
        self.textures = {}
        self.font = None
        self.debug_hud_text  = None
        self.keys = [0, 0, 0]
        self.zoom = 1

        self.show_debug = False
        self.show_grid  = False

        self.load_assets()

        self.running = True

    def load_assets(self):
        # Textures not in spritesheet
        for id, path in asset_files.items():
            self.textures[id] = sf.Texture.from_file(asset_files[id])

        # Text/Font
        self.font = sf.Font.from_file("assets/SourceCodePro-Regular.otf")

        self.debug_hud_text = sf.Text(font=self.font)
        self.debug_hud_text.color = sf.Color.BLACK
        self.debug_hud_text.scale(Vector2(0.5, 0.5))

        self.hud_text = sf.Text(font=self.font)
        self.hud_text.color = sf.Color.BLACK
        self.hud_text.scale(Vector2(0.5, 0.5))

        self.player = sf.Sprite(self.textures[pz.CAT])
        self.player.origin = self.textures[pz.CAT].size / 2.0

    def get_input(self):
        """ Returns a list of directional keys pressed\n
            Handles renderer specific events as well\n
            (restart game, new game, resize, grid, debug overlay, etc)
        """

        for event in self.window.events:
            if event == sf.Event.CLOSED:
                self.running = False

            if event == sf.Event.RESIZED:
                self.zoom = ceil((event['width'] + event['height']) / (1280 + 720))
                self.window.view.size = (event['width'] / self.zoom, event['height'] / self.zoom)

            if event in [sf.Event.KEY_PRESSED, sf.Event.KEY_RELEASED]:
                pressed = event == sf.Event.KEY_PRESSED
                key = event['code']

                # Renderer specific
                if key in [sf.Keyboard.ESCAPE]:
                    self.running = False
                if key in [sf.Keyboard.I]:
                    self.show_debug ^= pressed
                if key in [sf.Keyboard.G]:
                    self.show_grid ^= pressed
                
                # Movement
                if key in [sf.Keyboard.RIGHT, sf.Keyboard.D]:
                    self.keys[pz.RIGHT] = pressed
                if key in [sf.Keyboard.LEFT, sf.Keyboard.A]:
                    self.keys[pz.LEFT] = pressed
                if key in [sf.Keyboard.UP, sf.Keyboard.W, sf.Keyboard.SPACE, sf.Keyboard.W]:
                    self.keys[pz.JUMP] = pressed
        
        return self.keys

    def draw_overlay(self, game):
        if self.show_debug:
            self.debug_hud_text.string = (
                f"Player pos: {game.player.pos}\n"
                f"Player vel: ({game.player.vel.x:.1f}, {game.player.vel.y:.1f})\n"
                f"Tile: {game.player.tile}\n"
                f"Seed: {game.map_seed}"
            )
            self.debug_hud_text.position = self.window.view.center - self.window.view.size / 2
            self.window.draw(self.debug_hud_text)
        
        self.hud_text.string = (
            f"Time:    {game.player.time:06.2f}\n"
            f"Fitness: {int(game.player.fitness)}\n"
            f"{('←' if self.keys[pz.LEFT] else ''):<5}"
            f"{('↑' if self.keys[pz.JUMP] else ''):<5}"
            f"{('→' if self.keys[pz.RIGHT] else ''):<5}"
        )
        self.hud_text.position = Vector2(self.window.view.center.x, self.window.view.center.y - self.window.view.size.y / 2 )
        textRect = self.hud_text.local_bounds
        self.hud_text.origin = (textRect.left + textRect.width / 2.0, 0)
        self.window.draw(self.hud_text)

    def adjust_camera(self, game):
        center = self.player.position
        lbound = center.x - self.window.view.size.x / 2
        rbound = center.x + self.window.view.size.x / 2
        bbound = center.y + self.window.view.size.y / 2

        if bbound > game.height * pz.TILE_SIZE:
            center.y = game.height * pz.TILE_SIZE - self.window.view.size.y / 2
        
        if lbound < 0:
            center.x = self.window.view.size.x / 2

        if rbound > game.width * pz.TILE_SIZE:
            center.x = game.width * pz.TILE_SIZE - self.window.view.size.x / 2

        self.window.view.center = center
    
    def new_game_setup(self, game):
        """ Must call this when running a new game!
        """
        self.level_tilemap = TileMap(game.tiles)
        grid_ids = np.ndarray(shape=game.tiles.shape, dtype=np.int32)
        grid_ids[:, :] = pz.GRID
        self.grid_tilemap = TileMap(grid_ids)
    
    def draw_state(self, game):
        self.player.position = game.player.pos
        self.adjust_camera(game)

        self.window.clear(sf.Color(135, 206, 235))
        
        self.window.draw(self.level_tilemap)
        
        if self.show_grid:
            self.window.draw(self.grid_tilemap)
        
        self.window.draw(self.player)
        self.draw_overlay(game)
        self.window.display()

class TileMap(sf.Drawable):
    def __init__(self, tiles):
        super().__init__()
        
        self.m_tileset  = sf.Texture.from_file("assets/spritesheet.png")
        self.m_vertices = sf.VertexArray(sf.PrimitiveType.QUADS, tiles.size * 4)

        for i in range(tiles.shape[1]):
            for j in range(tiles.shape[0]):
                # get the current tile number
                id = tiles[j, i]

                # find its position in the tileset texture
                tu = int(id % (self.m_tileset.width / pz.TILE_SIZE))
                tv = int(id / (self.m_tileset.width / pz.TILE_SIZE))

                # get a pointer to the current tile's quad
                index = (i + j * tiles.shape[1]) * 4
                # define its 4 corners
                self.m_vertices[index + 0].position = (i * pz.TILE_SIZE, j * pz.TILE_SIZE)
                self.m_vertices[index + 1].position = ((i + 1) * pz.TILE_SIZE, j * pz.TILE_SIZE)
                self.m_vertices[index + 2].position = ((i + 1) * pz.TILE_SIZE, (j + 1) * pz.TILE_SIZE)
                self.m_vertices[index + 3].position = (i * pz.TILE_SIZE, (j + 1) * pz.TILE_SIZE)

                # define its 4 texture coordinates
                self.m_vertices[index + 0].tex_coords = (tu * pz.TILE_SIZE, tv * pz.TILE_SIZE);
                self.m_vertices[index + 1].tex_coords = ((tu + 1) * pz.TILE_SIZE, tv * pz.TILE_SIZE);
                self.m_vertices[index + 2].tex_coords = ((tu + 1) * pz.TILE_SIZE, (tv + 1) * pz.TILE_SIZE);
                self.m_vertices[index + 3].tex_coords = (tu * pz.TILE_SIZE, (tv + 1) * pz.TILE_SIZE);

    def draw(self, target, states):
        states.texture = self.m_tileset
        target.draw(self.m_vertices, states)
