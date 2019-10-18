from sfml import sf
from sfml.sf import Vector2 as SFVector2
import game.core.defs as pz
from math import ceil

asset_files = {
    pz.CAT: "game/assets/cat.png",
    pz.LAMP: "game/assets/lamp.png",
    pz.SQUARE: "game/assets/grid.png",
}

class Renderer():
    RESTART  = 1 # Restart game
    NEW_GAME = 2 # Generate new game (new seed)

    def __init__(self, width=800, height=600):
        self.window_size = SFVector2(width, height)
        self.window = sf.RenderWindow(sf.VideoMode(*self.window_size), "PettingZoo")
        self.window.key_repeat_enabled = False
        self.window.framerate_limit = 60

        self.player  = None
        self.level_tilemap = None
        self.tilegrid = None
        self.textures = {}
        self.font = None
        self.debug_hud_text  = None
        self.keys = [0, 0, 0]
        self.zoom = 1

        self.show_debug = False
        self.show_grid  = False
        self.show_help  = True

        self.load_assets()

        self.running = True

        self.game_seed = None

    def load_assets(self):
        # Textures not in spritesheet
        for id, path in asset_files.items():
            self.textures[id] = sf.Texture.from_file(asset_files[id])

        # Text/Font
        self.font = sf.Font.from_file("game/assets/SourceCodePro-Regular.otf")

        self.debug_hud_text = sf.Text(font=self.font)
        self.debug_hud_text.color = sf.Color.BLACK
        self.debug_hud_text.scale(SFVector2(0.5, 0.5))

        self.hud_stat_text = sf.Text(font=self.font)
        self.hud_stat_text.color = sf.Color.BLACK
        self.hud_stat_text.scale(SFVector2(0.5, 0.5))

        self.hud_help_text = sf.Text(font=self.font)
        self.hud_help_text.color = sf.Color.BLACK
        self.hud_help_text.scale(SFVector2(0.4, 0.4))

        self.textures[pz.SQUARE].repeated = True

        self.player = sf.Sprite(self.textures[pz.CAT])
        self.player.origin = self.textures[pz.CAT].size / 2.0

    def get_input(self):
        """ Returns a list of directional keys pressed\n
            Handles renderer specific events as well\n
            (restart game, new game, resize, grid, debug overlay, etc)
        """
        game_req = None # Ask to generate / restart game

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

                # Handle restarting / Generating a new game
                if event['code'] in [sf.Keyboard.R, sf.Keyboard.N]:
                    game_req = Renderer.RESTART if event['code'] == sf.Keyboard.R else Renderer.NEW_GAME

                if event['code'] == sf.Keyboard.H:
                    self.show_help ^= pressed
        
        return self.keys, game_req

    def draw_overlay(self, game, keys):
        textRect = self.hud_stat_text.local_bounds
        self.hud_stat_text.origin = (textRect.left + textRect.width / 2.0, 0)
        textRect = self.hud_help_text.local_bounds
        self.hud_help_text.origin = (textRect.left + textRect.width, 0)

        if self.show_debug:
            self.debug_hud_text.string = (
                f"Player pos: {game.player.pos}\n"
                f"Player vel: ({game.player.vel.x:.1f}, {game.player.vel.y:.1f})\n"
                f"Tile: {game.player.tile}\n"
                f"Seed: {game.level.seed}\n"
                f"Num chunks: {game.level.num_chunks}"
            )
            self.debug_hud_text.position = self.window.view.center - self.window.view.size / 2
            self.window.draw(self.debug_hud_text)

        if self.show_help:
            self.hud_help_text.string = (
                f"Arrow keys, WASD, space\n"
                f"all do what you expect.\n"
                f"  R: Restart level\n"
                f"  N: Generate new level\n"
                f"  I: Debug info\n"
                f"  G: Display grid\n"
                f"  H: Toggle help info\n"
            )
            self.hud_help_text.position = (self.window.view.center.x + self.window.view.size.x / 2 - 10, self.window.view.center.y - self.window.view.size.y / 2)
            self.window.draw(self.hud_help_text)
        
        self.hud_stat_text.string = (
            f"Time:    {game.player.time:06.2f}\n"
            f"Fitness: {int(game.player.fitness)}\n"
            f"{('←' if keys[pz.LEFT] else ''):<5}"
            f"{('↑' if keys[pz.JUMP] else ''):<5}"
            f"{('→' if keys[pz.RIGHT] else ''):<5}"
        )
        self.hud_stat_text.position = SFVector2(self.window.view.center.x, self.window.view.center.y - self.window.view.size.y / 2 )
        self.window.draw(self.hud_stat_text)


    def adjust_camera(self, game):
        center = Vector2_to_SFML(self.player.position)
        lbound = center.x - self.window.view.size.x / 2
        rbound = center.x + self.window.view.size.x / 2
        bbound = center.y + self.window.view.size.y / 2

        if bbound > game.level.size.y * pz.TILE_SIZE:
            center.y = game.level.size.y * pz.TILE_SIZE - self.window.view.size.y / 2
        
        if lbound < 0:
            center.x = self.window.view.size.x / 2

        if rbound > game.level.size.x * pz.TILE_SIZE:
            center.x = game.level.size.x * pz.TILE_SIZE - self.window.view.size.x / 2

        self.window.view.center = center
    
    def new_game_setup(self, game):
        """ Must call this when running a new game!
        """
        self.level_tilemap = TileMap(game.level.tiles)
        self.tilegrid = sf.Sprite(self.textures[pz.SQUARE])
        self.tilegrid.texture_rectangle = sf.Rect((0, 0), 
            (game.level.size.x * pz.TILE_SIZE, game.level.size.y * pz.TILE_SIZE))
        self.tilegrid.color = sf.Color(255, 255, 255, 50)
    
    def draw_state(self, game, keys):
        self.player.position = Vector2_to_SFML(game.player.pos)
        self.adjust_camera(game)

        self.window.clear(sf.Color(135, 206, 235))
        
        self.window.draw(self.level_tilemap)
        
        if self.show_grid:
            self.window.draw(self.tilegrid)
        
        self.window.draw(self.player)
        self.draw_overlay(game, keys)
        self.window.display()

class TileMap(sf.Drawable):
    def __init__(self, tiles):
        super().__init__()
        
        self.m_tileset  = sf.Texture.from_file("game/assets/spritesheet.png")
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

def Vector2_to_SFML(vec):
    return SFVector2(vec.x, vec.y)