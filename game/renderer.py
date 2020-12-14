from pygame import Vector2
import pygame as pg
import game.defs as pz
from game.RenderObjs import *

pg.init()
pg.event.set_allowed([pg.QUIT, pg.KEYDOWN, pg.KEYUP])

asset_files = {
    pz.CAT: "game/assets/cat.png",
    pz.LAMP: "game/assets/lamp.png",
    pz.SQUARE: "game/assets/grid.png",
}

class Renderer():
    RESTART  = 1 # Restart game
    NEW_GAME = 2 # Generate new game (new seed)

    def __init__(self):
        self.canvas = pygame.Surface((1280, 736))
        self.window = pg.display.set_mode((1280, 736), pg.DOUBLEBUF | pg.HWSURFACE | pg.RESIZABLE)
        pg.display.set_caption("PettingZoo")

        self.player = None

        self.level_tilemap = None
        self.sheet = None
        self.textures = {}
        
        self.debug_hud_text  = None
        self.keys = [0, 0, 0]
        self.zoom = 1

        self.show_debug = False
        self.show_grid  = False
        self.show_help  = True

        self.load_assets()

        self.running = True

        self.game_seed = None

        self.clock = pygame.time.Clock()

    def load_assets(self):
        # Textures not in spritesheet
        for id_, _ in asset_files.items():
            self.textures[id_] = pg.image.load(asset_files[id_]).convert_alpha()

        self.debug_hud_text = Text(self.window, "topleft")
        self.hud_stat_text  = Text(self.window, "topCenter")
        self.hud_help_text  = Text(self.window, "topRight")

        self.player = Player(self.textures[pz.CAT])

        sheet = pg.image.load("game/assets/spritesheet.png").convert_alpha()
        w = int(sheet.get_size()[0] / pz.TILE_SIZE)
        h = int(sheet.get_size()[1] / pz.TILE_SIZE)
        for r in range(h):
            for c in range(w):
                image = pygame.Surface([pz.TILE_SIZE, pz.TILE_SIZE], pygame.SRCALPHA, 32)
                image.blit(sheet, (0, 0), (c*pz.TILE_SIZE, r*pz.TILE_SIZE, pz.TILE_SIZE, pz.TILE_SIZE))
                self.textures[r * w + c] = image

    def get_input(self):
        """ Returns a list of directional keys pressed\n
            Handles renderer specific events as well\n
            (restart game, new game, resize, grid, debug overlay, etc)
        """
        game_req = None # Ask to generate / restart game

        for event in pygame.event.get():
            if event.type == pg.QUIT:
                self.running = False

            if event.type in [pg.KEYDOWN, pg.KEYUP]:
                pressed = event.type == pg.KEYDOWN
                key = event.key

                # Renderer specific
                if key in [pg.K_ESCAPE]:
                    self.running = False
                if key in [pg.K_i]:
                    self.show_debug ^= pressed
                if key in [pg.K_g]:
                    self.show_grid ^= pressed
                
                # Movement
                if key in [pg.K_RIGHT, pg.K_d]:
                    self.keys[pz.RIGHT] = pressed
                if key in [pg.K_LEFT, pg.K_a]:
                    self.keys[pz.LEFT] = pressed
                if key in [pg.K_UP, pg.K_w, pg.K_SPACE]:
                    self.keys[pz.JUMP] = pressed

                # Handle restarting / Generating a new game
                if key in [pg.K_r, pg.K_n]:
                    game_req = Renderer.RESTART if key == pg.K_r else Renderer.NEW_GAME

                if key == pg.K_h:
                    self.show_help ^= pressed
        
        return self.keys, game_req

    def draw_overlay(self, game, keys):
        screen = self.window.get_rect()
        if self.show_debug:
            self.debug_hud_text.update(
                f"Player pos: {game.player.pos}\n"
                f"Player vel: ({game.player.vel.x:.1f}, {game.player.vel.y:.1f})\n"
                f"Tile: {game.player.tile}\n"
                f"Seed: {game.level.seed}\n"
                f"Num chunks: {game.level.num_chunks}"
            )
            self.debug_hud_text.draw()

        if self.show_help:
            self.hud_help_text.update("""\
                Arrow keys, WASD, space
                all do what you expect.
                  R: Restart level     
                  N: Generate new level
                  I: Debug info         
                  G: Display grid      
                  H: Toggle help info"""
            )
            self.hud_help_text.draw()

        
        self.hud_stat_text.update(
            f"Time:    {game.player.time:06.2f}\n"
            f"Fitness: {int(game.player.fitness)}\n"
            f"{('←' if keys[pz.LEFT]  else ''):<5}"
            f"{('↑' if keys[pz.JUMP]  else ''):<5}"
            f"{('→' if keys[pz.RIGHT] else ''):<5}"
        )
        self.hud_stat_text.draw()

    def adjust_camera(self, game):
        center = Vector2(self.player.rect.center)

        lbound = center.x - self.canvas.get_rect()[0] / 2
        rbound = center.x + self.canvas.get_rect()[0] / 2
        bbound = center.y + self.canvas.get_rect()[1] / 2

        if bbound > game.level.size.y * pz.TILE_SIZE:
            center.y = game.level.size.y * pz.TILE_SIZE - self.canvas.view.size.y / 2
        
        if lbound < 0:
            center.x = self.canvas.view.size.x / 2

        if rbound > game.level.size.x * pz.TILE_SIZE:
            center.x = game.level.size.x * pz.TILE_SIZE - self.canvas.view.size.x / 2

        return None
    
    def new_game_setup(self, game):
        """ Must call this when running a new game!
        """

        self.level_tilemap = pg.sprite.Group()
        for r in range(0, game.level.tiles.shape[0]):
            for c in range(0, game.level.tiles.shape[1]):
                id_ = game.level.tiles[r, c]
                if id_ == pz.EMPTY:
                    continue

                tile = Tile(self.textures[game.level.tiles[r, c]], pos=Vector2(c*pz.TILE_SIZE, r*pz.TILE_SIZE))
                self.level_tilemap.add(tile)

    def draw_state(self, game, keys):
        self.canvas.fill((135, 206, 235))

        self.player.update(game.player.pos)
        self.adjust_camera(game)

        self.level_tilemap.draw(self.canvas)

        if self.show_grid:
            self.canvas.draw(self.tilegrid)
        
        self.player.draw(self.canvas)
        self.clock.tick(60)

        pg.transform.scale(self.canvas, self.window.get_size(), self.window)

        self.draw_overlay(game, keys)

        pg.display.flip()