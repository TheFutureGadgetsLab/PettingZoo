from sfml import sf
import defs as pz
import pysnooper

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
        self.window.framerate_limit = 60
        
        self.textures = {}
        self.load_assets()

        self.player = sf.Sprite(self.textures[pz.LAMP])
        self.player.position = (0, 0)
        self.running = True

        self.tilemap = None

    def load_assets(self):
        for id, path in asset_files.items():
            self.textures[id] = sf.Texture.from_file(asset_files[id])
        
    def handle_input(self):
        keys = [0, 0, 0]
        for event in self.window.events:
            if event == sf.Event.CLOSED:
                self.running = False
                
            if event == sf.Event.KEY_PRESSED:
                if sf.Keyboard.is_key_pressed(sf.Keyboard.ESCAPE):
                    self.running = False
                if self.check_key_list([sf.Keyboard.RIGHT, sf.Keyboard.D]):
                    keys[pz.RIGHT] = 1
                elif self.check_key_list([sf.Keyboard.LEFT, sf.Keyboard.A]):
                    keys[pz.LEFT] = 1
                if self.check_key_list([sf.Keyboard.UP, sf.Keyboard.W, sf.Keyboard.SPACE]):
                    keys[pz.JUMP] = 1
        
        return keys

    def check_key_list(self, keys):
        return True in [sf.Keyboard.is_key_pressed(key) for key in keys]

    def draw_map(self, game):
        map = game.get_map()
        for row in range(map.shape[0]):
            for col in range(map.shape[1]):
                x = col * 32
                y = row * 32

                self.draw_tile(map[row, col], x, y)
                self.draw_tile(pz.GRID, x, y)
    
    def draw_tile(self, id, x, y):
        if id == 0:
            return

        sprite = sf.Sprite(self.textures[id])
        sprite.position = (x, y)

        self.window.draw(sprite)

    def run(self, game):
        """ Begins rendering game and advances gameloop
        """
        self.tilemap = TileMap(game)
        
        while self.running:
            keys = self.handle_input()
            ret = game.update(keys)

            if ret in [pz.PLAYER_DEAD, pz.PLAYER_TIMEOUT, pz.PLAYER_COMPLETE]:
                game.setup_game()
                continue

            self.player.position = (game.player.px, game.player.py)

            self.window.clear(sf.Color(135, 206, 235))
            
            # self.draw_map(game)
            self.window.draw(self.tilemap)
            self.window.draw(self.player)

            self.window.display()
            