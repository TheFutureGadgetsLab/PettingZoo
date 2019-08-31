from game import Game
from renderer import Renderer

game = Game(20, 10)
game.setup_game()
game.print_map()

renderer = Renderer()

renderer.run(game)