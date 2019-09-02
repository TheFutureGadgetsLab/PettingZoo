from game import Game
from renderer import Renderer

game = Game(100, 30)
game.setup_game()

renderer = Renderer()

renderer.run(game)