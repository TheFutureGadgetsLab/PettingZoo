from game import Game
from renderer import Renderer

game = Game(40, 20)
game.setup_game()

renderer = Renderer()

renderer.run(game)