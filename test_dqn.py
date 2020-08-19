import numpy as np
import torch
from models.DQN import DQN, ReplayMemory
from game import Game, Vec2d
from game import Renderer

def main():
    model = DQN(11, 11)
    memory = ReplayMemory(1_000)

    renderer = Renderer()
    game = Game(num_chunks=10, seed=10, view_size=Vec2d(11, 11))
    renderer.new_game_setup(game)

    while renderer.running:
        renderer.get_input() # Run window event loop (resize 'n stuff)

        player_view = game.get_player_view()
        keys = model.evaluate(player_view)

        last_reward = game.player.fitness
        game.update(keys)

        memory.push(player_view, keys, game.get_player_view(), game.player.fitness - last_reward)

        if game.game_over:
            game = Game(num_chunks=10, seed=10, view_size=Vec2d(11, 11))
            renderer.new_game_setup(game)
            continue
        
        renderer.draw_state(game, keys)

if __name__ == "__main__":
    main()