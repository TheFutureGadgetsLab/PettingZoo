from game import Game
from game import Renderer
from joblib import load

def main():
    model, game_args = load("/home/supa/lin_storage/pettingzoo/runs/test/9_1041.43.joblib")

    renderer = Renderer()
    game = Game(**game_args)
    renderer.new_game_setup(game)

    while renderer.running:
        keys = renderer.get_input()

        player_view = game.get_player_view(11, 11)
        keys = model.evaluate(player_view)

        game.update(keys)

        if game.game_over:
            print(f"{game.player.fitness}")
            game = Game(**game_args)
            renderer.new_game_setup(game)
            continue
    
        renderer.draw_state(game, keys)

if __name__ == "__main__":
    main()