class Agent():
    """ A class to wrap different models for training

    `model`:     a model class to initialize (FFNN, CNN, etc.)  
    `kwargs`:    arguments to initialize the model with  
    `view_size`: A tuple describing how large the view size should be
    """
    def __init__(self, model, view_size, **kwargs):
        self.model = model(kwargs)
        self.view_size = view_size
        
        self.fitness = 0
        self.death_type = None
    
    def evaluate(self, game):
        while not game.game_over:
            player_view = game.get_player_view(*self.view_size)

            keys = model.evaluate(player_view)
            game.update(keys)
        
        self.fitness = game.player.fitness
        self.death_type = game.game_over_type