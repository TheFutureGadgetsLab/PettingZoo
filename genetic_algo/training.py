import numpy as np
import ray
from game import Game

def get_gen_stats(fitnesses, death_types):
    mean = np.mean(fitnesses)
    min_ = np.min(fitnesses)
    max_ = np.max(fitnesses)

    print(f"Avg: {mean:.2f}")
    print(f"Min: {min_:.2f}")
    print(f"Max: {max_:.2f}")

def evaluate_agent(agent, game_seed):
    game = Game(10, game_seed)

    while game.game_over == False:
        player_view = game.get_player_view(11, 11)
        keys = agent.evaluate(player_view)

        game.update(keys)

    return game.player.fitness, game.game_over_type

def evaluate_generation(agents, game_seed):
    fitnesses   = []
    death_types = []

    for agent in agents:
        fitness, death_type = evaluate_agent(agent, game_seed)

        fitnesses.append(fitness)
        death_types.append(death_type)

    return fitnesses, death_types

def breed_generation(agents, breeding_pairs):
    agent_class = type(agents[0])
    new_gen = []

    for pair in breeding_pairs:
        childA, childB = agent_class.breed(agents[pair[0]], agents[pair[1]], np.random.randint(1000))
        new_gen.append(childA)
        new_gen.append(childB)
    
    return new_gen