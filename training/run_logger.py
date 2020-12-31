import os
import shutil
from game import Game
from tqdm import tqdm
from collections import defaultdict
import time

class RunLogger():
    r"""A class to monitor and log the progress of a run.

    Args:
    ---
    output_dir:  A directory to output run progress and agents

    Features:
    ---
    Will dump stats of each generation to a run_log.txt file in the output directory. \
    Will dump top N agents of each generation. \
    Can be used to print out stats about each generation
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

        self.run_log_file = None
        # self.setup_output()

        self.n_gens = 0
        self.overall_best_fitness = 0

    def setup_output(self):
        self.output_dir = os.path.expanduser(self.output_dir)
        self.output_dir = os.path.abspath(self.output_dir)

        try:
            os.mkdir(self.output_dir)
        except FileExistsError:
            resp = input(f"Output dir '{self.output_dir}' exists!\nWould you like to delete? Y / N: ")
            if resp.lower() != "y":
                print("Please provide a new output path.")
                exit(-1)
            
            shutil.rmtree(self.output_dir)
            os.mkdir(self.output_dir)

        self.run_log_file = open(os.path.join(self.output_dir, "run_log.txt"), 'w')

    def log_generation(self, results, gameArgs):
        maxFit = results['Fitness'].max()
        minFit = results['Fitness'].min()
        avgFit = results['Fitness'].mean()

        deaths = results['Death Type'].value_counts()
        deaths = defaultdict(int, deaths)

        print_stats(minFit, maxFit, avgFit, deaths, gameArgs)
    
def print_stats(minFit, maxFit, avgFit, deaths, gameArgs):
    tqdm.write(f"Seed: {gameArgs['seed']}")
    tqdm.write(f"Avg: {avgFit:0.2f}")
    tqdm.write(f"Min: {minFit:0.2f}")
    tqdm.write(f"Max: {maxFit:0.2f}")
    tqdm.write("")
    tqdm.write(f"Died:      {deaths[Game.PLAYER_DEAD]}")
    tqdm.write(f"Completed: {deaths[Game.PLAYER_COMPLETE]}")
    tqdm.write(f"Timed out: {deaths[Game.PLAYER_TIMEOUT]}")