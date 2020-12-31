import ray
import numpy as np
from training.Section import Section
import pandas as pd
from functools import lru_cache
import time
from tqdm import tqdm
from bisect import bisect_right

class Orchestra:
    def __init__(self, nSections, nAgents, AgentClass, ss) -> None:
        ray.init(local_mode=False)

        if nAgents % 2 != 0:
            print("The number of agents must be an even number. Please update accordingly.")
            exit(-1)

        self.ss  = ss
        self.gen = np.random.default_rng(ss)

        self.nSections = nSections
        self.nAgents   = nAgents

        self.sizes  = []
        self.ranges = []
        self.getSizes()

        self.sections = [
            Section.remote(ID, n, AgentClass, nss) for ID, (n, nss) in enumerate(zip(self.sizes, self.ss.spawn(nSections)))
        ]
    
    def getSizes(self):
        # Borrowed from the 1.19.0 numpy function `array_split`
        Neach_section, extras = divmod(self.nAgents, self.nSections)
        self.sizes  = extras * [Neach_section+1] + (self.nSections-extras) * [Neach_section]
        self.ranges = np.cumsum([0] + self.sizes)

    def play(self, gameArgs):
        t0 = time.time()

        futures = []
        for sec in self.sections:
            futures.append(sec.play.remote(gameArgs))

        results = ray.get(futures)
        df = pd.concat(results, ignore_index=True)

        t1 = time.time()
        tqdm.write(f"Evaluation time: {(t1-t0):0.4f}")

        return df

    def breed(self, pairs):
        t0 = time.time()

        pairRefs = []
        for pair in pairs:
            pA = self.getAgent(pair[0])
            pB = self.getAgent(pair[1])
            pairRefs.append((pA, pB))

        bredChildren = []
        for i, ref in enumerate(pairRefs):
            children = self.sections[i % len(self.sections)].breed.remote(*ref)
            bredChildren.extend(children)

        packages = [p.tolist() for p in np.array_split(bredChildren, self.nSections)]
        for pack, sec in zip(packages, self.sections):
            sec.setAgents.remote(pack)
        
        self.getAgent.cache_clear()

        t1 = time.time()
        tqdm.write(f"Breed time: {(t1-t0):0.4f}")
    
    @lru_cache(5_000) # Need a better way to handle this
    def getAgent(self, ID):
        i = bisect_right(self.ranges, ID) - 1
        return self.sections[i].getAgent.remote(ID - self.ranges[i])