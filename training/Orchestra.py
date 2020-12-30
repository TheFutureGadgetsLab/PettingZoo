import ray
import numpy as np
from training.Section import Section
import pandas as pd
from functools import lru_cache

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

        tmp = np.cumsum([0] + self.sizes)
        self.ranges = [range(tmp[i], tmp[i+1]) for i in range(len(tmp)-1)]

    def play(self, gameArgs):
        futures = []
        for sec in self.sections:
            futures.append(sec.play.remote(gameArgs))

        results = ray.get(futures)
        results = [r for sec in results for r in sec] # Flatten
        df = pd.DataFrame(results, columns=('Fitness', 'Death Type'))

        return df

    def breed(self, pairs):
        pairRefs = []
        for pair in pairs:
            pA = self.getAgent(pair[0])
            pB = self.getAgent(pair[1])
            pairRefs.append((pA, pB))

        bredChildren = []
        for i, ref in enumerate(pairRefs):
            children = self.sections[i % len(self.sections)].breed.remote(*ref)
            bredChildren.extend(children)

        packages = [list(p) for p in np.array_split(bredChildren, self.nSections)]
        for pack, sec in zip(packages, self.sections):
            sec.setAgents.remote(pack)
        
        self.getAgent.cache_clear()
    
    @lru_cache(5_000) # Need a better way to handle this
    def getAgent(self, ID):
        for i, r in enumerate(self.ranges):
            if ID in r:
                return self.sections[i].getAgent.remote(ID - r.start)
        
        raise ValueError