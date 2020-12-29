import ray
import numpy as np
from training.Section import Section
import pandas as pd

class Orchestra:
    def __init__(self, nSections, nAgents, AgentClass, ss) -> None:
        ray.init(local_mode=False)

        if (nAgents % nSections) != 0 or (nAgents % 2) != 0:
            print("For now, the number of agents must be a multiple of the number of sections " 
            "AND an even number. Please update accordingly.")
            exit(-1)

        self.ss  = ss
        self.gen = np.random.default_rng(ss)

        self.nSections = nSections
        self.nAgents   = nAgents

        self.sizes  = [self.nAgents // self.nSections] * self.nSections
        self.ranges = []

        self.sections = [
            Section.remote(ID, n, AgentClass, nss) for ID, (n, nss) in enumerate(zip(self.sizes, self.ss.spawn(nSections)))
        ]

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
            aS, aI = divmod(pair[0], self.nAgents // self.nSections)
            bS, bI = divmod(pair[1], self.nAgents // self.nSections)
            ref = (
                self.sections[aS].getAgent.remote(aI),
                self.sections[bS].getAgent.remote(bI),
            )
            pairRefs.append(ref)


        futures = []
        for i, ref in enumerate(pairRefs):
            fut = self.sections[i % len(self.sections)].breed.remote(*ref)
            futures.append(fut)

        ray.get(futures)
        ray.get([sec.finalize.remote() for sec in self.sections])