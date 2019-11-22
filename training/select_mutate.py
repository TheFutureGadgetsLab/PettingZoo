from training.rng import get_seeds, construct_generator
import torch
import numpy as np

def select_mutate(parent, agents, generator):
    agents[0] = parent
    for agent in agents[1:]:
        breed(parent, agent, generator)

def breed(parent, child, generator):
    pA_params = parent.parameters()
    cA_params = child.parameters()

    for pA_param, cA_param in zip(pA_params, cA_params):
        combine_tensors(pA_param, cA_param, generator)

def combine_tensors(parentA, childA, generator):
    amnt = torch.Tensor(generator.normal(0.0, 0.2, size=parentA.shape))

    # Only copying parentB into childA because childA is a deepcopy of parentA
    # Same with childB (but reversed)
    childA.copy_(parentA + amnt)