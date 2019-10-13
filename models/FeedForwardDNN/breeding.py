from copy import deepcopy

def breed(parentA, parentB, generator):
    childA = deepcopy(parentA)
    childB = deepcopy(parentB)

    pA_params = parentA.parameters()
    pB_params = parentB.parameters()
    cA_params = childA.parameters()
    cB_params = childB.parameters()

    for param_group in zip(pA_params, pB_params, cA_params, cB_params):
        combine_tensors(*param_group, generator)

    return childA, childB

def combine_tensors(parentA, parentB, childA, childB, generator):
    split_loc = generator.integers(low=0, high=parentA.shape[0], endpoint=True)

    # Only copying parentB into childA because childA is a deepcopy of parentA
    # Same with childB (but reversed)
    childA[split_loc:] = parentB[split_loc:]
    childB[split_loc:] = parentA[split_loc:]

def combine_tensors_avg(parentA, parentB, childA, childB, generator):
    first_weight  = generator.uniform(0, 2)
    second_weight = 1 - first_weight

    # Only copying parentB into childA because childA is a deepcopy of parentA
    # Same with childB (but reversed)
    childA.copy_(first_weight  * parentA + second_weight * parentB)
    childB.copy_(second_weight * parentA + first_weight  * parentB)