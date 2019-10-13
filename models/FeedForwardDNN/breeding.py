from copy import deepcopy

def breed(parentA, parentB, generator):
    childA = deepcopy(parentA)
    childB = deepcopy(parentB)

    pA_params = list(parentA.parameters())
    pB_params = list(parentB.parameters())
    cA_params = list(childA.parameters())
    cB_params = list(childB.parameters())

    for pA_param, pB_param, cA_param, cB_param in zip(pA_params, pB_params, cA_params, cB_params):
        combine_tensors(pA_param, pB_param, cA_param, cB_param, generator)

    return childA, childB

def combine_tensors(parentA, parentB, childA, childB, generator):
    split_loc = generator.integers(low=1, high=parentA.shape[0])

    # Only copying parentB into childA because childA is a deepcopy of parentA
    # Same with childB (but reversed)
    childA[split_loc:] = parentB[split_loc:]
    childB[split_loc:] = parentA[split_loc:]

def combine_tensors_avg(parentA, parentB, childA, childB, generator):
    first_weight  = generator.uniform(0, 2)
    second_weight = 1 - first_weight

    # Only copying parentB into childA because childA is a deepcopy of parentA
    # Same with childB (but reversed)
    childA = first_weight  * parentA + second_weight * parentB
    childB = second_weight * parentA + first_weight  * parentB