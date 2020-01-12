from copy import deepcopy

def breed(parentA, parentB, generator):
    childA = deepcopy(parentA)
    childB = deepcopy(parentB)

    pA_params = parentA.named_parameters()
    pB_params = parentB.named_parameters()
    cA_params = childA.named_parameters()
    cB_params = childB.named_parameters()

    for param_group in zip(pA_params, pB_params, cA_params, cB_params):
        name = param_group[0][0]
        params = [tup[1] for tup in param_group]

        if "mask" in name:
            combine_tensors_avg(*params, generator)
        else:
            combine_tensors(*params, generator)

    return childA, childB

def combine_tensors(parentA, parentB, childA, childB, generator):
    split_loc = generator.integers(low=0, high=parentA.shape[0], endpoint=True)

    # Only copying parentB into childA because childA is a deepcopy of parentA
    # Same with childB (but reversed)
    childA[split_loc:] = parentB[split_loc:]
    childB[split_loc:] = parentA[split_loc:]

def combine_tensors_avg(parentA, parentB, childA, childB, generator):
    first_weight  = generator.uniform(0, 1)
    second_weight = 1 - first_weight

    # Only copying parentB into childA because childA is a deepcopy of parentA
    # Same with childB (but reversed)
    childA.copy_(first_weight  * parentA + second_weight * parentB)
    childB.copy_(second_weight * parentA + first_weight  * parentB)