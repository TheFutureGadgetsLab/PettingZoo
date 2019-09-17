from models.FFNN import FFNN
import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)

parentA = FFNN(3, 3, 2, 2, bias=True)
parentB = FFNN(3, 3, 2, 2, bias=True)

childA, childB = FFNN.breed(parentA, parentB, 10)
