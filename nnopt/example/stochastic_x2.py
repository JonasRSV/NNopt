import sys
import time

sys.path.append("/Users/jonval/WARNING/singularity/MLProjects/NNopt/nnopt")
import nnopt
import numpy as np

def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1


opt = nnopt.Optimizer(black_box_function, N=2, surrogate_hidden_layer=100, kernel_seperate=100, kernel_common=100)

print(opt.run(random=10, optimization=5, exploration=1, fitting=5000, lr=0.01))

