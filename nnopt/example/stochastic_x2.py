import sys
import time

sys.path.append("/Users/jonval/WARNING/singularity/MLProjects/NNopt/nnopt")
import nnopt
import numpy as np

def X2(x, y):
    return 10 - (x)**2 - (y)**2

opt = nnopt.Optimizer(X2, N=2, surrogate_hidden_layer=150)

print(opt.run(random=40, exploration=100, fitting=1000))


