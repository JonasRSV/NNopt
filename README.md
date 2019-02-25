Surrogate Optimization Method using NN as Surrogate
---

* Tries to model underlying function using a NN
* Tries to model uncertanty of the surrogate function by using a memory NN to predict the output of a random network (This works not as good as id want it to, it is a WIP)

#### Example
```python
from nnopt.nnopt import Optimizer

def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1

Nopt = Optimizer(black_box_function, N=2, 
								surrogate_hidden_layer=200, 
								memory_hidden_layer=200,
								model_hidden_layer=200,
								Rn=[(-10, 10), (-10, 10)])
best, target = Nopt.run(random=10, optimization=40, fitting=100, verbose=True)
print("Best", best, "target", target)

```

see 
- [bayes opt comparison](https://github.com/JonasRSV/NNopt/blob/master/nnopt/example/bayes_vs_nnopt.ipynb)
- [interactive sort of..](https://github.com/JonasRSV/NNopt/blob/master/nnopt/example/NNoptInteractive.ipynb)


#### Demo
(Gif is speed up 4x)
![demo](images/traindemo.gif)


#### Demo (Optimize linear transformation to land space crafts! :))
(Gif speed up 4x)
![demo2](images/moondemo.gif)


