Surrogate Optimization Method using NN as Surrogate
---

* Tries to model underlying function using a NN
* Tries to model uncertainty by training a kernel and using heuristics  

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
	kernel_seperate=200,
	kernel_common=200,
	Rn=[(-10, 10), (-10, 10)])
best, target = Nopt.run(random=10, optimization=40, fitting=100, verbose=True)
print("Best", best, "target", target)

```

see 
- [bayes opt comparison](https://github.com/JonasRSV/NNopt/blob/master/nnopt/example/bayes_vs_nnopt.ipynb)
- [interactive sort of..](https://github.com/JonasRSV/NNopt/blob/master/nnopt/example/NNoptInteractive.ipynb)
- [Ai Gyms](https://github.com/JonasRSV/NNopt/blob/master/nnopt/example/openAISimpleChallenge.ipynb)


#### Demo
(Gif is speed up 4x)
![demo](images/plotDemo.gif)


#### Demo (Optimize linear transformation to land space crafts! :))
(Gif speed up 4x)
![demo2](images/landingDemo.gif)


