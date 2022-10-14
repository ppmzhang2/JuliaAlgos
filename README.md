# Julia Algorithms

## CNN with MNIST

start training with the MNIST handwritten digit:

```julia
using JuliaAlgos.MnistClassifier
Trainer()()
```

result:

```
[ Info: training loop 1 / 5 starting ...
[ Info:   -- evaluating ...
[ Info:   -- accuracy=0.8806
[ Info:   -- new best record
[ Info: training loop 2 / 5 starting ...
[ Info:   -- evaluating ...
[ Info:   -- accuracy=0.9775
[ Info:   -- new best record
[ Info: training loop 3 / 5 starting ...
[ Info:   -- evaluating ...
[ Info:   -- accuracy=0.9808
[ Info:   -- new best record
[ Info: training loop 4 / 5 starting ...
[ Info:   -- evaluating ...
[ Info:   -- accuracy=0.9833
[ Info:   -- new best record
[ Info: training loop 5 / 5 starting ...
[ Info:   -- evaluating ...
[ Info:   -- accuracy=0.9842
[ Info:   -- new best record
```

## Pluto Notebook

Please refer to the [page](https://ppmzhang2.github.io/JuliaAlgos.jl/).
