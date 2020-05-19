My second neural network experiment (first was FIR filter). 
DFT output is just a linear combination of inputs, so it should be 
implementable by a single layer with no activation function.

![Topology of a 4-point complex DFT](https://i.stack.imgur.com/siCTx.gif)

Animation of weights being trained:

![Neural network weights heatmap](https://i.imgur.com/5SjyBsw.gif)

Red are positive, blue are negative.
The black squares (2336 out of 4096) are unused, and could be pruned out 
to save computation time (if I knew how to do that).

Even with pruning, it would be less efficient than an FFT, so if 
the FFT output is useful, probably best to do it externally and 
provide it as separate inputs?

This at least demonstrates that neural networks can figure out 
frequency content on their own, though, if it's useful to the problem.

The loss goes down for a while but then goes up.  I don't know why:

![loss vs epoch](https://i.imgur.com/hocsRli.png)