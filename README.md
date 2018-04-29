# digitNN
Simple neural network classifier on the MNIST digit set.

The neural network is a backpropagating softmax activated multimatrix with double variable sized hidden nodes layer that achieves 96% accuracy on default settings.

Comes with automatic training featuring epochs, mini-batches, automatic alpha adjustment and cutoff, validation error visualisation and hyperparamater configuration for many aspects of the training process.

### Version 1.5
- New dataset of around 300.000 handwritten letters
- Parameter tweaking and cleanup
- Epoch progress bar
- Converter for .csv to .idx.gz files
- 97% accuracy on the handwritten letters set

### Version 1.1
- Cross-entropy loss and softmax activation
- Epoch based learning, automatic training data shuffling, deterministic batching
- Automatic alpha decay
- Validation error visualiser, see which samples the network got wrong
- Average 96% accuracy after three minutes of training

### Version: 1.0
- Randomized mini-batch learning classifier by sigmoid activation
- Sampler for user image prediction

## Resources
[Download the MNIST set yourself here](http://yann.lecun.com/exdb/mnist/)
since I have no explicit right to redistribute it.
