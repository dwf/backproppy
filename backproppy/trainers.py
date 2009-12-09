"""Trainers that work on network objects."""
from itertools import izip
import numpy as np
import sys

class BatchNetworkTrainer(object):
    """A class that trains networks by batch gradient descent."""
    def __init__(self, network, inputs, targets, cost, decay=1e-3,
                 *args, **kwargs):
        self.network = network
        self.inputs = inputs
        self.targets = targets
        self.cost = cost
        self.decay = decay

    def train(self, niters=500, eta=0.1, alpha=0.9):
        """Train a network for a given number of iterations."""
        previous = None
        for iters in xrange(niters):
            predictions = self.network.fprop(self.inputs)
            cost = self.cost.evaluate(self.targets, predictions)
            print cost
            sys.stdout.flush()
            cost_gradient = self.cost.grad(self.targets, predictions)
            net_gradients = self.network.grad(cost_gradient, self.inputs)
            step = net_gradients.copy()
            step += 2 * self.decay * self.network.params
            step *= -eta
            if previous is not None:
                step *= (1. - alpha)
                step += alpha * previous
            else:
                previous = np.empty(step.shape)
            self.network.params += step
            previous[...] = step
            #for layer, gradient in izip(self.network.layers[::-1], net_gradients):
            #    if gradient is not None:
            #        decay = 2 * self.decay * layer.params
            #        layer.params -= eta * gradient.reshape(layer.params.shape)
            #        layer.params -= eta * decay

