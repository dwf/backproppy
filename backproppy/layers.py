"""
A set of classes for building feed forward networks.
"""

import numpy as np

class Layer(object):
    """Base class for network layers."""
    pass

class SoftmaxLayer(Layer):
    """
    An object that computes a softmax a.k.a. multinomial logit
    transformation, exp(a_i) / sum_j exp(a_j).
    """
    def __init__(self, *args, **kwargs):
        super(SoftmaxLayer, self).__init__(*args, **kwargs)

    def fprop(self, inputs):
        """
        Forward propagate input through this module.
        """
        expd = np.exp(inputs)
        expd /= expd.sum(axis=1)[:, np.newaxis]
        return expd

    def bprop(self, dout, inputs):
        """
        Given derivatives with respect to the output of this
        module as well as a set of inputs, calculate derivatives
        with respect to the inputs.
        """
        out = self.fprop(inputs)[:, np.newaxis, :]
        idx = np.arange(out.shape[2])
        values = out[:, 0, idx]
        out = out * -out[:, 0, :, np.newaxis]
        out[:, idx, idx] += values
        return out.sum(axis=-1)

class LogisticLayer(Layer):
    """
    A layer of elementwise nonlinearities using the logistic sigmoid
    function, 1/(1 + exp(-x)).
    """
    def __init__(self, inshape, *args, **kwargs):
        super(LogisticLayer, self).__init__(*args, **kwargs)
        self.inshape = (inshape,) if np.isscalar(inshape) else inshape
        self.outshape = (inshape,) if np.isscalar(inshape) else inshape
        self.params = np.empty(np.prod(inshape))
        self.biases = self.params.reshape(inshape)
        self._grad = np.empty(np.prod(inshape))
    
    def fprop(self, inputs):
        """Forward propagate input through this module."""
        return (1 + np.exp(-(inputs + self.biases[np.newaxis, ...])))**(-1)
    
    def bprop(self, dout, inputs):
        """
        Given derivatives with respect to the output of this
        module as well as a set of inputs, calculate derivatives
        with respect to the inputs.
        """
        fpropped = self.fprop(inputs)
        return fpropped * (1 - fpropped) * dout
    
    def grad(self, dout, inputs):
        """
        Given derivatives with respect to the output of this
        module as well as a set of inputs, calculate derivatives
        with respect to this module's internal parameters.
        """
        self._grad[...] = \
            self.bprop(dout, inputs).sum(axis=0).reshape(
                np.prod(self.inshape)
            )
        return self._grad

class LinearLayer(Layer):
    """
    An object representing a  linear transformation of the input,
    i.e. a matrix multiply. Each output is a weighted sum of inputs.
    """    
    def __init__(self, inshape, outshape, *args, **kwargs):
        super(LinearLayer, self).__init__(*args, **kwargs)
        self.inshape = (inshape,) if np.isscalar(inshape) else inshape
        self.outshape = (outshape,) if np.isscalar(outshape) else outshape
        self.params = np.empty(np.prod(self.inshape + self.outshape))
        self.weights = self.params.reshape(self.inshape[::-1] + self.outshape)

    def fprop(self, inputs):
        """
        Forward propagate input through this module.
        """
        return np.tensordot(
            inputs,
            self.weights,
            axes=len(self.inshape)
        )
    
    def bprop(self, dout, inputs):
        """
        Given derivatives with respect to the output of this
        module as well as a set of inputs, calculate derivatives
        with respect to the inputs.
        """
        return np.tensordot(dout, self.weights.transpose(), 1)

    def grad(self, dout, inputs):
        """
        Given derivatives with respect to the output of this
        module as well as a set of inputs, calculate derivatives
        with respect to this module's internal parameters.
        """
        return np.tensordot(inputs.transpose(), dout, 1)

