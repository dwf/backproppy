"""
A set of classes for building feed forward networks.
"""

import numpy as np

class Layer(object):
    """Base class for network layers."""
    def __init__(self, nparams=None, params=None, grad=None, *args, **kwargs):
        if params is None:
            if nparams is not None:
                self.params = np.empty(nparams)
            else:
                self.params = None
        else:
            if not hasattr(params, 'shape') or params.ndim != 1:
                raise ValueError('params must be rank 1 array if supplied')
            elif params.size < nparams:
                raise ValueError('params smaller than required (%d)' % nparams)
            self.params = params

        if grad is None:
            if nparams is not None:
                self._grad = np.empty((nparams,))
            else:
                self._grad = None
        else:
            if not hasattr(grad, 'shape') or grad.ndim != 1:
                raise ValueError('grad must be rank 1 array if supplied')
            elif grad.size < nparams:
                raise ValueError('grad smaller than required (%d)' % nparams)
            self._grad = grad


class SoftmaxLayer(Layer):
    """
    An object that computes a softmax a.k.a. multinomial logit
    transformation, exp(a_i) / sum_j exp(a_j).
    """
    def __init__(self, *args, **kwargs):
        super(SoftmaxLayer, self).__init__(*args, **kwargs)

    def fprop(self, inputs):
        expd = np.atleast_2d(inputs.copy())
        expd -= expd.max()
        expd = np.exp(expd)
        expd /= np.atleast_1d(expd.sum(axis=-1))[:, np.newaxis]
        return expd
    
    def bprop(self, dout, inputs):
        """
        Given derivatives with respect to the output of this
        module as well as a set of inputs, calculate derivatives
        with respect to the inputs.
        """
        dout = np.atleast_2d(dout)
        out = self.fprop(inputs)[:, np.newaxis, :]
        idx = np.arange(out.shape[2])
        values = out[:, 0, idx]
        out = out * -out[:, 0, :, np.newaxis]
        out[:, idx, idx] += values
        out *= dout[:, np.newaxis, :]
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
        super(LogisticLayer, self).__init__(
            np.prod(inshape),
            *args,
            **kwargs
        )
        self.biases = self.params.reshape(inshape)
    
    def fprop(self, inputs):
        """Forward propagate."""
        out = np.atleast_2d(inputs.copy())
        out += self.biases[np.newaxis, ...]
        out *= -1.
        np.exp(out, out)
        out += 1.
        out **= -1.
        return out
    
    def bprop(self, dout, inputs):
        """Backpropagate through this module."""
        out = self.fprop(np.atleast_2d(inputs))
        # Compute 1 - y_I = exp(-inputs) / (1 + exp(-inputs)) =>  more stable
        expd = np.atleast_2d(inputs.copy())
        expd += self.biases[np.newaxis, ...]
        expd *= -1.
        np.exp(expd, expd)
        oneminus = expd / (1 + expd)
        out *= oneminus
        out *= dout
        return out
    
    def grad(self, dout, inputs):
        """
        Given derivatives with respect to the output of this
        module as well as a set of inputs, calculate derivatives
        with respect to this module's internal parameters.
        """
        self._grad[...] = \
            self.bprop(dout, np.atleast_2d(inputs)).sum(axis=0).reshape(
                np.prod(self.inshape)
            )
        return self._grad

class LinearLayer(Layer):
    """
    An object representing a  linear transformation of the input,
    i.e. a matrix multiply. Each output is a weighted sum of inputs.
    """    
    def __init__(self, inshape, outshape, *args, **kwargs):
        self.inshape = (inshape,) if np.isscalar(inshape) else inshape
        self.outshape = (outshape,) if np.isscalar(outshape) else outshape
        ninputs = np.prod(self.inshape)
        super(LinearLayer, self).__init__(
            np.prod(self.inshape + self.outshape),
            *args,
            **kwargs
        )
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
        self._grad[...] = np.tensordot(
            inputs.transpose(),
            dout,
            1
        ).reshape(np.prod(self.weights.shape))
        return self._grad

