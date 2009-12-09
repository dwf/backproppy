import numpy as np
from layers import LogisticLayer, LinearLayer, SoftmaxLayer

"""Modules that represent entire networks."""

class ShallowLogisticSoftmaxNetwork(object):
    def __init__(self, inshape, hidshape, noutputs):
        ninput = np.prod(inshape)
        nhid = np.prod(hidshape)
        nparams = (ninput + 1) * nhid + (nhid * noutputs)
        # TODO: 
        self.params = np.empty(nparams)
        self._grad = np.empty(nparams)
        inhidwts = ninput * nhid
        hidoutwts = nhid * noutputs
        self.layers = [
            LinearLayer(
                inshape,
                hidshape,
                params=self.params[0:inhidwts],
                grad=self._grad[0:inhidwts]
            ),
            LogisticLayer(
                hidshape,
                params=self.params[inhidwts:(inhidwts + nhid)],
                grad=self._grad[inhidwts:(inhidwts + nhid)]
            ),
            LinearLayer(
                hidshape,
                noutputs,
                params=self.params[(inhidwts + nhid):],
                grad=self._grad[(inhidwts + nhid):]
            ),
            SoftmaxLayer()
        ]

    def fprop(self, inputs):
        curr = inputs
        for layer in self.layers:
            curr = layer.fprop(curr)
        return curr
    
    def bprop(self, dout, inputs):
        fpropped_inputs = []
        fpropped_inputs.append(inputs)
        for layer in self.layers[:-1]:
            fpropped_inputs.append(layer.fprop(fpropped_inputs[-1]))
        bpropped = dout
        for layer, input in zip(self.layers[::-1], fpropped_inputs[::-1]):
            bpropped = layer.bprop(bpropped, input)
        return bpropped

    def grad(self, dout, inputs):
        fpropped_inputs = []
        fpropped_inputs.append(inputs)
        for layer in self.layers[:-1]:
            fpropped_inputs.append(layer.fprop(fpropped_inputs[-1]))
        bpropped = dout
        grads = []
        for layer, input in zip(self.layers[::-1], fpropped_inputs[::-1]):
            if hasattr(layer, 'grad'):
                grads.append(layer.grad(bpropped, input))
            else:
                grads.append(None)
            bpropped = layer.bprop(bpropped, input)
        return self._grad

