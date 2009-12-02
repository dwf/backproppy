from layers import LogisticLayer, LinearLayer, SoftmaxLayer

"""Modules that represent entire networks."""

class ShallowLogisticSoftmaxNetwork(object):
    """
    A network with a single hidden layer of logistic sigmoids.
    """
    def __init__(self, inshape, hidshape, noutputs):
        self.layers = [
            LinearLayer(inshape, hidshape),
            LogisticLayer(hidshape),
            LinearLayer(hidshape, noutputs),
            SoftmaxLayer()
        ]
    
    def fprop(self, inputs):
        """
        Forward propagate input through this module.
        """
        curr = inputs
        for layer in self.layers:
            curr = layer.fprop(curr)
        return curr
    
    def bprop(self, dout, inputs):
        """
        Given derivatives with respect to the output of this
        module as well as a set of inputs, calculate derivatives
        with respect to the inputs.
        """
        fpropped_inputs = []
        fpropped_inputs.append(inputs)
        for layer in self.layers[:-1]:
            fpropped_inputs.append(layer.fprop(fpropped_inputs[-1]))
        bpropped = dout
        for layer, input in zip(self.layers[::-1], fpropped_inputs[::-1]):
            bpropped = layer.bprop(bpropped, input)        
        return bpropped
    
    def grad(self, dout, inputs):
        """
        Given derivatives with respect to the output of this
        module as well as a set of inputs, calculate derivatives
        with respect to this module's internal parameters.
        """
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
        return grads

