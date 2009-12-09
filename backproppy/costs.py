import numpy as np

class CostFunction(object):
    """Base class for a cost function."""
    def __init__(self, module, *args, **kwargs):
        super(CostFunction, self).__init__(*args, **kwargs)
        self.module = module

    def evaluate(self, targets, predictions=None, inputs=None):
        """
        Evaluate the cost function with respect to a set of targets
        and the module's predictions. If predictions are unspecified
        and inputs are given instead, the inputs are fprop()'d
        through the module.
        """
        raise NotImplementedError()
    
    def grad(self, targets, predictions=None, inputs=None):
        """
        Evaluate the gradient of the cost function with respect to
        a set of predictions.
        """
        raise NotImplementedError()

class MultiClassCrossEntropy(CostFunction):
    """
    Cost function for multinomial regression, or a softmax output 
    layer in a neural network.
    """
    def __init__(self, module, *args, **kwargs):
        super(MultiClassCrossEntropy, self).__init__(module, *args, **kwargs)

    def evaluate(self, targets, predictions=None, inputs=None):
        """
        Evaluate the cost function with respect to a set of targets
        and the module's predictions. If predictions are unspecified
        and inputs are given instead, the inputs are fprop()'d
        through the module.
        """
        if predictions is None:
            predictions = self.module.fprop(inputs)
        err = np.log(predictions)
        err *= targets
        err *= -1.
        return err.sum()

    def grad(self, targets, predictions=None, inputs=None):
        """
        Evaluate the gradient of the cost function with respect to
        a set of predictions.
        """
        if predictions is None:
            predictions = self.module.fprop(inputs)
        return -targets / predictions


