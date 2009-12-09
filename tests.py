import numpy as np

from backproppy.layers import SoftmaxLayer, LinearLayer, LogisticLayer
from backproppy.networks import ShallowLogisticSoftmaxNetwork
from backproppy.costs import MultiClassCrossEntropy
from backproppy.trainers import BatchNetworkTrainer

def fd_grad(func, x_in, tol=1.0e-5):
    """
    Approximates the gradient of f with finite differences, moving
    half of tol in either direction on each axis.
    """
    x_in = np.asarray(x_in)
    inshape = x_in.shape
    x_in = x_in.reshape(np.prod(x_in.shape))
    num = np.prod(x_in.shape)
    grad = np.zeros(num)
    for ii in xrange(num):
        aaa = x_in.copy()
        bbb = x_in.copy()
        aaa[ii] = aaa[ii] - tol / 2.0
        bbb[ii] = bbb[ii] + tol / 2.0
        grad[ii] = (func(bbb.reshape(inshape)) - func(aaa.reshape(inshape))) \
            / (bbb[ii] - aaa[ii])
    return grad

class FPropWithInputs(object):
    def __init__(self, module, params, dout):
        self.module = module
        if hasattr(module, 'params') and module.params is not None:
            module.params[:] = params
        self.dout = dout

    def __call__(self, changed):
        return (self.module.fprop(changed) * \
                self.dout).sum()

    def gradient(self, changed):
        return self.module.bprop(self.dout, changed)
    
    def fdgradient(self, changed):
        return fd_grad(self, changed)

class FPropWithParams(object):
    def __init__(self, module, inputs, dout):
        self.module = module
        self.inputs = inputs
        self.dout = dout

    def __call__(self, changed):
        self.module.params[:] = changed.reshape(np.prod(changed.shape))
        return (self.module.fprop(changed) * self.dout).sum()

    def gradient(self, changed):
        self.module.params[:] = changed.reshape(np.prod(changed.shape))
        return self.module.grad(self.dout, self.inputs)

if __name__ == "__main__":
    import sys
    from scipy.io.matlab import loadmat
    import matplotlib.pyplot as plt
    data = loadmat(sys.argv[1])
    digits = [data['train%d' % i][:int(sys.argv[2])] for i in range(10)]
    digits = (np.concatenate(digits) / 255.).reshape(10 * int(sys.argv[2]),
                                                     28, 28)
    labels = np.zeros((10 * int(sys.argv[2]), 10))
    for i in range(10):
        labels[(int(sys.argv[2]) * i):(int(sys.argv[2]) * (i + 1)), i] = 1.
    net = ShallowLogisticSoftmaxNetwork((28, 28), int(sys.argv[3]), 10)
    net.params[:] = np.random.randn(len(net.params)) * 0.001
    cost = MultiClassCrossEntropy(net)
    trainer = BatchNetworkTrainer(net, digits, labels, cost)
    trainer.train(eta=float(sys.argv[4]))
