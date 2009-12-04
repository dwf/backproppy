import numpy as np

from backproppy.layers import SoftmaxLayer

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
        grad[ii] = (func(bbb) - func(aaa))/(bbb[ii] - aaa[ii])
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

