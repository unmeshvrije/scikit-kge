import sys
import numpy as np
from numpy import sqrt, squeeze, zeros_like
from numpy.random import randn, uniform
import pdb
import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('EX-KG')

def init_unif(sz):
        """
        Uniform intialization

        Heuristic commonly used to initialize deep neural networks
        """
        bnd = 1 / sqrt(sz[0])
        p = uniform(low=-bnd, high=bnd, size=sz)
        return squeeze(p)


# C++: Armadillo lib function
def init_nunif(sz):
        """
        Normalized uniform initialization

        See Glorot X., Bengio Y.: "Understanding the difficulty of training
        deep feedforward neural networks". AISTATS, 2010
        """
        # This is the initialization that corresponds to algorithm in TransE paper 
        #pdb.set_trace()
        bnd = sqrt(6) / sqrt(sz[0] + sz[1])
        # Initial vector embedding for each entity
        # sz is a tuple of size N * N * M so, that many samples will be  drawn
        p = uniform(low=-bnd, high=bnd, size=sz)
        # Squeeze will remove the single dimensionan entries from the shape of an array
        '''
        Illustration
        >>> x = np.array([[1],[2],[3]])
        >>> x.shape
        (3, 1)
        >>> np.squeeze(x).shape
        (3,)
        >>> np.squeeze(x)
        array([1, 2, 3])
        '''
        init_nunif.counter += 1
        #pdb.set_trace()
        return squeeze(p)

init_nunif.counter = 0

def init_randn(sz):
        return squeeze(randn(*sz))


class Parameter(np.ndarray):

    def __new__(cls, *args, **kwargs):
        # TODO: hackish, find better way to handle higher-order parameters
        #pdb.set_trace();
        if len(args[0]) == 3:
                sz = (args[0][1], args[0][2])
                arr = np.array([Parameter._init_array(sz, args[1]) for _ in range(args[0][0])])
        else:
                arr = Parameter._init_array(args[0], args[1])
        #pdb.set_trace();
        arr = arr.view(cls)
        #pdb.set_trace();
        arr.name = kwargs.pop('name', None)
        arr.post = kwargs.pop('post', None)

        if arr.post is not None:
            arr = arr.post(arr)

        return arr

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.post = getattr(obj, 'post', None)

    @staticmethod
    def _init_array(shape, method):
        mod = sys.modules[__name__]
        #pdb.set_trace();
        method = 'init_%s' % method
        #pdb.set_trace();
        if not hasattr(mod, method):
            raise ValueError('Unknown initialization (%s)' % method)
        elif len(shape) != 2:
            raise ValueError('Shape must be of size 2')
        #pdb.set_trace();
        return getattr(mod, method)(shape)


class ParameterUpdate(object):

    def __init__(self, param, learning_rate):
        self.param = param
        self.learning_rate = learning_rate

    # This allows class's instance to be called as a function
    def __call__(self, gradient, idx=None):
        #pdb.set_trace();
        self._update(gradient, idx)
        if self.param.post is not None:
            self.param = self.param.post(self.param, idx)

    def reset(self):
        pass


class SGD(ParameterUpdate):
    """
    Class to perform SGD updates on a parameter
    """

    def _update(self, g, idx):
        self.param[idx] -= self.learning_rate * g


# Adaptive Gradient
class AdaGrad(ParameterUpdate):

    def __init__(self, param, learning_rate):
        super(AdaGrad, self).__init__(param, learning_rate)
        self.p2 = zeros_like(param)

    def _update(self, g, idx=None):
        #pdb.set_trace();
        # p2 is of type Parameter
        # g is an ndarray of shape (M X N).
        # * will yield the element wise multiplication of matrix
        # idx is array of ids
        # p2[idx] will extract the vector embeddings for each id in the array idx
        self.p2[idx] += g * g
        #pdb.set_trace();
        H = np.maximum(np.sqrt(self.p2[idx]), 1e-7)
        # Check the learning rate here
        #log.info("Learning rate = [%f] | " % (self.learning_rate))
        self.param[idx] -= self.learning_rate * g / H

    def reset(self):
        self.p2 = zeros_like(self.p2)


def normalize(M, idx=None):
    if idx is None:
        M = M / np.sqrt(np.sum(M ** 2, axis=1))[:, np.newaxis]
    else:
        nrm = np.sqrt(np.sum(M[idx, :] ** 2, axis=1))[:, np.newaxis]
        M[idx, :] = M[idx, :] / nrm
    return M


def normless1(M, idx=None):
    nrm = np.sum(M[idx] ** 2, axis=1)[:, np.newaxis]
    nrm[nrm < 1] = 1
    M[idx] = M[idx] / nrm
    return M
