import numpy as np
from numpy.fft import fft, ifft
import scipy.sparse as sp
import functools
import collections
import pdb

def cconv(a, b):
    """
    Circular convolution of vectors

    Computes the circular convolution of two vectors a and b via their
    fast fourier transforms

    a \ast b = \mathcal{F}^{-1}(\mathcal{F}(a) \odot \mathcal{F}(b))

    Parameter
    ---------
    a: real valued array (shape N)
    b: real valued array (shape N)

    Returns
    -------
    c: real valued array (shape N), representing the circular
       convolution of a and b
    """
    return ifft(fft(a) * fft(b)).real


def ccorr(a, b):
    """
    Circular correlation of vectors

    Computes the circular correlation of two vectors a and b via their
    fast fourier transforms

    a \ast b = \mathcal{F}^{-1}(\overline{\mathcal{F}(a)} \odot \mathcal{F}(b))

    Parameter
    ---------
    a: real valued array (shape N)
    b: real valued array (shape N)

    Returns
    -------
    c: real valued array (shape N), representing the circular
       correlation of a and b
    """

    return ifft(np.conj(fft(a)) * fft(b)).real


def grad_sum_matrix(idx):
    # idx is a list of N elements
    # np.unique will 
    #   1. store all unique elements (in sorted order in uidx)
    #   2. store indices of every element in uidx => iinv will be as long as idx
    """
    Illustration
    a = np.array([1,2,6,4,2,3,2])
    u,indices = np.unique(a, return_inverse=True)

    Then,
    >>> u
    array([1,2,3,4,6])
    >>> indices
    array([0,1,4,3,1,2,1]) # 0 is index of 1 in the new unique array, 1 is the index of 2 and so on
    """
    uidx, iinv = np.unique(idx, return_inverse=True)
    pdb.set_trace();
    sz = len(iinv)
    #pdb.set_trace();

    # create a COOrdinate matrix and convert it to CSR (Compressed Sparse Row matrix)
    #                   data        (row, col) where row is array of row indices and col is col indices.
    M = sp.coo_matrix((np.ones(sz), (iinv, np.arange(sz)))).tocsr()
    pdb.set_trace();
    # CSR Matrix
    '''
    Illustration
    >>> data = np.array([1, 2, 3, 4, 5, 6]) # data is traversed row wise
    >>> indices = np.array([0, 2, 2, 0, 1, 2]) # column indices of each element
    >>> indptr = np.array([0, 2, 3, 6]) # Contains N_ROWS + 1 elements. Last element is nnz
    >>> # Where nnz is Number of NonZero elements. Row 0 contains elements data[indptr[0]:indptr[1]]
    >>> # Which is data[0:2] i.e. all elements from index 0 to 1 in the data array
    >>> # Same for column indices
    >>> # For row 1, column indices are indices[indptr[1]:indptr[2]] = indices[2:3] = indices[2]
    >>> # For row 1, data indices are data[indptr[1]:indptr[2]] = data[2:3] = data[2]
    >>> mtx = sparse.csr_matrix((data, indices, indptr), shape=(3, 3))
    >>> mtx.todense()
    matrix([[1, 0, 2],
            [0, 0, 3],
            [4, 5, 6]])
    '''
    # Calculate the row sums of M and store them in the array n
    # axis = 0 would calculate column sums
    n = np.array(M.sum(axis=1))

    #pdb.set_trace();
    #M = M.T.dot(np.diag(n))
    return uidx, M, n


def unzip_triples(xys, with_ys=False):
    xs, ys = list(zip(*xys))
    ss, os, ps = list(zip(*xs))
    if with_ys:
        return np.array(ss), np.array(ps), np.array(os), np.array(ys)
    else:
        return np.array(ss), np.array(ps), np.array(os)


def to_tensor(xs, ys, sz):
    T = [sp.lil_matrix((sz[0], sz[1])) for _ in range(sz[2])]
    for i in range(len(xs)):
        i, j, k = xs[i]
        T[k][i, j] = ys[i]
    return T


def init_nvecs(xs, ys, sz, rank, with_T=False):
    from scipy.sparse.linalg import eigsh

    T = to_tensor(xs, ys, sz)
    T = [Tk.tocsr() for Tk in T]
    S = sum([T[k] + T[k].T for k in range(len(T))])
    _, E = eigsh(sp.csr_matrix(S), rank)
    if not with_T:
        return E
    else:
        return E, T


class memoized(object):
    '''
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    see https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    '''

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncachable, return direct function application
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            val = self.func(*args)
            self.cache[args] = val
            return val

    def __repr__(self):
        '''return function's docstring'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''support instance methods'''
        return functools.partial(self.__call__, obj)
