from __future__ import division, print_function
import numpy as np
import torch


def ind2sub(shape, index):
    """
    A PyTorch implementation of MATLAB's "ind2sub" function

    :param shape: [torch.Size] shape of the hypothetical 2D matrix
    :param index: [(n,) tensor] indices to convert
    :return:
        yi: [(n,) tensor] y sub-indices
        xi: [(n,) tensor] x sub-indices
    """
    # checks
    assert isinstance(index, torch.Tensor)
    assert isinstance(shape, torch.Size)
    if not len(shape) == 2:
        raise NotImplementedError('only implemented for 2D case.')
    # compute inds
    xi = index % shape[0]
    yi = index / shape[0]

    return yi, xi

def aeq(x, y, tol=2.22e-6):
    if isinstance(x, list):
        assert isinstance(y, list)
        diff = np.abs(np.asarray(x) - np.asarray(y))
        acceptable = diff < tol
        r = acceptable.all()
    elif isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray)
        assert x.shape == y.shape
        diff = np.abs(x.flatten() - y.flatten())
        acceptable = diff < tol
        r = acceptable.all()
    elif isinstance(x, torch.Tensor):
        assert isinstance(y, torch.Tensor)
        assert x.shape == y.shape
        diff = torch.abs(x.view(-1) - y.view(-1))
        acceptable = diff < tol
        r = acceptable.all()
    else:
        diff = np.abs(x - y)
        r = diff < tol

    return r

def inspect_dir(dir_name):
    raise NotImplementedError

def makestr(varargin):
    raise NotImplementedError

def rand_discrete(vcell, wts):
    raise NotImplementedError

def rand_reset():
    raise NotImplementedError

def randint(m, n, rg):
    raise NotImplementedError

def sptight(nrow, ncol, indx):
    raise NotImplementedError
