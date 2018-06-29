from __future__ import division, print_function
import numpy as np
from scipy.stats import multivariate_normal
import torch


# ----
# MATLAB functions
# ----

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
    assert isinstance(shape, torch.Size)
    assert isinstance(index, torch.Tensor)
    if not len(shape) == 2:
        raise NotImplementedError('only implemented for 2D case.')
    # compute inds
    rows = index % shape[0]
    cols = index / shape[0]

    return rows, cols

def sub2ind(shape, rows, cols):
    """
    A PyTorch implementation of MATLAB's "sub2ind" function

    :param shape:
    :param rows:
    :param cols:
    :return:
    """
    # checks
    assert isinstance(shape, torch.Size)
    assert isinstance(rows, torch.Tensor)
    assert isinstance(cols, torch.Tensor)
    if not len(shape) == 2:
        raise NotImplementedError('only implemented for 2D case.')
    # compute inds
    n_inds = shape[0]*shape[1]
    ind_mat = torch.arange(n_inds).view(shape[1], shape[0])
    ind_mat = torch.transpose(ind_mat, 0, 1)
    index = ind_mat[rows,cols]

    return index

def imfilter(A, h, mode='conv'):
    """
    A PyTorch implementation of MATLAB's "imfilter" function

    :param A: [tensor] image
    :param h: [tensor] filter kernel
    :return:
    """
    if not mode == 'conv':
        raise NotImplementedError("Only 'conv' mode imfilter implemented.")
    assert isinstance(A, torch.Tensor)
    assert isinstance(h, torch.Tensor)

    return torch.nn.functional.conv2d(A, h)

def fspecial(hsize, sigma, ftype='gaussian'):
    """
    Implementation of MATLAB's "fspecial" function. Create filter kernel using
    numpy, return as torch.Tensor

    :param hsize:
    :param sigma:
    :param ftype:
    :return:
    """
    if not ftype == 'gaussian':
        raise NotImplementedError("Only Gaussain kernel implemented.")
    assert isinstance(hsize, int)
    assert isinstance(sigma, float) or isinstance(sigma, int)
    assert hsize % 2 == 1, 'Image size must be odd'

    # store image midpoint
    mid = int((hsize-1)/2)
    # store 2D gaussian covariance matrix
    cov = sigma*np.eye(2)
    # initialize the kernel
    kernel = np.zeros(shape=(hsize, hsize))
    for xi in range(hsize):
        for yi in range(hsize):
            kernel[xi, yi] = multivariate_normal.pdf([mid-xi, mid-yi], cov=cov)

    return torch.tensor(kernel, dtype=torch.float32)

# ----
# Other functions
# ----

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
