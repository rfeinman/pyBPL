from __future__ import division, print_function
import numpy as np
import torch


def least_squares(a, b, rcond=None):
    """
    A PyTorch implementation of NumPy's "linalg.lstsq" function

    Parameters
    ----------
    a : (m,n) tensor
        "Coefficient" matrix
    b : (m,) or (m,k) tensor
        Ordinate or "dependent variable" values
    rcond : float
        Cutt-off ratio for small singular values of a. For the
        purposes of rank determination, singular values are treated
        as zero if they are smaller than rcond times the largest
        singular value of a. If `None`, the default will use the
        machine precision times max(m, n)

    Returns
    -------
    x : (n,) or (n,k) tensor
        Least-squares solution
    residuals: (1,) or (k,) or (0,) tensor
        Sums of residuals; squared Euclidean 2-norm for each column
        in b - a*x. If rank(a) < n or m <= n, this is an empty
        array. If b is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (k,)
    rank : int
        Rank of matrix a
    s : (min(m,n),) tensor
        Singular values of matrix a

    """
    m,n = a.shape
    if rcond is None:
        rcond = max(a.shape)*torch.finfo(a.dtype).eps
    U, s, V = torch.svd(a)
    rank = torch.sum(s > rcond*s[0]).item()
    s_inv = torch.where(s > rcond*s[0], s.reciprocal(), torch.zeros_like(s))
    x = V @ torch.diag(s_inv) @ U.transpose(0,1) @ b
    if rank < n or m <= n:
        residuals = torch.tensor([])
    else:
        residuals = torch.sum((a@x - b)**2, 0, keepdim=len(b.shape)==1)

    return x, residuals, rank, s

def ind2sub(shape, index):
    """
    A PyTorch implementation of MATLAB's "ind2sub" function

    Parameters
    ----------
    shape : torch.Size or list or tuple
        shape of the hypothetical 2D matrix
    index : (n,) tensor
        indices to convert

    Returns
    -------
    rows : (n,) tensor
    cols : (n,) tensor
    """
    # checks
    assert isinstance(shape, torch.Size) or \
           isinstance(shape, list) or \
           isinstance(shape, tuple)
    assert isinstance(index, torch.Tensor) and len(index.shape) == 1
    valid_index = index < shape[0]*shape[1]
    assert valid_index.all()
    if not len(shape) == 2:
        raise NotImplementedError('only implemented for 2D case.')
    # compute inds
    cols = index % shape[0]
    rows = index / shape[0]

    return rows, cols

def sub2ind(shape, rows, cols):
    """
    A PyTorch implementation of MATLAB's "sub2ind" function

    Parameters
    ----------
    shape : torch.Size or list or tuple
    rows : (n,) tensor
    cols : (n,) tensor

    Returns
    -------
    index : (n,) tensor
    """
    # checks
    assert isinstance(shape, torch.Size) or \
           isinstance(shape, list) or \
           isinstance(shape, tuple)
    assert isinstance(rows, torch.Tensor) and len(rows.shape) == 1
    assert isinstance(cols, torch.Tensor) and len(cols.shape) == 1
    assert len(rows) == len(cols)
    valid_rows = rows < shape[0]
    valid_cols = cols < shape[1]
    assert valid_cols.all() and valid_cols.all()
    if not len(shape) == 2:
        raise NotImplementedError('only implemented for 2D case.')
    # compute inds
    ind_mat = torch.arange(shape[0]*shape[1]).view(shape)
    index = ind_mat[rows.long(), cols.long()]

    return index

def imfilter(A, h, mode='conv'):
    """
    A PyTorch implementation of MATLAB's "imfilter" function

    :param A: [(m,n) tensor] image
    :param h: [(k,l) tensor] filter kernel
    :return:
    """
    if not mode == 'conv':
        raise NotImplementedError("Only 'conv' mode imfilter implemented.")
    assert isinstance(A, torch.Tensor)
    assert isinstance(h, torch.Tensor)
    assert len(A.shape) == 2
    assert len(h.shape) == 2
    m, n = A.shape
    k, l = h.shape
    pad_x = k // 2
    pad_y = l // 2

    A_filt = torch.nn.functional.conv2d(
        A.view(1,1,m,n), h.view(1,1,k,l), padding=(pad_x, pad_y)
    )
    A_filt = A_filt[0,0]

    return A_filt

def fspecial(hsize, sigma, ftype='gaussian'):
    """
    Implementation of MATLAB's "fspecial" function for option ftype='gaussian'.
    Calculate the 2-dimensional gaussian kernel which is the product of two
    gaussian distributions for two different variables (in this case called
    x and y)

    :param hsize:
    :param sigma:
    :param ftype:
    :return:
    """
    if not ftype == 'gaussian':
        raise NotImplementedError("Only Gaussain kernel implemented.")
    assert isinstance(hsize, int)
    if isinstance(sigma, torch.Tensor):
        assert sigma.shape == torch.Size([])
        assert sigma.dtype == torch.float
    else:
        assert isinstance(sigma, float) or isinstance(sigma, int)
    assert hsize % 2 == 1, 'Image size must be odd'

    # create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(hsize, dtype=torch.float)
    x_grid = x_cord.repeat(hsize).view(hsize, hsize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    # store the mean
    mean = (hsize-1)//2
    # compute the kernel
    kernel = torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*sigma**2)
    )
    kernel = kernel / (2.*np.pi*sigma**2)
    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    return kernel

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
