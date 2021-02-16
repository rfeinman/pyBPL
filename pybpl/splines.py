"""
B-splines utilities. For reference material on B-splines, see Kristin Branson's
"A Practical Review of Uniform B-splines":
http://vision.ucsd.edu/~kbranson/research/bsplines/bsplines.pdf
"""
import functools
import torch
from torch import Tensor

from .parameters import Parameters
from .util.general import least_squares, least_squares_qr
from .util.stroke import dist_along_traj

PM = Parameters()


@functools.lru_cache(maxsize=128)
def get_vi(neval, nland, device=None):
    vi = torch.arange(nland, dtype=torch.float, device=device)
    vi = vi.unsqueeze(0).repeat(neval,1)
    return vi


@functools.lru_cache(maxsize=128)
def s_to_vs(s, nland):
    vs = s.unsqueeze(1).repeat(1,nland)
    return vs


@functools.lru_cache(maxsize=128)
def vectorized_bspline_coeff(vi, vs):
    """Spline coefficients

    from Kristin Branson's "A Practical Review of Uniform B-splines"

    Inputs vi and vs are the spline evaluation indices and times (respectively),
    each with shape [neval, nland]. The output matrix has shape [neval,nland].
    """
    assert vi.shape == vs.shape
    assert vi.dtype == vs.dtype

    # step through the conditions
    # NOTE: in the following, * stands in for 'and'
    C = torch.zeros_like(vi, dtype=torch.float)

    # sel1
    sel = (vs >= vi)*(vs < vi+1)
    diff = vs[sel] - vi[sel]
    val = torch.pow(diff, 3)
    C[sel] = val/6.
    # sel2
    sel = (vs >= vi+1)*(vs < vi+2)
    diff = vs[sel] - vi[sel] - 1
    val = -3*torch.pow(diff, 3) + 3*torch.pow(diff, 2) + 3*diff + 1
    C[sel] = val/6.
    # sel3
    sel = (vs >= vi+2)*(vs < vi+3)
    diff = vs[sel] - vi[sel] - 2
    val = 3*torch.pow(diff, 3) - 6*torch.pow(diff, 2) + 4
    C[sel] = val/6.
    # sel4
    sel = (vs >= vi+3)*(vs < vi+4)
    diff = vs[sel] - vi[sel] - 3
    val = torch.pow(1-diff, 3)
    C[sel] = val/6.

    return C


@functools.lru_cache(maxsize=128)
def bspline_gen_s(nland, neval=200, device=None):
    """Generate time points for evaluating spline.

    The convex-combination of the endpoints with five control points are 80
    percent of the last cpt and 20 percent of the control point after that.
    We return the upper and lower bounds, in addition to the timepoints.
    """
    lb = 2
    ub = nland + 1
    if neval == 1:
        s = torch.tensor([lb], dtype=torch.float, device=device)
    else:
        s = torch.linspace(lb, ub, neval, device=device)

    return s, lb, ub


@functools.lru_cache(maxsize=128)
def coefficient_mat(nland, neval, s=None, device=None):
    """Generate the B-spline coefficient matrix"""
    if s is None:
        s, _, _ = bspline_gen_s(nland, neval, device=device)
    else:
        assert s.dim() == 1
        neval = s.size(0)
    S = s_to_vs(s, nland) # (neval, nland)
    I = get_vi(neval, nland, device=device) # (neval, nland)
    C = vectorized_bspline_coeff(I, S) # (neval, nland)
    C = C / C.sum(1, keepdim=True)

    return C




# ---------------------------------------------------
#    Core functions for spline fitting/evaluation
# ---------------------------------------------------

def _check_input(x):
    assert torch.is_tensor(x)
    assert x.dim() == 2
    assert x.size(1) == 2


def get_stk_from_bspline(Y, neval=None, s=None):
    """Produce a stroke trajectory by evaluating a B-spline.

    Parameters
    ----------
    Y : Tensor
        [nland,2] input spline (control points)
    neval : int
        number of eval points (optional)
    s : Tensor
        (optional) [neval] time points for spline evaluation

    Returns
    -------
    X : Tensor
        [neval,2] output trajectory
    """
    _check_input(Y)
    nland = Y.size(0)

    # if `neval` is None, set it adaptively according to stroke size
    if neval is None and s is None:
        X = get_stk_from_bspline(Y, neval=PM.spline_max_neval)
        dist = dist_along_traj(X)
        neval = (dist / PM.spline_grain).ceil().long()
        neval = neval.clamp(PM.spline_min_neval, PM.spline_max_neval)

    C = coefficient_mat(nland, neval, s=s, device=Y.device)
    X = torch.matmul(C, Y)  # (neval,2)

    return X


def fit_bspline_to_traj(X, nland, s=None, include_resid=False, lstsq_mode='svd'):
    """Produce a B-spline from a trajectory with least-squares.

    Parameters
    ----------
    X : Tensor
        [neval,2] input trajectory
    nland : int
        number of landmarks (control points)
    s : Tensor
        (optional) [neval] time points for spline evaluation
    include_resid : bool
        whether to return the residuals of the least-squares problem
    lstsq_mode : str
        algorithm for solving the least-squares problem. Must be either
        'svd' or 'qr' (default='svd').

    Returns
    -------
    Y : Tensor
        [neval,2] output spline
    residuals : Tensor
        [2,] residuals of the least-squares problem (optional)
    """
    _check_input(X)
    neval = X.size(0)

    C = coefficient_mat(nland, neval, s=s, device=X.device)
    if lstsq_mode == 'svd':
        Y, residuals, _, _ = least_squares(C, X) # (nland, 2)
    elif lstsq_mode == 'qr':
        Y, residuals = least_squares_qr(C, X)
    else:
        raise ValueError("lstsq_mode must be either 'svd' or 'qr'.")

    if include_resid:
        return Y, residuals

    return Y
