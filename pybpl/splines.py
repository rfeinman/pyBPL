"""
B-splines utilities. For reference material on B-splines, see Kristin Branson's
"A Practical Review of Uniform B-splines":
http://vision.ucsd.edu/~kbranson/research/bsplines/bsplines.pdf
"""
from __future__ import division, print_function
import functools
import math
import torch

from .parameters import defaultps
from .util.general import least_squares
from .util.stroke import dist_along_traj



PM = defaultps()


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
    """
    Compute spline coefficient matrix
    see Kristin Branson's "A Practical Review of Uniform B-splines"

    Parameters
    ----------
    vi : torch.Tensor
        [neval,nland] spline evaluation indices
    vs : torch.Tensor
        [neval,nland] spline evaluation times

    Returns
    -------
    C : torch.Tensor
        [neval,nland] spline coefficient matrix
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
    """
    Generate time points for evaluating spline.
    The convex-combination of the endpoints with five control points are 80
    percent of the last cpt and 20 percent of the control point after that.

    Parameters
    ----------
    nland : int
        number of landmarks (control points)
    neval : int
        number of eval points

    Returns
    -------
    s : torch.Tensor
        [neval,] time points for spline eval
    lb : float
        lower bound
    ub : float
        upper bound
    """
    lb = 2.
    ub = float(nland+1)
    if neval == 1:
        s = torch.tensor([lb], dtype=torch.float, device=device)
    else:
        s = torch.linspace(lb, ub, neval, device=device)

    return s, lb, ub

def bspline_eval(s, Y):
    """
    Produce a trajectory from a B-spline.

    Parameters
    ----------
    s : torch.Tensor
        [neval,] time points for spline eval
    Y : torch.Tensor
        [nland,2] input spline (control points)

    Returns
    -------
    X : torch.Tensor
        [neval,2] output trajectory
    """
    if s.shape == torch.Size([]):
        s = s.view(1)
    assert len(s.shape) == 1
    assert len(Y.shape) == 2 and Y.shape[1] == 2
    neval = s.shape[0]
    nland = Y.shape[0]

    # compute spline coefficients
    S = s_to_vs(s, nland) # (neval, nland)
    I = get_vi(neval, nland, device=Y.device) # (neval, nland)
    A = vectorized_bspline_coeff(I, S) # (neval, nland)
    Cof = A / torch.sum(A, dim=1, keepdim=True) # (neval, nland)

    X = Cof @ Y # (neval,nland) @ (nland,2) = (neval,2)

    return X

def bspline_fit(s, X, nland, include_resid=False):
    """
    Produce a B-spline from a trajectory (via least-squares).

    Parameters
    ----------
    s : torch.Tensor
        [neval,] time points for spline eval
    X : torch.Tensor
        [neval,2] input trajectory
    nland : int
        number of landmarks (control points) for the spline
    include_resid : bool
        whether to return the residuals of the least-squares problem

    Returns
    -------
    Y : torch.Tensor
        [nland,2] output spline
    residuals : torch.Tensor
        [2,] residuals of the least-squares problem (optional)
    """
    neval = s.size(0)
    assert X.shape == (neval, 2)

    # compute spline coefficients
    S = s_to_vs(s, nland) # (neval, nland)
    I = get_vi(neval, nland, device=X.device) # (neval, nland)
    A = vectorized_bspline_coeff(I, S) # (neval, nland)
    Cof = A / torch.sum(A, dim=1, keepdim=True) # (neval, nland)

    # solve least squares problem
    Y, residuals, _, _ = least_squares(Cof, X) # (nland, 2)

    if include_resid:
        return Y, residuals
    else:
        return Y

def get_stk_from_bspline(Y, neval=None):
    """
    Produce a trajectory from a B-spline.
    NOTE: this is a wrapper for bspline_eval (first produces time points)

    Parameters
    ----------
    Y : torch.Tensor
        [nland,2] input spline (control points)
    neval : int
        number of eval points (optional)

    Returns
    -------
    X : torch.Tensor
        [neval,2] output trajectory
    """
    assert isinstance(Y, torch.Tensor)
    assert len(Y.shape) == 2 and Y.shape[1] == 2
    nland = Y.shape[0]

    # if `neval` is None, set it adaptively according to stroke size
    if neval is None:
        # check the stroke size
        s, _, _ = bspline_gen_s(nland, PM.spline_max_neval, device=Y.device)
        stk = bspline_eval(s, Y.detach())
        dist = dist_along_traj(stk)
        # set neval based on stroke size
        neval = math.ceil(dist/PM.spline_grain)
        # threshold
        neval = max(neval, PM.spline_min_neval)
        neval = min(neval, PM.spline_max_neval)

    # generate time points
    s, _, _ = bspline_gen_s(nland, neval, device=Y.device)
    # compute trajectory
    X = bspline_eval(s, Y)

    return X

def fit_bspline_to_traj(X, nland, include_resid=False):
    """
    Produce a B-spline from a trajectory (via least-squares).
    NOTE: this is a wrapper for bspline_fit (first produces time points)

    Parameters
    ----------
    X : torch.Tensor
        [neval,2] input trajectory
    nland : int
        number of landmarks (control points)
    include_resid : bool
        whether to return the residuals of the least-squares problem

    Returns
    -------
    Y : torch.Tensor
        [neval,2] output spline
    residuals : torch.Tensor
        [2,] residuals of the least-squares problem (optional)
    """
    assert isinstance(X, torch.Tensor)
    assert len(X.shape) == 2 and X.shape[1] == 2

    # generate time points
    s, _, _ = bspline_gen_s(nland, neval=len(X), device=X.device)
    # compute spline
    if include_resid:
        Y, residuals = bspline_fit(s, X, nland, include_resid=True)
        return Y, residuals
    else:
        Y = bspline_fit(s, X, nland, include_resid=False)
        return Y
