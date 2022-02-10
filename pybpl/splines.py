"""
B-splines utilities. For reference material on B-splines, see Kristin Branson's
"A Practical Review of Uniform B-splines":
http://vision.ucsd.edu/~kbranson/research/bsplines/bsplines.pdf
"""
import functools
import torch
from torch import Tensor

from .parameters import Parameters
from .util.stroke import dist_along_traj

__all__ = ['vectorized_bspline_coeff', 'bspline_gen_s', 'coefficient_mat',
           'get_stk_from_bspline', 'fit_bspline_to_traj']

PM = Parameters()


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
    C = torch.zeros_like(vi)

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
    lb = float(2)
    ub = float(nland + 1)
    if neval == 1:
        s = torch.tensor([lb], device=device)
    else:
        s = torch.linspace(lb, ub, neval, device=device)

    return s, lb, ub


@functools.lru_cache(maxsize=128)
def coefficient_mat(nland, neval=None, s=None, device=None):
    """Generate the B-spline coefficient matrix"""

    # generate time vector
    if s is None:
        assert neval is not None, 'neval must be provided when s not provided.'
        s, _, _ = bspline_gen_s(nland, neval, device=device)
    else:
        if s.dim() == 0:
            s = s.view(1)
        assert s.dim() == 1

    # generate index vector
    i = torch.arange(nland, dtype=s.dtype, device=device)

    # generate coefficient matrix and normalize
    vs, vi = torch.meshgrid(s, i)  # (neval, nland)
    C = vectorized_bspline_coeff(vi, vs)  # (neval, nland)
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
        neval = neval.clamp(PM.spline_min_neval, PM.spline_max_neval).item()

    C = coefficient_mat(nland, neval, s=s, device=Y.device)
    X = torch.matmul(C, Y)  # (neval,2)

    return X


def fit_bspline_to_traj(X, nland, s=None, include_resid=False):
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
    Y, residuals, _, _ = torch.linalg.lstsq(C, X, driver='gels')

    if include_resid:
        return Y, residuals

    return Y
