"""
B-splines utilities. For reference material on B-splines, see Kristin Branson's
"A Practical Review of Uniform B-splines":
http://vision.ucsd.edu/~kbranson/research/bsplines/bsplines.pdf
"""
from __future__ import division, print_function
import warnings
import numpy as np
import torch

def bspline_eval(sval, cpts):
    """
    Fit a uniform, cubic B-spline

    :param sval: [(neval,) tensor] vector, where 0 <= sval(i) <= n
    :param cpts: [(ncpt,2) tensor] array of control points
    :return:
        y: [(neval,2) tensor] the output of spline
        Cof: [(neval,ncpt) tensor] TODO
    """
    if sval.shape == torch.Size([]):
        sval = sval.view(1)
    assert len(sval.shape) == 1
    assert len(cpts.shape) == 2
    neval = sval.shape[0]
    ncpt = cpts.shape[0]
    y = torch.zeros(neval, 2, dtype=torch.float)

    # these will both have shape (neval,ncpt)
    S = torch.cat(
        [sval.view(-1,1) for _ in range(ncpt)],
        dim=1
    )
    I = torch.cat(
        [torch.arange(ncpt, dtype=torch.float32).view(1,-1)
         for _ in range(neval)],
        dim=0
    )
    # this will have shape (neval,ncpt)
    Cof = vectorized_bspline_coeff(I, S)
    # normalize rows of Cof
    Cof = Cof / torch.sum(Cof, dim=1).view(-1,1)
    # multiply (neval,ncpt) x (ncpt,1) = (neval, 1)
    y[:,0] = torch.mm(Cof, cpts[:,0].view(-1,1)).view(-1)
    y[:,1] = torch.mm(Cof, cpts[:,1].view(-1,1)).view(-1)

    return y, Cof

def bspline_fit(sval, X, nland):
    """
    Fit a bspline using least-squares.
    TODO: update this to remain in PyTorch. Right now we convert to numpy

    :param sval: [(ntraj,) tensor] time points
    :param X: [(ntraj,2) tensor] data points
    :param nland: [int] number of control points to fit
    :return:
        P: [(L,2) tensor] optimal control points
        is_singular: [bool] whether the least-squares problem was singular
    """
    ntraj = sval.size(0)
    assert X.shape == (ntraj, 2)

    S = sval.unsqueeze(1).repeat(1,nland) # (ntraj, nland)
    I = torch.arange(nland).unsqueeze(0).repeat(ntraj, 1) # (ntraj, nland)
    A = vectorized_bspline_coeff(I, S) # (ntraj, nland)
    Cof = A / torch.sum(A, dim=1, keepdim=True) # (ntraj, nland)

    # solve least squares problem
    a = Cof.transpose(0,1) @ Cof # (nland, nland)
    b = Cof.transpose(0,1) @ X # (nland, 2)
    P, _, rank, _ = np.linalg.lstsq(a.numpy(), b.numpy(), rcond=None) # (nland, 2)
    P = torch.tensor(P, dtype=torch.float)

    # check singularity of least squares problem
    is_singular = rank < Cof.shape[1]

    return P, is_singular

def bspline_gen_s(nland, neval=200):
    """
    Generate time points for evaluating spline.

    The convex-combination of the endpoints with five control points are 80
    percent of the last cpt and 20 percent of the control point after that.

    :param nland: [int] number of landmarks
    :param neval: [int] number of evaluations
    :return:
        s: the time points used to evaluate spline
        lb: TODO
        ub: TODO
    """
    lb = torch.tensor(2, dtype=torch.float)
    ub = torch.tensor(nland+1, dtype=torch.float)
    assert ub > lb
    if neval == 1:
        s = torch.tensor([lb], dtype=torch.float)
    else:
        s = torch.linspace(lb, ub, neval)

    return s, lb, ub

def fit_bspline_to_traj(stk, nland):
    """
    Fit a b-spline to 'stk' with 'nland' landmarks

    :param stk: [(N,2) tensor] substroke trajectory
    :param nland: [int] number of landmarks
    :return:
        P: [(nland,2) tensor] spline control points
    """
    neval = len(stk)
    s, _, _ = bspline_gen_s(nland, neval)
    P = bspline_fit(s, stk, nland)

    return P

def get_stk_from_bspline(P, neval=None):
    """
    Get a motor trajectory from the b-spline control points, using an adaptive
    method to choose the number of evaluations based on the distance along the
    trajectory.

    :param P: [(ncpt,2) array] control points
    :param neval: [int] optional; number of evaluations. Otherwise, we choose
                    this adaptively
    :return:
        stk: [(m,2) array] trajectory
    """
    assert isinstance(P, torch.Tensor)
    assert len(P.shape) == 2
    assert P.shape[1] == 2
    nland = P.shape[0]

    # brenden's code finds number of eval points adaptively.
    # Can consider doing this if things take too long.
    # I worry it may mess with gradients by making them more piecewise
    if neval is None:
        # % set the number of evaluations adaptively,
        # % based on the size of the stroke
        # PM = defaultps;
        # neval = PM.spline_min_neval;
        # s = bspline_gen_s(nland,neval);
        # stk = bspline_eval(s,P);
        # sumdist = sum_pair_dist(stk);
        # neval = max(neval,ceil(sumdist./PM.spline_grain));
        # neval = min(neval,PM.spline_max_neval);
        warnings.warn(
            "cannot yet set 'neval' adaptively... using neval=200 for now."
        )
    # s has shape (neval,)
    s, _, _ = bspline_gen_s(nland, neval)
    # stk has shape (neval,2)
    stk, _ = bspline_eval(s, P)

    return stk

def vectorized_bspline_coeff(vi, vs):
    """
    TODO

    :param vi: [(neval, ncpt) tensor] TODO
    :param vs: [(neval, ncpt) tensor] TODO
    :return:
        C: [(neval, ncpt) tensor] the coefficients
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