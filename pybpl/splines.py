from __future__ import division, print_function
import warnings
import torch

def bspline_eval(sval, cpts):
    """
    Fit a uniform, cubic B-spline

    :param sval: [(k,) array] vector, where 0 <= sval(i) <= n
    :param cpts: [(n,2) array] array of control points
    :return:
        y: [(k,2) array] the output of spline
        Cof: TODO
    """
    raise NotImplementedError
    assert len(sval.shape) == 1
    L = cpts.shape[0]
    ns = len(sval)

    list_sval = [sval for _ in range(L)]
    S = Variable(torch.cat(list_sval, 1))  # wait, does sval need to be
    list_L = [torch.Tensor(np.arange(L)).view(1, -1) for _ in range(ns)]
    I = Variable(torch.cat(list_L, 0))
    Cof = vectorized_bspline_coeff(I, S)
    y1 = torch.mm(Cof, cpts[:, 0])
    y2 = torch.mm(Cof, cpts[:, 1])
    y = torch.cat((y1, y2), 1)

    return y, Cof

def bspline_fit(sval, X, L):
    """
    Fit a bspline using least-squares

    :param sval: [(N,) array] time points
    :param X: [(N,2) array] data points
    :param L: [int] number of control points to fit
    :return:
        P: [(L,2) array] optimal control points
    """
    raise NotImplementedError
    P = None

    return P

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
    lb = 2
    ub = nland + 1
    assert ub > lb
    if neval == 1:
        s = torch.tensor([lb], dtype=torch.float)
    else:
        s = torch.linspace(lb, ub, neval)

    return s, lb, ub

def fit_bspline_to_traj(stk, nland):
    """
    Fit a b-spline to 'stk' with 'nland' landmarks

    :param stk: TODO
    :param nland: TODO
    :return:
        P: TODO
    """
    raise NotImplementedError
    P = None

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
    raise NotImplementedError
    # brenden's code finds number of eval points adaptively.
    # Can consider doing this if things take too long.
    # I worry it may mess with gradients by making them more piecewise

    nland = P.size[0]

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
        raise NotImplementedError('dynamic n evaluation not implemented')

    s = bspline_gen_s(nland, neval)
    stk = bspline_eval(s, P)
    return stk

def vectorized_bspline_coeff(vi, vs):
    """
    TODO
    See Kristin Branson's "A Practical Review of Uniform B-splines"

    :param vi: [(n,m) array] TODO
    :param vs: [(n,m) array] TODO
    :return:
        C: [(n,) array] the coefficients
    """
    raise NotImplementedError
    assert vi.shape == vs.shape
    C = torch.zeros(vi.shape)

    # in the following, * stands in for 'and'
    sel1 = (vs >= vi) * (vs < vi + 1)
    C[sel1] = (1 / 6.) * torch.pow((vs[sel1] - vi[sel1]), 3)

    sel2 = (vs >= vi + 1) * (vs < vi + 2)
    C[sel2] = (1 / 6.) * (
            -3. * torch.pow((vs[sel2] - vi[sel2] - 1), 3) +
            3. * torch.pow((vs[sel2] - vi[sel2] - 1), 2) +
            3. * (vs[sel2] - vi[sel2] - 1) +
            1)

    sel3 = (vs >= vi + 2) * (vs < vi + 3)
    C[sel3] = (1 / 6.) * (3 * torch.pow((vs[sel3] - vi[sel3] - 2), 3) -
                          6 * torch.pow((vs[sel3] - vi[sel3] - 2), 2) + 4)

    sel4 = (vs >= vi + 3) * (vs < vi + 4)
    C[sel4] = (1 / 6.) * torch.pow((1 - (vs(sel4) - vi(sel4) - 3)), 3)

    return C