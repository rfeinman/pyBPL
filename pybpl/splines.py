from __future__ import division, print_function
import warnings

def bspline_eval(sval, cpts):
    """
    Fit a uniform, cubic B-spline

    :param sval: [(k,) array] vector, where 0 <= sval(i) <= n
    :param cpts: [(n,2) array] array of control points
    :return:
        y: [(k,2) array] the output of spline
        Cof: TODO
    """
    warnings.warn("'bspline_eval' function not yet implemented")
    y = None
    Cof = None

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
    warnings.warn("'bspline_fit' function not yet implemented")
    P = None

    return P

def bspline_gen_s(nland, neval):
    """
    Generate time points for evaluating spline.

    The convex-combination of the endpoints with five control points are 80
    percent of the last cpt and 20 percent of the control point after that.

    :param nland: [int] number of landmarks
    :param neval: [int] number of evaluations
    :return:
        s: the time points used to evaluate spline
    """
    warnings.warn("'bspline_gen_s' function not yet implemented")
    s = None
    lb = None
    ub = None

    return s, lb, ub

def fit_bspline_to_traj(stk, nland):
    """
    Fit a b-spline to 'stk' with 'nland' landmarks

    :param stk: TODO
    :param nland: TODO
    :return:
        P: TODO
    """
    warnings.warn("'fit_bspline_to_traj' function not yet implemented")
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
    warnings.warn("'get_stk_from_bspline' function not yet implemented")
    stk = None

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
    warnings.warn("'vectorized_bspline_coeff' function not yet implemented")
    C = None

    return C