from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
import torch
import torch.distributions as dist
import os
import scipy.io as io

int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def clear_shape_type(M):
    raise NotImplementedError

def clear_invscales_type(M):
    raise NotImplementedError

def set_affine_to_image(M, I):
    raise NotImplementedError

def set_affine(M, target_com, target_range, missing):
    raise NotImplementedError

def set_affine_motor(M, target_com, target_range):
    raise NotImplementedError

def apply_each_substroke(nested, fnc):
    raise NotImplementedError
    nested = None

    return nested

def flip_stroke(S):
    raise NotImplementedError

def merge_strokes(S1, S2):
    raise NotImplementedError
    Z = None

    return Z

def split_stroke(Z, bid_start_of_second):
    raise NotImplementedError
    Z1 = None
    Z2 = None

    return Z1, Z2

def all_merge_moves(M):
    raise NotImplementedError
    moves = None
    reverse_moves = None

    return moves, reverse_moves

def all_split_moves(M):
    raise NotImplementedError
    moves = None
    reverse_moves = None

    return moves, reverse_moves

def valid_merge(M, sid):
    raise NotImplementedError
    val = None

    return val

# ----
# MATLAB functions
# ----

def ind2sub(shape, index):
    """
    A PyTorch implementation of MATLAB's "ind2sub" function

    :param shape: [torch.Size or list or array] shape of the hypothetical 2D
                    matrix
    :param index: [(n,) tensor] indices to convert
    :return:
        yi: [(n,) tensor] y sub-indices
        xi: [(n,) tensor] x sub-indices
    """
    # checks
    assert isinstance(shape, torch.Size) or \
           isinstance(shape, list) or \
           isinstance(shape, tuple) or \
           isinstance(shape, np.ndarray)
    assert isinstance(index, torch.Tensor)
    valid_index = index < shape[0]*shape[1]
    assert valid_index.all()
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
    assert isinstance(shape, torch.Size) or \
           isinstance(shape, list) or \
           isinstance(shape, tuple) or \
           isinstance(shape, np.ndarray)
    assert isinstance(rows, torch.Tensor) and len(rows.shape) == 1
    assert isinstance(cols, torch.Tensor) and len(cols.shape) == 1
    assert len(rows) == len(cols)
    valid_rows = rows < shape[0]
    valid_cols = cols < shape[1]
    assert valid_cols.all() and valid_cols.all()
    if not len(shape) == 2:
        raise NotImplementedError('only implemented for 2D case.')
    # compute inds
    n_inds = shape[0]*shape[1]
    ind_mat = torch.arange(n_inds).view(shape[1], shape[0])
    ind_mat = torch.transpose(ind_mat, 0, 1)
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
    kernel /= (2.*np.pi*sigma**2)
    # Make sure sum of values in gaussian kernel equals 1.
    kernel /= torch.sum(kernel)

    return kernel

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

def logsumexp_t(tensor):
    """
    TODO

    :param tensor: [(n,) tensor] TODO
    :return:
        tensor1: [(n,) tensor] TODO

    """
    array = logsumexp(tensor.numpy())
    tensor1 = torch.tensor(array, dtype=torch.float32)

    return tensor1

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






"""
B-splines utilities. For reference material on B-splines, see Kristin Branson's
"A Practical Review of Uniform B-splines":
http://vision.ucsd.edu/~kbranson/research/bsplines/bsplines.pdf
"""


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















# ----
# vanilla to motor
# ----

def vanilla_to_motor(shapes, invscales, first_pos, neval=200):
    """
    Create the fine-motor trajectory of a stroke (denoted 'f()' in pseudocode)
    with 'nsub' sub-strokes

    :param shapes: [(ncpt,2,nsub) tensor] spline points in normalized space
    :param invscales: [(nsub,) tensor] inverse scales for each sub-stroke
    :param first_pos: [(2,) tensor] starting location of stroke
    :param neval: [int] number of evaluations to use for each motor
                    trajectory
    :return:
        motor: [(nsub,neval,2) tensor] fine motor sequence
        motor_spline: [(ncpt,2,nsub) tensor] fine motor sequence in spline space
    """
    for elt in [shapes, invscales, first_pos]:
        assert elt is not None
        assert isinstance(elt, torch.Tensor)
    assert len(shapes.shape) == 3
    assert shapes.shape[1] == 2
    assert len(invscales.shape) == 1
    assert first_pos.shape == torch.Size([2])
    ncpt, _, nsub = shapes.shape
    motor = torch.zeros(nsub, neval, 2, dtype=torch.float)
    motor_spline = torch.zeros_like(shapes, dtype=torch.float)
    previous_pos = first_pos
    for i in range(nsub):
        # re-scale the control points
        shapes_scaled = invscales[i]*shapes[:,:,i]
        # get trajectories from b-spline
        traj = get_stk_from_bspline(shapes_scaled, neval)
        # reposition; shift by offset
        offset = traj[0] - previous_pos
        motor[i] = traj - offset
        motor_spline[:,:,i] = shapes_scaled - offset
        # update previous_pos to be last position of current traj
        previous_pos = motor[i,-1]

    return motor, motor_spline


# ----
# affine warp
# ----

def com_char(char):
    raise NotImplementedError

def affine_warp(stk, affine):
    raise NotImplementedError

def apply_warp(motor_unwarped, affine):
    raise NotImplementedError('affine warping not yet implemented.')
    cell_traj = torch.cat(motor_unwarped) # flatten substrokes
    com = com_char(cell_traj)
    b = torch.zeros(4, dtype=torch.float)
    b[:2] = affine[:2]
    b[2:4] = affine[2:4] - (affine[:2]-1)*com
    fn = lambda stk: affine_warp(stk, b)
    #motor_warped = util_character.apply_each_substroke(motor_unwarped, fn)

    return motor_warped


# ----
# render the image
# ----

def check_bounds(myt, imsize):
    """

    :param myt: [(k,2) tensor]
    :param imsize: [list or tuple]
    :return:
        out: [(k,) Byte tensor]
    """
    xt = myt[:,0]
    yt = myt[:,1]
    x_out = (torch.floor(xt) < 0) | (torch.ceil(xt) > imsize[0])
    y_out = (torch.floor(yt) < 0) | (torch.ceil(yt) > imsize[1])
    out = x_out | y_out

    return out

def pair_dist(D):
    """

    :param D: [(k,2) tensor]
    :return:
        z: [(k,) tensor]
    """
    assert isinstance(D, torch.Tensor)
    assert len(D.shape) == 2
    assert D.shape[1] == 2
    x1 = D[:-1]
    x2 = D[1:]
    z = torch.sqrt(
        torch.sum(
            torch.pow(x1-x2, 2),
            dim=1
        )
    )

    return z

def seqadd(D, lind_x, lind_y, inkval):
    """

    :param D: [(m,n) tensor]
    :param lind_x: [(k,) tensor]
    :param lind_y: [(k,) tensor]
    :param inkval: [(k,) tensor]
    :return:
    """
    assert len(lind_x) == len(lind_y) == len(inkval)
    out = check_bounds(
        torch.cat([lind_x.view(-1,1), lind_y.view(-1,1)], dim=1),
        (D.shape[0]-1, D.shape[1]-1)
    )
    lind_x = lind_x[~out].long()
    lind_y = lind_y[~out].long()
    numel = len(lind_x)
    for i in range(numel):
        D[lind_x[i], lind_y[i]] = D[lind_x[i], lind_y[i]] + inkval[i]

    return D

def space_motor_to_img(pt):
    """
    Translate all control points from spline space to image space.
    Changes all points (x, -y) -> (y, x)

    Parameters
    ----------
    pt : (nsub,neval,2) tensor
        spline point sequence for each sub-stroke

    Returns
    -------
    new_pt : (nsub,neval,2) tensor
        image point sequence for each sub-stroke
    """
    assert isinstance(pt, torch.Tensor)
    assert len(pt.shape) == 3
    new_pt = torch.cat([-pt[:,:,1:], pt[:,:,:1]], dim=2)

    return new_pt

def render_image(cell_traj, epsilon, blur_sigma, parameters):
    """
    TODO

    Parameters
    ----------
    cell_traj : (nsub_total,neval,2) tensor
        TODO
    epsilon : float
        TODO
    blur_sigma : float
        TODO
    parameters : defaultps
        TODO

    Returns
    -------
    pimg : (H, W) tensor
        TODO
    ink_off_page : bool
        TODO
    """
    # convert to image space
    # Note: traj_img is still shape (nsub_total,neval,2)
    traj_img = space_motor_to_img(cell_traj)

    # get relevant parameters
    imsize = parameters.imsize
    ink = parameters.ink_pp
    max_dist = parameters.ink_max_dist

    # draw the trajectories on the image
    pimg = torch.zeros(imsize, dtype=torch.float)
    nsub_total = traj_img.shape[0]
    ink_off_page = False
    for i in range(nsub_total):
        # get trajectory for current sub-stroke
        myt = traj_img[i] # shape (neval,2)
        # reduce trajectory to only those points that are in bounds
        out = check_bounds(myt, imsize) # boolean; shape (neval,)
        if out.any():
            ink_off_page = True
        if out.all():
            continue
        myt = myt[~out]

        # compute distance between each trajectory point and the next one
        if myt.shape[0] == 1:
            myink = ink
        else:
            dist = pair_dist(myt) # shape (k,)
            dist = torch.min(dist, max_dist)
            dist = torch.cat([dist[:1], dist])
            myink = (ink/max_dist)*dist # shape (k,)

        # make sure we have the minimum amount of ink, if a particular
        # trajectory is very small
        sumink = torch.sum(myink)
        if torch.abs(sumink) < 1e-6:
            nink = myink.shape[0]
            myink = (ink/nink)*torch.ones_like(myink)
        elif sumink < ink:
            myink = (ink/sumink)*myink
        assert torch.sum(myink) > (ink-1e-4)

        # share ink with the neighboring 4 pixels
        x = myt[:,0]
        y = myt[:,1]
        xfloor = torch.floor(x)
        yfloor = torch.floor(y)
        xceil = torch.ceil(x)
        yceil = torch.ceil(y)
        x_c_ratio = x - xfloor
        y_c_ratio = y - yfloor
        x_f_ratio = 1 - x_c_ratio
        y_f_ratio = 1 - y_c_ratio

        # paint the image
        pimg = seqadd(pimg, xfloor, yfloor, myink*x_f_ratio*y_f_ratio)
        pimg = seqadd(pimg, xceil, yfloor, myink*x_c_ratio*y_f_ratio)
        pimg = seqadd(pimg, xfloor, yceil, myink*x_f_ratio*y_c_ratio)
        pimg = seqadd(pimg, xceil, yceil, myink*x_c_ratio*y_c_ratio)


    # filter the image to get the desired brush-stroke size
    a = parameters.ink_a
    b = parameters.ink_b
    ink_ncon = parameters.ink_ncon
    H_broaden = b*torch.tensor(
        [[a/12, a/6, a/12],[a/6, 1-a, a/6],[a/12, a/6, a/12]],
        dtype=torch.float
    )
    for i in range(ink_ncon):
        pimg = imfilter(pimg, H_broaden, mode='conv')

    # store min and maximum pimg values for truncation
    min_val = torch.tensor(0., dtype=torch.float)
    max_val = torch.tensor(1., dtype=torch.float)

    # truncate
    pimg = torch.min(pimg, max_val)

    # filter the image to get Gaussian
    # noise around the area with ink
    if blur_sigma > 0:
        fsize = 11
        H_gaussian = fspecial(fsize, blur_sigma, ftype='gaussian')
        pimg = imfilter(pimg, H_gaussian, mode='conv')
        pimg = imfilter(pimg, H_gaussian, mode='conv')

    # final truncation
    pimg = torch.min(pimg, max_val)
    pimg = torch.max(pimg, min_val)

    # probability of each pixel being on
    pimg = (1-epsilon)*pimg + epsilon*(1-pimg)

    return pimg, ink_off_page


# ----
# apply render
# ----

def apply_render(P, affine, epsilon, blur_sigma, parameters):
    """
    TODO

    Parameters
    ----------
    P : list of StrokeToken
        TODO
    affine : TODO
        TODO
    epsilon : TODO
        TODO
    blur_sigma : TODO
        TODO
    parameters : defaultps
        TODO

    Returns
    -------
    pimg : TODO
        TODO
    ink_off_page : TODO
        TODO
    """
    # get motor for each part
    motor = [p.motor for p in P]
    # apply affine transformation if needed
    if affine is not None:
        motor = apply_warp(motor, affine)
    motor_flat = torch.cat(motor) # flatten substrokes
    pimg, ink_off_page = render_image(
        motor_flat, epsilon, blur_sigma, parameters
    )

    return pimg, ink_off_page











# --------------------- #
# parent 'Part' classes
# --------------------- #

class PartToken(object):
    """
    An abstract base class for part tokens. Holds all token-level parameters
    of the part.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def optimizable_parameters(self, eps=1e-4):
        pass


class Part(object):
    """
    An abstract base class for parts. Holds all type-level parameters of the
    part.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def optimizable_parameters(self, eps=1e-4):
        pass

    @abstractmethod
    def sample_token(self):
        pass

    @abstractmethod
    def score_token(self, token):
        pass


# ---------------------- #
# child 'Stroke' classes
# ---------------------- #


class StrokeToken(PartToken):
    """
    Stroke tokens hold all token-level parameters of the stroke.

    Parameters
    ----------
    shapes : (ncpt, 2, nsub) tensor
        shapes tokens
    invscales : (nsub,) tensor
        invscales tokens
    """
    def __init__(self, shapes, invscales):
        super(StrokeToken, self).__init__()
        self.shapes = shapes
        self.invscales = invscales
        self.position = None

    def optimizable_parameters(self, eps=1e-4):
        """
        Returns a list of parameters that can be optimized via gradient descent.
        Includes lists of lower and upper bounds, with one per parameter.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        params : list
            optimizable parameters
        lbs : list
            lower bound for each parameter
        ubs : list
            upper bound for each parameter
                """
        params = [self.shapes, self.invscales]
        lbs = [None, torch.full(self.invscales.shape, eps)]
        ubs = [None, None]

        return params, lbs, ubs

    @property
    def motor(self):
        """
        TODO
        """
        assert self.position is not None
        motor, _ = vanilla_to_motor(
            self.shapes, self.invscales, self.position
        )

        return motor

    @property
    def motor_spline(self):
        """
        TODO
        """
        assert self.position is not None
        _, motor_spline = vanilla_to_motor(
            self.shapes, self.invscales, self.position
        )

        return motor_spline
# ----
# vanilla to motor
# ----

def vanilla_to_motor(shapes, invscales, first_pos, neval=200):
    """
    Create the fine-motor trajectory of a stroke (denoted 'f()' in pseudocode)
    with 'nsub' sub-strokes

    :param shapes: [(ncpt,2,nsub) tensor] spline points in normalized space
    :param invscales: [(nsub,) tensor] inverse scales for each sub-stroke
    :param first_pos: [(2,) tensor] starting location of stroke
    :param neval: [int] number of evaluations to use for each motor
                    trajectory
    :return:
        motor: [(nsub,neval,2) tensor] fine motor sequence
        motor_spline: [(ncpt,2,nsub) tensor] fine motor sequence in spline space
    """
    for elt in [shapes, invscales, first_pos]:
        assert elt is not None
        assert isinstance(elt, torch.Tensor)
    assert len(shapes.shape) == 3
    assert shapes.shape[1] == 2
    assert len(invscales.shape) == 1
    assert first_pos.shape == torch.Size([2])
    ncpt, _, nsub = shapes.shape
    motor = torch.zeros(nsub, neval, 2, dtype=torch.float)
    motor_spline = torch.zeros_like(shapes, dtype=torch.float)
    previous_pos = first_pos
    for i in range(nsub):
        # re-scale the control points
        shapes_scaled = invscales[i]*shapes[:,:,i]
        # get trajectories from b-spline
        traj = get_stk_from_bspline(shapes_scaled, neval)
        # reposition; shift by offset
        offset = traj[0] - previous_pos
        motor[i] = traj - offset
        motor_spline[:,:,i] = shapes_scaled - offset
        # update previous_pos to be last position of current traj
        previous_pos = motor[i,-1]

    return motor, motor_spline


# ----
# affine warp
# ----

def com_char(char):
    raise NotImplementedError

def affine_warp(stk, affine):
    raise NotImplementedError

def apply_warp(motor_unwarped, affine):
    raise NotImplementedError('affine warping not yet implemented.')
    cell_traj = torch.cat(motor_unwarped) # flatten substrokes
    com = com_char(cell_traj)
    b = torch.zeros(4, dtype=torch.float)
    b[:2] = affine[:2]
    b[2:4] = affine[2:4] - (affine[:2]-1)*com
    fn = lambda stk: affine_warp(stk, b)
    #motor_warped = util_character.apply_each_substroke(motor_unwarped, fn)

    return motor_warped


# ----
# render the image
# ----

def check_bounds(myt, imsize):
    """

    :param myt: [(k,2) tensor]
    :param imsize: [list or tuple]
    :return:
        out: [(k,) Byte tensor]
    """
    xt = myt[:,0]
    yt = myt[:,1]
    x_out = (torch.floor(xt) < 0) | (torch.ceil(xt) > imsize[0])
    y_out = (torch.floor(yt) < 0) | (torch.ceil(yt) > imsize[1])
    out = x_out | y_out

    return out

def pair_dist(D):
    """

    :param D: [(k,2) tensor]
    :return:
        z: [(k,) tensor]
    """
    assert isinstance(D, torch.Tensor)
    assert len(D.shape) == 2
    assert D.shape[1] == 2
    x1 = D[:-1]
    x2 = D[1:]
    z = torch.sqrt(
        torch.sum(
            torch.pow(x1-x2, 2),
            dim=1
        )
    )

    return z

def seqadd(D, lind_x, lind_y, inkval):
    """

    :param D: [(m,n) tensor]
    :param lind_x: [(k,) tensor]
    :param lind_y: [(k,) tensor]
    :param inkval: [(k,) tensor]
    :return:
    """
    assert len(lind_x) == len(lind_y) == len(inkval)
    out = check_bounds(
        torch.cat([lind_x.view(-1,1), lind_y.view(-1,1)], dim=1),
        (D.shape[0]-1, D.shape[1]-1)
    )
    lind_x = lind_x[~out].long()
    lind_y = lind_y[~out].long()
    numel = len(lind_x)
    for i in range(numel):
        D[lind_x[i], lind_y[i]] = D[lind_x[i], lind_y[i]] + inkval[i]

    return D

def space_motor_to_img(pt):
    """
    Translate all control points from spline space to image space.
    Changes all points (x, -y) -> (y, x)

    Parameters
    ----------
    pt : (nsub,neval,2) tensor
        spline point sequence for each sub-stroke

    Returns
    -------
    new_pt : (nsub,neval,2) tensor
        image point sequence for each sub-stroke
    """
    assert isinstance(pt, torch.Tensor)
    assert len(pt.shape) == 3
    new_pt = torch.cat([-pt[:,:,1:], pt[:,:,:1]], dim=2)

    return new_pt

def render_image(cell_traj, epsilon, blur_sigma, parameters):
    """
    TODO

    Parameters
    ----------
    cell_traj : (nsub_total,neval,2) tensor
        TODO
    epsilon : float
        TODO
    blur_sigma : float
        TODO
    parameters : defaultps
        TODO

    Returns
    -------
    pimg : (H, W) tensor
        TODO
    ink_off_page : bool
        TODO
    """
    # convert to image space
    # Note: traj_img is still shape (nsub_total,neval,2)
    traj_img = space_motor_to_img(cell_traj)

    # get relevant parameters
    imsize = parameters.imsize
    ink = parameters.ink_pp
    max_dist = parameters.ink_max_dist

    # draw the trajectories on the image
    pimg = torch.zeros(imsize, dtype=torch.float)
    nsub_total = traj_img.shape[0]
    ink_off_page = False
    for i in range(nsub_total):
        # get trajectory for current sub-stroke
        myt = traj_img[i] # shape (neval,2)
        # reduce trajectory to only those points that are in bounds
        out = check_bounds(myt, imsize) # boolean; shape (neval,)
        if out.any():
            ink_off_page = True
        if out.all():
            continue
        myt = myt[~out]

        # compute distance between each trajectory point and the next one
        if myt.shape[0] == 1:
            myink = ink
        else:
            dist = pair_dist(myt) # shape (k,)
            dist = torch.min(dist, max_dist)
            dist = torch.cat([dist[:1], dist])
            myink = (ink/max_dist)*dist # shape (k,)

        # make sure we have the minimum amount of ink, if a particular
        # trajectory is very small
        sumink = torch.sum(myink)
        if torch.abs(sumink) < 1e-6:
            nink = myink.shape[0]
            myink = (ink/nink)*torch.ones_like(myink)
        elif sumink < ink:
            myink = (ink/sumink)*myink
        assert torch.sum(myink) > (ink-1e-4)

        # share ink with the neighboring 4 pixels
        x = myt[:,0]
        y = myt[:,1]
        xfloor = torch.floor(x)
        yfloor = torch.floor(y)
        xceil = torch.ceil(x)
        yceil = torch.ceil(y)
        x_c_ratio = x - xfloor
        y_c_ratio = y - yfloor
        x_f_ratio = 1 - x_c_ratio
        y_f_ratio = 1 - y_c_ratio

        # paint the image
        pimg = seqadd(pimg, xfloor, yfloor, myink*x_f_ratio*y_f_ratio)
        pimg = seqadd(pimg, xceil, yfloor, myink*x_c_ratio*y_f_ratio)
        pimg = seqadd(pimg, xfloor, yceil, myink*x_f_ratio*y_c_ratio)
        pimg = seqadd(pimg, xceil, yceil, myink*x_c_ratio*y_c_ratio)


    # filter the image to get the desired brush-stroke size
    a = parameters.ink_a
    b = parameters.ink_b
    ink_ncon = parameters.ink_ncon
    H_broaden = b*torch.tensor(
        [[a/12, a/6, a/12],[a/6, 1-a, a/6],[a/12, a/6, a/12]],
        dtype=torch.float
    )
    for i in range(ink_ncon):
        pimg = imfilter(pimg, H_broaden, mode='conv')

    # store min and maximum pimg values for truncation
    min_val = torch.tensor(0., dtype=torch.float)
    max_val = torch.tensor(1., dtype=torch.float)

    # truncate
    pimg = torch.min(pimg, max_val)

    # filter the image to get Gaussian
    # noise around the area with ink
    if blur_sigma > 0:
        fsize = 11
        H_gaussian = fspecial(fsize, blur_sigma, ftype='gaussian')
        pimg = imfilter(pimg, H_gaussian, mode='conv')
        pimg = imfilter(pimg, H_gaussian, mode='conv')

    # final truncation
    pimg = torch.min(pimg, max_val)
    pimg = torch.max(pimg, min_val)

    # probability of each pixel being on
    pimg = (1-epsilon)*pimg + epsilon*(1-pimg)

    return pimg, ink_off_page


# ----
# apply render
# ----

def apply_render(P, affine, epsilon, blur_sigma, parameters):
    """
    TODO

    Parameters
    ----------
    P : list of StrokeToken
        TODO
    affine : TODO
        TODO
    epsilon : TODO
        TODO
    blur_sigma : TODO
        TODO
    parameters : defaultps
        TODO

    Returns
    -------
    pimg : TODO
        TODO
    ink_off_page : TODO
        TODO
    """
    # get motor for each part
    motor = [p.motor for p in P]
    # apply affine transformation if needed
    if affine is not None:
        motor = apply_warp(motor, affine)
    motor_flat = torch.cat(motor) # flatten substrokes
    pimg, ink_off_page = render_image(
        motor_flat, epsilon, blur_sigma, parameters
    )

    return pimg, ink_off_page



class Stroke(Part):
    """
    A Stroke is a probabilistic program that can generate and score
    stroke tokens

    Parameters
    ----------
    nsub : tensor
        scalar; number of sub-strokes
    ids : (nsub,) tensor
        sub-stroke ID sequence
    shapes : (ncpt, 2, nsub) tensor
        shapes types
    invscales : (nsub,) tensor
        invscales types
    lib : Library
        library instance
    """
    def __init__(self, nsub, ids, shapes, invscales, lib):
        super(Stroke, self).__init__()
        self.nsub = nsub
        self.ids = ids
        self.shapes = shapes
        self.invscales = invscales

        # distributions
        self.sigma_shape = lib.tokenvar['sigma_shape']
        self.sigma_invscale = lib.tokenvar['sigma_invscale']

    def optimizable_parameters(self, eps=1e-4):
        """
        Returns a list of parameters that can be optimized via gradient descent.
        Includes lists of lower and upper bounds, with one per parameter.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        params : list
            optimizable parameters
        lbs : list
            lower bound for each parameter
        ubs : list
            upper bound for each parameter
        """
        params = [self.shapes, self.invscales]
        lbs = [None, torch.full(self.invscales.shape, eps)]
        ubs = [None, None]

        return params, lbs, ubs

    def sample_shapes_token(self):
        """
        Sample a token of each sub-stroke's shapes

        Returns
        -------
        shapes_token : (ncpt, 2, nsub) tensor
            sampled shapes token
        """
        shapes_dist = dist.normal.Normal(self.shapes, self.sigma_shape)
        # sample shapes token
        shapes_token = shapes_dist.sample()

        return shapes_token

    def score_shapes_token(self, shapes_token):
        """
        Compute the log-likelihood of each sub-strokes's shapes

        Parameters
        ----------
        shapes_token : (ncpt, 2, nsub) tensor
            shapes tokens to score

        Returns
        -------
        ll : (nsub,) tensor
            vector of log-likelihood scores
        """
        shapes_dist = dist.normal.Normal(self.shapes, self.sigma_shape)
        # compute scores for every element in shapes_token
        ll = shapes_dist.log_prob(shapes_token)

        return ll

    def sample_invscales_token(self):
        """
        Sample a token of each sub-stroke's scale

        Returns
        -------
        invscales_token : (nsub,) tensor
            sampled scales tokens
        """
        scales_dist = dist.normal.Normal(self.invscales, self.sigma_invscale)
        while True:
            invscales_token = scales_dist.sample()
            ll = self.score_invscales_token(invscales_token)
            if not torch.any(ll == -float('inf')):
                break

        return invscales_token

    def score_invscales_token(self, invscales_token):
        """
        Compute the log-likelihood of each sub-stroke's scale

        Parameters
        ----------
        invscales_token : (nsub,) tensor
            scales tokens to score

        Returns
        -------
        ll : (nsub,) tensor
            vector of log-likelihood scores
        """
        scales_dist = dist.normal.Normal(self.invscales, self.sigma_invscale)
        # compute scores for every element in invscales_token
        ll = scales_dist.log_prob(invscales_token)

        # correction for positive only invscales
        p_below = scales_dist.cdf(0.)
        p_above = 1. - p_below
        ll = ll - torch.log(p_above)

        # don't allow invscales that are negative
        out_of_bounds = invscales_token <= 0
        ll[out_of_bounds] = -float('inf')

        return ll

    def sample_token(self):
        """
        Sample a stroke token

        Returns
        -------
        token : StrokeToken
            stroke token sample
        """
        shapes = self.sample_shapes_token()
        invscales = self.sample_invscales_token()
        token = StrokeToken(shapes, invscales)

        return token

    def score_token(self, token):
        """
        Compute the log-likelihood of a stroke token

        Parameters
        ----------
        token : StrokeToken
            stroke token to score

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the stroke token
        """
        shapes_scores = self.score_shapes_token(token.shapes)
        invscales_scores = self.score_invscales_token(token.invscales)
        ll = torch.sum(shapes_scores) + torch.sum(invscales_scores)

        return ll





class SpatialHist(object):
    """
    xlim : (2,) tensor
    ylim : (2,) tensor
    """

    def __init__(self, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim

    def fit(self, data):
        """
        Parameters
        ----------
        data : (n,2) tensor
        """
        raise NotImplementedError

    def initialize_unif(self):
        """
        """
        bounds = torch.cat([self.xlim.view(1, -1), self.ylim.view(1, -1)])
        self.dist = dist.Uniform(bounds[:, 0], bounds[:, 1])

    def sample(self, nsamp):
        """
        Parameters
        ----------
        nsamp : int

        Returns
        -------
        samples : (n,2) tensor
        """
        assert hasattr(self, 'dist'), 'model not yet fit'
        assert type(nsamp) == int or \
               (type(nsamp) == torch.Tensor and len(nsamp.shape) == 0)
        samples = self.dist.sample(torch.Size([nsamp]))

        return samples

    def score(self, data):
        """
        Parameters
        ----------
        data : (n,2) tensor

        Returns
        -------
        ll : (n,) tensor
        """
        assert hasattr(self, 'dist'), 'model not yet fit'
        assert len(data.shape) == 2
        assert data.shape[1] == 2
        ll = self.dist.log_prob(data)
        ll = ll.sum(dim=1)

        return ll


class SpatialModel(object):
    """
    Use specific distributions if 0 <= part_ID < clump_ID, or a clump
    clump distribution if part_ID >= clump_ID.

    Parameters
    ----------
    xlim :
    ylim :
    clump_ID : int
        part index at which we begin clumping
    """

    def __init__(self, xlim, ylim, clump_ID):
        self.xlim = xlim
        self.ylim = ylim
        self.clump_ID = clump_ID

    def fit(self, data, data_id):
        """
        """
        raise NotImplementedError

    def initialize_unif(self):
        """
        """
        list_SH = []
        for sid in range(self.clump_ID + 1):
            sh = SpatialHist(self.xlim, self.ylim)
            sh.initialize_unif()
            list_SH.append(sh)
        self.list_SH = list_SH

    def sample(self, part_IDs):
        """
        Parameters
        ----------
        part_IDs : (n,) tensor

        Returns
        -------
        samples : (n,2) tensor
        """
        assert hasattr(self, 'list_SH'), 'model not yet fit'
        assert isinstance(part_IDs, torch.Tensor)
        assert len(part_IDs.shape) == 1
        nsamp = len(part_IDs)
        new_IDs = self.__map_indx(part_IDs)

        # for each stroke ID
        samples = torch.zeros(nsamp, 2)
        for sid in range(self.clump_ID + 1):
            sel = new_IDs == sid
            nsel = torch.sum(sel)
            # if nsel > 0 then sample
            if nsel.byte():
                samples[sel] = self.list_SH[sid].sample(nsel.item())

        return samples

    def score(self, data, part_IDs):
        """
        Parameters
        ----------
        data : (n,2) tensor
        part_IDs : (n,) tensor

        Returns
        -------
        ll : (n,) tensor
        """
        assert hasattr(self, 'list_SH'), 'model not yet fit'
        assert isinstance(data, torch.Tensor)
        assert isinstance(part_IDs, torch.Tensor)
        assert len(data.shape) == 2
        assert len(part_IDs.shape) == 1
        assert data.shape[1] == 2
        nsamp = len(part_IDs)
        new_IDs = self.__map_indx(part_IDs)

        # for each stroke ID
        ll = torch.zeros(nsamp)
        for sid in range(self.clump_ID + 1):
            sel = new_IDs == sid
            nsel = torch.sum(sel)
            # if nsel > 0 then score
            if nsel.byte():
                ll[sel] = self.list_SH[sid].score(data[sel])

        return ll

    def __map_indx(self, old_IDs):
        """
        Parameters
        ----------
        old_IDs : (n,) tensor

        Returns
        -------
        new_IDs : (n,) tensor
        """
        new_IDs = old_IDs
        new_IDs[new_IDs > self.clump_ID] = self.clump_ID

        return new_IDs



class Library(object):
    """
    LIBRARY: hyper-parameters for the BPL model
    """
    def __init__(self, lib_dir):
        """
        Constructor

        :param lib_dir: [string] path to the library files
        """
        # get contents of dir
        contents = os.listdir(lib_dir)
        # save lists of structs and single elements
        structs = ['shape', 'scale', 'rel', 'tokenvar', 'affine', 'stat']
        singles = [
            'logT', 'logStart', 'pkappa', 'pmat_nsub', 'newscale',
            'smooth_bigrams', 'diagSigma'
        ]
        # load structs
        for elt in structs:
            assert elt in contents
            value = get_dict(os.path.join(lib_dir, elt))
            if elt == 'shape':
                value = fix_shape_params(value)
            setattr(self, elt, value)
        # load individual properties
        for elt in singles:
            assert elt+'.mat' in contents
            value = get_data(elt+'.mat', lib_dir)
            setattr(self, elt, value)
        # change type of 'diagSigma' to torch.uint8 since this is a boolean
        self.diagSigma = self.diagSigma.byte()

        # Finally, load SpatialModel
        clump_ID = 2
        xlim = torch.tensor([0, 105], dtype=torch.float)
        ylim = torch.tensor([-105, 0], dtype=torch.float)
        spatial_model = SpatialModel(xlim, ylim, clump_ID)
        spatial_model.initialize_unif()
        self.Spatial = spatial_model

        # Check consistency of the library
        self.check_consistent()

    @property
    def ncpt(self):
        """
        Get the number of control points

        :return:
            ncpt: [int] the number of control points
        """
        dim = self.shape['mu'].shape[1]
        assert dim % 2 == 0 # dimension must be even
        ncpt = int(dim/2)

        return ncpt

    @property
    def N(self):
        """
        Get the number of primitives

        :return:
            N: [int] the number of primitives
        """
        N = self.shape['mu'].shape[0]

        return N

    def check_consistent(self):
        """
        Check consistency of the number of primitives in the model
        """
        N = self.N
        ncpt = self.ncpt
        assert len(self.shape['mu'].shape) == 2
        assert len(self.shape['Sigma'].shape) == 3
        assert self.shape['mu'].shape[1] == ncpt*2
        assert self.shape['Sigma'].shape[0] == N
        assert self.shape['Sigma'].shape[1] == ncpt*2
        assert self.shape['Sigma'].shape[2] == ncpt*2
        assert self.logT.shape[0] == N
        assert self.logStart.shape[0] == N
        assert self.shape['mixprob'].shape[0] == N
        assert self.shape['freq'].shape[0] == N
        assert self.shape['vsd'].shape[0] == N
        assert self.scale['theta'].shape[0] == N
        assert aeq(torch.sum(torch.exp(self.logStart)), torch.tensor(1.))
        for sid in range(N):
            pT = self.pT(torch.tensor(sid))
            assert aeq(torch.sum(pT), torch.tensor(1.))

    def pT(self, prev_state):
        """
        Get the probability of transitioning to a new state, given your current
        state is "prev_state"

        :param prev_state: [tensor] current state of the model
        :return:
            p: [tensor] probability vector; probabilities of transitioning to
                        each potential new state
        """
        assert prev_state.shape == torch.Size([])
        logR = self.logT[prev_state]
        R = torch.exp(logR)
        p = R / torch.sum(R)

        return p

    @property
    def isunif(self):
        return torch.isnan(self.shape['mu']).any()


def get_dict(path):
    """
    load folder of arrays as dictionary of tensors
    """
    field = {}
    contents = os.listdir(path)
    for item in contents:
        key = item.split('.')[0]
        field[key] = get_data(item, path)

    return field

def get_data(item, path):
    """
    load single array as a tensor
    """
    item_path = os.path.join(path, item)
    data = io.loadmat(item_path)['value']
    data = data.astype(np.float32)  # convert to float32
    out = torch.squeeze(torch.tensor(data, dtype=torch.float))

    return out

def fix_shape_params(shape):
    """
    fix organization of shapes 'mu' and 'Sigma' arrays to account for
    differences in the 'reshape' operation between MATLAB and numpy/pytorch
    """
    shapes_mu = shape['mu']
    shapes_Cov = shape['Sigma']
    n, m = shapes_mu.shape
    assert m % 2 == 0
    ncpt = m // 2
    # fix shapes mean
    shapes_mu = shapes_mu.view(n, 2, ncpt)  # (n, 2, ncpt)
    shapes_mu = shapes_mu.permute(0, 2, 1)  # (n, ncpt, 2)
    shapes_mu = shapes_mu.contiguous()
    shapes_mu = shapes_mu.view(n, ncpt * 2)  # (n, ncpt*2)
    shapes_mu = shapes_mu.contiguous()
    # fix shapes covariance
    shapes_Cov = shapes_Cov.permute(2, 0, 1)  # (n, 2*ncpt, 2*ncpt)
    shapes_Cov = shapes_Cov.view(n, 2, ncpt, 2, ncpt)  # (n, 2, ncpt, 2, ncpt)
    shapes_Cov = shapes_Cov.permute(0, 2, 1, 4, 3)  # (n, ncpt, 2, ncpt, 2)
    shapes_Cov = shapes_Cov.contiguous()
    shapes_Cov = shapes_Cov.view(n, ncpt * 2, ncpt * 2)  # (n, ncpt*2, ncpt*2)
    shapes_Cov = shapes_Cov.contiguous()
    # re-assign
    shape['mu'] = shapes_mu
    shape['Sigma'] = shapes_Cov

    return shape





"""
Relations for sampling part positions. Relations, together with parts, make up
concepts.
"""


categories_allowed = ['unihist', 'start', 'end', 'mid']


class RelationToken(object):
    """
    RelationToken instances hold all of the token-level information for a
    relation

    Parameters
    ----------
    rel : Relation
        relation type
    eval_spot_token : tensor
        Optional parameter. Token-level evaluation spot for RelationAttachAlong
    """
    def __init__(self, rel, **kwargs):
        self.rel = rel
        if rel.category in ['unihist', 'start', 'end']:
            assert kwargs == {}
        else:
            assert set(kwargs.keys()) == {'eval_spot_token'}
            self.eval_spot_token = kwargs['eval_spot_token']

    def optimizable_parameters(self, eps=1e-4):
        """
        Returns a list of parameters that can be optimized via gradient descent.
        Includes lists of lower and upper bounds, with one per parameter.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        params : list
            optimizable parameters
        lbs : list
            lower bound for each parameter
        ubs : list
            upper bound for each parameter
        """
        if self.rel.category == 'mid':
            _, lb, ub = bspline_gen_s(self.rel.ncpt, 1)
            params = [self.eval_spot_token]
            lbs = [lb]
            ubs = [ub]
        else:
            params = []
            lbs = []
            ubs = []

        return params, lbs, ubs

    def sample_location(self, prev_parts):
        """
        Sample a location from the relation token

        Parameters
        ----------
        prev_parts : list of PartToken
            previous part tokens

        Returns
        -------
        loc : (2,) tensor
            location; x-y coordinates

        """
        for pt in prev_parts:
            assert isinstance(pt, PartToken)
        base = self.get_attach_point(prev_parts)
        assert base.shape == torch.Size([2])
        loc = base + self.rel.loc_dist.sample()

        return loc

    def score_location(self, loc, prev_parts):
        """
        Compute the log-likelihood of a location

        Parameters
        ----------
        loc : (2,) tensor
            location; x-y coordinates
        prev_parts : list of PartToken
            previous part tokens

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the location

        """
        for pt in prev_parts:
            assert isinstance(pt, PartToken)
        base = self.get_attach_point(prev_parts)
        assert base.shape == torch.Size([2])
        ll = self.rel.loc_dist.log_prob(loc - base)

        return ll

    def get_attach_point(self, prev_parts):
        """
        Get the mean attachment point of where the start of the next part
        should be, given the previous part tokens.

        Parameters
        ----------
        prev_parts : list of PartToken
            previous part tokens

        Returns
        -------
        loc : (2,) tensor
            attach point (location); x-y coordinates

        """
        if self.rel.category == 'unihist':
            loc = self.rel.gpos
        else:
            prev = prev_parts[self.rel.attach_ix]
            if self.rel.category == 'start':
                subtraj = prev.motor[0]
                loc = subtraj[0]
            elif self.rel.category == 'end':
                subtraj = prev.motor[-1]
                loc = subtraj[-1]
            else:
                assert self.rel.category == 'mid'
                bspline = prev.motor_spline[:, :, self.rel.attach_subix]
                loc, _ = bspline_eval(self.eval_spot_token, bspline)
                # convert (1,2) tensor -> (2,) tensor
                loc = torch.squeeze(loc, dim=0)

        return loc


class Relation(object):
    """
    Relations define the relationship between the current part and all previous
    parts. They fall into 4 categories: ['unihist','start','end','mid']. Holds
    all type-level parameters of the relation. This is an abstract base class
    that must be inherited from to build specific categories of relations.

    Parameters
    ----------
    category : string
        relation category
    lib : Library
        library instance, which holds token-level distribution parameters
    """
    __metaclass__ = ABCMeta

    def __init__(self, category, lib):
        # make sure type is valid
        assert category in categories_allowed
        self.category = category
        # token-level position distribution parameters
        sigma_x = lib.rel['sigma_x']
        sigma_y = lib.rel['sigma_y']
        loc_Cov = torch.diag(torch.stack([sigma_x, sigma_y]))
        self.loc_dist = dist.MultivariateNormal(torch.zeros(2), loc_Cov)

    @abstractmethod
    def optimizable_parameters(self, eps=1e-4):
        """
        Returns a list of parameters that can be optimized via gradient descent.
        Includes lists of lower and upper bounds, with one per parameter.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        params : list
            optimizable parameters
        lbs : list
            lower bound for each parameter
        ubs : list
            upper bound for each parameter
        """
        pass

    def sample_token(self):
        """
        Sample a token of the relation

        Returns
        -------
        token : RelationToken
            relation token sample
        """
        token = RelationToken(self)

        return token

    def score_token(self, token):
        """
        Compute the log-likelihood of a relation token

        Parameters
        ----------
        token : RelationToken
            relation token to score

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the relation token

        """
        # default this to 0. The only relation category that has token-level
        # parameters is the 'mid'. For 'mid' this function is over-ridden
        # (see RelationAttachAlong)
        ll = 0.

        return ll


class RelationIndependent(Relation):
    """
    RelationIndependent (or 'unihist' relations) are assigned when the part's
    location is independent of all previous parts. The global position (gpos)
    of the part is sampled at random from the prior on positions

    Parameters
    ----------
    category : string
        relation category
    gpos : (2,) tensor
        position; x-y coordinates
    xlim : (2,) tensor
        [lower, upper]; bounds for the x direction
    ylim : (2,) tensor
        [lower, upper]; bounds for the y direction
    lib : Library
        library instance, which holds token-level distribution parameters
    """
    def __init__(self, category, gpos, xlim, ylim, lib):
        super(RelationIndependent, self).__init__(category, lib)
        assert category == 'unihist'
        assert gpos.shape == torch.Size([2])
        self.gpos = gpos
        self.xlim = xlim
        self.ylim = ylim

    def optimizable_parameters(self, eps=1e-4):
        params = [self.gpos]
        bounds = torch.cat([self.xlim.view(1, -1), self.ylim.view(1, -1)])
        lbs = [bounds[:,0]]
        ubs = [bounds[:,1]]

        return params, lbs, ubs


class RelationAttach(Relation):
    """
    RelationAttach is assigned when the part will attach to a previous part

    Parameters
    ----------
    category : string
        relation category
    attach_ix : int
        index of previous part to which this part will attach
    lib : Library
        library instance, which holds token-level distribution parameters
    """
    def __init__(self, category, attach_ix, lib):
        super(RelationAttach, self).__init__(category, lib)
        assert category in ['start', 'end', 'mid']
        self.attach_ix = attach_ix

    def optimizable_parameters(self, eps=1e-4):
        params = []
        lbs = []
        ubs = []

        return params, lbs, ubs


class RelationAttachAlong(RelationAttach):
    """
    RelationAttachAlong is assigned when the part will attach to a previous
    part somewhere in the middle of that part (as opposed to the start or end)

    Parameters
    ----------
    category : string
        relation category
    attach_ix : int
        index of previous part to which this part will attach
    attach_subix : int
        index of sub-stroke from the selected previous part to which
        this part will attach
    eval_spot : tensor
        type-level spline coordinate
    lib : Library
        library instance, which holds token-level distribution parameters
    """
    def __init__(self, category, attach_ix, attach_subix, eval_spot, lib):
        super(RelationAttachAlong, self).__init__(category, attach_ix, lib)
        assert category == 'mid'
        self.attach_subix = attach_subix
        self.eval_spot = eval_spot
        # token-level eval_spot distribution parameters
        self.ncpt = lib.ncpt
        self.sigma_attach = lib.tokenvar['sigma_attach']

    def optimizable_parameters(self, eps=1e-4):
        _, lb, ub = bspline_gen_s(self.ncpt, 1)
        params = [self.eval_spot]
        lbs = [lb]
        ubs = [ub]

        return params, lbs, ubs

    def sample_token(self):
        """
        Sample a token of the relation

        Returns
        -------
        token : RelationToken
            sampled relation token
        """
        eval_spot_dist = dist.normal.Normal(self.eval_spot, self.sigma_attach)
        eval_spot_token = sample_eval_spot_token(eval_spot_dist, self.ncpt)
        token = RelationToken(self, eval_spot_token=eval_spot_token)

        return token

    def score_token(self, token):
        """
        Compute the log-likelihood of a relation token

        Parameters
        ----------
        token : RelationToken
            relation token to score

        Returns
        -------
        ll : tensor
            scalar; log-likelihood of the relation token
        """
        assert hasattr(token, 'eval_spot_token')
        eval_spot_dist = dist.normal.Normal(self.eval_spot, self.sigma_attach)
        ll = score_eval_spot_token(
            token.eval_spot_token, eval_spot_dist, self.ncpt
        )

        return ll



def sample_eval_spot_token(eval_spot_dist, ncpt):
    """
    Sample an evaluation spot token

    Parameters
    ----------
    eval_spot_dist : Distribution
        torch distribution; will be used to sample evaluation spot tokens
    ncpt : int
        number of control points

    Returns
    -------
    eval_spot_token : tensor
        scalar; token-level spline coordinate
    """
    while True:
        eval_spot_token = eval_spot_dist.sample()
        ll = score_eval_spot_token(eval_spot_token, eval_spot_dist, ncpt)
        if not ll == -float('inf'):
            break

    return eval_spot_token


def score_eval_spot_token(eval_spot_token, eval_spot_dist, ncpt):
    """
    Compute the log-likelihood of an evaluation spot token

    Parameters
    ----------
    eval_spot_token : tensor
        scalar; token-level spline coordinate
    eval_spot_dist : Distribution
        torch distribution; will be used to score evaluation spot tokens
    ncpt : int
        number of control points

    Returns
    -------
    ll : tensor
        scalar; log-likelihood of the evaluation spot token
    """
    assert type(eval_spot_token) in [int, float] or \
           (type(eval_spot_token) == torch.Tensor and
            eval_spot_token.shape == torch.Size([]))
    _, lb, ub = bspline_gen_s(ncpt, 1)
    if eval_spot_token < lb or eval_spot_token > ub:
        ll = torch.tensor(-float('inf'), dtype=torch.float)
    else:
        ll = eval_spot_dist.log_prob(eval_spot_token)
        # correction for bounds
        p_within = eval_spot_dist.cdf(ub) - eval_spot_dist.cdf(lb)
        ll = ll - torch.log(p_within)

    return ll






"""
Parameters...
"""


class defaultps(object):
    def __init__(self):
        # Library to use
        self.libname = 'library'

        # number of particles to use in search algorithm
        self.K = torch.tensor(5, dtype=torch.int)

        ## image model parameters ##
        # number of convolutions
        self.ink_ncon = torch.tensor(2, dtype=torch.int)
        # image size
        self.imsize = torch.Size([105, 105])
        # amount of ink per point
        self.ink_pp = torch.tensor(2, dtype=torch.float)
        # distance between points to which you get full ink
        self.ink_max_dist = torch.tensor(2, dtype=torch.float)
        # ink parameter 1
        self.ink_a = torch.tensor(0.5, dtype=torch.float)
        # ink parameter 2
        self.ink_b = torch.tensor(6, dtype=torch.float)

        ## Creating a trajectory from a spline ##
        # maxmium number of evaluations
        self.spline_max_neval = torch.tensor(200, dtype=torch.int)
        # minimum
        self.spline_min_neval = torch.tensor(10, dtype=torch.int)
        # 1 trajectory point for every this many units pixel distance)
        self.spline_grain = torch.tensor(1.5, dtype=torch.float)

        ## Max/min noise parameters for image model ##
        # blur kernel width
        self.max_blur_sigma = torch.tensor(16, dtype=torch.float)
        self.min_blur_sigma = torch.tensor(0.5, dtype=torch.float)
        # pixel flipping
        self.max_epsilon = torch.tensor(0.5, dtype=torch.float)
        self.min_epsilon = torch.tensor(1e-4, dtype=torch.float)

        ## search parameters ##
        # scale changes must be less than a factor of 2
        self.max_affine_scale_change = 2
        # shift changes must less than this
        self.max_affine_shift_change = 50

        ## MCMC PARAMETERS ##
        ## they were in mcmc. notation, but i changed it for convenience ##

        ## details about the chain ##
        # number of samples to take in the MCMC chain (for classif.)
        self.mcmc_nsamp_type_chain = 200
        # number of samples to store from this chain (for classif.)
        self.mcmc_nsamp_type_store = 10
        # for completion (we take last sample in this chain)
        self.mcmc_nsamp_token_chain = 25

        # mcmc proposal parameters (Note these are based on lib.tokenvar
        # parameters, although here they are hard-coded for convenience)

        # global position move
        self.mcmc_prop_gpos_sd = 1
        # shape move
        self.mcmc_prop_shape_sd = 3/2
        # scale move
        self.mcmc_prop_scale_sd = 0.0235
        # attach relation move
        self.mcmc_prop_relmid_sd = 0.2168
        # multiply the sd of the standard position noise by this to propose
        # new positions from prior
        self.mcmc_prop_relpos_mlty = 2
    



