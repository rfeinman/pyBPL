"""
All the functions and modules for differentiable rendering go here
"""
from __future__ import print_function, division

import torch

from .util_general import aeq, fspecial, imfilter
from . import splines
from . import util_character


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
        traj = splines.get_stk_from_bspline(shapes_scaled, neval)
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
    cell_traj = util_character.flatten_substrokes(motor_unwarped)
    com = com_char(cell_traj)
    b = torch.zeros(4, dtype=torch.float)
    b[:2] = affine[:2]
    b[2:4] = affine[2:4] - (affine[:2]-1)*com
    fn = lambda stk: affine_warp(stk, b)
    motor_warped = util_character.apply_each_substroke(motor_unwarped, fn)

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
    valid_lind_x = lind_x < D.shape[0]
    valid_lind_y = lind_y < D.shape[1]
    assert valid_lind_x.all() and valid_lind_y.all()
    lind_x = lind_x.long()
    lind_y = lind_y.long()
    numel = len(lind_x)
    for i in range(numel):
        D[lind_x[i],lind_y[i]] = D[lind_x[i],lind_y[i]] + inkval[i]

    return D

def space_motor_to_img(pt):
    """

    :param pt: [tensor] Fine motor sequence. This is either for an individual
                sub-stroke, in which case it has shape (neval,2), or for a
                sequence of sub-strokes, in which case it has shape (k,neval,2)
    :return:
    """
    assert isinstance(pt, torch.Tensor)
    new_pt = torch.zeros_like(pt, dtype=torch.float)
    if len(pt.shape) == 2:
        new_pt[:,0] = -pt[:,1]
        new_pt[:,1] = pt[:,0]
    elif len(pt.shape) == 3:
        new_pt[:,:,0] = -pt[:,:,1]
        new_pt[:,:,1] = pt[:,:,0]
    else:
        raise Exception('pt must be 2- or 3-dim tensor')
    new_pt = new_pt + 1

    return new_pt

def add_header(dist):
    assert isinstance(dist, torch.Tensor)
    assert len(dist.shape) == 1
    dist_p = torch.zeros(len(dist)+1)
    dist_p[0] = dist[0]
    dist_p[1:] = dist

    return dist_p

def render_image(cell_traj, epsilon, blur_sigma, parameters):
    """

    :param cell_traj: [(nsub_total,neval,2) tensor]
    :param epsilon: [float]
    :param blur_sigma: [float]
    :param parameters:
    :return:
    """
    # convert to image space
    traj_img = space_motor_to_img(cell_traj) # same shape as cell_traj

    # get relevant parameters
    imsize = parameters.imsize
    ink = parameters.ink_pp
    max_dist = parameters.ink_max_dist

    # draw the trajectories on the image
    template = torch.zeros(imsize, dtype=torch.float)
    nsub_total = traj_img.shape[0]
    ink_off_page = False
    for i in range(nsub_total):
        # check boundaries
        myt = traj_img[i] # shape (neval,2)
        out = check_bounds(myt, imsize)
        if out.any():
            ink_off_page = True
        if out.all():
            continue
        myt = myt[~out]

        # compute distance between each trajectory point and the next one
        if myt.shape[0] == 1:
            myink = torch.tensor(ink, dtype=torch.float32)
        else:
            dist = pair_dist(myt) # shape (k,)
            dist[dist>max_dist] = max_dist
            dist = add_header(dist)
            myink = (ink/max_dist)*dist # shape (k,)

        # make sure we have the minimum amount of ink, if a particular
        # trajectory is very small
        sumink = torch.sum(myink)
        if aeq(sumink, torch.tensor(0.)):
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
        # Reuben's fix... don't want to access last array index
        # TODO - update this?
        xceil[xceil == imsize[0]] = imsize[0] - 1
        yceil[yceil == imsize[1]] = imsize[1] - 1

        # paint the image
        template = seqadd(template, xfloor, yfloor, myink*x_f_ratio*y_f_ratio)
        template = seqadd(template, xceil, yfloor, myink*x_c_ratio*y_f_ratio)
        template = seqadd(template, xfloor, yceil, myink*x_f_ratio*y_c_ratio)
        template = seqadd(template, xceil, yceil, myink*x_c_ratio*y_c_ratio)

    # filter the image to get the desired brush-stroke size
    a = parameters.ink_a
    b = parameters.ink_b
    ink_ncon = parameters.ink_ncon
    H_broaden = b*torch.tensor(
        [[a/12, a/6, a/12],[a/6, 1-a, a/6],[a/12, a/6, a/12]]
    )
    widen = template
    for i in range(ink_ncon):
        widen = imfilter(widen, H_broaden, mode='conv')

    # threshold again
    widen[widen>1] = 1

    # filter the image to get Gaussian
    # noise around the area with ink
    pblur = widen
    if blur_sigma > 0:
        fsize = 11
        H_gaussian = fspecial(fsize, blur_sigma, ftype='gaussian')
        pblur = imfilter(pblur, H_gaussian, mode='conv')
        pblur = imfilter(pblur, H_gaussian, mode='conv')

    # final truncation
    pblur[pblur>1] = 1
    pblur[pblur<0] = 0

    # probability of each pixel being on
    prob_on = (1-epsilon)*pblur + epsilon*(1-pblur)

    return prob_on, ink_off_page


# ----
# apply render
# ----

def apply_render(token, parameters):
    """
    TODO

    Parameters
    ----------
    token : CharacterToken
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
    from .part import StrokeToken
    for rs in token.stroke_tokens:
        assert isinstance(rs, StrokeToken)

    # get motor for each part
    motor = [st.motor for st in token.stroke_tokens]
    # apply affine transformation if needed
    if token.affine is not None:
        motor = apply_warp(motor, token.affine)
    motor_flat = util_character.flatten_substrokes(motor)
    pimg, ink_off_page = render_image(
        motor_flat, token.epsilon, token.blur_sigma, parameters
    )

    return pimg, ink_off_page