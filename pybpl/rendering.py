"""
All the functions and modules for differentiable rendering go here
"""
from __future__ import print_function, division

import torch

from .general_util import aeq, sub2ind, fspecial, imfilter
from . import splines
from .concept.part import RenderedPart
from .character import util_character


# ----
# vanilla to motor
# ----

def offset_stk(traj, offset):
    print("'offset_stk' function is unnecessary.")
    raise NotImplementedError

def vanilla_to_motor(shapes, invscales, first_pos, neval=200):
    """
    Create the fine-motor trajectory of a stroke (denoted 'f()' in pseudocode)
    with 'nsub' sub-strokes

    :param shapes: [(ncpt,2,nsub) tensor] spline points in normalized space
    :param invscales: [(nsub,) tensor] inverse scales for each sub-stroke
    :param first_pos: [(2,) tensor] starting location of stroke
    :param neval: [tensor] int; the number of evaluations to use for each motor
                    trajectory
    :return:
        motor: [(nsub,neval,2) tensor] fine motor sequence
        motor_spline: [(ncpt,2,nsub) tensor] fine motor sequence in spline space
    """
    for elt in [shapes, invscales, first_pos]:
        assert elt is not None
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

def apply_warp(rendered_parts, affine):
    motor_unwarped = [rp.motor for rp in rendered_parts]
    if affine is None:
        motor_warped = motor_unwarped
    else:
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
    xt = myt[:,0]
    yt = myt[:,1]
    x_out = torch.floor(xt) < 0 | torch.ceil(xt) > imsize[0]
    y_out = torch.floor(yt) < 0 | torch.ceil(yt) > imsize[1]
    out = x_out | y_out

    return out

def pair_dist(D):
    x1 = D[:-1]
    x2 = D[1:]
    z = torch.sqrt(
        torch.sum(
            torch.pow(x1-x2, 2),
            dim=1
        )
    )

    return z

def seqadd(x, lind, inkval):
    numel = len(lind.view(-1))
    for i in range(numel):
        x[lind[i]] = x[lind[i]] + inkval[i]

    return x

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

def render_image(cell_traj, epsilon, blur_sigma, parameters):
    """

    :param cell_traj: [(nsub_total,neval,2) tensor]
    :param epsilon: [float]
    :param blur_sigma: [float]
    :param parameters:
    :return:
    """
    # convert to image space
    traj_img = space_motor_to_img(cell_traj)

    # get relevant parameters
    imsize = parameters.imsize
    ink = parameters.ink_pp
    max_dist = parameters.ink_max_dist
    a = parameters.ink_a
    b = parameters.ink_b
    ink_ncon = parameters.ink_ncon

    # draw the trajectories on the image
    template = torch.zeros(imsize, dtype=torch.float)
    nsub_total = traj_img.shape[0]
    ink_off_page = False
    for i in range(nsub_total):
        pass
        # check boundaries
        myt = traj_img[i]

        # compute distance between each trajectory point and the next one

        # make sure we have the minimum amount of ink, if a particular
        # trajectory is very small

        # share ink with the neighboring 4 pixels

        # paint the image

    # filter the image to get the desired brush-stroke size


    return prob_on, ink_off_page


# ----
# apply render
# ----

def apply_render(rendered_parts, affine, epsilon, blur_sigma, parameters):
    """

    :param rendered_parts: [list of RenderedStroke]
    :param affine: [(4,) tensor]
    :param epsilon: [float]
    :param blur_sigma: [float]
    :param parameters: []
    :return:
    """
    for rp in rendered_parts:
        assert isinstance(rp, RenderedPart)
    motor_warped = apply_warp(rendered_parts, affine)
    flat_warped = util_character.flatten_substrokes(motor_warped)
    pimg, ink_off_page = render_image(
        flat_warped, epsilon, blur_sigma, parameters
    )

    return pimg, ink_off_page