"""
All the functions and modules for differentiable rendering go here
"""
from __future__ import print_function, division
import torch

from . import splines
from .concept.part import RenderedPart
from .character import util_character


def com_char(char):
    raise NotImplementedError
    ns = len(char)

def offset_stk(traj, offset):
    print("'offset_stk' function is unnecessary.")
    raise NotImplementedError

def space_motor_to_img(pt):
    raise NotImplementedError

def render_image(cell_traj, epsilon, blur_sigma, PM):
    raise NotImplementedError
    traj_img = space_motor_to_img(cell_traj)
    #TODO
    ink = PM.ink_pp
    max_dist = PM.ink_max_dist

    return prob_on, ink_off_page

def sum_pair_dist(D):
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