"""
All the functions and modules for differentiable rendering go here
"""
from __future__ import print_function, division
import torch

from . import splines
from . import UtilMP

def offset_stk(traj, offset):
    n = traj.shape[0]
    list_sub = [offset for _ in range(n)]
    sub = torch.cat(list_sub, 0)
    stk = traj - sub

    return stk

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
        motor: [list] nsub-length fine motor sequence
        motor_spline: [list] nsub-length fine motor sequence in spline space
    """
    for elt in [shapes, invscales, first_pos]:
        assert elt is not None
    ncpt, _, nsub = shapes.shape
    motor = torch.zeros(nsub, neval, 2, dtype=torch.float)
    motor_spline = torch.zeros_like(shapes, dtype=torch.float)
    for i in range(nsub):
        # re-scale the control points
        shapes_scaled = invscales[i]*shapes[:,:,i]
        # get trajectories from b-spline
        motor[i] = splines.get_stk_from_bspline(shapes_scaled, neval)
        # reposition
        if i == 0:
            offset = motor[i,0] - first_pos
        else:
            offset = motor[i,0] - motor[i-1,-1]
        motor[i] = offset_stk(motor[i], offset)
        motor_spline[:,:,i] = shapes_scaled - offset

    return motor, motor_spline

def apply_warp(list_st, list_pos, affine):
    motor_unwarped = [st.motor(pos) for st, pos in zip(list_st, list_pos)]
    if affine is None:
        motor_warped = motor_unwarped
    else:
        raise NotImplementedError

    return motor_warped

def apply_render(list_st, list_pos, affine, epsilon, blur_sigma, parameters):
    motor_warped = apply_warp(list_st, list_pos, affine)
    flat_warped = UtilMP.flatten_substrokes(motor_warped)
    pimg, ink_off_page = render_image(
        flat_warped, epsilon, blur_sigma, parameters
    )

    return pimg, ink_off_page