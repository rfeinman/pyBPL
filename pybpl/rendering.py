"""
All the functions and modules for differentiable rendering go here
"""
from __future__ import print_function, division
import torch

from .splines import get_stk_from_bspline
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

def vanilla_to_motor(shapes, invscales, first_pos):
    """
    Create the fine-motor trajectory of a stroke (denoted 'f()' in pseudocode)
    with k sub-strokes

    :param shapes: [(ncpt,2,k) tensor] spline points in normalized space
    :param invscales: [(k,) tensor] inverse scales for each sub-stroke
    :param first_pos: [(2,) tensor] starting location of stroke
    :return:
        motor: [list] k-length fine motor sequence
        motor_spline: [list] k-length fine motor sequence in spline space
    """
    vanilla_traj = []
    motor = []
    ncpt,_,n = shapes.shape
    for i in range(n):
        shapes[:,:,i] = invscales[i] * shapes[:,:,i]
        vanilla_traj.append(get_stk_from_bspline(shapes[:,:,i]))

        # calculate offset
        if i == 0:
            offset = vanilla_traj[i][0,:] - first_pos
        else:
            offset = vanilla_traj[i-1][0,:] - motor[i-1][-1,:]
        motor.append(offset_stk(vanilla_traj[i],offset))

    motor_spline = None

    return motor, motor_spline

def apply_warp(list_st, list_rt, affine):
    motor_unwarped = [st.motor(rt) for st, rt in zip(list_st, list_rt)]
    if affine is None:
        motor_warped = motor_unwarped
    else:
        raise NotImplementedError

    return motor_warped

def apply_render(list_st, list_rt, affine, epsilon, blur_sigma, parameters):
    motor_warped = apply_warp(list_st, list_rt, affine)
    flat_warped = UtilMP.flatten_substrokes(motor_warped)
    pimg, ink_off_page = render_image(
        flat_warped, epsilon, blur_sigma, parameters
    )

    return pimg, ink_off_page