"""
All the functions and modules for differentiable rendering goes here
"""
from __future__ import print_function, division
import torch


def offset_stk(traj,offset):
    n = traj.shape[0]
    list_sub = [offset for _ in range(n)]
    sub = torch.cat(list_sub,0)
    stk = traj - sub

    return stk

def space_motor_to_img(pt): #TODO
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