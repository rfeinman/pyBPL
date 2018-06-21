"""
rendering.py

all the functions and modules for differentiable rendering goes here

"""
from __future__ import print_function, division
import numpy as np
import torch
from torch.autograd import Variable


#I think it is fine do to everything in pieces.

def offset_stk(traj,offset):
    n = traj.shape[0]
    list_sub = [offset for _ in range(n)]
    sub = torch.cat(list_sub,0)
    stk = traj - sub
    return stk

def apply_warp(MP):
    motor_unwarped = MP.motor
    if MP.A == []:
        return motor_unwarped
    else:
        #TODO - motor warped
        return []

def space_motor_to_img(pt): #TODO
    error("not implemented yet")
    return new_pt

def render_image(cell_traj, epsilon, blur_sigma, PM):
    traj_img = rendering.space_motor_to_img(cell_traj)
    #TODO
    ink = PM.ink_pp
    max_dist = PM.ink_max_dist


    return prob_on, ink_off_page

def motor_to_pimg(MP):
    motor_warped = rendering.apply_warp(MP)

    pimg, ink_off_page = rendering.render_image(
        motor_warped, MP.epsilon, MP.blur_sigma, MP.parameters
    )
    return pimg, ink_off_page

def vectorized_bspline_coeff(vi,vs):
    assert vi.shape == vs.shape
    C = torch.zeros(vi.shape)

    #in the following, * stands in for 'and'
    sel1 = (vs >= vi) * (vs < vi + 1)
    C[sel1] = (1/6.)*torch.pow((vs[sel1]-vi[sel1]),3)

    sel2 = (vs >= vi+1) * (vs < vi+2)
    C[sel2] = (1/6.)*(
        -3.*torch.pow((vs[sel2]-vi[sel2]-1),3) +
        3.*torch.pow((vs[sel2]-vi[sel2]-1),2) +
        3.*(vs[sel2]-vi[sel2]-1) +
        1)

    sel3 = (vs >= vi+2) * (vs < vi+3)
    C[sel3] = (1/6.)*(3*torch.pow((vs[sel3]-vi[sel3]-2),3) -
                      6*torch.pow((vs[sel3]-vi[sel3]-2),2) + 4)

    sel4 = (vs >= vi+3) * (vs < vi+4)
    C[sel4] = (1/6.)*torch.pow((1-(vs(sel4)-vi(sel4)-3)),3)

    return C

def bspline_eval(sval, cpts):
    # % Fit a uniform, cubic B-spline
    # %
    # % Input
    # %   sval: numpy array (k,) where 0 <= sval(i) <= n
    # %   cpts: [n x 2] array of control points
    # %
    # % Output
    # %    y: vector [k x 2] which is output of spline
    assert len(sval.shape) == 1
    L = cpts.shape[0]
    ns = len(sval)

    list_sval = [sval for _ in range(L)]
    S = Variable(torch.cat(list_sval,1)) #wait, does sval need to be
    list_L = [torch.Tensor(np.arange(L)).view(1,-1) for _ in range(ns)]
    I = Variable(torch.cat(list_L,0))
    Cof = vectorized_bspline_coeff(I,S)
    y1 = torch.mm(Cof, cpts[:,0])
    y2 = torch.mm(Cof, cpts[:,1])
    y = torch.cat((y1,y2),1)

    return y, Cof


def bspline_gen_s(nland,neval=200):
    lb = 2
    ub = nland + 1
    length = ub - lb
    try:
        interval = length / float(neval - 1)
        s = np.arange(lb,ub,interval)
    except ZeroDivisionError:
        s = []
    return s, ub, lb


def sum_pair_dist(D):
    error('not implemented')
    return s



def get_stk_from_bspline(P,neval=200):
    #brenden's code finds number of eval points adaptively.
    #Can consider doing this if things take too long.
    #I worry it may mess with gradients by making them more piecewise

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
        error('dynamic n evaluation not implemented')

    s = bspline_gen_s(nland,neval)
    stk = bspline_eval(s,P)
    return stk