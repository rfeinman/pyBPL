"""
All the functions and modules for differentiable rendering go here
"""
from __future__ import print_function, division

import torch

from .util import sub2ind, fspecial, imfilter
from . import splines


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
    Given a list of 2D points (x-y coordinates) and an image size, return
    a boolean vector indicating which points are out of the image boundary

    Parameters
    ----------
    myt : (k,2) tensor
        list of 2D points
    imsize : (2,) tensor
        image size; H x W

    Returns
    -------
    out : (k,) Byte tensor
        vector indicating which points are out of bounds
    """
    xt = myt[:,0]
    yt = myt[:,1]
    x_out = (torch.floor(xt) < 0) | (torch.ceil(xt) > imsize[0])
    y_out = (torch.floor(yt) < 0) | (torch.ceil(yt) > imsize[1])
    out = x_out | y_out

    return out

def pair_dist(D):
    """
    Given a list of 2D points (x-y coordinates), compute the euclidean distance
    between each point and the next point in the list

    Parameters
    ----------
    D : (k,2) tensor
        list of 2D points

    Returns
    -------
    z : (k-1,) tensor
        list of distances
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
    Add ink to an image at the indicated locations

    Parameters
    ----------
    D : (m,n) tensor
        image that we'll be adding to
    lind_x : (k,) tensor
        x-coordinate for each adding point
    lind_y : (k,) tensor
        y-coordinate for each adding point
    inkval : (k,) tensor
        amount of ink to add for each adding point

    Returns
    -------
    D : (m,n) tensor
        image with ink added to it
    """
    assert len(lind_x) == len(lind_y) == len(inkval)
    # check if any adding points are out of bounds
    out = check_bounds(
        torch.cat([lind_x.view(-1,1), lind_y.view(-1,1)], dim=1),
        (D.shape[0]-1, D.shape[1]-1)
    )
    # keep only the adding points that are in bounds
    lind_x = lind_x[~out].long()
    lind_y = lind_y[~out].long()
    inkval = inkval[~out]
    # store the original shape of the image and then flatten it from 2D to 1D
    shape = D.shape
    D = D.view(-1)
    # convert the indices from 2D to 1D
    lind = sub2ind(shape, lind_x, lind_y)
    # create a zeros vector of same size as inkval. needed for next step
    zero = torch.zeros_like(inkval)
    # sum all inkvals with same index
    lind_unique = torch.unique(lind)
    inkval_unique = torch.stack(
        [torch.sum(torch.where(lind==i, inkval, zero)) for i in lind_unique]
    )
    D[lind_unique] += inkval_unique
    # reshape the image back to 2D from 1D
    D = D.view(shape)

    return D

def space_motor_to_img(pt):
    """
    Translate all control points from spline space to image space.
    Changes all points (x, -y) -> (y, x)

    Parameters
    ----------
    pt : (...,neval,2) tensor
        spline point sequence for each sub-stroke

    Returns
    -------
    new_pt : (...,neval,2) tensor
        image point sequence for each sub-stroke
    """
    assert isinstance(pt, torch.Tensor)
    new_pt = torch.cat([-pt[...,1:], pt[...,:1]], dim=-1)

    return new_pt

def render_image(cell_traj, epsilon, blur_sigma, parameters):
    """
    TODO

    Parameters
    ----------
    cell_traj : (nsub_total,neval,2) tensor, or list of (neval,2) tensor
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
    traj_img = [space_motor_to_img(traj) for traj in cell_traj]

    # get relevant parameters
    imsize = parameters.imsize
    ink = parameters.ink_pp
    max_dist = parameters.ink_max_dist

    # draw the trajectories on the image
    pimg = torch.zeros(imsize, dtype=torch.float)
    nsub_total = len(traj_img)
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