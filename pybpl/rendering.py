"""
All the functions and modules for differentiable rendering go here
"""
import torch

from .util.general import sub2ind, fspecial, imfilter
from .util.stroke import com_char, affine_warp
from .parameters import Parameters



# ----
# render the image
# ----

def check_bounds(myt, imsize):
    """
    Given a list of 2D points (x-y coordinates) and an image size, return
    a boolean vector indicating which points are out of the image boundary

    Parameters
    ----------
    myt : torch.Tensor
        (k,2) matrix of 2D points
    imsize : tuple or torch.Size
        image size; H x W

    Returns
    -------
    out : (k,) Byte tensor
        vector indicating which points are out of bounds
    """
    xt = myt[:,0]
    yt = myt[:,1]
    x_out = (torch.floor(xt) < 0) | (torch.ceil(xt) >= imsize[0])
    y_out = (torch.floor(yt) < 0) | (torch.ceil(yt) >= imsize[1])
    out = x_out | y_out

    return out

def seqadd(D, lind_x, lind_y, inkval):
    """
    Add ink to an image at the indicated locations

    Parameters
    ----------
    D : torch.Tensor
        (H,W) image that we'll be adding to
    lind_x : torch.Tensor
        (k,) x-coordinate for each adding point
    lind_y : torch.Tensor
        (k,) y-coordinate for each adding point
    inkval : torch.Tensor
        (k,) amount of ink to add for each adding point

    Returns
    -------
    D : torch.Tensor
        (H,W) image with ink added to it
    """
    assert len(lind_x) == len(lind_y) == len(inkval)
    imsize = D.shape

    # keep only the adding points that are in bounds
    lind_stack = torch.stack([lind_x, lind_y], dim=-1)
    out = check_bounds(lind_stack, imsize=imsize)
    lind_x = lind_x[~out].long()
    lind_y = lind_y[~out].long()
    inkval = inkval[~out]

    # return D if all adding points are out of bounds
    if len(lind_x) == 0:
        return D

    # flatten x-y indices
    lind = sub2ind(imsize, lind_x, lind_y).to(inkval.device)

    # add to image
    D = D.view(-1)
    D = D.scatter_add(0, lind, inkval)
    D = D.view(imsize)

    return D

def space_motor_to_img(pt):
    """
    Translate all control points from spline space to image space.
    Changes all points (x, -y) -> (y, x)

    Parameters
    ----------
    pt : torch.Tensor
        (..., 2) spline point sequence for each sub-stroke

    Returns
    -------
    new_pt : torch.Tensor
        (..., 2) image point sequence for each sub-stroke
    """
    assert torch.is_tensor(pt)
    space_flip = torch.tensor([-1.,1.], device=pt.device)
    new_pt = torch.flip(pt, dims=[-1]) * space_flip

    return new_pt

def add_stroke(pimg, stk, ps):
    """
    Draw one stroke onto an image

    Parameters
    ----------
    pimg : torch.Tensor
        (H,W) current image probability map
    stk : torch.Tensor
        (n,2) stroke to be drawn on the image
    ps : Parameters
        bpl parameters

    Returns
    -------
    pimg : torch.Tensor
        (H,W) updated image probability map
    ink_off_page : bool
        boolean indicating whether the ink went off the page
    """
    device = stk.device
    ink = ps.ink_pp.to(device)
    max_dist = ps.ink_max_dist.to(device)
    ink_off_page = False

    # convert stroke to image coordinate space
    stk = space_motor_to_img(stk)

    # reduce trajectory to only those points that are in bounds
    out = check_bounds(stk, pimg.shape) # boolean; shape (neval,)
    if out.any():
        ink_off_page = True
    if out.all():
        return pimg, ink_off_page
    stk = stk[~out]

    # compute distance between each trajectory point and the next one
    if stk.shape[0] == 1:
        myink = ink
    else:
        dist = torch.norm(stk[1:] - stk[:-1], dim=-1)
        dist = torch.min(dist, max_dist)
        dist = torch.cat([dist[:1], dist])
        myink = (ink/max_dist)*dist # shape (k,)

    # make sure we have the minimum amount of ink, if a particular
    # trajectory is very small
    sumink = torch.sum(myink)
    if sumink < 2.22e-6:
        nink = myink.shape[0]
        myink = (ink/nink)*torch.ones_like(myink)
    elif sumink < ink:
        myink = (ink/sumink)*myink
    assert torch.sum(myink) > (ink-1e-4)

    # share ink with the neighboring 4 pixels
    x = stk[:,0]
    y = stk[:,1]
    xfloor = torch.floor(x).detach()
    yfloor = torch.floor(y).detach()
    xceil = torch.ceil(x).detach()
    yceil = torch.ceil(y).detach()
    x_c_ratio = x - xfloor
    y_c_ratio = y - yfloor
    x_f_ratio = 1 - x_c_ratio
    y_f_ratio = 1 - y_c_ratio

    # paint the image
    pimg = seqadd(pimg, xfloor, yfloor, myink*x_f_ratio*y_f_ratio)
    pimg = seqadd(pimg, xceil, yfloor, myink*x_c_ratio*y_f_ratio)
    pimg = seqadd(pimg, xfloor, yceil, myink*x_f_ratio*y_c_ratio)
    pimg = seqadd(pimg, xceil, yceil, myink*x_c_ratio*y_c_ratio)

    return pimg, ink_off_page

def broaden_and_blur(pimg, blur_sigma, ps):
    """
    Apply broadening and blurring transformations to the image

    Parameters
    ----------
    pimg : torch.Tensor
        (H,W) current image probability map
    blur_sigma : float
        image blur value
    ps : Parameters
        bpl parameters

    Returns
    -------
    pimg : torch.Tensor
        (H,W) updated image probability map
    """
    device = pimg.device

    # filter the image to get the desired brush-stroke size
    a = ps.ink_a
    b = ps.ink_b
    if ps.broaden_mode == 'Lake':
        h_scale = b
    elif ps.broaden_mode == 'Hinton':
        h_scale = b*(1+a)
    else:
        raise Exception("'broaden_mode' must be either 'Lake' or 'Hinton'")

    H_broaden = h_scale*torch.tensor(
        [[a/12, a/6, a/12],[a/6, 1-a, a/6],[a/12, a/6, a/12]],
        dtype=torch.float,
        device=device
    )
    for i in range(ps.ink_ncon):
        pimg = imfilter(pimg, H_broaden, mode='conv')

    # store min and maximum pimg values for truncation
    min_val = torch.tensor(0., dtype=torch.float, device=device)
    max_val = torch.tensor(1., dtype=torch.float, device=device)

    # truncate
    pimg = torch.min(pimg, max_val)

    # filter the image to get Gaussian
    # noise around the area with ink
    if blur_sigma > 0:
        H_gaussian = fspecial(ps.fsize, blur_sigma, ftype='gaussian',
                              device=device)
        pimg = imfilter(pimg, H_gaussian, mode='conv')
        pimg = imfilter(pimg, H_gaussian, mode='conv')

    # final truncation
    pimg = torch.min(pimg, max_val)
    pimg = torch.max(pimg, min_val)

    return pimg

def render_image(strokes, epsilon, blur_sigma, ps=None):
    """
    Render a list of stroke trajectories into a image probability map.
    Reference: BPL/misc/render_image.m

    Parameters
    ----------
    strokes : torch.Tensor | list[torch.Tensor]
        either a single (m,n,2) tensor or a list of (n,2) tensors with
        varying size n; collection of strokes that make up the character
    epsilon : float
        image noise value
    blur_sigma : float
        image blur value
    ps : Parameters
        bpl parameters

    Returns
    -------
    pimg : torch.Tensor
        (H,W) image probability map
    ink_off_page : bool
        boolean indicating whether the ink went off the page
    """
    # load default parameters if necessary
    if ps is None:
        ps = Parameters()
    # initialize the image pixel map
    device = strokes[0].device
    pimg = torch.zeros(ps.imsize, dtype=torch.float, device=device)
    ink_off_page = False

    # draw the strokes on the image
    for stk in strokes:
        pimg, ink_off_i = add_stroke(pimg, stk, ps)
        ink_off_page = ink_off_page or ink_off_i

    # broaden and blur the image
    pimg = broaden_and_blur(pimg, blur_sigma, ps)

    # probability of each pixel being on
    if epsilon > 0:
        pimg = (1-epsilon)*pimg + epsilon*(1-pimg)

    return pimg, ink_off_page
