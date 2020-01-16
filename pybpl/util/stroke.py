import numpy as np
import torch


def dist_along_traj(stk):
    """
    Compute the total euclidean dist. along a stroke (or sub-stroke) trajectory

    Parameters
    ----------
    stk : (n,2) array or tensor

    Returns
    -------
    total_dist : float

    """
    assert stk.shape[1] == 2
    if isinstance(stk, np.ndarray):
        distances = np.linalg.norm(stk[1:] - stk[:-1], axis=1)
    elif isinstance(stk, torch.Tensor):
        distances = torch.norm(stk[1:] - stk[:-1], dim=1)
    else:
        raise Exception
    dist = distances.sum()
    return dist

def remove_short_stks(slist, minlen, mindist):
    """
    Remove short strokes from a character, or sub-strokes from a stroke

    Parameters
    ----------
    slist
    minlen
    mindist

    Returns
    -------
    slist_new
    """
    slist_new = []
    for stk in slist:
        slen = len(stk)
        sdist = dist_along_traj(stk)
        if slen < minlen and sdist < mindist:
            # do not add this stroke
            continue
        slist_new.append(stk)

    return slist_new

def normalize_stk(stk, newscale=105.):
    """
    Normalize a stroke (or substroke) by subtracting the center of mass
    and re-scaling

    Parameters
    ----------
    stk : (n,2) array or tensor
    newscale : float

    Returns
    -------
    stk : (n,2) array
    center : (2,) array
    invscale : float

    """
    # subtract center of mass
    center = stk.mean(0)
    stk = stk - center

    # re-scale
    if isinstance(stk, np.ndarray):
        range_x, range_y = np.ptp(stk, 0)
    elif isinstance(stk, torch.Tensor):
        vmax, _ = stk.max(0)
        vmin, _ = stk.min(0)
        range_x, range_y = (vmax - vmin)
    else:
        raise Exception
    invscale = newscale / max(1, max(range_x, range_y))
    stk = stk * invscale

    return stk, center, invscale

def affine_warp(stk, affine):
    """
    Parameters
    ----------
    stk : (n,2) tensor
    affine : (4,) tensor

    Returns
    -------
    stk : (n,2) tensor

    """
    stk = stk * affine[:2] + affine[2:]
    return stk

def com_stk(stk):
    """
    Get center-of-mass for one stroke

    Parameters
    ----------
    stk : (ncpt, 2) tensor
        stroke

    Returns
    -------
    center : (2,) tensor
        center of mass
    """
    center = stk.mean(0)
    return center

def com_char(char):
    """
    Get the overall center-of-mass for a character

    Parameters
    ----------
    char : (nsub_total, ncpt, 2) tensor
        the substrokes that define the character

    Returns
    -------
    center : (2,) tensor
        center of mass of the character
    """
    char = char.view(-1,2) # (nsub_total*2, 2)
    center = char.mean(0) # (2,)
    return center