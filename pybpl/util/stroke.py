import numpy as np
import torch


def dist_along_traj(stk):
    """
    Compute the total euclidean dist. along a stroke (or sub-stroke) trajectory

    Parameters
    ----------
    stk : (n,2) array

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
    stk : (n,2) array
    newscale : float

    Returns
    -------
    stk : (n,2) array
    center : (2,) array
    invscale : float

    """
    # subtract center of mass
    center = stk.mean(axis=0)
    stk = stk - center

    # re-scale
    range_x, range_y = np.ptp(stk, axis=0)
    invscale = newscale / max(1, max(range_x, range_y))
    stk = stk * invscale

    return stk, center, invscale