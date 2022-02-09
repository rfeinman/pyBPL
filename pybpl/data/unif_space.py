import numpy as np
import torch


def unif_space(stroke, dist_int=1.):
    """
    Transform a stroke so that it is uniformly sampled in space.
    Translated from https://github.com/brendenlake/BPL/blob/master/data/uniform_space_lerp.m

    Parameters
    ----------
    stroke : np.ndarray | torch.Tensor
        (n,2) input stroke
    space_int : float
        space interval; we want approximately this much euclidean distance
        covered between successive points for spatial interpolation

    Returns
    -------
    new_stroke : np.ndarray
        (m,2) interpolated stroke
    """
    num_steps = len(stroke)
    if num_steps == 1:
        # quick return if stroke is too short
        return stroke

    if torch.is_tensor(stroke):
        stroke = stroke.numpy()
        format_output = lambda u: torch.from_numpy(u).float()
    else:
        stroke = np.asarray(stroke)
        format_output = lambda u: u

    # store data type for future ops
    dtype = stroke.dtype

    # compute distance between each point &
    # remove points that are too close to previous
    dist = np.empty(num_steps, dtype=dtype)
    dist[0] = 1
    dist[1:] = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    keep = np.where(dist >= 1e-4)[0]
    if keep.shape[0] == 1:
        # return if filtered stroke is too short
        return format_output(stroke)

    stroke = stroke[keep]
    dist = dist[keep]

    # cumulative distance
    cumdist = np.cumsum(dist)
    nint = int(round(cumdist[-1] / dist_int))
    nint = max(nint, 2)
    query_dist = np.linspace(0, cumdist[-1], nint, dtype=dtype)
    new_stroke = np.empty((nint, 2), dtype=dtype)
    new_stroke[:,0] = np.interp(query_dist, cumdist, stroke[:,0])
    new_stroke[:,1] = np.interp(query_dist, cumdist, stroke[:,1])

    return format_output(new_stroke)