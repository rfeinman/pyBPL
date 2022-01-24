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
    is_tensor = False
    if torch.is_tensor(stroke):
        stroke = stroke.numpy()
        is_tensor = True

    num_steps = len(stroke)

    # return if stroke is too short
    if len(stroke) == 1:
        return torch.from_numpy(stroke).float() if is_tensor else stroke

    # compute distance between each point &
    # remove points that are too close to previous
    dist = np.zeros(num_steps)
    dist[1:] = np.linalg.norm(stroke[1:] - stroke[:-1], axis=-1)
    remove = np.zeros(num_steps, dtype=bool)
    remove[1:] = dist[1:] < 1e-4
    stroke = stroke[~remove]
    dist = dist[~remove]

    # return if stroke is too short
    if len(stroke) == 1:
        return torch.from_numpy(stroke).float() if is_tensor else stroke

    # cumulative distance
    cumdist = np.cumsum(dist)
    nint = int(round(cumdist[-1] / dist_int))
    nint = max(nint, 2)
    query_dist = np.linspace(0, cumdist[-1], nint)
    new_stroke = np.zeros((nint, 2))
    new_stroke[:,0] = np.interp(query_dist, cumdist, stroke[:,0])
    new_stroke[:,1] = np.interp(query_dist, cumdist, stroke[:,1])

    if is_tensor:
        new_stroke = torch.from_numpy(new_stroke).float()

    return new_stroke