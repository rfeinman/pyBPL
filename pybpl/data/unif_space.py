import numpy as np
from scipy.interpolate import interp1d


def unif_space(stroke, dist_int=1.):
    """
    Transform a stroke so that it is uniformly sampled in space.
    Translated from https://github.com/brendenlake/BPL/blob/master/data/uniform_space_lerp.m

    Parameters
    ----------
    stroke : np.ndarray
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

    # return if stroke is too short
    if len(stroke) == 1:
        return stroke

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
        return stroke

    # cumulative distance
    cumdist = np.cumsum(dist)
    total_dist = cumdist[-1]
    nint = int(round(total_dist/dist_int))
    nint = max(nint, 2)
    fx = interp1d(cumdist, stroke[:,0])
    fy = interp1d(cumdist, stroke[:,1])

    # new stroke
    query_points = np.linspace(0, total_dist, nint)
    new_stroke = np.zeros((nint,2))
    new_stroke[:,0] = fx(query_points)
    new_stroke[:,1] = fy(query_points)

    return new_stroke