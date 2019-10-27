import numpy as np
from scipy.interpolate import interp1d


def unif_space(stroke, dist_int=1.):
    """
    Transform a stroke so that it is uniformly sampled in space.
    Translated from https://github.com/brendenlake/BPL/blob/master/data/uniform_space_lerp.m

    Parameters
    ----------
    stroke : (n,2) array
        input stroke
    space_int : float
        space interval; we want approximately this much euclidean distance
        covered between successive points for spatial interpolation

    Returns
    -------
    new_stroke : (m,2) array
        interpolated stroke
    """
    num_steps = len(stroke)

    # return if stroke is too short
    if len(stroke) == 1:
        return stroke

    # compute distance between each point
    dist = np.zeros((num_steps,1))
    to_remove = np.zeros((num_steps,1), dtype=bool) # array of false

    for i in range(1,num_steps):
        xy_1 = stroke[i]
        xy_2 = stroke[i-1]
        diff = xy_1 - xy_2
        dist[i] = np.linalg.norm(diff)
        to_remove[i] = dist[i] < 1e-4

    remove_indices = [i for i,b in enumerate(to_remove) if b==True]

    # remove points that are too close
    stroke = np.delete(stroke,remove_indices,axis=0)
    dist = np.delete(dist,remove_indices,axis=0)

    # return if stroke is too short
    if len(stroke) == 1:
        return stroke

    # cumulative distance
    cumdist = np.cumsum(dist)
    start_dist = cumdist[0]
    end_dist = cumdist[-1]
    nint = round(end_dist/dist_int)
    nint = int(max(nint,2))
    fx = interp1d(cumdist,stroke[:,0])
    fy = interp1d(cumdist,stroke[:,1])

    # new stroke
    query_points = np.linspace(start_dist,end_dist,nint,endpoint=True)
    new_stroke = np.zeros((len(query_points),2))
    new_stroke[:,0] = fx(query_points)
    new_stroke[:,1] = fy(query_points)

    return new_stroke