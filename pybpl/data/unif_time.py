import numpy as np


def unif_time(stroke, time_int=50.):
    """
    Transform a stroke so that it is uniformly sampled in time.
    Translated from https://github.com/brendenlake/BPL/blob/master/data/uniform_time_lerp.m

    Parameters
    ----------
    stroke : (n,3) array
        input stroke; each entry (x,y,t) includes an x-y coordinate and a time
    time_int : float
        time interval (milliseconds); strokes will be interpolated such that
        we have datapoints every interval

    Returns
    -------
    unif_stroke : (m,2) array
        new stroke's x-y coordinates
    unif_time : (m,1) array
        new stroke's time intervals
    """
    times = stroke[:,2]
    min_time = min(times)
    max_time = max(times)

    # range excludes endpoint
    unif_times = list(np.arange(min_time,max_time,time_int))
    unif_times.append(max_time)
    unif_times = np.array(unif_times)
    unif_steps = len(unif_times)
    unif_stroke = np.zeros((unif_steps,2))

    for t in range(unif_steps):
        new_time = unif_times[t]
        diffs = times - new_time

        # If some thing/things happen at this time just average their values.
        # NOTE: probably only 1 thing happens at this time
        if np.any(diffs == 0):
            # if some indices have the same time as this new time,
            # average their x,y together
            matches = [xyt for i,xyt in enumerate(stroke) if diffs[i] == 0]
            unif_stroke[t] = np.mean(matches,axis=0)[:2]

        # Otherwise interpolate
        else:
            # last index with time less than new_times[i]
            prev_idx = lt = np.where(diffs<0)[0][-1]

            # first index with time greater than new_times[i]
            post_idx = np.where(diffs>0)[0][0]

            prev_xy = stroke[prev_idx,:2]
            prev_time = times[prev_idx]
            post_xy = stroke[post_idx,:2]
            post_time = times[post_idx]

            # interpolate
            frac = (new_time - prev_time) / (post_time - prev_time)
            assert frac <= 1

            unif_stroke[t] = (1-frac) * prev_xy + frac * post_xy

    return unif_stroke, unif_times