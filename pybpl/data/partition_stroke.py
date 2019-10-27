import numpy as np


def partition_stroke(unif_stk, dthresh=1., max_sequence=np.infty):
    """
    Partition a stroke into sub-strokes based on pauses of the pen.
    Translated from https://github.com/brendenlake/BPL/blob/master/data/partition_strokes.m

    Parameters
    ----------
    unif_stk : (n,2) array
        input stroke; assumed to be uniform time-sampled
    dthresh : float
        space interval; if this much distance/norm is not covered at each
        time point then it's considered a pause
    max_sequence : float
        max length of a stop sequence before it is considered its own stroke

    Returns
    -------
    substrokes : list
        the partitioned sub-strokes
    unif_stk : (m,2) array
        modified version of the input stroke; pause sequences have been
        shorted to a single point
    breaks : (m,) array
        boolean array indicating where pauses occurred
    """
    unif_stk = np.copy(unif_stk) # don't overwrite the original
    n = len(unif_stk)

    # Special case
    if n == 1:
        substrokes = [unif_stk]
        breaks = True
        return substrokes, unif_stk, breaks

    # Compute norm of derivatives
    dxdt = get_deriv(unif_stk) # deriv
    norm_dxdt = np.linalg.norm(dxdt, axis=1) # deriv norm

    # compute candidate stop points
    stop_pt = norm_dxdt < dthresh
    for i in range(1,n):
        if stop_pt[i]:
            stop_pt[i-1] = True
    stop_pt[0] = True
    stop_pt[-1] = True

    # Partition the stop points into sequences.
    # Here, non-stops are denoted as zeros, the first stop is a sequence of 1s,
    # second is a sequence 2s, etc. until the pen is moving fast enough again
    stop_sequence = np.zeros(n, dtype=np.int32)
    stop_count = 1
    for i in range(n):
        if stop_pt[i]: # current point is a stop, it's the same stop
            stop_sequence[i] = stop_count
        elif stop_pt[i-1] and stop_pt[i+1]:
            # points surround it are a stop... it's the same stop
            stop_sequence[i] = stop_count
        elif stop_pt[i-1]:
            stop_count += 1 # just finished a stop

    # Special case where the entire stroke is a stop sequence
    if stop_count == 1:
        stop_sequence = np.zeros(n)
        stop_sequence[0] = 1
        stop_sequence[-1] = 2
        stop_count = 2

    # Make sure the stop sequences aren't too long. If they are,
    # we place a sub-stroke break at the beginning and end.
    for i in range(1, stop_count+1):
        idx = np.where(stop_sequence == i)[0]
        nsel = len(idx)
        if nsel > max_sequence:
            stop_sequence[idx[1:]] = 0
            stop_sequence[stop_sequence>i] += 1
            stop_sequence[idx[-1]] = i+1
            stop_count += 1

    # breaks are the average of the stop sequences
    mybreaks = []
    for i in range(1, stop_count+1):
        # find the break
        idx = np.where(stop_sequence == i)[0]
        if i == 1: # begin of stroke
            b = idx[0]
        elif i == stop_count: # end of stroke
            b = idx[-1]
        else: # all others
            b = int(np.round(idx.mean()))
        # append to break list
        mybreaks.append(b)
        # set break element to mean of sequence
        unif_stk[b] = unif_stk[idx].mean(axis=0)
        # mark to keep
        stop_sequence[b] = -1

    # Remove all other stop sequence elements,
    # except for the marked mean
    idx = np.where(stop_sequence > 0)[0]
    unif_stk = np.delete(unif_stk, idx, 0)
    stop_sequence = np.delete(stop_sequence, idx, 0)
    breaks = stop_sequence < 0

    # convert to cell array
    fbreaks = np.where(breaks)[0]
    nbreaks = len(fbreaks)
    if nbreaks == 1: # if this stroke was just a single stop sequence
        assert len(unif_stk) == 1
        substrokes = [unif_stk]
    elif nbreaks > 1:
        substrokes = []
        for subid in range(nbreaks-1):
            sub = unif_stk[fbreaks[subid]:fbreaks[subid+1], :]
            substrokes.append(sub)
    else:
        raise Exception

    # new_start = substrokes[0][0,:]
    # new_end = substrokes[-1][-1,:]
    # assert np.array_equal(new_start,unif_stk[0,:])
    # assert np.array_equal(new_end,unif_stk[-1,:])

    return substrokes, unif_stk, breaks


def get_deriv(X):
    """
    Get spatial derivatives w.r.t. time
    Translated from https://github.com/brendenlake/BPL/blob/master/data/partition_strokes.m
    NOTE: assumes dt is always 1
    """
    steps, dim = X.shape
    dxdt = np.zeros(X.shape)
    for i in range(1, steps):
        prev = X[i-1,:]
        next = X[i,:]
        dxdt[i,:] = next - prev
        # dt is always 1

    return dxdt