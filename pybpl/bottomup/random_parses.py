import os
import numpy as np
import matlab.engine

from .. import BPL_PATH


def generate_random_parses(I, verbose=False):
    if verbose:
        print('starting matlab engine...')
    # start matlab engine
    eng = matlab.engine.start_matlab()
    # add BPL code to matlab path
    eng.addpath(eng.genpath(BPL_PATH), nargout=0)
    # add current directory to matlab path
    eng.addpath(os.path.dirname(__file__), nargout=0)

    if verbose:
        print('generating parses...')
    # convert image to matlab format
    I = matlab.logical(I.tolist())
    # call matlab function
    S_walks = eng.generate_random_parses_RF(I, verbose)

    # post-process
    for i in range(len(S_walks)):
        for j in range(len(S_walks[i])):
            for k in range(len(S_walks[i][j])):
                S_walks[i][j][k] = np.asarray(S_walks[i][j][k])

    return S_walks