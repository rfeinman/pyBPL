import os
import warnings
import numpy as np

try:
    import matlab.engine
except:
    raise Exception('Must have the MATLAB Engine API for Python installed')

# start matlab engine
eng = matlab.engine.start_matlab()

try:
    # add BPL code to matlab path
    bpl_path = os.environ['BPL_PATH']
    eng.addpath(eng.genpath(bpl_path), nargout=0)
except:
    warnings.warn('BPL_PATH environment variable not set... therefore you'
                  'must have BPL matlab repository already added to your matlab '
                  'path')

# add current directory to matlab path
eng.addpath(os.path.dirname(__file__), nargout=0)


def generate_random_parses(I, seed=None):
    # convert image to matlab format
    I = matlab.logical(I.tolist())
    # call matlab function
    if seed is not None:
        S_walks = eng.generate_random_parses_RF(I,seed)
    else:
        S_walks = eng.generate_random_parses_RF(I)

    # post-process
    for i in range(len(S_walks)):
        for j in range(len(S_walks[i])):
            S_walks[i][j] = np.asarray(S_walks[i][j])

    return S_walks