import os
import numpy as np
import matlab.engine

from .. import BPL_PATH

eng = matlab.engine.start_matlab() # start matlab engine
eng.addpath(eng.genpath(BPL_PATH), nargout=0) # add BPL code to matlab path
eng.addpath(os.path.dirname(__file__), nargout=0) # add current directory to matlab path


def generate_random_parses(I):
    # convert image to matlab format
    I = matlab.logical(I.tolist())
    # call matlab function
    S_walks = eng.generate_random_parses_RF(I)

    # post-process
    for i in range(len(S_walks)):
        for j in range(len(S_walks[i])):
            S_walks[i][j] = np.asarray(S_walks[i][j])

    return S_walks