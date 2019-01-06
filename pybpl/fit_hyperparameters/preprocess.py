from __future__ import division, print_function
try:
    import pickle # python 3.x
except ImportError:
    import cPickle as pickle # python 2.x
import os
import numpy as np
import scipy.io as sio
import torch

from .. import splines
from .dataset import Dataset
from .get_primitive_IDs import PrimitiveClassifierSingle


def make_data_pickles():
    if os.path.isfile('drawings_dict.p') and os.path.isfile('substroke_dict.p'):
        print('Data pickles already exist.')
        return

    assert os.path.isfile('data_background.mat')
    print("Loading Data...")
    data = sio.loadmat(
        'data_background',
        variable_names=['drawings','images','names','timing']
    )
    D = Dataset(data['drawings'],data['images'],data['names'],data['timing'])

    print("Making substroke data...")
    D.make_substroke_dict()
    with open('drawings_dict.p','wb') as fp:
        pickle.dump(D.drawings, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('substroke_dict.p', 'wb') as fp:
        pickle.dump(D.substroke_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def norm_substk(substroke, newscale=105):
    mu = np.mean(substroke, axis=0)
    substroke = substroke - mu
    range_x, range_y = np.ptp(substroke, axis=0)
    scale = newscale / max(1, max(range_x, range_y))
    substroke = substroke * scale

    return substroke, mu, scale


def make_spline_dict():
    if os.path.isfile('spline_dict.p'):
        print('Spline dictionary already exists.')
        return

    with open('substroke_dict.p', 'rb') as fp:
        ss_dict = pickle.load(fp)

    print("Converting sub-strokes to splines...")
    spline_dict = {}
    n_alpha = len(ss_dict)
    for a in range(n_alpha):
        spline_dict[a] = {}
        alphabet = ss_dict[a]
        n_char = len(alphabet)
        for c in range(n_char):
            spline_dict[a][c] = {}
            char = alphabet[c]
            n_rend = len(char)
            for r in range(n_rend):
                spline_dict[a][c][r] = {}
                rendition = char[r]
                n_stroke = len(rendition)
                for s in range(n_stroke):
                    spline_dict[a][c][r][s] = {}
                    stroke = rendition[s]
                    n_substrokes = len(stroke)
                    for ss in range(n_substrokes):
                        num_steps = len(stroke[ss])
                        if num_steps >= 10:
                            spline_dict[a][c][r][s][ss] = np.zeros((5,2))
                            substk = stroke[ss]
                            substk, _, scale = norm_substk(substk)
                            spline = splines.fit_bspline_to_traj(substk,nland=5)
                            # PyTorch -> Numpy
                            spline = spline.numpy()
                            # Add 2 extra dimensions - scales weighted twice
                            spline = np.append(spline,[[scale,scale]],axis=0)
                            spline_dict[a][c][r][s][ss] = spline

    with open('spline_dict.p', 'wb') as fp:
        pickle.dump(spline_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def make_subid_dict():
    if os.path.isfile('subid_dict.p'):
        print('SubID dictionary already exists.')
        return

    with open('substroke_dict.p', 'rb') as fp:
        ss_dict = pickle.load(fp)
    with open('spline_dict.p', 'rb') as fp:
        spline_dict = pickle.load(fp)

    clf = PrimitiveClassifierSingle(lib_dir='../../lib_data')
    subid_dict = {}
    n_alpha = len(ss_dict)
    for a in range(n_alpha):
        subid_dict[a] = {}
        alphabet = ss_dict[a]
        n_char = len(alphabet)
        for c in range(n_char):
            subid_dict[a][c] = {}
            char = alphabet[c]
            n_rend = len(char)
            for r in range(n_rend):
                subid_dict[a][c][r] = {}
                rendition = char[r]
                n_stroke = len(rendition)
                for s in range(n_stroke):

                    ids = []
                    stroke = rendition[s]
                    n_substrokes = len(stroke)
                    for ss in range(n_substrokes):
                        num_steps = len(stroke[ss])
                        if num_steps >= 10:
                            spline = torch.tensor(
                                spline_dict[a][c][r][s][ss],
                                dtype=torch.float32
                            )
                            prim_ID = clf.predict(spline)
                            ids.append(prim_ID)
                    subid_dict[a][c][r][s] = ids

    with open('subid_dict.p', 'wb') as fp:
        pickle.dump(subid_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


def preprocess_omniglot():
    make_data_pickles()
    make_spline_dict()
    make_subid_dict()