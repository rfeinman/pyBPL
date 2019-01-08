from __future__ import division, print_function
try:
    import pickle # python 3.x
except ImportError:
    import cPickle as pickle # python 2.x
import os
import numpy as np
import scipy.io as sio
import torch

from .dataset import PreprocessedDataset
from .primitive_classifier import PrimitiveClassifierSingle
from .primitive_classifier import PrimitiveClassifierBatch


def make_subid_dict(save_dir):
    data_path = os.path.join(save_dir, 'data_background_splines.mat')
    sid_path = os.path.join(save_dir, 'subid_dict.p')

    if os.path.isfile(sid_path):
        print('SubID dictionary already exists.')
        return

    assert os.path.isfile(data_path), "Must first create " \
                                      "'data_background_splines.mat'"
    D = sio.loadmat(data_path)
    dataset = PreprocessedDataset(
        splines=D['bspline_substks'],
        drawings=D['pdrawings_norm'],
        scales=D['pdrawings_scales']
    )
    spline_dict = dataset.splines
    scales_dict = dataset.scales

    clf = PrimitiveClassifierSingle()
    subid_dict = {}
    for a in spline_dict.keys():
        subid_dict[a] = {}
        for c in spline_dict[a].keys():
            subid_dict[a][c] = {}
            for r in spline_dict[a][c].keys():
                subid_dict[a][c][r] = {}
                for s in spline_dict[a][c][r].keys():
                    ids = []
                    for ss in spline_dict[a][c][r][s].keys():
                        spline = spline_dict[a][c][r][s][ss]
                        scales = scales_dict[a][c][r][s][ss]
                        x = torch.tensor(
                            np.vstack([spline, scales]),
                            dtype=torch.float32
                        )
                        prim_ID = clf.predict(x)
                        ids.append(prim_ID)
                    subid_dict[a][c][r][s] = ids

    with open(sid_path, 'wb') as fp:
        pickle.dump(subid_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

def preprocess_omniglot(save_dir):
    make_subid_dict(save_dir)

def mat2list(X, counts):
    i = 0
    x = []
    for c in counts:
        x.append(list(X[i:i+c]))
        i += c

    return x

def make_subid_dataset(spline_dict, scales_dict):
    clf = PrimitiveClassifierBatch()
    counts = []
    X = []
    for a in spline_dict.keys():
        for c in spline_dict[a].keys():
            for r in spline_dict[a][c].keys():
                for s in spline_dict[a][c][r].keys():
                    n_substrokes = len(spline_dict[a][c][r][s].keys())
                    counts.append(n_substrokes)
                    for ss in spline_dict[a][c][r][s].keys():
                        spline = spline_dict[a][c][r][s][ss]
                        scales = scales_dict[a][c][r][s][ss]
                        X.append(np.vstack([spline, scales]))
    X = torch.tensor(X, dtype=torch.float32)
    prim_IDs = clf.predict(X)
    prim_IDs_list = mat2list(prim_IDs, counts)

    return prim_IDs_list

def make_cpt_dataset(spline_dict, scales_dict):
    X = []
    for a in spline_dict.keys():
        for c in spline_dict[a].keys():
            for r in spline_dict[a][c].keys():
                for s in spline_dict[a][c][r].keys():
                    x = []
                    for i, ss in enumerate(spline_dict[a][c][r][s].keys()):
                        spline = spline_dict[a][c][r][s][ss].reshape(-1)
                        scales = scales_dict[a][c][r][s][ss]
                        spline = np.append(spline, scales[0])
                        x.append(spline)
                        #x.append(np.vstack([spline, scales]))
                    x = np.asarray(x, dtype=np.float32)
                    X.append(x)
    return X