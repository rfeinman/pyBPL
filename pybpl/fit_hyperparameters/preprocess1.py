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