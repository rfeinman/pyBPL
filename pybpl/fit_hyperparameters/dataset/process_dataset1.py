from __future__ import division, print_function
try:
    import pickle # python 3.x
except ImportError:
    import cPickle as pickle # python 2.x
import os
import numpy as np
import scipy.io as sio

from .preprocessed_dataset import PreprocessedDataset

def preprocess_omniglot(save_dir):
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
    dataset.make_subid_dict()
    with open(sid_path, 'wb') as fp:
        pickle.dump(dataset.subids, fp, protocol=pickle.HIGHEST_PROTOCOL)