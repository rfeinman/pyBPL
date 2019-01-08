from __future__ import division, print_function
try:
    import pickle # python 3.x
except ImportError:
    import cPickle as pickle # python 2.x
import os
import scipy.io as sio
from .dataset import Dataset


def preprocess_omniglot(save_dir):
    data_path = os.path.join(save_dir, 'data_background.mat')
    ssd_path = os.path.join(save_dir, 'substroke_dict.p')
    sd_path = os.path.join(save_dir, 'spline_dict.p')
    sid_path = os.path.join(save_dir, 'subid_dict.p')

    # create the dataset
    assert os.path.isfile(data_path)
    print("Loading Data...")
    data = sio.loadmat(
        data_path,
        variable_names=['drawings', 'images', 'names', 'timing']
    )
    D = Dataset(data['drawings'],data['images'],data['names'],data['timing'])

    # create drawing and substroke dictionaries
    if os.path.isfile(ssd_path):
        print('Sub-stroke dictionary already exists.')
    else:
        ss_dict = D.make_substroke_dict()
        with open(ssd_path, 'wb') as fp:
            pickle.dump(ss_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # create spline dictionary
    if os.path.isfile(sd_path):
        print('Spline dictionary already exists.')
    else:
        assert os.path.isfile(ssd_path)
        with open(ssd_path, 'rb') as fp:
            ss_dict = pickle.load(fp)
        D.substroke_dict = ss_dict
        spline_dict = D.make_spline_dict()
        with open(sd_path, 'wb') as fp:
            pickle.dump(spline_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # create subID dictionary
    if os.path.isfile(sid_path):
        print('SubID dictionary already exists.')
    else:
        assert os.path.isfile(sd_path)
        with open(sd_path, 'rb') as fp:
            spline_dict = pickle.load(fp)
        D.spline_dict = spline_dict
        subid_dict = D.make_subid_dict()
        with open(sid_path, 'wb') as fp:
            pickle.dump(subid_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)