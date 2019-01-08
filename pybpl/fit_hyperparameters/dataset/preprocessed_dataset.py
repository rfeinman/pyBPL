from __future__ import division, print_function
try:
    import pickle # python 3.x
except ImportError:
    import cPickle as pickle # python 2.x
import os
import scipy.io as sio
import numpy as np
import torch

from ..primitives.primitive_classifier import PrimitiveClassifierSingle
from ..primitives.primitive_classifier import PrimitiveClassifierBatch


def generate_preprocessed_dataset(save_dir):
    data_path = os.path.join(save_dir, 'data_background_splines.mat')
    sid_path = os.path.join(save_dir, 'subid_dict.p')

    assert os.path.isfile(data_path)
    print('Loading data...')
    D = sio.loadmat(data_path)
    dataset = PreprocessedDataset(
        splines=D['bspline_substks'],
        drawings=D['pdrawings_norm'],
        scales=D['pdrawings_scales']
    )

    if os.path.isfile(sid_path):
        print('SubID dictionary already exists.')
    else:
        dataset.make_subid_dict()
        with open(sid_path, 'wb') as fp:
            pickle.dump(dataset.subids, fp, protocol=pickle.HIGHEST_PROTOCOL)


class PreprocessedDataset(object):
    def __init__(self, splines, drawings, scales):
        '''
        Use this on the output of omniglot_extract_splines.m

        import numpy as np
        from scipy.io import loadmat
        data = loadmat(
            'data_background_splines.mat',
            variable_names=['bspline_substks','pdrawings_norm','pdrawings_scales']
        )
        D = PreprocessedDataset(
            data['bspline_substks'],
            data['pdrawings_norm'],
            data['pdrawings_scales']
        )
        '''
        self.splines = {}
        self.drawings = {}
        self.scales = {}

        n_alpha = len(splines)
        for a in range(n_alpha):
            alphabet_sp = splines[a, 0]
            alphabet_d = drawings[a, 0]
            alphabet_s = scales[a, 0]
            n_char = len(alphabet_sp)
            self.splines[a] = {}
            self.drawings[a] = {}
            self.scales[a] = {}
            for c in range(n_char):
                character_sp = alphabet_sp[c, 0]
                character_d = alphabet_d[c, 0]
                character_s = alphabet_s[c, 0]
                n_rend = len(character_sp)
                self.splines[a][c] = {}
                self.drawings[a][c] = {}
                self.scales[a][c] = {}
                for r in range(n_rend):
                    rendition_sp = character_sp[r, 0]
                    rendition_d = character_d[r, 0]
                    rendition_s = character_s[r, 0]
                    num_strokes = len(rendition_sp)
                    self.splines[a][c][r] = {}
                    self.drawings[a][c][r] = {}
                    self.scales[a][c][r] = {}
                    for s in range(num_strokes):
                        if rendition_sp[s, 0].size == 0:
                            continue
                        stroke_sp = rendition_sp[s, 0]
                        stroke_d = rendition_d[s, 0]
                        stroke_s = rendition_s[s, 0]
                        num_substrokes = len(stroke_sp)
                        self.splines[a][c][r][s] = {}
                        self.drawings[a][c][r][s] = {}
                        self.scales[a][c][r][s] = {}
                        for ss in range(num_substrokes):
                            self.splines[a][c][r][s][ss] = stroke_sp[ss, 0]
                            self.drawings[a][c][r][s][ss] = stroke_d[ss, 0]
                            self.scales[a][c][r][s][ss] = stroke_s[ss, 0]

    def make_subid_dict(self):
        clf = PrimitiveClassifierSingle()
        subid_dict = {}
        for a in self.splines.keys():
            subid_dict[a] = {}
            for c in self.splines[a].keys():
                subid_dict[a][c] = {}
                for r in self.splines[a][c].keys():
                    subid_dict[a][c][r] = {}
                    for s in self.splines[a][c][r].keys():
                        ids = []
                        for ss in self.splines[a][c][r][s].keys():
                            spline = self.splines[a][c][r][s][ss]
                            scales = self.scales[a][c][r][s][ss]
                            x = torch.tensor(
                                np.vstack([spline, scales]),
                                dtype=torch.float32
                            )
                            prim_ID = clf.predict(x)
                            ids.append(prim_ID)
                        subid_dict[a][c][r][s] = ids
        self.subids = subid_dict

    def make_subid_dataset(self):
        spline_dict = self.splines
        scales_dict = self.scales
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
        prim_IDs = mat2list(prim_IDs, counts)

        return prim_IDs

    def make_cpt_dataset(self):
        spline_dict = self.splines
        scales_dict = self.scales
        prim_cpts = []
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
                            # x.append(np.vstack([spline, scales]))
                        x = np.asarray(x, dtype=np.float32)
                        prim_cpts.append(x)

        return prim_cpts

def mat2list(X, counts):
    i = 0
    x = []
    for c in counts:
        x.append(list(X[i:i+c]))
        i += c

    return x