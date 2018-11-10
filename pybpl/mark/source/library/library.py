from __future__ import division, print_function
import os
import scipy.io as io
import numpy as np
import torch

from .spatial import SpatialModel
from ..util import aeq


class Library(object):
    """
    LIBRARY: hyper-parameters for the BPL model
    """
    def __init__(self, lib_dir):
        """
        Constructor

        :param lib_dir: [string] path to the library files
        """
        # get contents of dir
        contents = os.listdir(lib_dir)
        # save lists of structs and single elements
        structs = ['shape', 'scale', 'rel', 'tokenvar', 'affine', 'stat']
        singles = [
            'logT', 'logStart', 'pkappa', 'pmat_nsub', 'newscale',
            'smooth_bigrams', 'diagSigma'
        ]
        # load structs
        for elt in structs:
            assert elt in contents
            value = get_dict(os.path.join(lib_dir, elt))
            if elt == 'shape':
                value = fix_shape_params(value)
            setattr(self, elt, value)
        # load individual properties
        for elt in singles:
            assert elt+'.mat' in contents
            value = get_data(elt+'.mat', lib_dir)
            setattr(self, elt, value)
        # change type of 'diagSigma' to torch.uint8 since this is a boolean
        self.diagSigma = self.diagSigma.byte()

        # Finally, load SpatialModel
        clump_ID = 2
        xlim = torch.tensor([0, 105], dtype=torch.float)
        ylim = torch.tensor([-105, 0], dtype=torch.float)
        spatial_model = SpatialModel(xlim, ylim, clump_ID)
        spatial_model.initialize_unif()
        self.Spatial = spatial_model

        # Check consistency of the library
        self.check_consistent()

    @property
    def ncpt(self):
        """
        Get the number of control points

        :return:
            ncpt: [int] the number of control points
        """
        dim = self.shape['mu'].shape[1]
        assert dim % 2 == 0 # dimension must be even
        ncpt = int(dim/2)

        return ncpt

    @property
    def N(self):
        """
        Get the number of primitives

        :return:
            N: [int] the number of primitives
        """
        N = self.shape['mu'].shape[0]

        return N

    def check_consistent(self):
        """
        Check consistency of the number of primitives in the model
        """
        N = self.N
        ncpt = self.ncpt
        assert len(self.shape['mu'].shape) == 2
        assert len(self.shape['Sigma'].shape) == 3
        assert self.shape['mu'].shape[1] == ncpt*2
        assert self.shape['Sigma'].shape[0] == N
        assert self.shape['Sigma'].shape[1] == ncpt*2
        assert self.shape['Sigma'].shape[2] == ncpt*2
        assert self.logT.shape[0] == N
        assert self.logStart.shape[0] == N
        assert self.shape['mixprob'].shape[0] == N
        assert self.shape['freq'].shape[0] == N
        assert self.shape['vsd'].shape[0] == N
        assert self.scale['theta'].shape[0] == N
        assert aeq(torch.sum(torch.exp(self.logStart)), torch.tensor(1.))
        for sid in range(N):
            pT = self.pT(torch.tensor(sid))
            assert aeq(torch.sum(pT), torch.tensor(1.))

    def pT(self, prev_state):
        """
        Get the probability of transitioning to a new state, given your current
        state is "prev_state"

        :param prev_state: [tensor] current state of the model
        :return:
            p: [tensor] probability vector; probabilities of transitioning to
                        each potential new state
        """
        assert prev_state.shape == torch.Size([])
        logR = self.logT[prev_state]
        R = torch.exp(logR)
        p = R / torch.sum(R)

        return p

    @property
    def isunif(self):
        return torch.isnan(self.shape['mu']).any()


def get_dict(path):
    """
    load folder of arrays as dictionary of tensors
    """
    field = {}
    contents = os.listdir(path)
    for item in contents:
        key = item.split('.')[0]
        field[key] = get_data(item, path)

    return field

def get_data(item, path):
    """
    load single array as a tensor
    """
    item_path = os.path.join(path, item)
    data = io.loadmat(item_path)['value']
    data = data.astype(np.float32)  # convert to float32
    out = torch.squeeze(torch.tensor(data, dtype=torch.float))

    return out

def fix_shape_params(shape):
    """
    fix organization of shapes 'mu' and 'Sigma' arrays to account for
    differences in the 'reshape' operation between MATLAB and numpy/pytorch
    """
    shapes_mu = shape['mu']
    shapes_Cov = shape['Sigma']
    n, m = shapes_mu.shape
    assert m % 2 == 0
    ncpt = m // 2
    # fix shapes mean
    shapes_mu = shapes_mu.view(n, 2, ncpt)  # (n, 2, ncpt)
    shapes_mu = shapes_mu.permute(0, 2, 1)  # (n, ncpt, 2)
    shapes_mu = shapes_mu.contiguous()
    shapes_mu = shapes_mu.view(n, ncpt * 2)  # (n, ncpt*2)
    shapes_mu = shapes_mu.contiguous()
    # fix shapes covariance
    shapes_Cov = shapes_Cov.permute(2, 0, 1)  # (n, 2*ncpt, 2*ncpt)
    shapes_Cov = shapes_Cov.view(n, 2, ncpt, 2, ncpt)  # (n, 2, ncpt, 2, ncpt)
    shapes_Cov = shapes_Cov.permute(0, 2, 1, 4, 3)  # (n, ncpt, 2, ncpt, 2)
    shapes_Cov = shapes_Cov.contiguous()
    shapes_Cov = shapes_Cov.view(n, ncpt * 2, ncpt * 2)  # (n, ncpt*2, ncpt*2)
    shapes_Cov = shapes_Cov.contiguous()
    # re-assign
    shape['mu'] = shapes_mu
    shape['Sigma'] = shapes_Cov

    return shape
