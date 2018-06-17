from __future__ import division, print_function
import warnings
import os
import scipy.io as io
import numpy as np
import torch

from .spatial_hist import SpatialHist
from .spatial_model import SpatialModel
from ..splines import bspline_gen_s


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
            setattr(self, elt, value)
        # load individual properties
        for elt in singles:
            assert elt+'.mat' in contents
            value = get_data(elt+'.mat', lib_dir)
            setattr(self, elt, value)
        # change type of 'diagSigma' to torch.uint8 since this is a boolean
        self.diagSigma = self.diagSigma.byte()

        # Finally, load SpatialModel
        spatial_path = os.path.join(lib_dir, 'Spatial')
        hists = sorted(os.listdir(spatial_path))
        list_SH = []
        for hist in hists:
            SH = load_hist(os.path.join(spatial_path, hist))
            list_SH.append(SH)
        SM = SpatialModel()
        SM.set_properties(list_SH)
        self.Spatial = SM

        # private properties
        self.__eval_int = 1e-2

        # Learned ink model
        self.check_consistent()

        # Caching structure
        self.__create_eval_list()

    def legacylib(self, oldlib):
        # TODO - do we need this?
        return

    def restrict_library(self, keep):
        """
        Remove primitives from library, except for those in "keep"
        TODO

        :param keep: [(N,) array] array of bools; true for an entry if we want
                        to keep that primitive
        :return: None
        """
        return

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
        TODO

        :return: None
        """
        return

    def pT(self, prev_state):
        """
        Get the probability of transitioning to a new state, given your current
        state is "prev_state"

        :param prev_state: [] current state of the model
        :return:
            p: [float] probability of transitioning to a new state
        """
        logR = self.logT[prev_state]
        R = np.exp(logR)
        # TODO

        return

    def score_eval_marg(self, eval_spot_token):
        return

    def __create_eval_list(self):
        """
        Create caching structure for efficiently computing marginal likelihood
        of attachment

        :return: None
        """
        warnings.warn(
            "'__create_eval_list' method not yet implemented. Library "
            "properties '__int_eval_marg' and '__prob_eval_mar' not created.")
        return
        # TODO - fix bspline_gen_s function so we can use this.
        _, lb, ub = bspline_gen_s(self.ncpt, 1)
        x = np.arange(lb, ub+self.__eval_int, self.__eval_int)
        nint = len(x)
        logy = np.zeros(nint)
        for i in range(nint):
            logy[i] = self.__score_relation_eval_marginalize_exact(x[i])
        self.__int_eval_marg = x
        self.__prob_eval_marg = np.exp(logy)

    def __score_relation_eval_marginalize_exact(self, eval_spot_token):
        return


def get_dict(path):
    field = {}
    contents = os.listdir(path)
    for item in contents:
        key = item.split('.')[0]
        field[key] = get_data(item, path)

    return field

def get_data(item, path):
    item_path = os.path.join(path, item)
    data = io.loadmat(item_path)['value']
    data = data.astype(np.float32)  # convert to float32
    out = torch.squeeze(torch.tensor(data, requires_grad=True))

    return out

def load_hist(path):
    # load all hist properties
    logpYX = io.loadmat(os.path.join(path, 'logpYX'))['value']
    xlab = io.loadmat(os.path.join(path, 'xlab'))['value']
    ylab = io.loadmat(os.path.join(path, 'ylab'))['value']
    rg_bin = io.loadmat(os.path.join(path, 'rg_bin'))['value']
    prior_count = io.loadmat(os.path.join(path, 'prior_count'))['value']
    # fix some of the properties
    xlab = xlab[0]
    ylab = ylab[0]
    rg_bin = list(rg_bin[0])
    prior_count = prior_count.item()
    # build the SpatialHist instance
    H = SpatialHist()
    H.set_properties(logpYX, xlab, ylab, rg_bin, prior_count)

    return H