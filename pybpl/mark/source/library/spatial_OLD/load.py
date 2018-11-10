import os
import scipy.io as sio
import torch

from .spatial_hist import SpatialHist
from .spatial_model import SpatialModel

def load_SpatialHist(path):
    """
    load spatial histogram
    """
    # load all hist properties
    logpYX = sio.loadmat(os.path.join(path, 'logpYX'))['value']
    xlab = sio.loadmat(os.path.join(path, 'xlab'))['value']
    ylab = sio.loadmat(os.path.join(path, 'ylab'))['value']
    rg_bin = sio.loadmat(os.path.join(path, 'rg_bin'))['value']
    prior_count = sio.loadmat(os.path.join(path, 'prior_count'))['value']
    # fix some of the properties, convert to torch tensors
    logpYX = torch.tensor(logpYX, dtype=torch.float)
    xlab = torch.tensor(xlab[0], dtype=torch.float)
    ylab = torch.tensor(ylab[0], dtype=torch.float)
    rg_bin = torch.tensor(rg_bin[0], dtype=torch.float)
    prior_count = prior_count.item()
    # build the SpatialHist instance
    H = SpatialHist()
    H.set_properties(logpYX, xlab, ylab, rg_bin, prior_count)

    return H

def load_SpatialModel(path):
    hists = sorted(os.listdir(path))
    list_SH = []
    for hist in hists:
        SH = load_SpatialHist(os.path.join(path, hist))
        list_SH.append(SH)
    SM = SpatialModel()
    SM.set_properties(list_SH)

    return SM