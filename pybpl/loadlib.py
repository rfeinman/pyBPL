"""
A function that loads the library.
"""
from __future__ import print_function, division
import os
import scipy.io as io

from pybpl.classes import SpatialHist, SpatialModel


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

def loadlib(lib_dir='./library'):
    lib = {}
    contents = os.listdir(lib_dir)
    # First, load everything except SpatialModel
    contents.remove('Spatial')
    for item in contents:
        item_path = os.path.join(lib_dir, item)
        if item_path.endswith('.mat'):
            data = io.loadmat(item_path)['value']
            key = item.split('.')[0]
            lib[key] = data
        else:
            if item not in lib:
                lib[item] = {}
            dir_content1 = os.listdir(item_path)
            for item1 in dir_content1:
                item_path1 = os.path.join(item_path, item1)
                data = io.loadmat(item_path1)['value']
                key = item1.split('.')[0]
                lib[item][key] = data

    # Finally, load SpatialModel
    hists_path = os.path.join(lib_dir, 'Spatial')
    hists = sorted(os.listdir(hists_path))
    list_SH = []
    for hist in hists:
        SH = load_hist(os.path.join(hists_path, hist))
        list_SH.append(SH)
    SM = SpatialModel()
    SM.set_properties(list_SH)
    lib['Spatial'] = SM

    return lib