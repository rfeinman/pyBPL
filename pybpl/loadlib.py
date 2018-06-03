"""
A function that loads the library.
"""
from __future__ import print_function, division
import os
import scipy.io as io


def loadlib(lib_dir='./library'):
    lib = {}
    files = os.listdir(lib_dir)
    for file in files:
        data = io.loadmat(os.path.join(lib_dir, file))['value']
        key = file.split('.')[0]
        if '-' in key:
            base, sub = key.split('-')
            if base not in lib:
                lib[base] = {}
            lib[base][sub] = data
        else:
            lib[key] = data

    return lib

def loadlib_old(libtype=1250):
    #filename = '/Users/Maxwell/Documents/BPL_inf/pylib250'
    if libtype == 1250:
        filename = '/Users/Maxwell/Documents/BPL_inf/pylib1250'
    elif libtype ==250:
        filename = '/Users/Maxwell/Documents/BPL_inf/pylib250'

    lib = io.loadmat(filename)
    # does not fix problem of everything being embedding in multiple arrays.
    # Will need to squeeze it when put in tensor.

    #cleanup, doing it manually for now
    #shape
    # lib['shape']['mu'] = lib['shape']['mu'][0,0]
    # lib['shape']['Sigma'] = lib['shape']['Sigma'][0,0]
    # lib['shape']['vsd'] = lib['shape']['vsd'][0,0]
    # lib['shape']['mixprob'] = lib['shape']['mixprob'][0,0]
    # lib['shape']['freq'] = lib['shape']['freq'][0,0]

    # #scale
    # lib['scale']['theta'] = lib['scale']['theta'][0,0]

    # #rel
    # lib['rel']['sigma_x'] = lib['rel']['sigma_x'][0,0] #still an array
    # lib['rel']['sigma_y'] = lib['rel']['sigma_x'][0,0]
    # lib['rel']['mixprob'] = lib['rel']['mixprob'][0,0]
    #tokenvar

    #for all of the library stuff, will need to sqeeze.
    return lib
