"""
Parameters...
"""
from __future__ import division, print_function
import torch

class defaultps(object):
    def __init__(self):
        # Library to use
        self.libname = 'library'

        # number of particles to use in search algorithm
        self.K = torch.tensor(5, dtype=torch.int)

        ## image model parameters ##
        # number of convolutions
        self.ink_ncon = torch.tensor(2, dtype=torch.int)
        # image size
        self.imsize = torch.Size([105, 105])
        # amount of ink per point
        self.ink_pp = torch.tensor(2, dtype=torch.float)
        # distance between points to which you get full ink
        self.ink_max_dist = torch.tensor(2, dtype=torch.float)
        # ink parameter 1
        self.ink_a = torch.tensor(0.5, dtype=torch.float)
        # ink parameter 2
        self.ink_b = torch.tensor(6, dtype=torch.float)
        # convolution size for blurring
        self.fsize = 11

        ## Creating a trajectory from a spline ##
        # maxmium number of evaluations
        self.spline_max_neval = torch.tensor(200, dtype=torch.int)
        # minimum
        self.spline_min_neval = torch.tensor(10, dtype=torch.int)
        # 1 trajectory point for every this many units pixel distance)
        self.spline_grain = torch.tensor(1.5, dtype=torch.float)

        ## Max/min noise parameters for image model ##
        # blur kernel width
        self.max_blur_sigma = torch.tensor(16, dtype=torch.float)
        self.min_blur_sigma = torch.tensor(0.5, dtype=torch.float)
        # pixel flipping
        self.max_epsilon = torch.tensor(0.5, dtype=torch.float)
        self.min_epsilon = torch.tensor(1e-4, dtype=torch.float)

        ## search parameters ##
        # scale changes must be less than a factor of 2
        self.max_affine_scale_change = 2
        # shift changes must less than this
        self.max_affine_shift_change = 50

        ## MCMC PARAMETERS ##
        ## they were in mcmc. notation, but i changed it for convenience ##

        ## details about the chain ##
        # number of samples to take in the MCMC chain (for classif.)
        self.mcmc_nsamp_type_chain = 200
        # number of samples to store from this chain (for classif.)
        self.mcmc_nsamp_type_store = 10
        # for completion (we take last sample in this chain)
        self.mcmc_nsamp_token_chain = 25

        # mcmc proposal parameters (Note these are based on lib.tokenvar
        # parameters, although here they are hard-coded for convenience)

        # global position move
        self.mcmc_prop_gpos_sd = 1
        # shape move
        self.mcmc_prop_shape_sd = 3/2
        # scale move
        self.mcmc_prop_scale_sd = 0.0235
        # attach relation move
        self.mcmc_prop_relmid_sd = 0.2168
        # multiply the sd of the standard position noise by this to propose
        # new positions from prior
        self.mcmc_prop_relpos_mlty = 2
    
