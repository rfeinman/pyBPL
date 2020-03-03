"""
Parameters...
"""
from __future__ import division, print_function
import torch

class defaultps(object):
    def __init__(self):
        # Library to use
        self.libname = 'library'
        self.set_rendering_params()
        self.set_spline_params()
        self.set_image_model_params()
        self.set_mcmc_params()
        self.set_search_params()


    def set_rendering_params(self):
        # image size
        self.imsize = torch.Size([105, 105])

        ## ink-add parameters
        # amount of ink per point
        self.ink_pp = torch.tensor(2, dtype=torch.float)
        # distance between points to which you get full ink
        self.ink_max_dist = torch.tensor(2, dtype=torch.float)

        ## broadening parameters
        # number of convolutions
        self.ink_ncon = torch.tensor(2, dtype=torch.int)
        # parameter 1
        self.ink_a = torch.tensor(0.5, dtype=torch.float)
        # parameter 2
        self.ink_b = torch.tensor(6, dtype=torch.float)
        # broadening version (must be either "Lake" or "Hinton")
        self.broaden_mode = 'Lake'

        ## blurring parameters
        # convolution size for blurring
        self.fsize = 11

    def set_spline_params(self):
        """
        Parameters for creating a trajectory from a spline
        """
        # maxmium number of evaluations
        self.spline_max_neval = torch.tensor(200, dtype=torch.int)
        # minimum number of evaluations
        self.spline_min_neval = torch.tensor(10, dtype=torch.int)
        # 1 trajectory point for every this many units pixel distance
        self.spline_grain = torch.tensor(1.5, dtype=torch.float)

    def set_image_model_params(self):
        """
        Max/min noise parameters for image model
        """
        # min/max blur sigma
        self.max_blur_sigma = torch.tensor(16, dtype=torch.float)
        self.min_blur_sigma = torch.tensor(0.5, dtype=torch.float)
        # min/max pixel epsilon
        self.max_epsilon = torch.tensor(0.5, dtype=torch.float)
        self.min_epsilon = torch.tensor(1e-4, dtype=torch.float)

    def set_mcmc_params(self):
        """
        MCMC parameters
        """
        ## chain parameters
        # number of samples to take in the MCMC chain (for classif.)
        self.mcmc_nsamp_type_chain = 200
        # number of samples to store from this chain (for classif.)
        self.mcmc_nsamp_type_store = 10
        # for completion (we take last sample in this chain)
        self.mcmc_nsamp_token_chain = 25

        ## mcmc proposal parameters
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

    def set_search_params(self):
        """
        Parameters of search algorithm (part of inference)
        """
        # number of particles to use in search algorithm
        self.K = torch.tensor(5, dtype=torch.int)
        # scale changes must be less than a factor of 2
        self.max_affine_scale_change = 2
        # shift changes must less than this
        self.max_affine_shift_change = 50
    
