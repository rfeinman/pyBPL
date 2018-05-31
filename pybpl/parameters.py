"""
Parameters...
"""
from __future__ import division, print_function

class defaultps():
    def __init__(self):
        # Library to use
        self.libname = 'library'

        # number of particles to use in search algorithm
        self.K = 5

        ## image model parameters ##
        # number of convolutions
        self.ink_ncon = 2
        # image size
        self.imsize = [105, 105]
        # amount of ink per point
        self.ink_pp = 2
        # distance between points to which you get full ink
        self.ink_max_dist = 2
        # ink parameter 1
        self.ink_a = 0.5
        # ink parameter 2
        self.ink_b = 6

        ## Creating a trajectory from a spline ##
        # maxmium number of evaluations
        self.spline_max_neval = 200
        # minimum
        self.spline_min_neval = 10
        # 1 trajectory point for every this many units pixel distance)
        self.spline_grain = 1.5

        ## Max/min noise parameters for image model ##
        # blur kernel width
        self.max_blur_sigma = 16
        self.min_blur_sigma = 0.5
        # pixel flipping
        self.max_epsilon = 0.5
        self.min_epsilon = 1e-4

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
    
