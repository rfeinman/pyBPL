#parameters 

class defaultps():
    def __init__(self):
	    # Library to use    
	    self.libname = 'library'
	    
	    # Model parameters
	    self.K = 5  #number of particles to use in search algorithm

	    # image model parameters
	    self.ink_ncon = 2 # number of convolutions
	    self.imsize = [105, 105] # image size
	    self.ink_pp = 2       # amount of ink per point
	    self.ink_max_dist = 2 # distance between points to which you get full ink
	    self.ink_a = 0.5      # ink parameter 1
	    self.ink_b = 6        # ink parameter 2
	    
	    # Creating a trajectory from a spline
	    self.spline_max_neval = 200 # maxmium number of evaluations
	    self.spline_min_neval = 10  # minimum
	    self.spline_grain = 1.5     # 1 traj. point for every this many units pixel distance)
	    
	    # Max/min noise parameters for image model
	    self.max_blur_sigma = 16 # blur kernel width
	    self.min_blur_sigma = 0.5
	    self.max_epsilon = 0.5   # pixel flipping
	    self.min_epsilon = 1e-4
	    
	    # search parameters
	    self.max_affine_scale_change = 2  # scale changes must be less than a factor of 2
	    self.max_affine_shift_change = 50 # shift changes must less than this
	    
	    # MCMC PARAMETERS #they were in mcmc. notation, but i changed it for convenience
	    
	    # details about the chain
	    self.mcmc_nsamp_type_chain = 200 # number of samples to take in the MCMC chain (for classif.)
	    self.mcmc_nsamp_type_store = 10 # number of samples to store from this chain (for classif.)
	    self.mcmc_nsamp_token_chain = 25 # for completion (we take last sample in this chain) 
	    
	    # mcmc proposal parameters (Note these are based on lib.tokenvar
	    # parameters, although here they are hard-coded for convenience)
	    self.mcmc_prop_gpos_sd = 1 # global position move
	    self.mcmc_prop_shape_sd = 3. / 2. # shape move
	    self.mcmc_prop_scale_sd = 0.0235 # scale move
	    self.mcmc_prop_relmid_sd = 0.2168 # attach relation move
	    self.mcmc_prop_relpos_mlty = 2 # multiply the sd of the standard position noise by this to propose new positions from prior
    
