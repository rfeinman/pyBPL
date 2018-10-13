"""
The Library module contains a Library class (and its dependencies) for
loading the hyperparameters of a pre-fit BPL model. This is designed primarily
for the Omniglot use case, enabling the transfer of parameters from the matlab
BPL repo (see https://github.com/brendenlake/BPL).

Holds all parameters of the conditional probability
distribution. These parameters have been learned from the training data.
"""
from .library import Library
from .spatial_dist import SpatialDist
from .spatial_hist import SpatialHist, logsumexp_t
from .spatial_model import SpatialModel

__all__ = ["Library", "SpatialDist", "SpatialHist", "SpatialModel",
           "logsumexp_t"]