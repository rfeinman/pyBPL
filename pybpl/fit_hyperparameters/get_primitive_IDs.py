import numpy as np
import torch
import torch.distributions as dist

from pybpl.library import Library


class Classifier(object):
    def __init__(self, lib_dir='../../lib_data/'):
        # library
        lib = Library(lib_dir)

        # shapes params
        shapes_mu = lib.shape['mu']
        shapes_cov = lib.shape['Sigma']

        # scales params
        scales_theta = lib.scale['theta']
        scales_con = scales_theta[:,0]  # gamma concentration
        scales_rate = 1 / scales_theta[:,1]  # gamma rate

        # get distributions for each subid
        self.mvn = dist.MultivariateNormal(shapes_mu, shapes_cov)
        self.gamma = dist.Gamma(scales_con, scales_rate)

    def predict(self, x):
        assert x.shape == torch.Size([6, 2])
        scale = x[-1, 0]
        cpts = x[:5].view(-1)
        log_probs = self.mvn.log_prob(cpts) + self.gamma.log_prob(1./scale)
        _, prim_ID = log_probs.max(0)

        return prim_ID.item()

def get_IDs(X):
    """
    Parameters
    ----------
    X : (N,6,2) ndarray
        TODO

    Returns
    -------
    prim_IDs : (N,) ndarray
        TODO

    """
    clf = Classifier()
    prim_IDs = np.asarray([clf.predict(x) for x in X], dtype=np.int16)

    return prim_IDs