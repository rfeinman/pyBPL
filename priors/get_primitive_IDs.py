import numpy as np
import torch
import torch.distributions as dist

from ..library import Library

class Classifier(object):
    def __init__(self, lib_dir='../../lib_data/'):
        # number of clusters
        N = 1212

        # library
        lib = Library(lib_dir)

        # shapes params
        shapes_mu = lib.shape['mu']
        shapes_cov = lib.shape['Sigma']

        # scales params
        scales_theta = lib.scale['theta']
        scales_con = scales_theta[:, 0]  # gamma concentration
        scales_rate = 1 / scales_theta[:, 1]  # gamma rate

        # get distributions for each subid
        mvns = []
        gammas = []
        for subid in range(N):
            mvn = dist.MultivariateNormal(shapes_mu[subid], shapes_cov[subid])
            gamma = dist.Gamma(scales_con[subid], scales_rate[subid])
            mvns.append(mvn)
            gammas.append(gamma)
        self.mvns = mvns
        self.gammas = gammas
        self.N = N

    def score(self, x, subid):
        assert x.shape == torch.Size([6, 2])
        assert x[-1,0] == x[-1,1]
        scale = x[-1,0]
        cpts = x[:5].view(-1)
        log_prob = self.mvns[subid].log_prob(cpts) + \
                   self.gammas[subid].log_prob(1./scale)

        return log_prob

    def predict(self, x):
        scores = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            scores[i] = self.score(x, subid=i)

        return np.argmax(scores)

def get_IDs(X):
    """
    Parameters
    ----------
    X : (N,6,2) ndarray
        TODO

    Returns
    -------

    """
    return