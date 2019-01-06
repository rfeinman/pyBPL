import numpy as np
import torch.distributions as dist

from pybpl.library import Library


class PrimitiveClassifierBatch(object):
    """
    Classifier for predicting a batch of splines. Input X will have shape
    (n,6,2), where 'n' is the number of splines.

    Parameters
    ----------
    lib_dir : str
        path to library data folder
    """
    def __init__(self, lib_dir):
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
        mvns = []
        gammas = []
        for subid in range(lib.N):
            mvn = dist.MultivariateNormal(shapes_mu[subid], shapes_cov[subid])
            gamma = dist.Gamma(scales_con[subid], scales_rate[subid])
            mvns.append(mvn)
            gammas.append(gamma)
        self.mvns = mvns
        self.gammas = gammas
        self.N = lib.N

    def score(self, X, subid):
        """
        Parameters
        ----------
        X : (n,6,2) ndarray
        subid : int

        Returns
        -------
        log_prob : (n,) ndarray
        """
        n,m,d = X.shape
        assert (m,d) == (6,2)
        scale = X[:,-1,0]
        cpts = X[:,:5].view(-1,10)
        log_prob = self.mvns[subid].log_prob(cpts) + \
                   self.gammas[subid].log_prob(1./scale)
        log_prob = log_prob.numpy()

        return log_prob

    def predict(self, X):
        """
        Parameters
        ----------
        X : (n,6,2) ndarray
            splines to classify

        Returns
        -------
        prim_IDs : (n,) ndarray
            primitive ID labels, one per spline
        """
        n = X.shape[0]
        scores = np.zeros((self.N, n), dtype=np.float32)
        for i in range(self.N):
            scores[i] = self.score(X, subid=i)

        prim_IDs = np.argmax(scores, axis=0).astype(np.int16)

        return prim_IDs


class PrimitiveClassifierSingle(object):
    """
    Classifier for predicting a single spline. Input x will have shape (6,2).

    Parameters
    ----------
    lib_dir : str
        path to library data folder
    """
    def __init__(self, lib_dir):
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
        """
        Parameters
        ----------
        x : (6,2) ndarray
            spline to classify

        Returns
        -------
        prim_IDs : int
            primitive ID label
        """
        m,d = x.shape
        assert (m,d) == (6,2)
        scale = x[-1, 0]
        cpts = x[:5].view(-1)
        log_probs = self.mvn.log_prob(cpts) + self.gamma.log_prob(1./scale)
        _, prim_ID = log_probs.max(0)
        prim_ID = prim_ID.item()

        return prim_ID