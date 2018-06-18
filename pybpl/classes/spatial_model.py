"""
Spatial model class definition.
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import torch

from .spatial_hist import SpatialHist


class SpatialModel(object):
    """
    Stores a set of SpatialHist's, one for each stroke position, and can
    evaluate the likelihood/sample new positions.
    """

    def __init__(
            self, data_start=None, data_id=None, clump_id=None,
            xlim=None, ylim=None, nbin_per_side=None, prior_count=None
    ):
        """
        Initialize the SpatialModel class instance.

        :param data_start: [(n,2) tensor] input data array
        :param data_id: [(n,) tensor] index array
        :param clump_id: [int] the id of...
        :param xlim: [list of 2 ints] (xmin,xmax); range of x-dimension
        :param ylim: [list of 2 ints] (ymin,ymax); range of y-dimension
        :param nbin_per_side: [int] number of bins per dimension
        :param prior_count: [float] prior counts in each cell (not added to
                            edge cells)
        """
        # if params are empty, return; model properties will be set later
        # using 'set_properties' method
        params = [
            data_start, data_id, clump_id, xlim, ylim, nbin_per_side,
            prior_count
        ]
        if all([item is None for item in params]):
            return

        assert isinstance(data_start, torch.Tensor)
        assert isinstance(data_id, torch.Tensor)
        assert len(data_id) == len(data_start)
        assert len(xlim) == 2 and len(ylim) == 2

        # Learn specific spatial models
        self.list_SH = []
        for i in range(clump_id):
            sh = SpatialHist(
                data_start[data_id==i], xlim, ylim, nbin_per_side, prior_count
            )
            self.list_SH.append(sh)

        # lump together datapoints from strokes after and including clump_id
        sh = SpatialHist(
            data_start[data_id>=clump_id], xlim, ylim, nbin_per_side,
            prior_count
        )
        self.list_SH.append(sh)

    def set_properties(self, list_SH):
        """
        Set the list_SH property of the model manually

        :param list_SH: [list of SpatialHist instances]
        :return: None
        """
        assert type(list_SH) == list
        for elt in list_SH:
            assert isinstance(elt, SpatialHist)
        self.list_SH = list_SH

    @property
    def last_model_id(self):
        """
        Stroke ids after this are given to the same model (inclusive)

        :return:
            out: [int] id of the last model
        """
        out = len(self.list_SH)

        return out

    def score(self, data_start, data_id):
        """
        Compute log-likelihood of new points

        :param data_start: [(n,2) tensor] positions
        :param data_id: [(n,) tensor] the stroke index of each position
        :return:
            ll: [float] total log-likelihood
        """
        assert isinstance(data_start, torch.Tensor)
        assert isinstance(data_id, torch.Tensor)
        new_id = self.__map_indx(data_id)
        ndat = len(data_start)

        # for each stroke id
        ll = 0
        for sid in range(self.last_model_id):
            data = data_start[new_id==sid]
            ll += self.list_SH[sid].score(data)

        return ll

    def score_vec(self, data_start, data_id):
        """
        Compute log-likelihood of new points, and return breakdown for each one

        :param data_start: [(n,2) tensor] positions
        :param data_id: [(n,) tensor] the stroke index of each position
        :return:
            ll: [(n,) array] the log-likelihood of each position
        """
        assert isinstance(data_start, torch.Tensor)
        assert isinstance(data_id, torch.Tensor)
        new_id = self.__map_indx(data_id)
        ndat = len(data_start)
        ll = torch.zeros(ndat)
        for sid in range(self.last_model_id):
            data = data_start[new_id==sid]
            _, ll[new_id==sid] = self.list_SH[sid].get_id(data)

        return ll

    def sample(self, data_id):
        """
        Sample new stroke start positions

        :param data_id: [(nsamp,) tensor] the stroke index of each position
        :return:
            samples: [(nsamp,2) tensor] positions drawn from the model
        """
        assert isinstance(data_id, torch.Tensor)
        assert len(data_id.shape) == 1
        nsamp = len(data_id)
        new_id = self.__map_indx(data_id)
        print('new_id: ', new_id)

        # for each stroke id
        samples = torch.zeros(nsamp, 2)
        for sid in range(self.last_model_id):
            # TODO - fix this
            sel = new_id == sid
            print('sel: ', sel)
            nsel = torch.sum(sel)
            samp, _, _ = self.list_SH[sid].sample(nsel.item())
            samples[new_id==sid] = samp

        return samples

    def plot(self):
        """
        Plot the array of position models

        :return: None
        """
        n = self.last_model_id
        nrow = np.ceil(np.sqrt(n))
        plt.figure()
        for sid in range(self.last_model_id):
            plt.subplot(nrow, nrow, sid+1)
            self.list_SH[sid].plot(subplot=True)
            plt.title("%i" % sid)
        plt.show()

    def __map_indx(self, old_id):
        """
        Map stroke ids to new ids

        :param old_id:
        :return:
            new_id:
        """
        new_id = old_id
        new_id[new_id>self.last_model_id] = self.last_model_id

        return new_id