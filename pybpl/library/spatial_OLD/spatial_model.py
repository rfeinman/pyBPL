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
    Stores a set of SpatialHist's, one for each stroke number. Can
    evaluate the likelihood of a stroke position and sample new positions.
    """

    def __init__(
            self, data_start=None, data_id=None, clump_id=None,
            xlim=None, ylim=None, nbin_per_side=None, prior_count=None
    ):
        """
        Initialize the SpatialModel class instance.

        :param data_start: [(n,2) tensor] input data array
        :param data_id: [(n,) tensor] index array
        :param clump_id: [int] the part index at which we start to clump.
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

        # learn separate spatial histograms for each stroke num up to 'clump_id'
        self.list_SH = []
        for sid in range(clump_id):
            sh = SpatialHist(
                data_start[data_id==sid], xlim, ylim, nbin_per_side, prior_count
            )
            self.list_SH.append(sh)

        # lump together datapoints from strokes after and including clump_id
        sh = SpatialHist(
            data_start[data_id>=clump_id], xlim, ylim, nbin_per_side,
            prior_count
        )
        self.list_SH.append(sh)
        self.xlim = xlim
        self.ylim = ylim
        self.clump_id = clump_id

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
        # make sure all spatial hists have same bounds
        xlim = list_SH[0].xlim
        ylim = list_SH[0].ylim
        for elt in list_SH[1:]:
            assert torch.all(elt.xlim == xlim)
            assert torch.all(elt.ylim == ylim)
        self.xlim = xlim
        self.ylim = ylim
        self.clump_id = len(list_SH)-1

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
        assert len(data_start.shape) == 2
        assert len(data_id.shape) == 1
        new_id = self.__map_indx(data_id)
        ndat = len(data_start)

        # for each stroke id
        ll = 0.
        for sid in range(self.clump_id+1):
            sel = new_id == sid
            nsel = torch.sum(sel)
            # if nsel > 0 then score
            if nsel.byte():
                data = data_start[sel]
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
        assert len(data_start.shape) == 2
        assert len(data_id.shape) == 1
        new_id = self.__map_indx(data_id)
        ndat = len(data_start)
        ll = torch.zeros(ndat)
        for sid in range(self.clump_id+1):
            sel = new_id == sid
            nsel = torch.sum(sel)
            # if nsel > 0 then score
            if nsel.byte():
                data = data_start[sel]
                _, ll[sel] = self.list_SH[sid].get_id(data)

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

        # for each stroke id
        samples = torch.zeros(nsamp, 2)
        for sid in range(self.clump_id+1):
            sel = new_id == sid
            nsel = torch.sum(sel)
            # if nsel > 0 then sample
            if nsel.byte():
                samp, _, _ = self.list_SH[sid].sample(nsel.item())
                # PYPROB CUDA
                samples[sel] = samp.float().cpu()

        return samples

    def plot(self):
        """
        Plot the array of position models

        :return: None
        """
        nrow = np.ceil(np.sqrt(self.clump_id+1))
        plt.figure()
        for sid in range(self.clump_id+1):
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
        new_id[new_id>self.clump_id] = self.clump_id

        return new_id