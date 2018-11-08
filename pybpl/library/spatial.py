import torch
import torch.distributions as dist

class SpatialHist(object):
    """
    xlim : (2,) tensor
    ylim : (2,) tensor
    """

    def __init__(self, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim

    def fit(self, data):
        """
        Parameters
        ----------
        data : (n,2) tensor
        """
        raise NotImplementedError

    def initialize_unif(self):
        """
        """
        bounds = torch.cat([self.xlim.view(1, -1), self.ylim.view(1, -1)])
        self.dist = dist.Uniform(bounds[:, 0], bounds[:, 1])

    def sample(self, nsamp):
        """
        Parameters
        ----------
        nsamp : int

        Returns
        -------
        samples : (n,2) tensor
        """
        assert hasattr(self, 'dist'), 'model not yet fit'
        assert type(nsamp) == int or \
               (type(nsamp) == torch.Tensor and len(nsamp.shape) == 0)
        samples = self.dist.sample(torch.Size([nsamp]))

        return samples

    def score(self, data):
        """
        Parameters
        ----------
        data : (n,2) tensor

        Returns
        -------
        ll : (n,) tensor
        """
        assert hasattr(self, 'dist'), 'model not yet fit'
        assert len(data.shape) == 2
        assert data.shape[1] == 2
        ll = self.dist.log_prob(data)
        ll = ll.sum(dim=1)

        return ll


class SpatialModel(object):
    """
    Use specific distributions if 0 <= part_ID < clump_ID, or a clump
    clump distribution if part_ID >= clump_ID.

    Parameters
    ----------
    xlim :
    ylim :
    clump_ID : int
        part index at which we begin clumping
    """

    def __init__(self, xlim, ylim, clump_ID):
        self.xlim = xlim
        self.ylim = ylim
        self.clump_ID = clump_ID

    def fit(self, data, data_id):
        """
        """
        raise NotImplementedError

    def initialize_unif(self):
        """
        """
        list_SH = []
        for sid in range(self.clump_ID + 1):
            sh = SpatialHist(self.xlim, self.ylim)
            sh.initialize_unif()
            list_SH.append(sh)
        self.list_SH = list_SH

    def sample(self, part_IDs):
        """
        Parameters
        ----------
        part_IDs : (n,) tensor

        Returns
        -------
        samples : (n,2) tensor
        """
        assert hasattr(self, 'list_SH'), 'model not yet fit'
        assert isinstance(part_IDs, torch.Tensor)
        assert len(part_IDs.shape) == 1
        nsamp = len(part_IDs)
        new_IDs = self.__map_indx(part_IDs)

        # for each stroke ID
        samples = torch.zeros(nsamp, 2)
        for sid in range(self.clump_ID + 1):
            sel = new_IDs == sid
            nsel = torch.sum(sel)
            # if nsel > 0 then sample
            if nsel.byte():
                samples[sel] = self.list_SH[sid].sample(nsel.item())

        return samples

    def score(self, data, part_IDs):
        """
        Parameters
        ----------
        data : (n,2) tensor
        part_IDs : (n,) tensor

        Returns
        -------
        ll : (n,) tensor
        """
        assert hasattr(self, 'list_SH'), 'model not yet fit'
        assert isinstance(data, torch.Tensor)
        assert isinstance(part_IDs, torch.Tensor)
        assert len(data.shape) == 2
        assert len(part_IDs.shape) == 1
        assert data.shape[1] == 2
        nsamp = len(part_IDs)
        new_IDs = self.__map_indx(part_IDs)

        # for each stroke ID
        ll = torch.zeros(nsamp)
        for sid in range(self.clump_ID + 1):
            sel = new_IDs == sid
            nsel = torch.sum(sel)
            # if nsel > 0 then score
            if nsel.byte():
                ll[sel] = self.list_SH[sid].score(data[sel])

        return ll

    def __map_indx(self, old_IDs):
        """
        Parameters
        ----------
        old_IDs : (n,) tensor

        Returns
        -------
        new_IDs : (n,) tensor
        """
        new_IDs = old_IDs
        new_IDs[new_IDs > self.clump_ID] = self.clump_ID

        return new_IDs