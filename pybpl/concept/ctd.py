from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import torch


# list of acceptable dtypes for 'np' parameter
int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]

class ConceptTypeDist(object):
    __metaclass__ = ABCMeta

    def __init__(self, lib):
        pass

    @abstractmethod
    def sample_np(self, nsamp=1):
        """
        Sample the number of parts

        :param nsamp: [int] number of samples to draw
        :return:
            np: [(nsamp,) tensor]
        """
        pass

    @abstractmethod
    def score_np(self, np):
        """
        Score the log-probability of the number of parts under the prior

        :param np:
        :return:
            ll: [(nsamp,) tensor]
        """
        pass

    @abstractmethod
    def sample_part_type(self, np):
        """


        :param np:
        :return:
        """

    @abstractmethod
    def sample_relation_type(self, prev_parts):
        """
        Sample a relation type for the current part

        :param prev_parts: [list of Parts] TODO
        :return:
            r: [Relation] TODO
        """
        pass

    def sample_type(self, np=None):
        if np is None:
            # sample the number of parts 'np'
            np = self.sample_np()
        elif isinstance(np, int):
            np = torch.tensor(np)
        else:
            assert isinstance(np, torch.Tensor)
            assert np.shape == torch.Size([])
            assert np.dtype in int_types

        # initialize part and relation lists
        P = []
        R = []
        # for each part, sample part parameters
        for _ in range(np):
            # sample the part type
            part = self.sample_part_type(np)
            # sample the relation type
            relation = self.sample_relation_type(P)
            # append part and relation types to their respective lists
            P.append(part)
            R.append(relation)

        # return the concept type (a stencil for a concept)
        return P, R

