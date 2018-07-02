from __future__ import division, print_function
from abc import ABCMeta, abstractmethod
import torch

# list of acceptable dtypes for 'k' parameter
int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


class ConceptTypeDist(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def sample_k(self):
        """
        Sample a number of parts from the prior

        :return:
            k: [tensor] scalar; part count
        """
        pass

    @abstractmethod
    def score_k(self, k):
        """
        Score the log-probability of the number of parts under the prior

        :param k: [tensor] scalar; part count to score
        :return:
            ll: [tensor] scalar; log-probability of the part count
        """
        pass

    @abstractmethod
    def sample_part_type(self, k):
        """
        Sample a part type from the prior, conditioned on a number of parts

        :param k: [tensor]
        :return:
            p: [Part]
        """
        pass

    @abstractmethod
    def score_part_type(self, k, p):
        """
        Score the log-probability of the part type under the prior, conditioned
        on a number of parts

        :param k: [tensor] scalar; part count
        :param p: [Part] part type to score
        :return:
            ll: [tensor] scalar; log-probability of the part type
        """
        pass

    @abstractmethod
    def sample_relation_type(self, prev_parts):
        """
        Sample a relation type from the prior for the current part, conditioned
        on the previous parts

        :param prev_parts: [list of Parts] TODO
        :return:
            r: [Relation] relation type
        """
        pass

    @abstractmethod
    def score_relation_type(self, prev_parts, r):
        """
        Score the relation type of the current part under the prior

        :param prev_parts: [list of Parts] TODO
        :param r: [Relation] TODO
        :return:
            ll: [tensor] log-probability of the relation type
        """
        pass

    def sample_type(self, k=None):
        """
        Sample a concept type from the prior

        :param k: [int or tensor] scalar; the number of parts to use
        :return:
            P: [list of Part] TODO
            R: [list of Part] TODO
        """
        if k is None:
            # sample the number of parts 'k'
            k = self.sample_k()
        elif isinstance(k, int):
            k = torch.tensor(k)
        else:
            assert isinstance(k, torch.Tensor)
            assert k.shape == torch.Size([])
            assert k.dtype in int_types

        # initialize part and relation lists
        P = []
        R = []
        # for each part, sample part parameters
        for _ in range(k):
            # sample the part type
            part = self.sample_part_type(k)
            # sample the relation type
            relation = self.sample_relation_type(P)
            # append part and relation types to their respective lists
            P.append(part)
            R.append(relation)

        # return the concept type (a stencil for a concept)
        return P, R

    def score_type(self, P, R):
        """
        Score a concept type under the prior
        P(type) = P(k)*\prod_{i=1}^k [P(S_i)P(R_i|S_{0:i-1})]

        :param P: [list of Part] TODO
        :param R: [list of Relation] TODO
        :return:
            ll: [tensor] scalar; the log-probability of the concept type
        """
        # score the number of parts
        k = len(P)
        ll = self.score_k(k)
        for i in range(k):
            ll = ll + self.score_part_type(k, P[i])
            ll = ll + self.score_relation_type(P[:i], R[i])

        return ll

