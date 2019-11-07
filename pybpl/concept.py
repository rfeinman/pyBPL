from __future__ import division, print_function
from abc import ABCMeta, abstractmethod

from .part import StrokeType, StrokeToken
from .relation import RelationType, RelationToken


class CharacterType:
    """
    Character types are made up of strokes (parts) and relations. Relations are
    each either [independent, attach, attach-along].

    Parameters
    ----------
    k : tensor
        scalar; part count
    stroke_types: list of StrokeType
        part type list
    relation_types : list of RelationType
        relation type list
    """

    def __init__(self, k, stroke_types, relation_types):
        assert isinstance(stroke_types, list)
        assert isinstance(relation_types, list)
        assert len(stroke_types) == len(relation_types)
        assert k > 0
        for stroke_type, relation_type in zip(stroke_types, relation_types):
            assert isinstance(stroke_type, StrokeType)
            assert isinstance(relation_type, RelationType)
        self.k = k
        self.stroke_types = stroke_types
        self.relation_types = relation_types

    def parameters(self):
        """
        Returns a list of parameters that can be optimized via gradient
            descent.

        Returns
        -------
        params : list
            optimizable parameters
        """
        parameters = []
        for stroke_type, relation_type in zip(self.stroke_types,
                                              self.relation_types):
            parameters.extend(stroke_type.parameters())
            parameters.extend(relation_type.parameters())

        return parameters

    def lbs(self, eps=1e-4):
        """
        Returns a list of lower bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        lbs : list
            lower bound for each parameter
        """
        lbs = []
        for stroke_type, relation_type in zip(self.stroke_types,
                                              self.relation_types):
            lbs.extend(stroke_type.lbs(eps))
            lbs.extend(relation_type.lbs(eps))

        return lbs

    def ubs(self, eps=1e-4):
        """
        Returns a list of upper bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        ubs : list
            upper bound for each parameter
        """
        ubs = []
        for stroke_type, relation_type in zip(self.stroke_types,
                                              self.relation_types):
            ubs.extend(stroke_type.ubs(eps))
            ubs.extend(relation_type.ubs(eps))

        return ubs

    def train(self):
        """
        makes params require grad
        """
        for param in self.parameters():
            param.requires_grad_(True)

    def eval(self):
        """
        makes params require no grad
        """
        for param in self.parameters():
            param.requires_grad_(False)

    def to(self, device):
        """
        moves parameters to device
        TODO
        """
        pass


class CharacterToken:
    """
    Character tokens hold all token-level parameters of the character. They
    consist of a list of StrokeTokens and a list of RelationTokens.

    Parameters
    ----------
    P : list of StrokeToken
        stroke tokens
    R : list of RelationToken
        relation tokens
    affine : (4,) tensor
        affine transformation
    epsilon : tensor
        scalar; image noise quantity
    blur_sigma : tensor
        scalar; image blur quantity
    """
    def __init__(self, P, R, affine, epsilon, blur_sigma):
        assert isinstance(P, list)
        assert isinstance(R, list)
        assert len(P) == len(R)
        for ptoken, rtoken in zip(P, R):
            assert isinstance(ptoken, StrokeToken)
            assert isinstance(rtoken, RelationToken)
        self.part_tokens = P
        self.relation_tokens = R

        for ptoken in P:
            assert isinstance(ptoken, StrokeToken)
        self.affine = affine
        self.epsilon = epsilon
        self.blur_sigma = blur_sigma

    def parameters(self):
        """
        Returns a list of parameters that can be optimized via gradient descent.

        Returns
        -------
        params : list
            optimizable parameters
        """
        parameters = []
        for p, r in zip(self.part_tokens, self.relation_tokens):
            parameters.extend(p.parameters())
            parameters.extend(r.parameters())
        parameters.append(self.blur_sigma)

        return parameters

    def lbs(self, eps=1e-4):
        """
        Returns a list of lower bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        lbs : list
            lower bound for each parameter
        """
        lbs = []
        for p, r in zip(self.part_tokens, self.relation_tokens):
            lbs.extend(p.lbs(eps))
            lbs.extend(r.lbs(eps))
        lbs.append(None)

        return lbs

    def ubs(self, eps=1e-4):
        """
        Returns a list of upper bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        ubs : list
            upper bound for each parameter
        """
        ubs = []
        for p, r in zip(self.part_tokens, self.relation_tokens):
            ubs.extend(p.ubs(eps))
            ubs.extend(r.ubs(eps))
        ubs.append(None)

        return ubs

    def train(self):
        """
        makes params require grad
        """
        for param in self.parameters():
            param.requires_grad_(True)

    def eval(self):
        """
        makes params require no grad
        """
        for param in self.parameters():
            param.requires_grad_(False)

    def to(self, device):
        """
        moves parameters to device
        TODO
        """
        pass
