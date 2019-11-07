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
    def __init__(self, stroke_tokens, relation_tokens, affine, epsilon,
                 blur_sigma):
        assert isinstance(stroke_tokens, list)
        assert isinstance(relation_tokens, list)
        assert len(stroke_tokens) == len(relation_tokens)
        for stroke_token, relation_token in zip(stroke_tokens, relation_tokens):
            assert isinstance(stroke_token, StrokeToken)
            assert isinstance(relation_token, RelationToken)
        self.stroke_tokens = stroke_tokens
        self.relation_tokens = relation_tokens

        for stroke_token in stroke_tokens:
            assert isinstance(stroke_token, StrokeToken)
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
        for stroke_token, relation_token in zip(self.stroke_tokens,
                                                self.relation_tokens):
            parameters.extend(stroke_token.parameters())
            parameters.extend(relation_token.parameters())
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
        for stroke_token, relation_token in zip(self.stroke_tokens,
                                                self.relation_tokens):
            lbs.extend(stroke_token.lbs(eps))
            lbs.extend(relation_token.lbs(eps))
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
        for stroke_token, relation_token in zip(self.stroke_tokens,
                                                self.relation_tokens):
            ubs.extend(stroke_token.ubs(eps))
            ubs.extend(relation_token.ubs(eps))
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
