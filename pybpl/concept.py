from __future__ import division, print_function
from abc import ABCMeta, abstractmethod


class ConceptType(object):
    '''
    An abstract base class for concept types. Concept types are made up of
    parts and relations.

    Parameters
    ----------
    k : tensor
        scalar; part count
    P : list of Part
        part type list
    R : list of Relation
        relation type list
    '''
    def __init__(self, k, P, R):
        assert isinstance(P, list)
        assert isinstance(R, list)
        assert len(P) == len(R)
        assert k > 0
        for ptype, rtype in zip(P, R):
            assert isinstance(ptype, Part)
            assert isinstance(rtype, Relation)
        self.k = k
        self.part_types = P
        self.relation_types = R

    def parameters(self):
        '''
        return list of parameters
        '''
        pass

    def train(self):
        '''
        makes params require grad
        '''
        pass

    def eval(self):
        '''
        makes params require no grad
        '''
        pass

    def to(self, device):
        '''
        moves parameters to device
        '''
        pass

class CharacterType(ConceptType):
    '''
    Character types are made up of strokes (parts) and relations. Relations are
    each either [independent, attach, attach-along].

    Parameters
    ----------
    k : tensor
        scalar; part count
    P : list of Stroke
        part type list
    R : list of Relation
        relation type list
    '''

    def __init__(self, k, P, R):
        super(CharacterType, self).__init__(k, P, R)
        for p in P:
            assert isinstance(p, Stroke)

    def parameters(self):
        '''
        Returns a list of parameters that can be optimized via gradient descent.

        Returns
        -------
        params : list
            optimizable parameters
        '''
        parameters = []
        for p, r in zip(self.part_types, self.relation_types):
            parameters.extend(p.parameters())
            parameters.extend(r.parameters())

        return parameters

    def lbs(self, eps=1e-4):
        '''
        Returns a list of lower bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        lbs : list
            lower bound for each parameter
        '''
        lbs = []
        for p, r in zip(self.part_types, self.relation_types):
            lbs.extend(p.lbs(eps))
            lbs.extend(r.lbs(eps))

        return lbs

    def ubs(self, eps=1e-4):
        '''
        Returns a list of upper bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        ubs : list
            upper bound for each parameter
        '''
        ubs = []
        for p, r in zip(self.part_types, self.relation_types):
            ubs.extend(p.ubs(eps))
            ubs.extend(r.ubs(eps))

        return ubs


class ConceptToken(object):
    '''
    Abstract base class for concept tokens. Concept tokens consist of a list
    of PartTokens and a list of RelationTokens.

    Parameters
    ----------
    P : list of PartToken
        part tokens
    R : list of RelationToken
        relation tokens
    '''
    __metaclass__ = ABCMeta

    def __init__(self, P, R):
        assert isinstance(P, list)
        assert isinstance(R, list)
        assert len(P) == len(R)
        for ptoken, rtoken in zip(P, R):
            assert isinstance(ptoken, PartToken)
            assert isinstance(rtoken, RelationToken)
        self.part_tokens = P
        self.relation_tokens = R

    def parameters(self):
        '''
        return list of parameters
        '''
        pass

    def train(self):
        '''
        makes params require grad
        '''
        pass

    def eval(self):
        '''
        makes params require no grad
        '''
        pass

    def to(self, device):
        '''
        moves parameters to device
        '''
        pass


class CharacterToken(ConceptToken):
    """
    Character tokens hold all token-level parameters of the character. They
    consist of a list of PartTokens and a list of RelationTokens.

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
        super(CharacterToken, self).__init__(P, R)
        for ptoken in P:
            assert isinstance(ptoken, StrokeToken)
        self.affine = affine
        self.epsilon = epsilon
        self.blur_sigma = blur_sigma

    def parameters(self):
        '''
        Returns a list of parameters that can be optimized via gradient descent.

        Returns
        -------
        params : list
            optimizable parameters
        '''
        parameters = []
        for p, r in zip(self.part_tokens, self.relation_tokens):
            parameters.extend(p.parameters())
            parameters.extend(r.parameters())

        return parameters

    def lbs(self, eps=1e-4):
        '''
        Returns a list of lower bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        lbs : list
            lower bound for each parameter

        '''
        lbs = []
        for p, r in zip(self.part_tokens, self.relation_tokens):
            lbs.extend(p.lbs(eps))
            lbs.extend(r.lbs(eps))

        return lbs

    def ubs(self, eps=1e-4):
        '''
        Returns a list of upper bounds for each of the optimizable parameters.

        Parameters
        ----------
        eps : float
            tolerance for constrained optimization

        Returns
        -------
        ubs : list
            upper bound for each parameter
        '''
        ubs = []
        for p, r in zip(self.part_tokens, self.relation_tokens):
            ubs.extend(p.ubs(eps))
            ubs.extend(r.ubs(eps))

        return ubs