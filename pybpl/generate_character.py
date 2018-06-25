from __future__ import division, print_function
import torch
from . import CPD
from .character.character import Character


# list of acceptable dtypes for 'ns' parameter
int_types = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]

def generate_type(lib, ns=None):
    """
    Generate a character type by sampling from the prior.

    :param lib: [Library] library class instance
    :param ns: [int or tensor] number of strokes for the character
    :return:
        ctype: [CharacterType] character type
    """

    if ns is None:
        # sample the number of strokes 'ns'
        ns = CPD.sample_number(lib)
    else:
        # make sure 'ns' is correct dtype
        isint = isinstance(ns, int)
        istensorint = isinstance(ns, torch.Tensor) and \
                      ns.shape == torch.Size([]) and \
                      ns.dtype in int_types
        assert isint or istensorint

    # initialize stroke and relation lists
    S = []
    R = []
    # for each stroke, sample stroke parameters
    for _ in range(ns):
        # sample the stroke type
        stroke = CPD.sample_stroke_type(lib, ns)
        # sample the relation of this stroke to previous strokes
        relation = CPD.sample_relation_type(lib, S)
        # append stroke type and relation to their respective lists
        S.append(stroke)
        R.append(relation)

    # return the character type (a stencil for a character)
    return S, R

def generate_token(lib, ns=None):
    """
    Generate a character token by sampling from the prior. First, sample a
    character type from the prior. Then, sample a token of that type from
    the prior.

    :param lib:
    :param ns:
    :return:
    """
    S, R = generate_type(lib, ns)
    char = Character(S, R, lib)
    exemplar = char.sample_token()

    return exemplar

def generate_program(lib, ns=None):
    """
    Generate a character by sampling from the prior. A character is a motor
    program that can sample new character tokens.

    :param lib:
    :param ns:
    :return:
    """
    S, R = generate_type(lib, ns)
    char = Character(S, R, lib)

    return char
