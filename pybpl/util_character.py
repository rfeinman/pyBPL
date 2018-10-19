from __future__ import division, print_function
import torch

def clear_shape_type(M):
    raise NotImplementedError

def clear_invscales_type(M):
    raise NotImplementedError

def set_affine_to_image(M, I):
    raise NotImplementedError

def set_affine(M, target_com, target_range, missing):
    raise NotImplementedError

def set_affine_motor(M, target_com, target_range):
    raise NotImplementedError

def apply_each_substroke(nested, fnc):
    raise NotImplementedError
    nested = None

    return nested

def flip_stroke(S):
    raise NotImplementedError

def merge_strokes(S1, S2):
    raise NotImplementedError
    Z = None

    return Z

def split_stroke(Z, bid_start_of_second):
    raise NotImplementedError
    Z1 = None
    Z2 = None

    return Z1, Z2

def all_merge_moves(M):
    raise NotImplementedError
    moves = None
    reverse_moves = None

    return moves, reverse_moves

def all_split_moves(M):
    raise NotImplementedError
    moves = None
    reverse_moves = None

    return moves, reverse_moves

def valid_merge(M, sid):
    raise NotImplementedError
    val = None

    return val