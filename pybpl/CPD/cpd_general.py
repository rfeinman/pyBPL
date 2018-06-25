"""
Defines the conditional probability distributions that make up the
BPL model
"""
from __future__ import division, print_function
import numpy as np
import torch
import torch.distributions as dist

from ..character.stroke import Stroke
from .cpd_substrokes import sample_nsub, sample_sequence
from .cpd_shape import sample_shape_type
from .cpd_scale import sample_invscale_type

# ----
# Stroke
# ----

def sample_stroke_type(lib, ns):
    # sample the number of sub-strokes
    nsub = sample_nsub(lib, ns)
    # sample the sub-stroke sequence
    ss_seq = sample_sequence(lib, nsub)
    # sample control points for each sub-stroke in the sequence
    cpts = sample_shape_type(lib, ss_seq)
    # sample scales for each sub-stroke in the sequence
    scales = sample_invscale_type(lib, ss_seq)
    # initialize the stroke type
    stroke = Stroke(
        ss_seq, cpts, scales,
        sigma_shape=lib.tokenvar['sigma_shape'],
        sigma_invscale=lib.tokenvar['sigma_invscale']
    )

    return stroke