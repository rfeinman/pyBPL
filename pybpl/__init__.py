"""
pyBPL - Python-based Bayesian Program Learning.

Python implementation of Bayesian Program Learning (BPL), using PyTorch.
The BPL framework has been generalized to work with various types of concepts.
Character learning with Omniglot is one manifestation of the BPL framework,
and it is included here as the preliminary use case (see Lake et al. 2015
"Human-level concept learning through probabilistic program induction").
"""
__version__ = "0.1"
import os

LIB_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'lib_data'
)