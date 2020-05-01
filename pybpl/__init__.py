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

# path to BPL matlab repository (https://github.com/brendenlake/BPL)
# must be set by user
if 'BPL_PATH' in os.environ:
    BPL_PATH = os.environ['BPL_PATH']
elif os.path.exists('/Users/rfeinman'):
    BPL_PATH = '/Users/rfeinman/src/BayesianProgramLearning/BPL'
elif os.path.exists('/home/feinman'):
    BPL_PATH = '/home/feinman/src/BPL'
else:
    raise Exception('BPL_PATH environment variable must be set by user')