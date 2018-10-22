# pyBPL: Python-based Bayesian Program Learning

pyBPL is a Python 3 package to implement Bayesian Program Learning (BPL)
using PyTorch backend. The BPL framework has been generalized to work with
various types of concepts. Character learning with Omniglot is one
manifestation of the BPL framework, and it is included here as the preliminary
use case (see Lake et al. 2015 "Human-level concept learning through
probabilistic program induction" and the [original BPL repository](https://github.com/brendenlake/BPL)).

*NOTE:* This is a work in progress. Not yet fully operational.



## Setup

This code repository requires PyTorch >= 0.4.0. A full list of requirements can
be found in `requirements.txt`. To install, run the following command to clone
the repository into a folder of your choice:
```
git clone https://github.com/mtensor/pyBPL.git
```
On UNIX machines, after cloning this repository, it is recommended that you
add the path to the repository to your `PYTHONPATH` environment variable to
enable imports from any folder:
```
export PYTHONPATH="/path/to/pyPBL:$PYTHONPATH"
```



## Documentation
In order to generate the documentation site for the pyBPL library, execute the
following demands from the root folder:
```
cd docs/
make html
```



## Demo
Currently there are 2 working demos.

#### Generate character
You can generate a character type and sample a few tokens of the type by
running the following command from the root folder:
```
python demo_generate_character.py
```
The script will sample a character type from the prior and then sample 4 tokens
of the type, displaying the images.

#### Optimize character type
You can generate a character type and then optimize its parameters to maximize
the likelihood of the type under the prior by running the following
command from the root folder:
```
python demo_optimize_type.py
```
Optionally, you may add the integer parameter `--ns=<int>` to specify how many
strokes you would like the generated character type to have.



## Status Notes

#### General

All functions required to sample character types, tokens and images are now
complete. The `inference` package is incomplete - this is where all functions
for infering motor programs will be contained. At the moment, I only have a
simple function for optimizing the continuous parameters of a character type
in order to maximize the log-likelihood under the prior.

#### Library

The library contains all of the parameters of the character learning BPL
model. These parameters have been learned from the Omniglot dataset. 
The library data is stored as a 
series of `.mat` files in the subfolder `lib_data/`. 
I've included a Matlab script, `process_library.m`, which can be
run inside the [original BPL repository](https://github.com/brendenlake/BPL) to 
obtain this folder of files. For an example of how to load the library, see
`demo_generate_character.py`.


## Unit Tests
Unit tests are found in the `pybpl.tests` module. They can be run using
`python -m`. For example, to run the test `test_general_util.py`, use the
following command:
```
python -m pybpl.tests.test_general_util
```