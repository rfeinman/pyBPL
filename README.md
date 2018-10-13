# pyBPL: Python-based Bayesian Program Learning

pyBPL is a Python 3 package to implement Bayesian Program Learning (BPL)
using PyTorch backend. The BPL framework has been generalized to work with
various types of concepts. Character learning with Omniglot is one
manifestation of the BPL framework, and it is included here as the preliminary
use case (see Lake et al. 2015 "Human-level concept learning through
probabilistic program induction").

Not yet fully operational.



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

NOTE: unit tests require an additional environment variable, `PYBPL_DIR` to
be set:
```
export PYBPL_DIR="/path/to/pyBPL"
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

#### Library

The library data is stored as a series of `.mat` files in the subfolder
`lib_data/`. I've included a Matlab script, `process_library.m`, which can be
run inside the original BPL repository to obtain this folder of files.
Library loading is fully functional... see loadlib.py for an example of how to
load the library.

#### General

Most things required to sample character types and tokens (images) are now
complete. I have not tested backprop through the level of character tokens,
although it works well through types for now.
