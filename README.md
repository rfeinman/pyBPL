# pyBPL: Python-based Bayesian Program Learning

pyBPL is a package of tools to implement Bayesian Program Learning (BPL) in Python 3
using PyTorch backend. The original BPL implementation was written in MATLAB (see Lake et al. 2015 "Human-level concept learning through
probabilistic program induction" and the [original BPL repository](https://github.com/brendenlake/BPL)). I'm a Ph.D. student with Brenden Lake and I've developed this modern implementation to use for our ongoing research projects.

The key contributions of this repository are:
1. A fully-differentiable implementation of the original BPL character learning tools including symbolic rendering, spline fitting/evaluation, and model scoring (log-likelihoods).
2. A generalized framework for BPL that can work with various types of concepts. Character learning is one manifestation of the BPL framework, and it is included here as the preliminary use case.

I am thankful to Maxwell Nye, Mark Goldstein and Tuan-Anh Le for their help developing this library.

## Setup

This code repository requires PyTorch >= 1.0.0. A full list of requirements can
be found in `requirements.txt`. To install, first run the following command to clone
the repository into a folder of your choice:
```
git clone https://github.com/rfeinman/pyBPL.git
```
Then, run the following command to install the package:
```
python setup.py install
```



## Documentation
In order to generate the documentation site for the pyBPL library, execute the
following commands from the root folder:
```
cd docs/
make html
```
HELP WANTED: documentation build is broken right now, needs to be fixed.

## Usage Example

The following code loads the BPL model with pre-defined hyperparameters 
and samples a token

```python
from pybpl.library import Library
from pybpl.model import CharacterModel

# load the hyperparameters of the BPL graphical model (i.e. the "library")
lib = Library(lib_dir='/path/to/lib_dir', use_hist=True)

# create the BPL graphical model
model = CharacterModel(lib)

# sample a character type from the prior P(Type) and score its log-probability
char_type = model.sample_type()
ll_type = model.score_type(char_type)

# sample a character token from P(Token | Type=type) and score its log-probability
char_token = model.sample_token(char_type)
ll_token_given_type = model.score_token(char_type, char_token)

# sample an image from P(Image | Token=token)
image = model.sample_image(char_token)
ll_image_given_token = model.score_image(char_token, image)

```

## Demo
Currently there are 2 working demos, both found in the `examples` subfolder.

#### 1. generate character

You can generate a character type and sample a few tokens of the type by
running the following command from the root folder:
```
python examples/generate_character.py
```
The script will sample a character type from the prior and then sample 4 tokens
of the type, displaying the images.

#### 2. optimize character type
You can generate a character type and then optimize its parameters to maximize
the likelihood of the type under the prior by running the following
command from the root folder:
```
python examples/optimize_type.py
```
Optionally, you may add the integer parameter `--ns=<int>` to specify how many
strokes you would like the generated character type to have.


## Status Notes

#### General

All functions required to sample character types, tokens and images are now
complete. 
Currently, independent relations sample their position from a uniform distribution over the entire image window by default. 
To use the original spatial histogram from BPL, make sure to load the Library object with `use_hist=True`. 
Note, however, that log-likelihoods for spatial histograms are not differentiable.

My Python implementations of the bottum-up image parsing algorithms are not yet complete (HELP WANTED! see `pybpl/bottomup` for current status).
However, I have provided some wrapper functions in that call the original matlab code using the [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html). 
You must have the MATLAB bindings installed to use this code. With the bindings installed, you can run the parser as follows:

```python
import numpy as np
from pybpl.bottomup import generate_random_parses

# load your binary image
I = np.zeros((105,105), dtype=np.bool) 

# call bottum-up parser (optional random seed for reproducibility)
parses = generate_random_parses(I, seed=3)
```


#### Library

The library contains all of the parameters of the character learning BPL
model. These parameters have been learned from the Omniglot dataset. 
The library data is stored as a 
series of `.mat` files in the subfolder `lib_data/`. 
I've included a Matlab script, `process_library.m`, which can be
run inside the original BPL repository to 
obtain this folder of files. For an example of how to load the library, see
`examples/generate_character.py`.


## Citing
If you use pyBPL for your research, you are encouraged (though not required) to use the following BibTeX reference:

```
@misc{feinman2020pybpl,
    title={{pyBPL}},
    author={Feinman, Reuben},
    year={2020},
    version={0.1},
    url={https://github.com/rfeinman/pyBPL}
}
```
