# pyBPL
Python implementation of BPL for omniglot, using PyTorch. Not yet operational.

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

## Demo
Currently there are two working demos.

### Generate character type
You can generate a character type (i.e., a motor program) by
running the following command from the root folder:
```
python demo_generate_character.py
```

### Optimize character type
You can generate a character type and then optimize its parameters to maximize
the likelihood of the character under the prior by running the following
command from the root folder:
```
python demo_optimize_type.py
```

## Status notes

### General

Most things which exist are implemented. CPD.py should work,
but has some shortcuts taken, meaning it is not entirely faithful to the
original model. rendering.py is incomplete. The conversion from control points
to a motor path is implemented, but the differential rendering, which takes a
motor path and outputs an image, is unimplemented.

### Library

The library data is stored as a series of `.mat` files in the subfolder
`library/`. I've included a Matlab script, `process_library.m`, which can be
run inside the original BPL repository to obtain this folder of files.
Library loading is fully functional... see loadlib.py for an example of how to
load the library.