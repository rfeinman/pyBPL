# pyBPL
Pyro/pytorch implementation of BPL for omniglot. 
Not yet operational. In current setup, can be run by typing into terminal:
```
python generate_character.py
```
while in the main folder. 


Current status: most things which exist are implemented. CPD.py should work, but has some shortcuts taken, meaning it is not entirely faithful to the original model. rendering.py is incomplete. The conversion from control points to a motor path is implemented, but the differential rendering, which takes a motor path and outputs an image, is unimplemented.

## Reuben's notes

The library data is now stored as a series of `.mat` files in the subfolder
`library/`. Currently the only piece of data missing is the `Spatial` entry of
the library. This will be dealt with soon. I've included a Matlab script,
`process_library.m`, which can be run with the original BPL repository to obtain
these data files. Library loading is now functional.