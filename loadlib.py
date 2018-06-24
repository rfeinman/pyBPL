"""
A function that loads the library.
"""
from __future__ import print_function, division

from pybpl.classes import Library


def loadlib(lib_dir='./library'):
    lib = Library(lib_dir)
    assert isinstance(lib, Library)

    return lib