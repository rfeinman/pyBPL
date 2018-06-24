from __future__ import division, print_function
import warnings

from pybpl.classes import Library
from pybpl.forward_model import generate_type


def main():
    print('generating character...')
    warnings.warn(
        "'generate_character' not yet fully implemented. Generating character "
        "type (template) for now."
    )
    lib = Library(lib_dir='./library')
    ctype = generate_type(lib)
    print('Character type: ', ctype)

if __name__ == '__main__':
    main()