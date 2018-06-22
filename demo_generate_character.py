from __future__ import division, print_function

from pybpl.classes import Library
from pybpl.forward_model import generate_character


def main():
    print('generating character type...')
    lib = Library(lib_dir='./library')
    x = generate_character(lib)

if __name__ == '__main__':
    main()