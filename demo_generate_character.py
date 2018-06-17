from __future__ import division, print_function

from pybpl.classes import Library
from pybpl.generate_character import generate_character


def main():
    lib = Library(lib_dir='./library')
    x, y = generate_character(lib)
    print('generating exemplar:')
    character = y()


if __name__ == '__main__':
    main()