from __future__ import division, print_function

import pybpl.loadlib as lb
from pybpl.generate_character import generate_character

def main():
    lib = lb.loadlib()
    x,y = generate_character(lib)
    print('generating exemplar:')
    character = y()


if __name__ == '__main__':
    main()