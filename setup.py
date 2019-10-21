# This script is mostly copied from Atilim Gunes Baydin

import os
import sys
import setuptools

PACKAGE_NAME = 'pybpl'
MINIMUM_PYTHON_VERSION = 3, 5


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert 0, "'{0}' not found in '{1}'".format(key, module_path)


check_python_version()
setuptools.setup(
    name=PACKAGE_NAME,
    version=read_package_variable('__version__'),
    description='',
    packages=['pybpl'],
    install_requires=['torch', 'numpy', 'scipy', 'matplotlib'])
