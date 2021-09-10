# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:00:39 2021

setup_mcmc.py

A build program for the Cython compiler to build 
'markov_chain_monte_carlo.pyx'.
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize('markov_chain_monte_carlo.pyx', annotate=True,
    compiler_directives={'cdivision': True}), zip_safe=False,
    include_dirs=[numpy.get_include()])

## 'annotate=True' generates a HTML highlighting code that draws heavily on
## Python, vs that which is entirely C.

## 'cdivision' flag tells compiler to use native C modulo division, which
## gives a moderate speed up but doesn't affect our usage (differs from 
## Python % only when negative ints are involved).