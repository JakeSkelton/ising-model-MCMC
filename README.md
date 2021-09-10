# ising-model-MCMC
Python (and Cython) code to simulate a 2D Ising model using Markov chain Monte Carlo (MCMC) methods, namely the metropolis algorithm.

## Structure

The repository is made up of four programs. *markov_chain_monte_carlo.pyx* is a Cython file (see below) containing cour numerical routines - the Metropolis-Hastings algorithm; *comp_project_0_js2443.py* defines the important *lattice* class and is in some sense the mother program; *comp_project_1_js2443.py*'s role is to produce checkerboad animations of the time evolution of an Ising lattice; *comp_project_2_js2443.py* produces graphs of the time evolution of important lattice-wide quantities, like magnetiation; finally, *comp_project_3_js2443.py* can be used to investigate how a time-variable temperature affects the lattice magnetisation and total energy.

## Cython compilation

To compile 'markov_chain_monte_carlo.pyx' into a .c and a .pyd (or .so on Linux) run

>python setup_mcmc.py build_ext --inplace

on the command line in the directory containing both 
'markov_chain_monte_carlo.pyx' and 'setup_mcmc.py', as described in the Cython documentation
https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html.
Once the .pyd is created, it is imported by Python programs just like a module of pure Python.

If all else fails and the .pyx cannot be compiled, I suppose one could remove all Cython-specific syntax in 'markov_chain_monte_carlo.pyx' and save it as a .py, but it will run about 100x slower.
