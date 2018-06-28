# Climate modeling

This Python package investigates the role that cloud seeding may play in Earth's
climate trends, and how such a model compares to traditional models that
consider primarily the greenhouse gas (GHG) effect.

   - For: Data Analysis (PHYS 201), Spring 2018, Harvard University.
   - Authors: Brendon Bullard, Yu Li, and Michael Rowan
   - License: GNU General Public License v3


Prerequisites
-------------

To use this package, a Fortran compiler is required.  Depending on your OS, you
can install with one of the following commands:

   - [Linux] sudo apt-get install gfortran

or for Mac, if `Homebrew` is installed,

   - [Mac OS X] brew install


Installation
------------
From the `/climate` directory, enter these commands:

     python setup.py build
     python setup.py install


To check that the installation was successful, navigate to `/climate/climate/tests`
and (if `nose` is installed) run `nosetests`.


Acknowledgements
----------------
This package uses the [`pySCM`](https://pythonhosted.org/pySCM/#) (Simple Climate Model)
package.  The `pySCM` package is authored by Jared Lewis (License: BSD).
