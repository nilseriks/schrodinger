.. Schrodinger documentation master file, created by
   sphinx-quickstart on Mon Sep 21 15:49:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***********************
Welcome to Schrodinger!
***********************

Goal of Schrodinger
===================

Schrodinger is a students project and the first time for said students to
create a somewhat bigger program. Schrodinger is capable of solving the one
dimensional time independet schrodinger equation for any potentails. The user
has to create one input file containing the needed data, which will be read and
organized in different files. The solution of the problem desbribed in the
input file will be displayed graphically aswell.

However, since this is a first time students program, the focus of this project
was primarily learning to code and especially parallel development aswell as a
comprehensible way of documentation of this progress. Consequently, the code is
rather simple and possibly not optimal for solving complicated problems.


Version and packages
====================

Schrodinger requires Python 3.6 or higher aswell as the packages numpy, scipy,
os and matplotlib.


Input file
==========

The data describing the problem the user wants schrodinger to solve has to be
written into an input file named 'schrodinger.inp'. Schrodinger will search
the main directory for that file. It is also possible to pass a different
directory to the program using a command line argument. The file has to follow
a special formatting style. Here is an example on how the content of the input
file can look:

.. code-block:: shell

   2.0			# mass
   -2.0 2.0 1999	# xMin xMax nPoint
   1 5			# first and last eigenvalue to print
   linear 		# interpolation type
   2			# nr. of interpolation points
   -2.0 0.0
    2.0 0.0


Notes
=====
Here is some aditional information on the calculations:

* The wavefunctions run towards 0 at the chosen borders

* All spline calculations are calculated as natural splines


API documentation
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
