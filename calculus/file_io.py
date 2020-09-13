#!/usr/bin/env python3

import os.path
import numpy as np
import os


def read_schrodinger(directory, fname):
    """Function to read out the schrodinger.inp file and return the
    informations as a dictionary.

    Args:
        directory: Name of the directory containing 'fname' file.
        fname: Name of the input file.

    Returns:
        Directory with all informations: mass of the particle, settings of
        the potential, lower and upper bound of the eigenvalues, type of the
        interpolation, number of given points of the potential, points of the
        potential."""
    filename = os.path.join(directory, fname)
    with open(filename) as fp:
        lines = fp.readlines()
        ii = 0
        # Read out the lines of the file.
        for item in lines:
            newitem = item.replace("\n", "")
            lines[ii] = newitem
            ii += 1
        lines = lines[:5]
    _MASS = float(lines[0])
    plot_set = lines[1].split()
    evalues = lines[2].split()
    regression = lines[3]
    interpolate_nr = int(lines[4])
    # Read the points of the potential.
    pot = np.loadtxt(filename, skiprows=5)
    return {"_MASS": _MASS, "plot_set": plot_set, "evalues": evalues,
            "regression": regression, "interpolate_nr": interpolate_nr,
            "pot": pot}


def write_result(directory, fname, array):
    """Write an array in a file.

    Args:
        directory: Directory where to write the file.
        fname: Name of the file.
        array: Array to write into fname.
    """
    np.savetxt(os.path.join(directory, fname), array)


def read_data(directory, fname):
    """Read a file and extract the contant as an array.

    Args:
        directory: Directory which contains the file.
        fname. Name of the file.

    Returns:
        Array of content."""
    return np.loadtxt(os.path.join(directory, fname))
