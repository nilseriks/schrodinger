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


def read_files(filepath):
    '''reads four different files and converts them into arrays

    Args:
        filepath: Filepath of , `energies.dat`, `expvalues.dat, `potential.dat`
        and `wavefuncs.dat`

    Returns:
        endata: array containing energies of their corresponding eigenstates
        expxdata: array containing expected values for x and their
        uncerainnities
        potdata: array containing potentials and corresponding x-values
        wfdata: array containing eigenstates and corresponding x-values
    '''
    enfile = os.path.join(filepath, "energies.dat")
    expxfile = os.path.join(filepath, "expvalues.dat")
    potfile = os.path.join(filepath, "potential.dat")
    wffile = os.path.join(filepath, "wavefuncs.dat")

    endata = np.loadtxt(enfile)
    expxdata = np.loadtxt(expxfile)
    potdata = np.loadtxt(potfile)
    wfdata = np.loadtxt(wffile)

    return endata, expxdata, potdata, wfdata


def create_files(filepath, endata, expxdata, potdata, wfdata):
    '''
    Creates files containing the energies (`energies.dat`), expected values for
    x and thier uncertainities (`expvalues.dat`), the potentials and their
    corresponding x-values (`potential.dat`) and the eigenstates with their
    corresponding x_values (`wavefuncs.dat`)

    Args:
        filepath: Filepath of the destination, in which the files should be
        saved
        endata: array containing data dedicated to `energies.dat`
        expxdata: array containing data dedicated to `expvalues.dat`
        potdata: array containing data dedicated to `potential.dat`
        wfdata: array containing data dedicated to `wavefuncs.dat`
    '''
    enfile = os.path.join(filepath, "energies.dat")
    expxfile = os.path.join(filepath, "expvalues.dat")
    potfile = os.path.join(filepath, "potential.dat")
    wffile = os.path.join(filepath, "wavefuncs.dat")

    np.savetxt(enfile, endata)
    np.savetxt(expxfile, expxdata)
    np.savetxt(potfile, potdata)
    np.savetxt(wffile, wfdata)
