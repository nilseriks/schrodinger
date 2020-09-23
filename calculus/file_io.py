'''File_IO contains various filereading and writing routines. The goal is to
read a special formated file containing the user input and creating different
files containing the calculated eigenstates, potential, etc.'''

import os.path
import os
import numpy as np


def _getvalue(string_with_data):
    """Searches a string for numbers and strips other characters off of them.

    Args:
        string_with_data (str): String the numbers should be read out of

    Returns:
        int: Numbers that were included in the input string
    """
    aa = string_with_data
    hashindex = string_with_data.find('#')
    newstr = ''.join((ch if ch in '0123456789.-e' else ' ')
                     for ch in aa[0:hashindex])

    numbers = [float(i) for i in newstr.split()]

    return numbers


def read_schrodinger(directory, file):
    """Reads the file "schrodinger.inp" containing special formated user data
    describing the problem

    Args:
        filepath (str): Filepath of "schrodinger.inp"

    Returns:
        dict: Dictionary containing the needed data for further calculations
    """

    filepath = directory + '/' + file

    list_of_data = open(filepath, 'r').readlines()

    alldata = dict()

    massstring = list_of_data[0]
    alldata['_MASS'] = _getvalue(massstring)[0]

    interpolationstring = list_of_data[1]
    alldata['_XMIN'] = _getvalue(interpolationstring)[0]
    alldata['_XMAX'] = _getvalue(interpolationstring)[1]
    alldata['_NPOINT'] = int(_getvalue(interpolationstring)[2])

    ev_string = list_of_data[2]
    alldata['_MIN_EV'] = int(_getvalue(ev_string)[0])
    alldata['_MAX_EV'] = int(_getvalue(ev_string)[1])

    inttypestring = list_of_data[3]
    seperator = '\t' if '\t' in inttypestring else ' '
    alldata['_REG_TYPE'] = inttypestring.split(seperator)[0]
    alldata['_REG_TYPE'] = alldata['_REG_TYPE'].replace("\n", "")

    intpointsstring = list_of_data[4]
    alldata['_INTERPOLATE_NR'] = _getvalue(intpointsstring)[0]

    alldata['_POT'] = np.loadtxt(filepath, skiprows=5)

    return alldata


def read_data(directory, fname):
    """Read a file and extract the content as an array.

    Args:
        directory (str): Directory which contains the file.
        fname (str): Name of the file.

    Returns:
        ndarray: Array of content.
    """
    data = np.loadtxt(os.path.join(directory, fname))
    return data


def read_files(filepath):
    """Reads four different files and converts them into arrays

    Args:
        filepath (str): Filepath of , `energies.dat`, `expvalues.dat,
         `potential.dat` and `wavefuncs.dat`

    Returns:
        1darray: array containing energies of their corresponding eigenstates
        1darray: array containing expected values for x and their uncerainties
        2darray: array containing potentials and corresponding x-values
        ndarray: array containing eigenstates and corresponding x-values
    """
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
    """
    Creates files containing the energies (`energies.dat`), expected values for
    x and thier uncertainities (`expvalues.dat`), the potentials and their
    corresponding x-values (`potential.dat`) and the eigenstates with their
    corresponding x_values (`wavefuncs.dat`)

    Args:
        filepath (str): Filepath of the destination, in which the files should
          be saved
        endata (1darray): array containing data dedicated to `energies.dat`
        expxdata (1darray): array containing data dedicated to `expvalues.dat`
        potdata (2darray): array containing data dedicated to `potential.dat`
        wfdata (ndarray): array containing data dedicated to `wavefuncs.dat`
    """
    enfile = os.path.join(filepath, "energies.dat")
    expxfile = os.path.join(filepath, "expvalues.dat")
    potfile = os.path.join(filepath, "potential.dat")
    wffile = os.path.join(filepath, "wavefuncs.dat")

    np.savetxt(enfile, endata)
    np.savetxt(expxfile, expxdata)
    np.savetxt(potfile, potdata)
    np.savetxt(wffile, wfdata)
