#!/usr/bin/env python3
"""Module which contains all functions for numerical calculations."""

import numpy as np
import scipy as sp
import scipy.interpolate
import calculus


def pot_calc(xplot, discrete_pot, interpoltype):
    """Interpolates the Potential for given data points.

    Args:
        xplot: Array of the x values.
        pot: Array with the data points of the potential.
        interpoltype: Type of the interpolation.

    Returns:
        VV: Array with values of the potential at the points of xplot.
    """
    xx = discrete_pot[:, 0]
    yy = discrete_pot[:, 1]
    if interpoltype == 'linear':
        VV = scipy.interpolate.interp1d(xx, yy, 'linear')
        return VV(xplot)
    elif interpoltype == 'polynomial':
        VV = sp.interpolate.barycentric_interpolate(xx, yy, xplot)
        return VV
    elif interpoltype == 'cspline':
        VV = sp.interpolate.CubicSpline(xx, yy, bc_type='natural')
        return VV(xplot)


def _solve_seq(xmin, xmax, npoint, mass, pot):
    """Solve the discrete time independent schrodinger equation and return the
    eigenvalues and eigenvectors.
    Note: For the discret solution it assume that the eigenvectors are zero at
    the bounds.

    Args:
        xmin: Minimum of x values of the potential.
        xmax: Maximum of x values of the potential.
        npoint: Number of discret points of x.
        pot: Discret potential at the x values.

    Returns:
        EVAL: Array containing the eigenvalues.
        EVEC: Array containing the eigenvectors as column vectors.
    """
    _DELTA = abs(xmin - xmax) / npoint
    _CONST = 1 / (mass * _DELTA**2)
    # Calculating the off diagonal values.
    OD = - 1 / 2 * _CONST * np.ones((npoint - 1,), dtype=float)
    # Calculating the main diagonal values.
    MD = pot + _CONST
    EVAL, EVEC = sp.linalg.eigh_tridiagonal(MD, OD)
    return EVAL, EVEC


def get_WF_array(xplot, minEV, maxEV, evec):
    """Calculating the Array of the wavefunctions in the
        x1 Psi1(x1) Psi2(x1)
        x2 Psi1(x2) Psi2(x2)
    format.

    Args:
        xplot: Array of the x values.
        minEV: Lower bound of eigenvalues.
        maxEV: Upper bound of eigenvalues.
        evec: Array of the eigenvectors.

    Returns:
        WF: Array with the described format.
    """
    delta = abs(xplot[0] - xplot[1])
    WF = np.array([xplot])
    for ii in range(minEV - 1, maxEV):
        norm = np.sqrt(delta * np.sum(evec[:, ii] * evec[:, ii]))
        evec[:, ii] /= norm
        WF = np.vstack((WF, evec[:, ii]))
    WF = np.transpose(WF)
    return WF


def _energy_inf_square_well(maxEV):
    """Calculates the energies of the infinite square well for the first maxEV
    eigenvalues.

    Args:
        EVmax: Set the upper bound for the eigenvalues to calculate.

    Returns:
        energy: Array containing the calculated eigenvalues."""
    energy = np.zeros((maxEV, ), dtype=float)
    for nn in range(0, maxEV):
        energy[nn] = np.pi**2 * (nn + 1)**2 / (2 * 2 * (-2 - 2)**2)
    calculus.file_io.write_result('tests/test_energy',
                                  'E_inf_square_well.dat', energy)
    return energy


def _energy_harm_osc(maxEV):
    """Calculates the energies of the harmonic oscillator for given number of
    eigenvalues.

    Args:
        EVmin: Set the lower bound for the eigenvalues to calculate.
        EVmax: Set the upper bound for the eigenvalues to calculate.

    Returns:
        energy: Array containing the calculated eigenvalues."""
    energy = np.zeros((maxEV, ), dtype=float)
    for nn in range(maxEV):
        energy[nn] = 0.5 * nn + 0.25
    calculus.file_io.write_result('tests/test_energy', 'E_harm_osc.dat',
                                  energy)
    return energy


def _pot_inf_square_well():
    """Calculates the potential of the ininite square well problem."""

    pot = np.zeros((1999, ), dtype=float)
    calculus.file_io.write_result('tests/test_potential',
                                  'pot_inf_square_well.dat', pot)
    return pot


def _pot_fin_square_well():
    """Calculates the potential of the finite square well problem."""

    pot1 = np.zeros((750, ), dtype=float)
    pot2 = -10 * np.ones((499, ), dtype=float)
    pot3 = np.zeros((750, ), dtype=float)
    pot = np.concatenate((pot1, pot2, pot3), axis=0)
    calculus.file_io.write_result('./../tests/test_potential',
                                  'pot_fin_square_well.dat', pot)
    return pot


def _pot_harm_osc():
    """Calculates the potential of the harmonic oscillator."""

    _XPLOT = _XPLOT = np.linspace(-5, 5, num=1999, endpoint=True)
    pot = 0.5 * _XPLOT**2
    calculus.file_io.write_result('tests/test_potential',
                                  'pot_harm_osc.dat', pot)
    return pot


def expected_values(xplot, EVEC, minEV, maxEV):
    """Calculates the expected value of the position.

    Args:
        xplot: x values.
        EVEC: Array of the eigenvectors to calculate the expected position of.
        minEV: Lower bound of the eigenvalues.
        maxEV: Upper bound of the eigenvalues.

    Returns:
        expectedx: Array containing the expected values of the position."""
    delta = abs(xplot[0] - xplot[1])
    expectedx = np.zeros((maxEV - minEV + 1, ), dtype=float)
    for ii in range(minEV - 1, maxEV):
        xx = delta * np.sum(EVEC[:, ii] * xplot * EVEC[:, ii])
        expectedx[ii] = xx
    return expectedx


def expected_value(xvalues, wavefcts, exp=1):
    '''
    Calculates the expected value of x

    Args:
        xvalues: array containing the x-values
        wavefcts: array containing the eigenstates
    Returns:
        expectedx: arry containing expected values of x
    '''
    delta = np.abs(xvalues[0] - xvalues[-1]) / len(xvalues)
    expectedx = np.array([])
    for wf, xx in zip(wavefcts, xvalues):
        expectedx = np.append(expectedx, [np.sum((wf ** 2) * (xx ** exp))
                                            * delta], axis=0)

    return expectedx


def uncertainties(xvalues, wavefcts):
    '''
    Calculates the uncertainties of the x-values

    Args:
        xvalues: array containing the x-values
        wavefcts: array containing the eigenstates
    Returns:
        uncert: Uncertainties of x
    '''
    expx = expected_value(xvalues, wavefcts, 1)
    expxsquared = expected_value(xvalue, wavefcts, 2)

    uncert = expx - expxsquared

    return uncert