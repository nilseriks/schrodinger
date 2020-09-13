#!/usr/bin/env python3
"""Module which contains all functions for numerical calculations."""

import numpy as np
import scipy as sp
import scipy.interpolate
import calculus


def pot_calc(xplot, pot, interpoltype):
    """Interpolates the Potential for given data points.

    Args:
        xplot: Array of the x values.
        pot: Array with the data points of the potential.
        interpoltype: Type of the interpolation.

    Returns:
        VV: Array with values of the potential at the points of xplot.
    """
    xx = pot[:, 0]
    yy = pot[:, 1]
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
    WF = np.array([xplot])
    for ii in range(minEV - 1, maxEV):
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

    pot1 = np.zeros((1999, ), dtype=float)
    #for nn in range()
