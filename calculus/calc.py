#!/usr/bin/env python3

import numpy as np
import scipy as sp


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
        VV = sp.interpolate.interp1d(xx, yy, 'linear')
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
