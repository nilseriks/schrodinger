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
    DD = - 1 / 2 * _CONST * np.ones((npoint - 1,), dtype=float)
    VV = pot + _CONST
    EVAL, EVEC = sp.linalg.eigh_tridiagonal(VV, DD)
    return EVAL, EVEC
