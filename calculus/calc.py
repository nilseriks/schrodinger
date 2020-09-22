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


def expected_x_square(xplot, EVEC, minEV, maxEV):
    """Calculates the expected values of the square of the position.

    Args:
        xplot: x values.
        EVEC (ndarray): Array of the wavefunctions.
        minEV (int): Lower bound of eigenvalues to calculate the expected
            square positions of.
        maxEV (int): Upper bound of eigenvalues to calculate the expected
            square postions of.

    Returns:
        expx2 (1darray): Array containing the expected values of the square
            position from the minEV eigenvalue to the maxEV eigenvalue.
    """
    delta = abs(xplot[0] - xplot[1])
    expx2 = np.zeros((maxEV - minEV + 1, ), dtype=float)
    for ii in range(minEV - 1, maxEV):
        xx = delta * np.sum(EVEC[:, ii] * xplot**2 * EVEC[:, ii])
        expx2[ii] = xx
    return expx2


def uncertainty(xplot, EVEC, minEV, maxEV):
    """Calculates the uncertainty of the expected position.

    Args:
        xplot: x values.
        EVEC (ndarray): Array of the wavefunctions.
        minEV (int): Lower bound of eigenvalues to calculate the uncertainty
            of.
        maxEV (int): Upper bound of eigenvalues to calculate the uncertainty
            of.

    Retruns:
        uncertainty (1darray): Array containing the uncertainties of the
            the expected positions."""
    expx = expected_values(xplot, EVEC, minEV, maxEV)
    expx2 = expected_x_square(xplot, EVEC, minEV, maxEV)
    uncertainty = np.sqrt(expx2 - expx * expx)
    return uncertainty


def get_exp_unc(expx, unc):
    """Combines the arrays of the expected values and the array of the
    corresponding uncertainties.

    Args:
        expx (1darray): Array containing the expected values.
        unc (1darray): Array containing the uncertainties corresponding to the
            expected values.

    Returns:
        expvalues (2darray): Expected values as colum vector and uncertainties
            as second colum vector."""
    expvalues = np.vstack((expx, unc))
    expvalues = np.transpose(expvalues)
    return expvalues
