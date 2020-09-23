"""Module containing all functions for numerical calculations."""

import numpy as np
import scipy as sp
import scipy.interpolate


def pot_calc(xplot, discrete_pot, interpoltype):
    """Interpolates the potential for given data points.

    Args:
        xplot (1darray): Array containing the x values.
        discrete_pot (1darry) : Array containing data points of the potential.
        interpoltype (str): Type of the interpolation.

    Returns:
        1darray: Array with values of the potential at the points of xplot.
    """
    xx = discrete_pot[:, 0]
    yy = discrete_pot[:, 1]
    if interpoltype == 'linear':
        vv = scipy.interpolate.interp1d(xx, yy, 'linear')
    elif interpoltype == 'polynomial':
        vv = sp.interpolate.barycentric_interpolate(xx, yy, xplot)
    elif interpoltype == 'cspline':
        vv = sp.interpolate.CubicSpline(xx, yy, bc_type='natural')
    if interpoltype in ['linear', 'cspline']:
        vv = vv(xplot)
    return vv


def solve_seq(xmin, xmax, npoint, mass, pot):
    """Solves the discrete time independent schrodinger equation and returns
    the eigenvalues and eigenvectors.
    Note: For the discret solution it assumes that the eigenvectors are zero at
    the bounds.

    Args:
        xmin (int): Minimum x value of the potential.
        xmax (int): Maximum x value of the potential.
        npoint (int): Number of discret points of x.
        pot (1darray): Discret potential at the x values.

    Returns:
        1darray: Array containing the eigenvalues.
        ndarray: Array containing the eigenvectors as column vectors.
    """
    delta = abs(xmin - xmax) / npoint
    const = 1 / (mass * delta**2)
    # Calculating the off diagonal values.
    OD = - 1 / 2 * const * np.ones((npoint - 1,), dtype=float)
    # Calculating the main diagonal values.
    MD = pot + const
    EVAL, EVEC = sp.linalg.eigh_tridiagonal(MD, OD)
    return EVAL, EVEC


def get_wf_array(xplot, min_ev, max_ev, evec):
    """Calculates the array of the wavefunctions in the\n
    x1 Psi1(x1) Psi2(x1)\n
    x2 Psi1(x2) Psi2(x2)\n
    format.

    Args:
        xplot (1darray): Array containing x values.
        min_ev (int): Lower bound of eigenvalues.
        max_ev (int): Upper bound of eigenvalues.
        evec (ndarray): Array of the eigenvectors.

    Returns:
        ndarray: Array in the described format.
    """
    delta = abs(xplot[0] - xplot[1])
    wf_array = np.array([xplot])
    for ii in range(min_ev - 1, max_ev):
        norm = np.sqrt(delta * np.sum(evec[:, ii] * evec[:, ii]))
        evec[:, ii] /= norm
        wf_array = np.vstack((wf_array, evec[:, ii]))
    wf_array = np.transpose(wf_array)
    return wf_array


def expected_values(xplot, evec, min_ev, max_ev):
    """Calculates the expected value of the position.

    Args:
        xplot (1darray): x values.
        evec (ndarray): Array of the eigenvectors to calculate the expected
          position of.
        min_ev (int): Lower bound of the eigenvalues.
        max_ev (int): Upper bound of the eigenvalues.

    Returns:
        1darray: Array containing expected values of the position.
    """
    delta = abs(xplot[0] - xplot[1])
    expectedx = np.zeros((max_ev - min_ev + 1, ), dtype=float)
    for ii in range(min_ev - 1, max_ev):
        xx = delta * np.sum(evec[:, ii] * xplot * evec[:, ii])
        expectedx[ii] = xx
    return expectedx


def expected_x_square(xplot, evec, min_ev, max_ev):
    """Calculates the expected values of the square of the position.

    Args:
        xplot (1darray): x values.
        evec (ndarray): Array containing the wavefunctions.
        min_ev (int): Lower bound of eigenvalues to calculate the expected
            square positions of.
        max_ev (int): Upper bound of eigenvalues to calculate the expected
            square postions of.

    Returns:
        1darray: Array containing the expected values of the square
            position from the minEV eigenvalue to the maxEV eigenvalue.
    """
    delta = abs(xplot[0] - xplot[1])
    expx2 = np.zeros((max_ev - min_ev + 1, ), dtype=float)
    for ii in range(min_ev - 1, max_ev):
        xx = delta * np.sum(evec[:, ii] * xplot**2 * evec[:, ii])
        expx2[ii] = xx
    return expx2


def uncertainty(xplot, evec, min_ev, max_ev):
    """Calculates the uncertainty of the expected position.

    Args:
        xplot (1darray): x values.
        evec (ndarray): Array containing the wavefunctions.
        min_ev (int): Lower bound of eigenvalues to calculate the uncertainty
            of.
        max_ev (int): Upper bound of eigenvalues to calculate the uncertainty
            of.

    Retruns:
        1darray: Array containing the uncertainties of the expected positions.
    """
    expx = expected_values(xplot, evec, min_ev, max_ev)
    expx2 = expected_x_square(xplot, evec, min_ev, max_ev)
    uncert = np.sqrt(expx2 - expx * expx)
    return uncert


def get_exp_unc(expx, unc):
    """Combines the arrays of the expected values and the array of the
    corresponding uncertainties.

    Args:
        expx (1darray): Array containing the expected values.
        unc (1darray): Array containing the uncertainties corresponding to the
            expected values.

    Returns:
        2darray: Expected values as colum vector and uncertainties as second
          colum vector.
    """
    expvalues = np.vstack((expx, unc))
    expvalues = np.transpose(expvalues)
    return expvalues
