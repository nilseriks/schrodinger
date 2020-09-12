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
